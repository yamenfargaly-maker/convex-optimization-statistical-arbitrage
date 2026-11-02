"""The daily portfolio problem (a convex QP/SOCP), compiled once with cvxpy and re-solved
every day with new parameter values (Disciplined Parametrized Programming).

    maximize   alpha' w                                  expected H-day residual return
             - lam * H * (||G w||^2 + sum_i d_i w_i^2)   risk (shrinkage covariance, H-day)
             - tc_mult * sum_i c_i |w_i - w0_i|          transaction costs (linear, per name)
             - rho * ||w - w0||_1                        extra turnover penalty
             - u * sum_i (sa_i w_i)^2                    alpha-uncertainty (robust) penalty
    subject to ||w||_1 <= gross                          gross leverage
               |w_i| <= ub_i                             position limit (0 for untradable names)
               1' w = 0                                  dollar neutral (optional)
               ||w - w0||_1 <= turnover_cap              daily turnover limit (optional)
               ||G w||^2 + sum d_i w_i^2 <= vol_cap^2    ex-ante risk limit (optional)

Without the cost terms the solution jumps to a new portfolio every day. With
them, the optimum has a "no-trade region" around the current book: a name is
only traded when its marginal alpha exceeds its marginal cost, which is how
the optimizer cuts turnover without throwing away the signal.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import warnings

import cvxpy as cp
import numpy as np

warnings.filterwarnings("ignore", message="Solution may be inaccurate")


@dataclass(frozen=True)
class OptConfig:
    horizon: float = 10.0  # days of OU reversion the alpha forecast covers
    risk_aversion: float = 1.0
    tc_mult: float = 1.0  # >1 overweights costs (amortization / alpha decay)
    turnover_penalty: float = 0.0  # extra L1 penalty on trades, in return units
    alpha_uncertainty: float = 0.0  # weight on the robust penalty (0 = off)
    gross: float = 2.0
    max_weight: float = 0.02
    dollar_neutral: bool = True
    turnover_cap: float | None = None  # max one-way daily turnover (sum |dw|)
    vol_cap_annual: float | None = None  # ex-ante annualized vol limit
    cov_method: str = "pca_mp"
    cov_window: int = 120
    max_factors: int = 20
    use_costs: bool = True
    solver: str = "auto"  # OSQP for pure QPs, CLARABEL when a risk cap makes it an SOCP

    def with_(self, **kw) -> "OptConfig":
        return replace(self, **kw)


class PortfolioOptimizer:
    def __init__(self, cfg: OptConfig, n: int, k: int):
        self.cfg, self.n, self.k = cfg, n, k
        self.alpha = cp.Parameter(n)
        self.w0 = cp.Parameter(n)
        self.G = cp.Parameter((k, n))
        self.sqrt_d = cp.Parameter(n, nonneg=True)
        self.ub = cp.Parameter(n, nonneg=True)
        self.c = cp.Parameter(n, nonneg=True)
        self.sa = cp.Parameter(n, nonneg=True)
        self.w = cp.Variable(n)

        w = self.w
        dw = w - self.w0
        trade = cp.Variable(n, nonneg=True)  # trade >= |w - w0|  (keeps the problem DPP)
        y = cp.Variable(k)  # factor exposures, y = G w
        risk = cp.sum_squares(y) + cp.sum_squares(cp.multiply(self.sqrt_d, w))
        obj = self.alpha @ w - cfg.risk_aversion * cfg.horizon * risk
        cons = [y == self.G @ w, trade >= dw, trade >= -dw,
                cp.norm1(w) <= cfg.gross, w <= self.ub, w >= -self.ub]
        if cfg.use_costs:
            obj -= cfg.tc_mult * (self.c @ trade)
            if cfg.turnover_penalty > 0:
                obj -= cfg.turnover_penalty * cp.sum(trade)
        if cfg.alpha_uncertainty > 0:
            obj -= cfg.alpha_uncertainty * cp.sum_squares(cp.multiply(self.sa, w))
        if cfg.dollar_neutral:
            cons.append(cp.sum(w) == 0)
        if cfg.turnover_cap is not None:
            cons.append(cp.sum(trade) <= cfg.turnover_cap)
        if cfg.vol_cap_annual is not None:
            cons.append(risk <= (cfg.vol_cap_annual ** 2) / 252.0)
        self.problem = cp.Problem(cp.Maximize(obj), cons)
        assert self.problem.is_dpp()
        self.failures = 0

    def solve(self, alpha, w0, G, d, ub, c, sa=None) -> np.ndarray:
        n, k = self.n, self.k
        Gp = np.zeros((k, n))
        kk = min(k, G.shape[0])
        Gp[:kk, : G.shape[1]] = G[:kk]
        self.alpha.value = alpha
        self.w0.value = w0
        self.G.value = Gp
        self.sqrt_d.value = np.sqrt(np.maximum(d, 0.0))
        self.ub.value = ub
        self.c.value = c
        self.sa.value = np.zeros(n) if sa is None else sa
        try:
            solver = self.cfg.solver
            if solver == "auto":
                solver = "CLARABEL" if self.cfg.vol_cap_annual is not None else "OSQP"
            kw = dict(eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, polishing=False) if solver == "OSQP" else {}
            self.problem.solve(solver=solver, warm_start=True, **kw)
            ok = self.problem.status in ("optimal", "optimal_inaccurate") and self.w.value is not None
        except cp.error.SolverError:
            ok = False
        if not ok:
            self.failures += 1
            return np.clip(w0, -ub, ub)  # hold (within limits) if the solve fails
        w = np.clip(np.asarray(self.w.value), -ub, ub)  # remove solver-tolerance overshoot
        w[np.abs(w) < 1e-6] = 0.0
        return w
