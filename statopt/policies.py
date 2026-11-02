"""Portfolio construction rules. Each maps (date t, panel, current weights) -> target weights."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alpha import POOLED_HALF_LIFE_DAYS, AlphaModel, alpha_uncertainty, ou_expected_return
from .costs import CostModel
from .optimizer import OptConfig, PortfolioOptimizer
from .panel import Panel
from .risk import estimate


class Policy:
    name = "policy"

    def reset(self, panel: Panel) -> None:
        pass

    def weights(self, t: int, panel: Panel, w_prev: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ThresholdPolicy(Policy):
    """Avellaneda-Lee bang-bang rule, as in the screening paper's backtest:
    open long below -1.25, close long above -0.50, open short above +1.25,
    close short below +0.75; equal weight at gross/n_active, capped at max_weight."""

    open_long: float = -1.25
    open_short: float = 1.25
    close_long: float = -0.50
    close_short: float = 0.75
    gross: float = 2.0
    max_weight: float = 0.02
    name: str = "threshold (Avellaneda-Lee)"

    def reset(self, panel: Panel) -> None:
        self.pos = np.zeros(panel.N, dtype=np.int8)

    def weights(self, t, panel, w_prev):
        s, ok = panel.s[t], panel.tradable[t]
        pos = self.pos
        s = np.where(np.isfinite(s), s, 0.0)
        new = pos.copy()
        flat = pos == 0
        new[flat & (s < self.open_long)] = 1
        new[flat & (s > self.open_short)] = -1
        new[(pos == 1) & (s > self.close_long)] = 0
        new[(pos == -1) & (s < self.close_short)] = 0
        new[~ok] = 0
        self.pos = new
        n_act = int((new != 0).sum())
        if n_act == 0:
            return np.zeros(panel.N)
        return new.astype(float) * min(self.gross / n_act, self.max_weight)


@dataclass
class LinearPolicy(Policy):
    """Weights proportional to the OU alpha, dollar-neutral, scaled to gross; no optimizer."""

    gross: float = 2.0
    max_weight: float = 0.02
    name: str = "linear in alpha"

    def weights(self, t, panel, w_prev):
        ok = panel.tradable[t]
        a = np.where(ok, ou_expected_return(panel.s[t], panel.sigma_eq[t], 10.0), 0.0)  # same shaped forecast
        a[ok] -= a[ok].mean()
        g = np.abs(a).sum()
        if g == 0:
            return np.zeros(panel.N)
        w = np.clip(a * self.gross / g, -self.max_weight, self.max_weight)
        return w


class OptimizerPolicy(Policy):
    def __init__(self, cfg: OptConfig, costs: CostModel, name: str = "optimizer",
                 half_life: float = POOLED_HALF_LIFE_DAYS, s_noise: float = 1.0, alpha_scale: float = 1.0,
                 alpha_model: AlphaModel | None = None):
        self.cfg, self.costs, self.name = cfg, costs, name
        self.alpha_scale = alpha_scale
        self.alpha_model = alpha_model
        self.half_life, self.s_noise = half_life, s_noise
        self.opt: PortfolioOptimizer | None = None

    def reset(self, panel: Panel) -> None:
        k = self.cfg.cov_window if self.cfg.cov_method in ("sample", "ledoit_wolf") else self.cfg.max_factors
        self.opt = PortfolioOptimizer(self.cfg, panel.N, max(k, 1))
        self.cov_shrinkage: list[float] = []

    def weights(self, t, panel, w_prev):
        cfg = self.cfg
        ok = panel.tradable[t] & np.isfinite(panel.s[t])
        if self.alpha_model is not None:
            am = self.alpha_model.at(t) if hasattr(self.alpha_model, "at") else self.alpha_model
            alpha = np.where(ok, am(panel.s[t], panel.sigma_eq[t]), 0.0)
            sa = np.where(ok, am.uncertainty(panel.s[t], panel.sigma_eq[t]), 0.0)
        else:
            alpha = self.alpha_scale * np.where(
                ok, ou_expected_return(panel.s[t], panel.sigma_eq[t], cfg.horizon, self.half_life), 0.0)
            sa = abs(self.alpha_scale) * np.where(
                ok, alpha_uncertainty(panel.sigma_eq[t], cfg.horizon, self.half_life, self.s_noise), 0.0)
        R = panel.realized_resid(t, cfg.cov_window)
        rm = estimate(R, cfg.cov_method, cfg.max_factors)
        self.cov_shrinkage.append(rm.shrinkage)
        ub = np.where(ok, cfg.max_weight, 0.0)
        c = self.costs.per_name_linear(panel.N, None if panel.spread is None else panel.spread[t])
        return self.opt.solve(alpha, w_prev, rm.G, rm.d, ub, c, sa)
