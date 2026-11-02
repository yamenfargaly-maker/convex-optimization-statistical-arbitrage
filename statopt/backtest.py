"""Daily backtest: weights at t earn resid_fwd[t] over t -> t+1, minus costs of moving to them."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .costs import CostModel
from .panel import Panel
from .policies import Policy

TD = 252


def run(policy: Policy, panel: Panel, costs: CostModel, start: int = 0, end: int | None = None,
        warmup: int = 0) -> dict[str, np.ndarray]:
    """Run [start - warmup, end). Days in the warm-up build state (positions) but are not scored."""
    end = panel.T - 1 if end is None else min(end, panel.T - 1)
    t0 = max(0, start - warmup)
    policy.reset(panel)
    w_prev = np.zeros(panel.N)
    n = end - start
    gross = np.zeros(n); cost = np.zeros(n); turn = np.zeros(n); lev = np.zeros(n); npos = np.zeros(n)
    tic = time.perf_counter()
    for t in range(t0, end):
        w = policy.weights(t, panel, w_prev)
        dw = w - w_prev
        f = np.where(np.isfinite(panel.fwd[t]), panel.fwd[t], 0.0)
        if t >= start:
            i = t - start
            spr = None if panel.spread is None else panel.spread[t]
            cost[i] = costs.trade_cost(dw, spr) + costs.borrow_cost(w)
            gross[i] = float(w @ f)
            turn[i] = float(np.abs(dw).sum())
            lev[i] = float(np.abs(w).sum())
            npos[i] = int((np.abs(w) > 1e-6).sum())
        # drift weights with returns so tomorrow's trade is measured from the drifted book
        w_prev = w * (1.0 + f) / (1.0 + float(w @ f)) if np.isfinite(w @ f) else w
    return {"gross": gross, "cost": cost, "net": gross - cost, "turnover": turn, "leverage": lev,
            "n_positions": npos, "seconds": time.perf_counter() - tic}


def nw_tstat(r: np.ndarray, lags: int = 5) -> float:
    m = r.mean()
    x = r - m
    n = r.size
    v = np.dot(x, x) / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * np.dot(x[l:], x[:-l]) / n
    return float(np.sqrt(n) * m / np.sqrt(v)) if v > 0 else float("nan")


def sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(np.sqrt(TD) * r.mean() / s) if s > 0 else float("nan")


def summarize(res: dict[str, np.ndarray], label: str) -> dict:
    net, g = res["net"], res["gross"]
    eq = np.cumprod(1 + net)
    dd = eq / np.maximum.accumulate(eq) - 1
    turn = res["turnover"].mean()
    return {
        "strategy": label,
        "sharpe_net": sharpe(net),
        "sharpe_gross": sharpe(g),
        "ann_return_net": float(net.mean() * TD),
        "ann_vol": float(net.std(ddof=1) * np.sqrt(TD)),
        "max_drawdown": float(dd.min()),
        "turnover_annual": float(turn * TD),
        "cost_drag_annual": float(res["cost"].mean() * TD),
        "breakeven_bps_per_leg": float(1e4 * g.mean() / turn) if turn > 0 else float("nan"),
        "avg_gross_leverage": float(res["leverage"].mean()),
        "avg_positions": float(res["n_positions"].mean()),
        "nw_tstat_net": nw_tstat(net),
        "solve_ms_per_day": 1e3 * res["seconds"] / max(len(net), 1),
    }


def table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("strategy")
