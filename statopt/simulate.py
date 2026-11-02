"""Synthetic residual stat-arb panel in the exact format of statarb-screening's signals_*.npz.

Each name's cumulative residual is  X = Y + Z :
  Y  Ornstein-Uhlenbeck, half-lives drawn around the paper's pooled 22 days
  Z  a random walk (non-reverting drift the factor model fails to remove)
Shocks are correlated within clusters (residual "sector" structure the PCA
factors miss), which is what makes a good covariance estimate matter.

s-scores are then *estimated* exactly the way the pipeline does it: a rolling
60-day AR(1) fit of X, giving kappa_hat, m_hat, sigma_eq and s = (X - m)/sigma_eq.
Per-name kappa_hat is therefore as noisy as in the paper, while the pooled
reversion is real. Default parameters were calibrated so the paper's threshold
rule reproduces its headline numbers (gross Sharpe ~1.6, ~120x annual turnover,
net Sharpe ~1.1 at 2.5 bps per leg); see scripts/calibrate_sim.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel


@dataclass(frozen=True)
class SimConfig:
    n_names: int = 300
    n_days: int = 2520  # 10 years scored
    burn_in: int = 260
    reg_window: int = 60
    half_life_median: float = 22.0
    half_life_disp: float = 0.5  # lognormal sd of per-name half-lives
    ou_vol: tuple[float, float] = (0.008, 0.016)  # daily vol of the reverting part
    rw_ratio: float = 1.3  # random-walk vol as a fraction of OU vol
    n_clusters: int = 10
    cluster_loading: float = 0.45  # within-cluster shock correlation ~ loading^2
    seed: int = 20260818


def simulate(cfg: SimConfig = SimConfig()) -> Panel:
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_names, cfg.n_days + cfg.burn_in + cfg.reg_window
    hl = cfg.half_life_median * np.exp(cfg.half_life_disp * rng.standard_normal(N))
    phi = np.exp(-np.log(2) / hl)  # AR(1) coefficient of the OU part
    sig_y = rng.uniform(*cfg.ou_vol, size=N)
    sig_z = cfg.rw_ratio * sig_y
    cluster = rng.integers(0, cfg.n_clusters, size=N)
    load = cfg.cluster_loading

    def shocks():
        f = rng.standard_normal((T, cfg.n_clusters))
        e = rng.standard_normal((T, N))
        return load * f[:, cluster] + np.sqrt(1 - load ** 2) * e

    ey, ez = shocks(), shocks()
    Y = np.zeros((T, N)); Z = np.zeros((T, N))
    for t in range(1, T):
        Y[t] = phi * Y[t - 1] + sig_y * ey[t]
        Z[t] = Z[t - 1] + sig_z * ez[t]
    X = Y + Z

    L = cfg.reg_window
    s = np.full((T, N), np.nan); sig_eq = np.full((T, N), np.nan); kap = np.full((T, N), np.nan)
    for t in range(L, T):
        x0 = X[t - L:t]  # X_tau
        x1 = X[t - L + 1:t + 1]  # X_{tau+1}
        mx0, mx1 = x0.mean(0), x1.mean(0)
        b = ((x0 - mx0) * (x1 - mx1)).sum(0) / np.maximum(((x0 - mx0) ** 2).sum(0), 1e-18)
        a = mx1 - b * mx0
        res = x1 - (a + b * x0)
        good = (b > 0) & (b < 1)
        bb = np.where(good, b, 0.5)
        m = a / (1 - bb)
        seq = res.std(0) / np.sqrt(1 - bb ** 2)
        s[t] = np.where(good, (X[t] - m) / seq, np.nan)
        sig_eq[t] = np.where(good, seq, np.nan)
        kap[t] = np.where(good, -np.log(bb) * 252, np.nan)

    fwd = np.full((T, N), np.nan)
    fwd[:-1] = X[1:] - X[:-1]
    keep = slice(cfg.burn_in + L, T)
    s, sig_eq, fwd = s[keep], sig_eq[keep], fwd[keep]
    tradable = np.isfinite(s)
    dates = np.arange(s.shape[0])
    return Panel(s=s, fwd=fwd, tradable=tradable, sigma_eq=sig_eq, dates=dates)
