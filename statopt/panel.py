"""Signal panels in the format written by statarb-screening's ``build_signals.py``.

Arrays, shape (T, N) unless noted:
    s_scores   OU s-score at date t (data through t)
    resid_fwd  factor-neutral return from t to t+1 of a unit long opened at t
    tradable   bool, name may be traded at t
    sigma_eq   OU equilibrium standard deviation of the cumulative residual
    dates      (T,)
Optional cost inputs (same shape): ``spread`` (proportional quoted spread).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Panel:
    s: np.ndarray
    fwd: np.ndarray
    tradable: np.ndarray
    sigma_eq: np.ndarray
    dates: np.ndarray
    spread: np.ndarray | None = None

    @property
    def T(self) -> int:
        return self.s.shape[0]

    @property
    def N(self) -> int:
        return self.s.shape[1]

    def realized_resid(self, t: int, window: int) -> np.ndarray:
        """Residual returns realized up to date t (rows t-window .. t-1 of resid_fwd)."""
        lo = max(0, t - window)
        return self.fwd[lo:t]

    def split(self, frac: float) -> int:
        return int(self.T * frac)


def load_npz(path: str | Path) -> Panel:
    z = np.load(path, allow_pickle=True)
    s = z["s_scores"].astype(float)
    sig = z["sigma_eq"].astype(float) if "sigma_eq" in z.files else np.ones_like(s)
    dates = z["dates"] if "dates" in z.files else np.arange(s.shape[0])
    return Panel(
        s=s,
        fwd=z["resid_fwd"].astype(float),
        tradable=z["tradable"].astype(bool) & np.isfinite(s) & np.isfinite(sig),
        sigma_eq=sig,
        dates=dates,
        spread=z["spread"].astype(float) if "spread" in z.files else None,
    )


def save_npz(panel: Panel, path: str | Path, **extra: np.ndarray) -> None:
    arrays = dict(s_scores=panel.s.astype(np.float32), resid_fwd=panel.fwd.astype(np.float32),
                  tradable=panel.tradable, sigma_eq=panel.sigma_eq.astype(np.float32), dates=panel.dates)
    if panel.spread is not None:
        arrays["spread"] = panel.spread.astype(np.float32)
    np.savez_compressed(path, **arrays, **extra)
