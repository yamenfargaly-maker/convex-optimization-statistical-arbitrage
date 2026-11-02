"""Expected residual returns from OU s-scores.

For an OU cumulative residual  dX = kappa (m - X) dt + sigma dW  with
s = (X - m) / sigma_eq, the expected residual return over the next H days is

    E[X_{t+H} - X_t] = -(1 - exp(-kappa H)) * s * sigma_eq.

The screening paper's central finding is that kappa is *not identified name by
name* at a 60-day window, but a pooled half-life is (22 business days, 95% CI
20-26). So the default here is a single pooled kappa for every name rather than
the noisy per-name estimate, and no kappa screen.
"""
from __future__ import annotations

import numpy as np

POOLED_HALF_LIFE_DAYS = 22.0


S_CAP = 2.5  # s beyond this adds no further conviction
S_TRUST = 3.0  # |s| beyond this: treat as a regime break, not a reversion opportunity


def shape_s(s: np.ndarray, cap: float = S_CAP, trust: float = S_TRUST) -> np.ndarray:
    """Clip moderate s-scores and zero out extreme ones.

    Empirically (and in the synthetic panel) forward residual returns are monotone in s
    for |s| < ~2.5 but *reverse sign* beyond ~3: a name that far from its 60-day mean
    has usually broken out (its "mean" moved), so a linear forecast that keeps
    scaling up with |s| bets hardest exactly where reversion has stopped working."""
    return np.where(np.abs(s) <= trust, np.clip(s, -cap, cap), 0.0)


def robust_sigma(sigma_eq: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    """Winsorize sigma_eq cross-sectionally. When the fitted AR(1) coefficient is near 1,
    sigma_eq = sd / sqrt(1 - b^2) explodes; left alone, a handful of such names dominate
    both the forecast and any regression used to calibrate it."""
    v = sigma_eq[np.isfinite(sigma_eq)]
    if v.size == 0:
        return sigma_eq
    a, b = np.percentile(v, [lo, hi])
    return np.clip(sigma_eq, a, b)


def ou_expected_return(s: np.ndarray, sigma_eq: np.ndarray, horizon: float,
                       half_life: float = POOLED_HALF_LIFE_DAYS) -> np.ndarray:
    kappa = np.log(2.0) / half_life
    a = -(1.0 - np.exp(-kappa * horizon)) * shape_s(s) * robust_sigma(sigma_eq)
    return np.where(np.isfinite(a), a, 0.0)


def alpha_uncertainty(sigma_eq: np.ndarray, horizon: float, half_life: float = POOLED_HALF_LIFE_DAYS,
                      s_noise: float = 1.0) -> np.ndarray:
    """Std. error of the alpha forecast if the s-score itself is measured with noise ``s_noise``.

    Used for a robust (uncertainty-penalized) objective: positions in names whose
    forecast is noisiest are shrunk the most.
    """
    kappa = np.log(2.0) / half_life
    u = (1.0 - np.exp(-kappa * horizon)) * s_noise * robust_sigma(sigma_eq)
    return np.where(np.isfinite(u), u, 0.0)


def fit_alpha_scale(s: np.ndarray, sigma_eq: np.ndarray, fwd: np.ndarray, tradable: np.ndarray,
                    start: int, end: int, horizon: int, half_life: float = POOLED_HALF_LIFE_DAYS,
                    step: int = 5) -> float:
    """Pooled no-intercept regression of realized H-day residual returns on the OU forecast,
    over [start, end - horizon). The slope is the forecast's realized "IC-style" shrinkage:
    the raw OU formula overstates the edge because s-scores are noisy, and the optimizer
    can only trade off alpha against costs correctly if alpha is in the right units."""
    num = den = 0.0
    csum = np.vstack([np.zeros(fwd.shape[1]), np.nancumsum(np.where(np.isfinite(fwd), fwd, 0.0), axis=0)])
    for t in range(start, end - horizon, step):
        ok = tradable[t] & np.isfinite(s[t]) & np.isfinite(sigma_eq[t])
        a = ou_expected_return(s[t, ok], sigma_eq[t, ok], horizon, half_life)
        r = csum[t + horizon, ok] - csum[t, ok]
        num += float(a @ r)
        den += float(a @ a)
    return num / den if den > 0 else 0.0


class AlphaModel:
    """Maps (s, sigma_eq) -> expected H-day residual return.

    kind="binned" (default): nonparametric. On the training window, average the realized
        H-day residual return, in units of (winsorized) sigma_eq, within s-score bins, then
        interpolate between bin centers. This learns *both* the reversion zone and the
        breakout/continuation zone at |s| > ~3, and puts alpha in the right units so the
        optimizer's alpha-vs-cost trade-off is meaningful.
    kind="ou": the closed-form OU forecast with a trust region, times a fitted scale.
    """

    def __init__(self, kind: str = "binned", horizon: int = 10, half_life: float = POOLED_HALF_LIFE_DAYS,
                 edges: tuple[float, ...] = (-6, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6),
                 shrink_n: float = 2000.0):
        self.kind, self.horizon, self.half_life = kind, horizon, half_life
        self.edges = np.asarray(edges, dtype=float)
        self.shrink_n = shrink_n
        self.scale = 1.0
        self.centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.g = np.zeros(len(self.centers))
        self.g_se = np.zeros(len(self.centers))

    def fit(self, s, sigma_eq, fwd, tradable, start: int, end: int, step: int = 1) -> "AlphaModel":
        H = self.horizon
        if self.kind == "ou":
            self.scale = fit_alpha_scale(s, sigma_eq, fwd, tradable, start, end, H, self.half_life, step)
            return self
        csum = np.vstack([np.zeros(fwd.shape[1]), np.nancumsum(np.where(np.isfinite(fwd), fwd, 0.0), axis=0)])
        nb = len(self.centers)
        sums, sq, cnt = np.zeros(nb), np.zeros(nb), np.zeros(nb)
        for t in range(start, end - H, step):
            ok = tradable[t] & np.isfinite(s[t]) & np.isfinite(sigma_eq[t])
            sig = robust_sigma(sigma_eq[t])[ok]
            z = (csum[t + H, ok] - csum[t, ok]) / sig
            b = np.clip(np.digitize(s[t, ok], self.edges) - 1, 0, nb - 1)
            sums += np.bincount(b, z, nb); sq += np.bincount(b, z * z, nb); cnt += np.bincount(b, minlength=nb)
        mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
        var = np.where(cnt > 1, sq / np.maximum(cnt, 1) - mean ** 2, 0.0)
        self.g = mean * cnt / (cnt + self.shrink_n)  # shrink thin bins toward zero
        self.g_se = np.sqrt(np.maximum(var, 0) / np.maximum(cnt, 1))
        self.counts = cnt
        return self

    def __call__(self, s: np.ndarray, sigma_eq: np.ndarray) -> np.ndarray:
        if self.kind == "ou":
            return self.scale * ou_expected_return(s, sigma_eq, self.horizon, self.half_life)
        x = np.clip(np.where(np.isfinite(s), s, 0.0), self.centers[0], self.centers[-1])
        a = np.interp(x, self.centers, self.g) * robust_sigma(sigma_eq)
        return np.where(np.isfinite(a) & np.isfinite(s), a, 0.0)

    def uncertainty(self, s: np.ndarray, sigma_eq: np.ndarray) -> np.ndarray:
        if self.kind == "ou":
            return abs(self.scale) * alpha_uncertainty(sigma_eq, self.horizon, self.half_life)
        x = np.clip(np.where(np.isfinite(s), s, 0.0), self.centers[0], self.centers[-1])
        u = np.interp(x, self.centers, self.g_se) * robust_sigma(sigma_eq)
        return np.where(np.isfinite(u), u, 0.0)

    def table(self):
        import pandas as pd
        return pd.DataFrame({"s_lo": self.edges[:-1], "s_hi": self.edges[1:], "n": getattr(self, "counts", 0),
                             "mean_H_day_return_in_sigma_eq": self.g, "se": self.g_se})


class WalkForwardAlpha:
    """Re-fit an AlphaModel at the start of every calendar year on all data before it
    (expanding window, leaving an H-day gap so no forward return leaks across the refit).

    A static forecast fit once on 2015-2020 goes stale as the signal decays; refitting
    lets the optimizer *shrink its own positions* when the realized reversion weakens."""

    def __init__(self, s, sigma_eq, fwd, tradable, dates, first_fit_end: int, start: int = 130,
                 horizon: int = 10, min_years: int = 0, window_days: int | None = None, **kw):
        import pandas as pd

        years = pd.to_datetime(pd.Series(dates)).dt.year.to_numpy()
        refits = [first_fit_end] + [t for t in range(first_fit_end + 1, len(years)) if years[t] != years[t - 1]]
        self.points, self.models = [], []
        for t in refits:
            lo = start if window_days is None else max(start, t - window_days)
            m = AlphaModel("binned", horizon, **kw).fit(s, sigma_eq, fwd, tradable, lo, t - horizon)
            self.points.append(t)
            self.models.append(m)
        self.horizon = horizon

    def at(self, t: int) -> AlphaModel:
        i = max(0, int(np.searchsorted(self.points, t, side="right")) - 1)
        return self.models[i]

    def strength(self):
        """Average |forecast| slope in the reversion zone per refit, for reporting."""
        out = []
        for t, m in zip(self.points, self.models):
            core = (m.centers > -2) & (m.centers < 2)
            slope = -np.polyfit(m.centers[core], m.g[core], 1)[0]
            out.append((t, slope))
        return out
