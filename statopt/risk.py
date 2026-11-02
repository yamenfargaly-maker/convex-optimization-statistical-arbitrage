"""Covariance estimators for residual returns, all returned in low-rank-plus-diagonal form

    Sigma = G^T G + diag(d)          G: (k, n),  d: (n,)

which keeps the optimizer's risk term cheap (sum of squares of k + n terms)
regardless of which estimator produced it.

With n ~ 300-600 names and a 60-250 day window, the sample covariance is
singular or close to it, and an optimizer will happily load up on the
spurious zero-variance directions. Shrinkage fixes that:

* ``ledoit_wolf``  Ledoit & Wolf (2004) optimal shrinkage toward a scaled identity.
* ``pca_mp``       keep eigenvalues above the Marchenko-Pastur edge (signal), treat
                   the rest as idiosyncratic noise (random-matrix "clipping").
* ``diagonal``     ignore correlations entirely.
* ``sample``       no shrinkage (the baseline that goes wrong).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskModel:
    G: np.ndarray  # (k, n)
    d: np.ndarray  # (n,)
    method: str
    shrinkage: float = float("nan")  # LW intensity, or number of factors kept for PCA

    def cov(self) -> np.ndarray:
        return self.G.T @ self.G + np.diag(self.d)

    def variance(self, w: np.ndarray) -> float:
        return float(np.sum((self.G @ w) ** 2) + np.sum(self.d * w * w))


def _clean(R: np.ndarray) -> np.ndarray:
    R = np.where(np.isfinite(R), R, 0.0)
    return R - R.mean(axis=0, keepdims=True)


def ledoit_wolf_intensity(X: np.ndarray) -> tuple[float, float]:
    """Return (delta, mu): Sigma_LW = delta*mu*I + (1-delta)*S, S = X'X/L (X centered)."""
    L, n = X.shape
    S = X.T @ X / L
    mu = np.trace(S) / n
    d2 = np.sum((S - mu * np.eye(n)) ** 2)
    # b_bar^2 = (1/L^2) sum_t || x_t x_t' - S ||_F^2, without forming n x n per row
    sq = np.sum(X * X, axis=1)  # ||x_t||^2
    xSx = np.sum((X @ S) * X, axis=1)
    b_bar2 = np.sum(sq ** 2 - 2 * xSx + np.sum(S * S)) / L ** 2
    b2 = min(b_bar2, d2)
    delta = float(b2 / d2) if d2 > 0 else 1.0
    return delta, float(mu)


def estimate(R: np.ndarray, method: str = "ledoit_wolf", max_factors: int = 20) -> RiskModel:
    """R: (L, n) trailing residual returns (rows = days)."""
    X = _clean(R)
    L, n = X.shape
    var = np.maximum(X.var(axis=0), 1e-10)
    if method == "diagonal" or L < 5:
        return RiskModel(np.zeros((0, n)), var, "diagonal")
    if method == "sample":
        return RiskModel(X / np.sqrt(L), np.full(n, 1e-12), "sample")
    if method == "ledoit_wolf":
        delta, mu = ledoit_wolf_intensity(X)
        return RiskModel(np.sqrt((1 - delta) / L) * X, np.full(n, delta * mu), "ledoit_wolf", delta)
    if method == "pca_mp":
        # eigen-decomposition of the correlation matrix via SVD of standardized returns
        sd = np.sqrt(var)
        Z = X / sd
        _, sv, Vt = np.linalg.svd(Z / np.sqrt(L), full_matrices=False)
        lam = sv ** 2
        edge = (1 + np.sqrt(n / L)) ** 2  # MP upper edge for unit-variance noise
        k = int(np.clip(np.sum(lam > edge), 1, max_factors))
        G = (np.sqrt(lam[:k])[:, None] * Vt[:k]) * sd[None, :]
        spec = var - np.sum(G * G, axis=0)
        d = np.maximum(spec, 0.05 * np.median(var))
        return RiskModel(G, d, "pca_mp", float(k))
    raise ValueError(f"unknown covariance method {method!r}")
