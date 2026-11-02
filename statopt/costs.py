"""Transaction-cost model used for *evaluation* (the optimizer gets its own copy as a parameter).

Matches the screening paper's cost-sensitivity table: a flat cost per leg in
basis points of traded notional (2.5 bps per leg = 5 bps round trip is the
headline), plus optional short-borrow and optional per-name spread costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    bps_per_leg: float = 2.5
    borrow_bps_annual: float = 0.0
    spread_capture: float = 0.5  # fraction of quoted spread paid when a spread panel is supplied

    def per_name_linear(self, n: int, spread: np.ndarray | None = None) -> np.ndarray:
        c = np.full(n, self.bps_per_leg * 1e-4)
        if spread is not None:
            c = c + self.spread_capture * np.where(np.isfinite(spread), spread, 0.0)
        return c

    def trade_cost(self, dw: np.ndarray, spread: np.ndarray | None = None) -> float:
        return float(np.sum(self.per_name_linear(dw.size, spread) * np.abs(dw)))

    def borrow_cost(self, w: np.ndarray) -> float:
        return float(self.borrow_bps_annual * 1e-4 / 252 * np.sum(np.maximum(-w, 0.0)))
