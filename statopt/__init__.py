"""statopt: convex portfolio construction for residual statistical arbitrage.

Replaces the bang-bang s-score threshold rule (Avellaneda & Lee, 2010) with a
mean-variance optimizer that prices transaction costs, penalizes turnover,
enforces risk limits, and uses shrinkage covariance estimates.
"""
__version__ = "0.1.0"
