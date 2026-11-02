"""Build Avellaneda-Lee residual s-scores from Yahoo OHLCV, in the signals_*.npz format.

    python scripts/build_signals_yahoo.py data/yahoo_sp500.parquet --out data/signals_yahoo.npz

Same construction as the screening paper, on whatever universe the file contains:
  * returns from adjusted closes; universe on day t = names with >=95% of the trailing
    252 days present, unadjusted price > $5, 20-day median dollar volume > $1m
  * factors: first 15 eigenportfolios of the trailing 252-day correlation matrix,
    weights scaled by inverse volatility, re-estimated daily
  * residuals: 60-day regression of returns on factor returns (with intercept);
    X = cumulative residual; AR(1) fit gives kappa, m, sigma_eq
  * s = -(m - mean m) / sigma_eq  (Avellaneda-Lee, m centered across the universe);
    X_L is 0 by construction because in-window residuals sum to zero (the paper's
    Brownian-bridge point), so s is driven entirely by the fitted intercept m
  * resid_fwd[t] = r_{t+1} - beta' F_{t+1} with eigenportfolios and betas fixed at t
No screen on kappa: tradable = a usable AR(1) fit (0 < b < 1), per the paper's finding.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

TD = 252


def load_long(path: str | Path) -> pd.DataFrame:
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def panels(df: pd.DataFrame, drop: tuple[str, ...] = ("SPY",)):
    df = df[~df["ticker"].isin(drop)]
    wide = lambda c: df.pivot_table(index="date", columns="ticker", values=c, aggfunc="last").sort_index()
    adj, close, vol = wide("adj_close"), wide("close"), wide("volume")
    tickers = adj.columns
    close, vol = close.reindex(columns=tickers), vol.reindex(columns=tickers)
    ret = adj.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 0.5)  # drop bad prints / unadjusted splits
    dv = (close * vol).rolling(20, min_periods=10).median()
    return ret.to_numpy(float), close.to_numpy(float), dv.to_numpy(float), adj.index.to_numpy(), np.asarray(tickers)


def build(df: pd.DataFrame, corr_window: int = 252, reg_window: int = 60, n_factors: int = 15,
          min_names: int = 50, min_price: float = 5.0, min_dollar_volume: float = 1e6, verbose: bool = True):
    R, close, dv, dates, tickers = panels(df)
    T, N = R.shape
    s_out = np.full((T, N), np.nan, np.float32)
    fwd = np.full((T, N), np.nan, np.float32)
    sig_out = np.full((T, N), np.nan, np.float32)
    kap_out = np.full((T, N), np.nan, np.float32)
    tradable = np.zeros((T, N), bool)
    var_share = np.full(T, np.nan, np.float32)
    M, L = corr_window, reg_window
    t0 = time.time()
    for t in range(M, T - 1):
        win = R[t - M + 1:t + 1]
        ok = (np.isfinite(win).mean(0) >= 0.95) & (close[t] > min_price) & (dv[t] > min_dollar_volume) \
            & np.isfinite(R[t])
        idx = np.where(ok)[0]
        if idx.size < min_names:
            continue
        w = np.nan_to_num(win[:, idx])
        mu, sd = w.mean(0), w.std(0)
        good = sd > 0
        idx, w, mu, sd = idx[good], w[:, good], mu[good], sd[good]
        Y = (w - mu) / sd  # (M, n) standardized
        # top eigenvectors of the n x n correlation matrix via the M x M dual problem
        # (same eigenvalues; 252^3 instead of n^3 work when n >> 252)
        ev, U = np.linalg.eigh(Y @ Y.T / M)
        ev, U = np.maximum(ev[::-1], 0.0), U[:, ::-1]
        k = min(n_factors, idx.size - 1, M - 1)
        V = (Y.T @ U[:, :k]) / np.sqrt(M * np.maximum(ev[:k], 1e-12))
        ev_total = float((Y * Y).sum() / M)  # trace of the correlation matrix
        Q = V[:, :k] / sd[:, None]  # eigenportfolio weights, inverse-vol scaled
        Q = Q / np.abs(Q).sum(0, keepdims=True)
        F = w @ Q  # (M, k) factor returns
        Rr, Fr = w[-L:], F[-L:]
        A = np.column_stack([np.ones(L), Fr])  # (L, k+1)
        beta, *_ = np.linalg.lstsq(A, Rr, rcond=None)  # (k+1, n)
        eps = Rr - A @ beta
        X = np.cumsum(eps, axis=0)  # (L, n)
        x0, x1 = X[:-1], X[1:]
        m0, m1 = x0.mean(0), x1.mean(0)
        b = ((x0 - m0) * (x1 - m1)).sum(0) / np.maximum(((x0 - m0) ** 2).sum(0), 1e-18)
        a = m1 - b * m0
        z = x1 - (a + b * x0)
        fit = (b > 0) & (b < 1)
        bb = np.where(fit, b, 0.5)
        m = a / (1 - bb)
        seq = np.sqrt(z.var(0) / (1 - bb ** 2))
        m_c = m - np.nanmean(np.where(fit, m, np.nan))
        s = np.where(fit, (X[-1] - m_c) / seq, np.nan)

        r_next = np.nan_to_num(R[t + 1, idx])
        f_next = r_next @ Q
        fwd[t, idx] = r_next - beta[1:].T @ f_next
        s_out[t, idx] = s
        sig_out[t, idx] = np.where(fit, seq, np.nan)
        kap_out[t, idx] = np.where(fit, -np.log(bb) * TD, np.nan)
        tradable[t, idx] = fit
        var_share[t] = ev[:k].sum() / ev_total
        if verbose and (t - M) % 250 == 0:
            print(f"  {str(dates[t])[:10]}  n={idx.size}  fit={fit.mean():.0%}  var15={var_share[t]:.0%}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return dict(s_scores=s_out, resid_fwd=fwd, tradable=tradable, sigma_eq=sig_out, kappa=kap_out,
                var_share=var_share, dates=np.array([str(d)[:10] for d in dates]), tickers=tickers.astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="yahoo_sp500.parquet (or .csv.gz) from fetch_yahoo.py")
    ap.add_argument("--out", default="data/signals_yahoo.npz")
    a = ap.parse_args()
    df = load_long(a.path)
    print(f"{df['ticker'].nunique()} tickers, {df['date'].min().date()} -> {df['date'].max().date()}", flush=True)
    sig = build(df)
    first = int(np.argmax(sig["tradable"].any(1)))
    keep = slice(first, None)  # drop the factor burn-in
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == sig["dates"].shape[0] else v)
           for k, v in sig.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    tr = out["tradable"]
    print(f"saved {a.out}: {tr.shape[0]} days x {tr.shape[1]} names, {int(tr.sum()):,} stock-days with signals, "
          f"mean names/day {tr.sum(1).mean():.0f}")


if __name__ == "__main__":
    main()
