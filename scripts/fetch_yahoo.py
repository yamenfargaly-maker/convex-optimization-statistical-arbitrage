"""Download daily OHLCV for current S&P 500 / 400 / 600 constituents from Yahoo Finance.

Run on your own machine or in Colab (needs internet):

    pip install yfinance pandas pyarrow lxml
    python fetch_yahoo.py                                  # S&P 1500 -> data/yahoo_sp1500.parquet
    python fetch_yahoo.py --index sp500                    # large caps only
    python fetch_yahoo.py --index sp400 sp600 --start 2010-01-01

Output (long format, one row per ticker-day):
    date, ticker, open, high, low, close, adj_close, volume
plus data/constituents.csv (ticker, name, GICS sector, sub-industry, index).

CAVEAT: this is *today's* index membership applied to the past, so the sample is
survivorship-biased (no delisted names). Fine for comparing portfolio-construction
rules on the same data; not fine for quoting an absolute strategy Sharpe.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd

WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
FALLBACK_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def _pick_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """The constituents table is the first one with a ticker column and a few hundred rows."""
    for t in tables:
        cols = {str(c).strip().lower(): c for c in t.columns}
        tick = next((cols[k] for k in cols if k in ("symbol", "ticker", "ticker symbol")), None)
        if tick is not None and len(t) >= 100:
            ren = {tick: "ticker"}
            for k, v in cols.items():
                if k in ("security", "company", "name"):
                    ren[v] = "name"
                elif k in ("gics sector", "sector"):
                    ren[v] = "sector"
                elif k in ("gics sub-industry", "sub-industry"):
                    ren[v] = "sub_industry"
            return t.rename(columns=ren)
    raise ValueError("no constituents table found")


def constituents(indexes: list[str]) -> pd.DataFrame:
    import requests

    frames = []
    for ix in indexes:
        try:
            html = requests.get(WIKI[ix], headers=UA, timeout=30).text
            df = _pick_table(pd.read_html(io.StringIO(html)))
        except Exception as e:  # Wikipedia layout changes / lxml missing
            if ix != "sp500":
                print(f"{ix}: Wikipedia failed ({e}); skipping", file=sys.stderr)
                continue
            print(f"sp500: Wikipedia failed ({e}); using GitHub fallback list", file=sys.stderr)
            df = _pick_table([pd.read_csv(FALLBACK_CSV)])
        df["index"] = ix
        print(f"{ix}: {len(df)} constituents", flush=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()  # BRK.B -> BRK-B
    cols = [c for c in ("ticker", "name", "sector", "sub_industry", "index") if c in df.columns]
    return df[cols].drop_duplicates("ticker").reset_index(drop=True)


def download(tickers: list[str], start: str, end: str | None, chunk: int, retries: int) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        for attempt in range(retries):
            try:
                raw = yf.download(batch, start=start, end=end, auto_adjust=False, actions=False,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"  batch {i // chunk + 1}: {e}; retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
        else:
            print(f"  batch {i // chunk + 1} failed after {retries} tries; skipping", file=sys.stderr)
            continue
        if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
            raw.columns = pd.MultiIndex.from_product([[batch[0]], raw.columns])
        long = raw.stack(level=0, future_stack=True).reset_index()
        long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
        long = long.rename(columns={"level_1": "ticker", "price": "ticker"})
        frames.append(long)
        got = long.dropna(subset=["close"])["ticker"].nunique() if "close" in long else 0
        print(f"  batch {i // chunk + 1}/{-(-len(tickers) // chunk)}: {got}/{len(batch)} tickers", flush=True)
        time.sleep(1.0)  # be polite to Yahoo
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"adj_close": "adj_close", "adjclose": "adj_close"})
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["close"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="+", default=["sp500", "sp400", "sp600"], choices=list(WIKI))
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--extra", nargs="*", default=["SPY"], help="extra tickers (e.g. SPY for the market)")
    a = ap.parse_args(args=[]) if "ipykernel" in sys.modules else ap.parse_args()  # works in Colab/Jupyter too

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = constituents(a.index)
    meta.to_csv(out / "constituents.csv", index=False)
    tag = "sp1500" if set(a.index) == set(WIKI) else "_".join(a.index)
    tickers = list(meta["ticker"]) + [t for t in a.extra if t not in set(meta["ticker"])]
    print(f"{len(tickers)} tickers, {a.start} -> {a.end or 'today'}", flush=True)

    t0 = time.time()
    df = download(tickers, a.start, a.end, a.chunk, a.retries)
    try:
        path = out / f"yahoo_{tag}.parquet"
        df.to_parquet(path, index=False)
    except Exception:  # no pyarrow
        path = out / f"yahoo_{tag}.csv.gz"
        df.to_csv(path, index=False)

    missing = sorted(set(tickers) - set(df["ticker"]))
    print(f"\nsaved {len(df):,} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} -> {df['date'].max().date()} to {path}  ({time.time() - t0:.0f}s)")
    if missing:
        print(f"no data for {len(missing)}: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    main()
