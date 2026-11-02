"""Real-data study on Yahoo S&P 1500 signals (scripts/build_signals_yahoo.py output).

Train 2015-01 -> 2020-11 (tune risk aversion and cost multiplier), test 2020-11 -> 2026-10.
Static forecast = fit once on train. Walk-forward = re-fit every January on all prior data.

    python scripts/run_real.py data/signals_sp1500.npz --out results_sp1500
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statopt.alpha import AlphaModel, WalkForwardAlpha  # noqa: E402
from statopt.backtest import run, summarize  # noqa: E402
from statopt.costs import CostModel  # noqa: E402
from statopt.optimizer import OptConfig  # noqa: E402
from statopt.panel import load_npz  # noqa: E402
from statopt.policies import LinearPolicy, OptimizerPolicy, ThresholdPolicy  # noqa: E402

P = None
MODELS: dict = {}
WARMUP = 20
H = 10
MAXW = 0.01  # 1% name cap; with ~1,300 names a 2% cap lets the book concentrate in ~100 names


def job(a):
    label, kind, cfg, bps, model_key, start, end = a
    cm = CostModel(bps)
    pol = {"threshold": ThresholdPolicy, "linear": LinearPolicy}.get(kind)
    pol = pol() if pol else OptimizerPolicy(cfg, cm, label, alpha_model=MODELS[model_key])
    res = run(pol, P, cm, start=start, end=end, warmup=WARMUP)
    row = summarize(res, label)
    row["bps_per_leg"] = bps
    return row, res["net"]


def pmap(jobs, procs):
    with Pool(procs) as pool:
        return pool.map(job, jobs)


def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--out", default=str(ROOT / "results_sp1500"))
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    log = lambda m: print(f"[{time.time() - t_all:5.0f}s] {m}", flush=True)

    P = load_npz(a.signals)
    T = P.T - 1
    split = T // 2
    train, test = (130, split), (split, T)
    d = lambda t: str(P.dates[min(t, P.T - 1)])[:10]
    log(f"panel {P.T}x{P.N}; train {d(train[0])}..{d(train[1])}; test {d(test[0])}..{d(test[1])}")

    MODELS["static"] = AlphaModel("binned", H).fit(P.s, P.sigma_eq, P.fwd, P.tradable, train[0], train[1] - H)
    MODELS["wf"] = WalkForwardAlpha(P.s, P.sigma_eq, P.fwd, P.tradable, P.dates, first_fit_end=split, horizon=H)
    MODELS["static"].table().to_csv(out / "tables" / "alpha_model_train.csv", index=False)
    log("alpha models fit")

    # 1. tune on train (static model == walk-forward model on train)
    grid = [(lam, tc) for lam in (4.0, 16.0, 64.0) for tc in (1.0, 2.0)]
    rows = [r for r, _ in pmap([(f"lam={l} tc={c}", "opt", OptConfig(horizon=H, risk_aversion=l, tc_mult=c, max_weight=MAXW),
                                 2.5, "static", *train) for l, c in grid], a.procs)]
    rows += [r for r, _ in pmap([("Threshold rule", "threshold", None, 2.5, None, *train)], 1)]
    tg = pd.DataFrame(rows)
    tg.to_csv(out / "tables" / "train_grid.csv", index=False)
    i = int(tg.iloc[:len(grid)]["sharpe_net"].idxmax())
    lam, tc = grid[i]
    tuned = OptConfig(horizon=H, risk_aversion=lam, tc_mult=tc, max_weight=MAXW)
    log(f"train: best lam={lam} tc={tc} net SR {tg.iloc[i]['sharpe_net']:.2f} "
        f"(threshold {tg.iloc[-1]['sharpe_net']:.2f})")

    # 2. out-of-sample ablation at 2.5 bps/leg
    abl = [
        ("Threshold rule (paper baseline)", "threshold", None, None),
        ("Linear in alpha (no optimizer)", "linear", None, None),
        ("Optimizer, cost-unaware (walk-forward)", "opt", tuned.with_(use_costs=False), "wf"),
        ("Optimizer + costs, static forecast", "opt", tuned, "static"),
        ("Optimizer + costs, walk-forward", "opt", tuned, "wf"),
        ("  walk-forward + Ledoit-Wolf cov", "opt", tuned.with_(cov_method="ledoit_wolf"), "wf"),
        ("  walk-forward + 50% daily turnover cap", "opt", tuned.with_(turnover_cap=0.5), "wf"),
        ("  walk-forward + alpha-uncertainty penalty", "opt", tuned.with_(alpha_uncertainty=2e4), "wf"),
    ]
    res = pmap([(l, k, c, 2.5, m, *test) for l, k, c, m in abl], a.procs)
    ab = pd.DataFrame([r for r, _ in res])
    ab.to_csv(out / "tables" / "ablation_test.csv", index=False)
    curves = {r["strategy"]: n for r, n in res}
    log("ablation done")

    # 3. cost sensitivity (test): threshold vs walk-forward optimizer, tc_mult fixed at the tuned value
    levels = [0.0, 1.0, 2.5, 5.0, 10.0]
    jobs = [(f"threshold @{b}", "threshold", None, b, None, *test) for b in levels]
    jobs += [(f"optimizer @{b}", "opt", tuned, b, "wf", *test) for b in levels if b != 2.5]
    cs = {r["strategy"]: r for r, _ in pmap(jobs, a.procs)}
    cs["optimizer @2.5"] = ab.set_index("strategy").loc["Optimizer + costs, walk-forward"].to_dict()
    cost_tab = pd.DataFrame({
        "bps_per_leg": levels,
        "threshold_net_SR": [cs[f"threshold @{b}"]["sharpe_net"] for b in levels],
        "optimizer_net_SR": [cs[f"optimizer @{b}"]["sharpe_net"] for b in levels],
        "threshold_turnover": [cs[f"threshold @{b}"]["turnover_annual"] for b in levels],
        "optimizer_turnover": [cs[f"optimizer @{b}"]["turnover_annual"] for b in levels],
    })
    cost_tab.to_csv(out / "tables" / "cost_sensitivity.csv", index=False)
    log("cost sensitivity done")

    # 4. frontier (test)
    sweep = [0.5, 1.0, 2.0, 4.0]
    fr_jobs = [(f"tc={c}", "opt", tuned.with_(tc_mult=c), 2.5, "wf", *test) for c in sweep if c != tc]
    fr = {r["strategy"]: r for r, _ in pmap(fr_jobs, a.procs)}
    fr[f"tc={tc}"] = cs["optimizer @2.5"]
    frt = pd.DataFrame([{"tc_mult": c, **{k: fr[f"tc={c}"][k] for k in ("sharpe_net", "sharpe_gross", "turnover_annual")}}
                        for c in sweep])
    frt.to_csv(out / "tables" / "frontier_test.csv", index=False)
    log("frontier done")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = pd.to_datetime(pd.Series(P.dates[test[0]:test[1]]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name in ["Threshold rule (paper baseline)", "Optimizer + costs, static forecast",
                 "Optimizer + costs, walk-forward"]:
        ax[0].plot(dates, np.cumsum(curves[name]), label=name, lw=1.3)
    ax[0].set_title("S&P 1500, out of sample: cumulative net return (2.5 bps/leg)", fontsize=9)
    ax[0].legend(fontsize=7); ax[0].tick_params(axis="x", labelsize=7)
    thr = ab.iloc[0]
    ax[1].plot(frt["turnover_annual"], frt["sharpe_net"], "o-", label="optimizer (cost multiplier sweep)")
    for x, y, c in zip(frt["turnover_annual"], frt["sharpe_net"], frt["tc_mult"]):
        ax[1].annotate(f"{c:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax[1].scatter([thr["turnover_annual"]], [thr["sharpe_net"]], color="crimson", zorder=5, label="threshold rule")
    ax[1].set_xlabel("annual turnover (x)"); ax[1].set_ylabel("net Sharpe")
    ax[1].set_title("Turnover frontier (test)", fontsize=9); ax[1].legend(fontsize=7)
    ax[2].plot(cost_tab["bps_per_leg"], cost_tab["threshold_net_SR"], "o-", color="crimson", label="threshold rule")
    ax[2].plot(cost_tab["bps_per_leg"], cost_tab["optimizer_net_SR"], "o-", label="optimizer (walk-forward)")
    ax[2].axhline(0, color="grey", lw=0.8); ax[2].set_xlabel("cost per leg (bps)"); ax[2].set_ylabel("net Sharpe")
    ax[2].set_title("Cost sensitivity (test)", fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figures" / "results.png", dpi=150); plt.close(fig)

    json.dump({"signals": a.signals, "panel": [P.T, P.N], "train": [d(train[0]), d(train[1])],
               "test": [d(test[0]), d(test[1])], "horizon": H, "tuned": {"risk_aversion": lam, "tc_mult": tc},
               "walk_forward_refits": [d(t) for t in MODELS["wf"].points],
               "runtime_s": round(time.time() - t_all)}, open(out / "summary.json", "w"), indent=2)
    pd.set_option("display.width", 220)
    cols = ["sharpe_net", "sharpe_gross", "turnover_annual", "breakeven_bps_per_leg", "max_drawdown", "ann_vol",
            "ann_return_net", "avg_positions", "nw_tstat_net", "solve_ms_per_day"]
    print("\nTRAIN GRID\n" + tg[["strategy", "sharpe_net", "sharpe_gross", "turnover_annual"]].round(2).to_string(index=False))
    print("\nOUT-OF-SAMPLE ABLATION (2.5 bps/leg)\n" + ab.set_index("strategy")[cols].round(2).to_string())
    print("\nCOST SENSITIVITY (test)\n" + cost_tab.round(2).to_string(index=False))
    print("\nFRONTIER (test)\n" + frt.round(2).to_string(index=False))
    log("done")


if __name__ == "__main__":
    main()
