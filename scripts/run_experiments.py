"""End-to-end experiment: calibrate on the first half, evaluate out of sample on the second.

    python scripts/run_experiments.py                     # synthetic panel calibrated to the paper
    python scripts/run_experiments.py --signals PATH.npz  # your real signals_primary.npz

Outputs (results/):
    tables/train_grid.csv         risk-aversion x cost-multiplier grid on the training half
    tables/ablation_test.csv      out-of-sample, 2.5 bps/leg: threshold rule vs optimizer variants
    tables/cost_sensitivity.csv   net Sharpe vs cost level (optimizer re-tuned on train per level)
    tables/frontier_test.csv      turnover vs net Sharpe as the cost multiplier is swept
    figures/*.png
    summary.json
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

from statopt.alpha import AlphaModel  # noqa: E402
from statopt.backtest import run, summarize  # noqa: E402
from statopt.costs import CostModel  # noqa: E402
from statopt.optimizer import OptConfig  # noqa: E402
from statopt.panel import Panel, load_npz, save_npz  # noqa: E402
from statopt.policies import LinearPolicy, OptimizerPolicy, ThresholdPolicy  # noqa: E402
from statopt.simulate import SimConfig, simulate  # noqa: E402

PANEL: Panel | None = None
WARMUP = 20


def _job(args):
    label, kind, cfg, bps, am, start, end = args
    cm = CostModel(bps)
    if kind == "threshold":
        pol = ThresholdPolicy()
    elif kind == "linear":
        pol = LinearPolicy()
    else:
        pol = OptimizerPolicy(cfg, cm, label, alpha_model=am)
    res = run(pol, PANEL, cm, start=start, end=end, warmup=WARMUP)
    out = summarize(res, label)
    out["bps_per_leg"] = bps
    return out, res["net"]


def pmap(jobs, procs):
    with Pool(procs) as pool:
        return pool.map(_job, jobs)


def main():
    global PANEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", help="path to a signals_*.npz from statarb-screening")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--n-days", type=int, default=2520)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="short panel smoke test")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    if a.signals:
        PANEL = load_npz(a.signals)
        source = a.signals
    else:
        PANEL = simulate(SimConfig(n_days=a.n_days))
        save_npz(PANEL, out / "synthetic_signals.npz")
        source = "synthetic (statopt.simulate, calibrated to the screening paper)"
    T = PANEL.T - 1
    split = T // 2
    train, test = (130, split), (split, T)  # 130d burn-in so the covariance window is full
    print(f"panel {PANEL.T} x {PANEL.N} | train {train} | test {test}", flush=True)

    H = a.horizon
    scale = AlphaModel("binned", H).fit(PANEL.s, PANEL.sigma_eq, PANEL.fwd, PANEL.tradable, *train)
    scale.table().to_csv(out / "tables" / "alpha_model_train.csv", index=False)
    print("alpha model (train): mean H-day residual return by s-bucket, in sigma_eq units", flush=True)
    print(scale.table().round(3).to_string(index=False), flush=True)
    base = OptConfig(horizon=H)

    # 1. tune risk aversion x cost multiplier on train at the headline cost --------------------
    grid = [(lam, tc) for lam in (0.5, 1.0, 2.0) for tc in (0.5, 1.0, 1.5, 2.0)]
    jobs = [(f"lam={lam} tc={tc}", "opt", base.with_(risk_aversion=lam, tc_mult=tc), 2.5, scale, *train)
            for lam, tc in grid]
    rows = [r for r, _ in pmap(jobs, a.procs)]
    tg = pd.DataFrame(rows)
    tg.to_csv(out / "tables" / "train_grid.csv", index=False)
    best = tg.loc[tg["sharpe_net"].idxmax()]
    lam_star, tc_star = grid[int(tg["sharpe_net"].idxmax())]
    print(f"train best: {best['strategy']} net SR {best['sharpe_net']:.2f} ({time.time()-t_all:.0f}s)", flush=True)
    tuned = base.with_(risk_aversion=lam_star, tc_mult=tc_star)

    # 2. out-of-sample ablation at 2.5 bps/leg ------------------------------------------------
    abl = [
        ("Threshold rule (paper baseline)", "threshold", None),
        ("Linear in alpha (no optimizer)", "linear", None),
        ("Optimizer, cost-unaware", "opt", tuned.with_(use_costs=False)),
        ("Optimizer + costs, diagonal cov", "opt", tuned.with_(cov_method="diagonal")),
        ("Optimizer + costs, sample cov", "opt", tuned.with_(cov_method="sample")),
        ("Optimizer + costs, Ledoit-Wolf cov", "opt", tuned.with_(cov_method="ledoit_wolf")),
        ("Optimizer + costs, PCA/Marchenko-Pastur cov", "opt", tuned),
        ("  + alpha-uncertainty penalty", "opt", tuned.with_(alpha_uncertainty=2e4)),
        ("  + 5% ex-ante vol cap", "opt", tuned.with_(vol_cap_annual=0.05)),
        ("  + 50% daily turnover cap", "opt", tuned.with_(turnover_cap=0.5)),
    ]
    jobs = [(lbl, kind, cfg, 2.5, scale, *test) for lbl, kind, cfg in abl]
    res = pmap(jobs, a.procs)
    ablation = pd.DataFrame([r for r, _ in res])
    ablation.to_csv(out / "tables" / "ablation_test.csv", index=False)
    curves = {r["strategy"]: net for r, net in res}
    print(f"ablation done ({time.time()-t_all:.0f}s)", flush=True)

    # 3. cost sensitivity: re-tune the cost multiplier on train for each cost level ----------
    levels = [0.0, 1.0, 2.5, 5.0, 10.0]
    tcs = (0.5, 1.0, 2.0, 4.0)
    jobs = [(f"tc={tc}", "opt", tuned.with_(tc_mult=tc), b, scale, *train) for b in levels for tc in tcs]
    tr = [r for r, _ in pmap(jobs, a.procs)]
    best_tc = {}
    for b in levels:
        cand = [(r["sharpe_net"], tcs[i]) for i, r in enumerate(tr[levels.index(b) * len(tcs):(levels.index(b) + 1) * len(tcs)])]
        best_tc[b] = max(cand)[1]
    jobs = [(f"threshold @{b}", "threshold", None, b, scale, *test) for b in levels] + \
           [(f"optimizer @{b}", "opt", tuned.with_(tc_mult=best_tc[b]), b, scale, *test) for b in levels]
    cs = [r for r, _ in pmap(jobs, a.procs)]
    cost_tab = pd.DataFrame({
        "bps_per_leg": levels,
        "threshold_net_SR": [cs[i]["sharpe_net"] for i in range(len(levels))],
        "optimizer_net_SR": [cs[len(levels) + i]["sharpe_net"] for i in range(len(levels))],
        "threshold_turnover": [cs[i]["turnover_annual"] for i in range(len(levels))],
        "optimizer_turnover": [cs[len(levels) + i]["turnover_annual"] for i in range(len(levels))],
        "optimizer_tc_mult": [best_tc[b] for b in levels],
    })
    cost_tab.to_csv(out / "tables" / "cost_sensitivity.csv", index=False)
    print(f"cost sensitivity done ({time.time()-t_all:.0f}s)", flush=True)

    # 4. turnover frontier on test --------------------------------------------------------------
    sweep = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    jobs = [(f"tc={tc}", "opt", tuned.with_(tc_mult=tc, use_costs=tc > 0), 2.5, scale, *test) for tc in sweep]
    fr = pd.DataFrame([r for r, _ in pmap(jobs, a.procs)])
    fr["tc_mult"] = sweep
    fr.to_csv(out / "tables" / "frontier_test.csv", index=False)

    # figures ----------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr = ablation.iloc[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name in ["Threshold rule (paper baseline)", "Optimizer, cost-unaware", "Optimizer + costs, PCA/Marchenko-Pastur cov"]:
        ax[0].plot(np.cumsum(curves[name]), label=name, lw=1.4)
    ax[0].set_title("Out-of-sample cumulative net return (2.5 bps/leg)", fontsize=10)
    ax[0].set_xlabel("test day"); ax[0].legend(fontsize=8)
    ax[1].plot(fr["turnover_annual"], fr["sharpe_net"], "o-", label="optimizer (cost multiplier sweep)")
    for x, y, t in zip(fr["turnover_annual"], fr["sharpe_net"], fr["tc_mult"]):
        ax[1].annotate(f"{t:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax[1].scatter([thr["turnover_annual"]], [thr["sharpe_net"]], color="crimson", zorder=5, label="threshold rule")
    ax[1].set_xlabel("annual turnover (x)"); ax[1].set_ylabel("net Sharpe")
    ax[1].set_title("Turnover frontier (test, 2.5 bps/leg)", fontsize=10); ax[1].legend(fontsize=8)
    ax[2].plot(cost_tab["bps_per_leg"], cost_tab["threshold_net_SR"], "o-", color="crimson", label="threshold rule")
    ax[2].plot(cost_tab["bps_per_leg"], cost_tab["optimizer_net_SR"], "o-", label="optimizer (tc tuned on train)")
    ax[2].axhline(0, color="grey", lw=0.8)
    ax[2].set_xlabel("cost per leg (bps)"); ax[2].set_ylabel("net Sharpe")
    ax[2].set_title("Cost sensitivity (test)", fontsize=10); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "results.png", dpi=150)
    plt.close(fig)

    summary = {
        "source": source, "panel_shape": [PANEL.T, PANEL.N], "train": train, "test": test,
        "horizon_days": H, "alpha_model": "binned (train)", "tuned": {"risk_aversion": lam_star, "tc_mult": tc_star},
        "best_tc_by_cost": best_tc, "runtime_seconds": round(time.time() - t_all, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 220)
    cols = ["sharpe_net", "sharpe_gross", "turnover_annual", "breakeven_bps_per_leg", "max_drawdown",
            "ann_vol", "avg_positions", "solve_ms_per_day"]
    print("\nOUT-OF-SAMPLE ABLATION (2.5 bps/leg)")
    print(ablation.set_index("strategy")[cols].to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nCOST SENSITIVITY (test)")
    print(cost_tab.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nFRONTIER (test)")
    print(fr[["tc_mult", "sharpe_net", "sharpe_gross", "turnover_annual"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\ntotal {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
