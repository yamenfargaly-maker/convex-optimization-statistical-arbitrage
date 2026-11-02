"""Grid over the random-walk share so the threshold rule matches the paper's headline numbers
(gross Sharpe 1.62, net 1.08 at 2.5 bps/leg, ~122x annual turnover)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from statopt.simulate import SimConfig, simulate
from statopt.policies import ThresholdPolicy
from statopt.costs import CostModel
from statopt.backtest import run, summarize

for rw in [float(x) for x in (sys.argv[1:] or ["0.4", "0.55", "0.7", "0.85"])]:
    p = simulate(SimConfig(rw_ratio=rw, n_days=1260))
    r = summarize(run(ThresholdPolicy(), p, CostModel(2.5)), f"rw={rw}")
    print(f"rw={rw}: gross SR {r['sharpe_gross']:.2f} net SR {r['sharpe_net']:.2f} "
          f"turnover {r['turnover_annual']:.0f}x breakeven {r['breakeven_bps_per_leg']:.1f}bps pos {r['avg_positions']:.0f}")
