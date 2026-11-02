import numpy as np
import pandas as pd
import pytest

from statopt.alpha import fit_alpha_scale, ou_expected_return, shape_s
from statopt.backtest import nw_tstat, run, sharpe, summarize
from statopt.costs import CostModel
from statopt.optimizer import OptConfig, PortfolioOptimizer
from statopt.panel import Panel, load_npz, save_npz
from statopt.policies import OptimizerPolicy, ThresholdPolicy
from statopt.risk import estimate, ledoit_wolf_intensity
from statopt.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def panel():
    return simulate(SimConfig(n_names=60, n_days=260, seed=1))


# ---------------------------------------------------------------- alpha
def test_alpha_sign_and_trust_region():
    s = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.5, -4.0])
    a = ou_expected_return(s, np.full(7, 0.05), horizon=10)
    assert a[0] > a[1] > 0 and a[2] == 0 and a[3] < 0
    assert a[5] == 0 and a[6] == 0  # beyond the trust region
    assert shape_s(np.array([2.9]))[0] == pytest.approx(2.5)


def test_alpha_scale_recovers_planted_signal():
    rng = np.random.default_rng(0)
    T, N, H = 400, 200, 5
    s = rng.standard_normal((T, N))
    sig = np.full((T, N), 0.05)
    true_scale = 0.2
    fwd = np.zeros((T, N))
    a = ou_expected_return(s, sig, 1)  # per-day forecast
    fwd[:] = true_scale * a + 0.002 * rng.standard_normal((T, N))
    est = fit_alpha_scale(s, sig, fwd, np.ones((T, N), bool), 0, T, 1, step=1)
    assert est == pytest.approx(true_scale, rel=0.1)


# ---------------------------------------------------------------- risk
def test_ledoit_wolf_bounds_and_psd():
    X = np.random.default_rng(0).standard_normal((60, 120))
    X -= X.mean(0)
    delta, mu = ledoit_wolf_intensity(X)
    assert 0 <= delta <= 1 and mu > 0
    rm = estimate(X, "ledoit_wolf")
    ev = np.linalg.eigvalsh(rm.cov())
    assert ev.min() > 0  # sample cov (n > L) is singular, LW is not


@pytest.mark.parametrize("method", ["sample", "diagonal", "ledoit_wolf", "pca_mp"])
def test_risk_models_match_variance_formula(method):
    R = np.random.default_rng(1).standard_normal((100, 30)) * 0.01
    rm = estimate(R, method)
    w = np.random.default_rng(2).standard_normal(30)
    assert rm.variance(w) == pytest.approx(w @ rm.cov() @ w)


def test_pca_mp_finds_planted_factors():
    rng = np.random.default_rng(3)
    L, n, k = 250, 100, 3
    B = rng.standard_normal((n, k))
    R = rng.standard_normal((L, k)) @ B.T * 2 + rng.standard_normal((L, n))
    assert estimate(R, "pca_mp").shrinkage == 3


# ---------------------------------------------------------------- optimizer
def test_optimizer_respects_constraints():
    rng = np.random.default_rng(4)
    n = 50
    cfg = OptConfig(gross=1.0, max_weight=0.05, turnover_cap=0.3, cov_method="diagonal")
    opt = PortfolioOptimizer(cfg, n, 1)
    ub = np.full(n, 0.05); ub[:5] = 0.0  # untradable names
    w = opt.solve(rng.standard_normal(n) * 0.01, np.zeros(n), np.zeros((1, n)), np.full(n, 1e-4), ub,
                  np.full(n, 2.5e-4))
    assert np.abs(w).sum() <= 1.0 + 1e-5
    assert np.all(np.abs(w) <= 0.05 + 1e-6)
    assert np.allclose(w[:5], 0, atol=1e-6)
    assert abs(w.sum()) < 1e-4  # dollar neutral to within solver tolerance (0.01% of NAV)
    assert np.abs(w).sum() <= 0.3 + 1e-5  # turnover from flat


def test_costs_create_no_trade_region():
    """A tiny alpha change should not move the book once costs are priced."""
    n = 20
    rng = np.random.default_rng(5)
    a = rng.standard_normal(n) * 1e-3
    args = (np.zeros((1, n)), np.full(n, 1e-4), np.full(n, 0.1), np.full(n, 5e-4))
    opt = PortfolioOptimizer(OptConfig(cov_method="diagonal", gross=1.0, max_weight=0.1), n, 1)
    w1 = opt.solve(a, np.zeros(n), *args)
    w2 = opt.solve(a + 1e-5 * rng.standard_normal(n), w1, *args)
    assert np.abs(w2 - w1).sum() < 1e-3
    free = PortfolioOptimizer(OptConfig(cov_method="diagonal", gross=1.0, max_weight=0.1, use_costs=False), n, 1)
    assert opt.problem.is_dpp() and free.problem.is_dpp()


def test_vol_cap_binds():
    n = 30
    rng = np.random.default_rng(6)
    cfg = OptConfig(cov_method="diagonal", vol_cap_annual=0.02, risk_aversion=0.0, use_costs=False)
    opt = PortfolioOptimizer(cfg, n, 1)
    d = np.full(n, (0.02) ** 2)
    w = opt.solve(rng.standard_normal(n) * 0.01, np.zeros(n), np.zeros((1, n)), d, np.full(n, 0.2), np.zeros(n))
    assert np.sqrt(252 * np.sum(d * w * w)) <= 0.02 * 1.01


# ---------------------------------------------------------------- policies / backtest
def test_threshold_policy_hysteresis():
    s = np.array([[-1.5, 1.5, 0.0], [-0.8, 1.0, 0.0], [-0.3, 0.5, 0.0]])
    p = Panel(s=s, fwd=np.zeros_like(s), tradable=np.ones_like(s, bool), sigma_eq=np.ones_like(s),
              dates=np.arange(3))
    pol = ThresholdPolicy(max_weight=1.0)
    pol.reset(p)
    w = [pol.weights(t, p, None) for t in range(3)]
    assert w[0][0] > 0 and w[0][1] < 0
    assert w[1][0] > 0 and w[1][1] < 0  # still inside the holding band
    assert w[2][0] == 0 and w[2][1] == 0  # closed


def test_backtest_accounting(panel):
    res = run(ThresholdPolicy(), panel, CostModel(2.5), start=60)
    assert np.allclose(res["net"], res["gross"] - res["cost"])
    assert np.allclose(res["cost"], 2.5e-4 * res["turnover"], rtol=1e-9, atol=1e-12)
    zero = run(ThresholdPolicy(), panel, CostModel(0.0), start=60)
    assert np.allclose(zero["net"], zero["gross"])


def test_optimizer_policy_runs_and_trades_less_with_costs(panel):
    cm = CostModel(5.0)
    lo = run(OptimizerPolicy(OptConfig(tc_mult=3.0), cm, alpha_scale=0.1), panel, cm, start=150, end=200)
    hi = run(OptimizerPolicy(OptConfig(use_costs=False), cm, alpha_scale=0.1), panel, cm, start=150, end=200)
    assert lo["turnover"].mean() < hi["turnover"].mean()


def test_npz_roundtrip(tmp_path, panel):
    save_npz(panel, tmp_path / "sig.npz")
    q = load_npz(tmp_path / "sig.npz")
    assert q.s.shape == panel.s.shape and np.allclose(np.nan_to_num(q.fwd), np.nan_to_num(panel.fwd), atol=1e-6)


def test_stats():
    r = np.random.default_rng(0).normal(0.001, 0.01, 5000)
    assert sharpe(r) == pytest.approx(np.sqrt(252) * 0.1, rel=0.2)
    assert nw_tstat(r) > 3


def test_binned_alpha_learns_reversion_and_breakout():
    rng = np.random.default_rng(7)
    T, N, H = 300, 300, 1
    s = rng.uniform(-5, 5, (T, N))
    sig = np.full((T, N), 0.05)
    # reversion inside |s|<2.5, continuation beyond 3
    g = np.where(np.abs(s) < 2.5, -0.1 * s, np.where(np.abs(s) > 3, 0.1 * np.sign(s), 0.0))
    fwd = g * 0.05 + 0.001 * rng.standard_normal((T, N))
    from statopt.alpha import AlphaModel
    am = AlphaModel("binned", horizon=H, shrink_n=0).fit(s, sig, fwd, np.ones((T, N), bool), 0, T)
    a = am(np.array([-2.0, 2.0, -4.5, 4.5]), np.full(4, 0.05))
    assert a[0] > 0 and a[1] < 0  # reversion
    assert a[2] < 0 and a[3] > 0  # breakout continuation


def test_walk_forward_refits_each_year_without_lookahead():
    from statopt.alpha import WalkForwardAlpha
    rng = np.random.default_rng(8)
    T, N = 800, 50
    dates = pd.bdate_range("2019-01-01", periods=T).strftime("%Y-%m-%d").to_numpy()
    s = rng.standard_normal((T, N)); sig = np.full((T, N), 0.05)
    fwd = 0.001 * rng.standard_normal((T, N))
    wf = WalkForwardAlpha(s, sig, fwd, np.ones((T, N), bool), dates, first_fit_end=300, start=0, horizon=5)
    years = pd.to_datetime(pd.Series(dates[wf.points])).dt.year.tolist()
    assert wf.points[0] == 300 and len(set(years)) == len(years)
    assert wf.at(299) is wf.models[0] and wf.at(wf.points[-1]) is wf.models[-1]
