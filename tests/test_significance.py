"""Data-snooping tests (White's Reality Check, Hansen's SPA, Romano-Wolf stepdown).

All data are seeded synthetic returns. Reference results come from deliberately
naive re-implementations (explicit loops over lags, paths and blocks) that share
nothing with the module except ``stationary_bootstrap_indices``.
"""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pandas as pd
import pytest

import quantlab.significance as sig
from quantlab.metrics import daily_risk_free
from quantlab.robustness import stationary_bootstrap_indices
from quantlab.significance import (
    hansen_spa,
    loss_differentials,
    romano_wolf_stepdown,
    stationary_bootstrap_variance,
    white_reality_check,
)


def _noise(n=500, k=20, seed=0, sd=0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sd, (n, k))


def _frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2019-01-02", periods=len(values))
    return pd.DataFrame(values, index=idx, columns=[f"cfg{j}" for j in range(values.shape[1])])


# --------------------------------------------------------------------------- #
# Brute-force references
# --------------------------------------------------------------------------- #


def _naive_omega2(x: np.ndarray, mean_block: float) -> float:
    """Hansen's kernel estimator with an explicit loop over every lag."""
    n = len(x)
    z = x - x.mean()
    q = 1.0 / mean_block
    total = float(z @ z) / n
    for i in range(1, n):
        kappa = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
        total += 2 * kappa * float(z[:-i] @ z[i:]) / n
    return total


def _naive_paths(x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap means and block-sum variances by materialising every path and walking its blocks."""
    n_paths, n = idx.shape
    means = np.empty((n_paths, x.shape[1]))
    var = np.empty_like(means)
    for b in range(n_paths):
        path = x[idx[b]]
        means[b] = path.mean(axis=0)
        cuts = [0] + [t for t in range(1, n) if idx[b, t] != (idx[b, t - 1] + 1) % n] + [n]
        var[b] = sum(((path[a:c] - means[b]).sum(axis=0)) ** 2 for a, c in zip(cuts[:-1], cuts[1:])) / n
    return means, var


def _naive_spa(x, idx, mean_block, restudentise):
    n, k = x.shape
    means = x.mean(axis=0)
    omega = np.sqrt([_naive_omega2(x[:, j], mean_block) for j in range(k)])
    t = math.sqrt(n) * means / omega
    statistic = max(0.0, t.max())
    threshold = math.sqrt(2 * math.log(math.log(n)))
    boot_means, boot_var = _naive_paths(x, idx)
    out = {"omega": omega, "t": t, "statistic": statistic}
    for name, mu in (
        ("lower", np.minimum(means, 0)),
        ("consistent", np.where(t <= -threshold, means, 0.0)),
        ("upper", np.zeros(k)),
    ):
        stars = []
        for b in range(len(idx)):
            scale = np.sqrt(boot_var[b]) if restudentise else omega
            path_t = [math.sqrt(n) * (boot_means[b, j] - means[j] + mu[j]) / scale[j] for j in range(k)]
            stars.append(max(0.0, max(path_t)))
        out[name] = np.mean(np.array(stars) >= statistic)
        out[f"boot_{name}"] = np.array(stars)
    return out


def _naive_romano_wolf(x, idx, mean_block):
    n, k = x.shape
    means = x.mean(axis=0)
    omega = np.sqrt([_naive_omega2(x[:, j], mean_block) for j in range(k)])
    t = math.sqrt(n) * means / omega
    boot_means, boot_var = _naive_paths(x, idx)
    t_star = math.sqrt(n) * (boot_means - means) / np.sqrt(boot_var)
    order = sorted(range(k), key=lambda j: (-t[j], j))
    adjusted, running = np.empty(k), 0.0
    for pos, j in enumerate(order):
        p = np.mean([t_star[b, order[pos:]].max() >= t[j] for b in range(len(idx))])
        running = max(running, p)
        adjusted[j] = running
    return adjusted


# --------------------------------------------------------------------------- #
# Differentials
# --------------------------------------------------------------------------- #


def test_loss_differentials_against_cash_and_against_a_benchmark():
    returns = _frame(_noise(n=60, k=3, seed=1))
    vs_cash = loss_differentials(returns, risk_free_rate=0.03)
    pd.testing.assert_frame_equal(vs_cash, returns - daily_risk_free(0.03))

    bench = pd.Series(np.random.default_rng(2).normal(0, 0.01, 60), index=returns.index)
    shuffled = bench.sample(frac=1.0, random_state=0)  # alignment is by date, not position
    vs_bench = loss_differentials(returns, shuffled)
    np.testing.assert_allclose(vs_bench.to_numpy(), returns.to_numpy() - bench.to_numpy()[:, None])
    pd.testing.assert_frame_equal(loss_differentials(returns, bench.to_numpy()), vs_bench)

    with pytest.raises(ValueError, match="cover"):
        loss_differentials(returns, bench.iloc[1:])
    with pytest.raises(ValueError):
        loss_differentials(returns, bench.to_numpy()[1:])
    broken = returns.copy()
    broken.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        loss_differentials(broken)


# --------------------------------------------------------------------------- #
# Formulas against brute force
# --------------------------------------------------------------------------- #


def test_kernel_variance_matches_explicit_lags_and_the_bootstrap_it_describes():
    rng = np.random.default_rng(4)
    e = rng.normal(size=300)
    x = np.empty(300)
    x[0] = e[0]
    for i in range(1, 300):
        x[i] = 0.5 * x[i - 1] + e[i]  # autocorrelated, so the kernel weights matter
    w2 = stationary_bootstrap_variance(x, 10.0)[0]
    assert w2 == pytest.approx(_naive_omega2(x, 10.0), rel=1e-10)
    # It is the variance of sqrt(T) * mean* under the stationary bootstrap itself.
    idx = stationary_bootstrap_indices(300, 20_000, 10.0, np.random.default_rng(5))
    assert np.var(math.sqrt(300) * x[idx].mean(axis=1)) == pytest.approx(w2, rel=0.04)
    assert stationary_bootstrap_variance(x, 1.0)[0] == pytest.approx(np.var(x), rel=1e-10)  # iid bootstrap


def test_reality_check_matches_brute_force():
    x = _noise(n=60, k=4, seed=7)
    x[:, 2] += 0.002
    res = white_reality_check(x, n_boot=50, mean_block=5.0, seed=3)
    idx = stationary_bootstrap_indices(60, 50, 5.0, np.random.default_rng(3))
    means = x.mean(axis=0)
    stars = np.array([math.sqrt(60) * (x[idx[b]].mean(axis=0) - means).max() for b in range(50)])
    np.testing.assert_allclose(res.bootstrap_statistics, stars, rtol=1e-10, atol=1e-14)
    assert res.statistic == pytest.approx(math.sqrt(60) * means.max())
    assert res.p_value == np.mean(stars >= res.statistic)
    assert res.best_index == 2 and res.best_label == 2
    assert (res.n_obs, res.n_configs, res.n_boot) == (60, 4, 50)


@pytest.mark.parametrize("restudentise", [True, False])
def test_spa_matches_brute_force(restudentise):
    x = _noise(n=80, k=4, seed=8)
    x[:, 0] += 0.002  # the candidate
    x[:, 1] -= 0.01  # hopeless: recentred away by the consistent and lower p-values
    x[:, 3] -= 0.001
    res = hansen_spa(_frame(x), n_boot=60, mean_block=6.0, seed=11, restudentise=restudentise)
    idx = stationary_bootstrap_indices(80, 60, 6.0, np.random.default_rng(11))
    ref = _naive_spa(x, idx, 6.0, restudentise)
    np.testing.assert_allclose(res.omega.to_numpy(), ref["omega"], rtol=1e-9)
    np.testing.assert_allclose(res.t_statistics.to_numpy(), ref["t"], rtol=1e-9)
    assert res.statistic == pytest.approx(ref["statistic"])
    assert res.p_value_lower == ref["lower"]
    assert res.p_value == ref["consistent"]
    assert res.p_value_upper == ref["upper"]
    np.testing.assert_allclose(res.bootstrap_statistics, ref["boot_consistent"], rtol=1e-9, atol=1e-12)
    assert res.threshold == pytest.approx(math.sqrt(2 * math.log(math.log(80))))
    assert res.best_label == "cfg0" and res.n_excluded == 0 and res.restudentised is restudentise


def test_romano_wolf_matches_brute_force_and_spa_upper():
    x = _noise(n=80, k=5, seed=9)
    x[:, 1] += 0.004
    x[:, 3] += 0.002
    res = romano_wolf_stepdown(x, n_boot=60, mean_block=6.0, seed=13)
    idx = stationary_bootstrap_indices(80, 60, 6.0, np.random.default_rng(13))
    np.testing.assert_allclose(res.adjusted_p_values.to_numpy(), _naive_romano_wolf(x, idx, 6.0))
    spa = hansen_spa(x, n_boot=60, mean_block=6.0, seed=13)
    top = int(res.t_statistics.idxmax())
    assert res.adjusted_p_values[top] == spa.p_value_upper  # step 1 is the studentised Reality Check
    order = res.t_statistics.sort_values(ascending=False).index
    assert res.adjusted_p_values[order].is_monotonic_increasing
    assert (res.adjusted_p_values <= res.single_step_p_values + 1e-15).all()


def test_chunked_bootstrap_equals_one_pass_over_the_same_draws(monkeypatch):
    x = _noise(n=50, k=3, seed=10)
    x[:, 0] += 0.003
    monkeypatch.setattr(sig, "_CHUNK_BYTES", 8 * 50 * 7)  # 7 paths per chunk: 30 = 7 + 7 + 7 + 7 + 2
    rc = white_reality_check(x, n_boot=30, mean_block=4.0, seed=21)
    spa = hansen_spa(x, n_boot=30, mean_block=4.0, seed=21)
    rng = np.random.default_rng(21)
    idx = np.vstack([stationary_bootstrap_indices(50, m, 4.0, rng) for m in (7, 7, 7, 7, 2)])
    means = x.mean(axis=0)
    stars = np.array([math.sqrt(50) * (x[idx[b]].mean(axis=0) - means).max() for b in range(30)])
    np.testing.assert_allclose(rc.bootstrap_statistics, stars, rtol=1e-10, atol=1e-14)
    ref = _naive_spa(x, idx, 4.0, restudentise=True)
    np.testing.assert_allclose(spa.bootstrap_statistics, ref["boot_consistent"], rtol=1e-9, atol=1e-12)
    assert spa.p_value == ref["consistent"]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_size_under_the_null_is_close_to_nominal():
    # 20 configurations of pure zero-mean noise: a 5% test should reject in about 5% of grids.
    n_seeds, rc_rejections, spa_rejections = 150, 0, 0
    for s in range(n_seeds):
        d = _noise(n=500, k=20, seed=10_000 + s)
        rc_rejections += white_reality_check(d, n_boot=199, seed=s).p_value <= 0.05
        spa_rejections += hansen_spa(d, n_boot=199, seed=s).p_value <= 0.05
    assert 0.01 <= rc_rejections / n_seeds <= 0.12
    assert 0.01 <= spa_rejections / n_seeds <= 0.12


def test_power_against_one_planted_edge():
    d = _frame(_noise(n=750, k=20, seed=5))
    d["cfg7"] += 0.0015  # daily Sharpe 0.15 hidden among 19 useless configurations
    assert white_reality_check(d, n_boot=499, seed=1).p_value < 0.05
    spa = hansen_spa(d, n_boot=499, seed=1)
    assert spa.p_value < 0.05 and spa.best_label == "cfg7"
    assert romano_wolf_stepdown(d, n_boot=499, seed=1).rejected(0.05) == ["cfg7"]


def test_romano_wolf_controls_the_nulls_while_finding_two_edges():
    d = _frame(_noise(n=1000, k=12, seed=6))
    d["cfg2"] += 0.002
    d["cfg9"] += 0.0025
    res = romano_wolf_stepdown(d, n_boot=499, seed=2)
    assert sorted(res.rejected(0.05)) == ["cfg2", "cfg9"]
    assert (res.adjusted_p_values.drop(["cfg2", "cfg9"]) > 0.05).all()


# --------------------------------------------------------------------------- #
# Invariances and properties
# --------------------------------------------------------------------------- #


def test_duplicating_a_column_changes_nothing():
    d = _noise(n=400, k=6, seed=12)
    d[:, 4] += 0.001
    dup = np.column_stack([d[:, :3], d[:, 4], d[:, 3:]])  # cfg4 repeated in the middle
    a, b = white_reality_check(d, n_boot=300, seed=4), white_reality_check(dup, n_boot=300, seed=4)
    assert a.p_value == b.p_value and a.statistic == b.statistic
    np.testing.assert_allclose(a.bootstrap_statistics, b.bootstrap_statistics, rtol=1e-12, atol=1e-15)
    s1, s2 = hansen_spa(d, n_boot=300, seed=4), hansen_spa(dup, n_boot=300, seed=4)
    assert (s1.p_value_lower, s1.p_value, s1.p_value_upper) == (s2.p_value_lower, s2.p_value, s2.p_value_upper)


def test_same_seed_same_result_and_different_seed_different_draws():
    d = _noise(n=300, k=5, seed=14)
    for fn, field in ((white_reality_check, "bootstrap_statistics"), (hansen_spa, "bootstrap_statistics")):
        one, two, other = fn(d, n_boot=200, seed=7), fn(d, n_boot=200, seed=7), fn(d, n_boot=200, seed=8)
        np.testing.assert_array_equal(getattr(one, field), getattr(two, field))
        assert one.p_value == two.p_value
        assert not np.array_equal(getattr(one, field), getattr(other, field))
    rw1, rw2 = romano_wolf_stepdown(d, n_boot=200, seed=7), romano_wolf_stepdown(d, n_boot=200, seed=7)
    pd.testing.assert_series_equal(rw1.adjusted_p_values, rw2.adjusted_p_values)


def test_spa_consistent_p_value_lies_between_lower_and_upper():
    for s in range(8):
        d = _noise(n=400, k=10, seed=100 + s)
        d[:, :6] -= np.linspace(0.0005, 0.004, 6)  # a spread of poor configurations
        d[:, 8] += 0.0006 * (s % 3)
        res = hansen_spa(d, n_boot=200, seed=s)
        assert res.p_value_lower <= res.p_value <= res.p_value_upper
    d = _noise(n=500, k=10, seed=3)
    d[:, :8] -= 0.003  # eight clearly inferior configurations
    d[:, 8] += 0.0008
    res = hansen_spa(d, n_boot=499, seed=1)
    assert res.p_value < res.p_value_upper  # the consistent p-value is not diluted by them


def test_constant_differentials_are_excluded_not_fatal():
    d = _noise(n=200, k=4, seed=15)
    d[:, 1] = 0.0  # identical to the benchmark
    d[:, 3] = 0.0002  # a constant spread
    with np.errstate(divide="raise", invalid="raise"):
        spa = hansen_spa(d, n_boot=100, seed=1)
        rw = romano_wolf_stepdown(d, n_boot=100, seed=1)
        rc = white_reality_check(d, n_boot=100, seed=1)
    assert spa.n_excluded == 2 and spa.t_statistics.isna().sum() == 2
    assert 0.0 <= spa.p_value <= 1.0 and np.isfinite(spa.bootstrap_statistics).all()
    assert rw.adjusted_p_values[1] == 1.0 and rw.adjusted_p_values[3] == 1.0
    assert np.isfinite(rc.p_value)

    flat = np.zeros((100, 3))
    spa = hansen_spa(flat, n_boot=50)
    assert (spa.p_value_lower, spa.p_value, spa.p_value_upper) == (1.0, 1.0, 1.0) and spa.best_index is None
    assert white_reality_check(flat, n_boot=50).p_value == 1.0
    assert (romano_wolf_stepdown(flat, n_boot=50).adjusted_p_values == 1.0).all()


def test_memory_stays_bounded_for_a_large_grid():
    d = _noise(n=5000, k=200, seed=16)
    tracemalloc.start()
    try:
        hansen_spa(d, n_boot=1000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 200 * 2**20


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("test", [white_reality_check, hansen_spa, romano_wolf_stepdown])
def test_invalid_inputs_raise(test):
    good = _noise(n=100, k=3, seed=17)
    with_nan = good.copy()
    with_nan[10, 1] = np.nan
    with_inf = good.copy()
    with_inf[10, 1] = np.inf
    for bad in (with_nan, with_inf, np.empty((100, 0)), good[:19], np.zeros((100, 2, 2))):
        with pytest.raises(ValueError):
            test(bad, n_boot=10)
    with pytest.raises(ValueError):
        test(pd.DataFrame({"a": ["x"] * 100}), n_boot=10)
    for kwargs in ({"n_boot": 0}, {"n_boot": 2.5}, {"mean_block": 0.5}, {"mean_block": 101.0}, {"mean_block": np.nan}):
        with pytest.raises(ValueError):
            test(good, **{"n_boot": 10, **kwargs})
    assert test(good[:, 0], n_boot=10).n_configs == 1  # a single series is one configuration
