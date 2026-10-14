"""Bootstrap and regime analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.metrics import sharpe_ratio
from quantlab.robustness import (
    bootstrap_metrics,
    bootstrap_summary,
    regime_labels,
    regime_performance,
    stationary_bootstrap_indices,
)


def _returns(n=1000, seed=0, mu=0.0004, sd=0.01):
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=idx)


def test_bootstrap_indices_shape_range_and_block_length():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 200, mean_block=10, rng=rng)
    assert idx.shape == (200, 500) and idx.min() >= 0 and idx.max() < 500
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 500).mean()
    assert continues == pytest.approx(1 - 1 / 10, abs=0.02)  # average block length ~10


def test_bootstrap_is_reproducible_with_a_seed():
    r = _returns()
    a = bootstrap_metrics(r, n_boot=200, seed=7)
    b = bootstrap_metrics(r, n_boot=200, seed=7)
    c = bootstrap_metrics(r, n_boot=200, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_bootstrap_distribution_brackets_the_observed_sharpe():
    r = _returns(n=1500, seed=3)
    samples = bootstrap_metrics(r, n_boot=1000, seed=1)
    observed = sharpe_ratio(r)
    lo, hi = samples["sharpe"].quantile([0.05, 0.95])
    assert lo < observed < hi
    assert samples["sharpe"].median() == pytest.approx(observed, abs=0.25)


def test_paired_resampling_preserves_identity_with_benchmark():
    r = _returns()
    samples = bootstrap_metrics(r, benchmark_returns=r, n_boot=100, seed=2)
    assert (samples["excess_cagr"].abs() < 1e-12).all()
    summary = bootstrap_summary(samples, {"sharpe": sharpe_ratio(r)})
    assert {"Sharpe ratio", "Excess annualised return vs benchmark"} <= set(summary["Metric"])


def test_zero_volatility_paths_do_not_crash():
    r = pd.Series(0.0005, index=pd.bdate_range("2020-01-01", periods=100))
    samples = bootstrap_metrics(r, n_boot=50, seed=1)
    assert samples["sharpe"].isna().all()
    assert np.allclose(samples["cagr"], samples["cagr"].iloc[0])


def test_regimes_use_information_known_before_each_return():
    n = 600
    up = np.linspace(0, 0.6, 400)
    down = 0.6 - np.linspace(0, 0.5, 200)
    close = pd.Series(100 * np.exp(np.r_[up, down]), index=pd.bdate_range("2015-01-02", periods=n))
    rets = close.pct_change().dropna()
    regimes = regime_labels(close)
    table = regime_performance(rets, rets, regimes)
    trend = table[table["Dimension"] == "Trend"].set_index("Regime")
    assert trend.loc["Up-trend", "Strategy return (ann.)"] > 0 > trend.loc["Down-trend", "Strategy return (ann.)"]
    assert trend["Sessions"].sum() == (regimes["trend"].shift(1).reindex(rets.index).notna()).sum()
    assert (table["Strategy return (ann.)"] == table["Benchmark return (ann.)"]).all()


def test_bootstrap_requires_enough_data():
    with pytest.raises(ValueError):
        bootstrap_metrics(_returns(n=10))
