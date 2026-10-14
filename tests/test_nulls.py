"""Honest nulls for timing strategies: exposure-matched benchmark and circular-shift permutation test.

All markets are synthetic and seeded. Nothing here is real market data.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import run_backtest
from quantlab.config import BacktestConfig
from quantlab.metrics import compute_metrics
from quantlab.nulls import (
    MIN_VALID_SHIFTS,
    PermutationResult,
    admissible_shifts,
    average_exposure,
    circular_shift,
    exposure_matched_benchmark,
    timing_permutation_test,
)
from quantlab.strategies import create_strategy
from tests.helpers import make_prices, random_walk

CFG = BacktestConfig()
VOL = 0.012


def _run(prices, key="ma_crossover", params=None, start=100, config=CFG):
    return run_backtest(prices, create_strategy(key, params), config, prices.index[start])


def _noise_market(n: int, seed: int) -> pd.DataFrame:
    # A null market must have ZERO ARITHMETIC drift: with log drift -sigma^2/2 the expected simple
    # return is exp(-sigma^2/2 + sigma^2/2) - 1 = 0, so no exposure pattern is rewarded on average.
    # A zero-LOG-drift walk is NOT a null: its expected simple return is exp(sigma^2/2) - 1 > 0,
    # so simply being invested earns a premium.
    return random_walk(n=n, seed=seed, drift=-(VOL**2) / 2, vol=VOL)


def _momentum_market(n: int, seed: int, d: float = 0.001, noise: float = 0.01, lookback: int = 63):
    """Planted time-series momentum (as in scripts/synthetic_model_study.py): the expected daily
    return is +d after a rising quarter and -d after a falling one, with zero arithmetic drift
    otherwise."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, noise, n)
    logp = np.zeros(n)
    for t in range(1, n):
        mu = 0.0 if t <= lookback else (d if logp[t - 1] > logp[t - 1 - lookback] else -d)
        logp[t] = logp[t - 1] + mu - noise**2 / 2 + shocks[t]
    close = 100 * np.exp(logp)
    opens = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0.0, 0.002, n))
    return make_prices(close, opens, start="2010-01-04")


def _circular_changes(values: np.ndarray) -> Counter:
    """Multiset of (from, to) target changes, including the wrap-around pair."""
    previous = np.roll(values, 1)
    return Counter((float(a), float(b)) for a, b in zip(previous, values) if a != b)


# --------------------------------------------------------------------------- #
# Circular shifts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", ["ma_crossover_with_warmup", "multi_valued"])
def test_shift_preserves_exposure_sum_values_and_changes(source):
    if source == "ma_crossover_with_warmup":
        # Evaluation starts before the 50-session average is ready: NaN warm-up targets in the window.
        result = _run(random_walk(n=700, seed=5), start=10)
        targets = result.daily["target"].to_numpy()
        assert np.isnan(targets).any()
    else:
        rng = np.random.default_rng(3)
        targets = np.repeat(rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 60), rng.integers(1, 30, 60))
        targets[:15] = np.nan
    base = np.nan_to_num(targets, nan=0.0)
    for shift in (1, 63, 250, len(base) - 1, len(base), -17):
        shifted = circular_shift(targets, shift)
        assert not np.isnan(shifted).any()
        assert math.fsum(shifted) == math.fsum(base)  # exact, independent of summation order
        np.testing.assert_array_equal(np.sort(shifted), np.sort(base))  # same exposure distribution
        assert _circular_changes(shifted) == _circular_changes(base)  # same target changes
    np.testing.assert_array_equal(circular_shift(targets, len(base)), base)


def test_admissible_shifts_respect_min_shift_in_both_directions():
    shifts = admissible_shifts(300, 63)
    assert shifts.min() == 63 and shifts.max() == 300 - 63
    assert len(admissible_shifts(100, 63)) == 0


# --------------------------------------------------------------------------- #
# Exposure-matched benchmark
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["ma_crossover", "adaptive_ensemble"])
@pytest.mark.parametrize(
    "config",
    [
        CFG,
        BacktestConfig(fractional_shares=False, cash_interest_rate=0.03, risk_free_rate=0.02,
                       rebalance_threshold=0.10, fee_rate=0.002),
    ],
    ids=["default", "whole_shares_interest_wide_band"],
)
def test_exposure_matched_benchmark_matches_exposure_and_reconciles(key, config):
    prices = random_walk(n=1100, seed=21)
    start = prices.index[300]
    result = run_backtest(prices, create_strategy(key), config, start)
    bench = exposure_matched_benchmark(prices, result, config, start)

    strategy_exposure = float(result.daily["exposure"].iloc[1:].mean())
    assert 0.05 < strategy_exposure < 0.95  # a genuinely partially-invested strategy
    assert average_exposure(result) == pytest.approx(strategy_exposure)
    assert bench.daily.index.equals(result.daily.index)
    assert bench.config == config
    assert bench.is_valid and all(row["passed"] for row in bench.reconciliation)
    realised = float(bench.daily["exposure"].iloc[1:].mean())
    assert abs(realised - strategy_exposure) <= config.rebalance_threshold + 0.01
    # A constant target from the anchor: first fill at the first evaluation open, like buy-and-hold.
    assert np.allclose(bench.daily["target"], strategy_exposure)
    assert bench.ledger["execution_date"].iloc[0] == result.daily.index[1]
    assert bench.strategy_label == f"Constant {strategy_exposure:.0%} exposure (exposure-matched)"
    assert bench.meta["strategy_average_exposure"] == pytest.approx(strategy_exposure)
    # Same benchmark whether the window comes from the result or from the same dates.
    implicit = exposure_matched_benchmark(prices, result, config)
    pd.testing.assert_series_equal(implicit.equity, bench.equity)


def test_exposure_matched_benchmark_rejects_a_different_window_or_prices():
    prices = random_walk(n=700, seed=4)
    result = _run(prices, start=200)
    with pytest.raises(ValueError, match="different evaluation window"):
        exposure_matched_benchmark(prices, result, CFG, prices.index[250])
    shifted_prices = prices.assign(close=prices["close"] * 1.01)
    with pytest.raises(ValueError, match="differ"):
        exposure_matched_benchmark(shifted_prices, result, CFG)


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("statistic", ["sharpe", "cagr", "total_return"])
def test_observed_statistic_equals_the_results_own_metric(statistic):
    prices = random_walk(n=900, seed=8)
    result = _run(prices)
    perm = timing_permutation_test(prices, result, CFG, n_shifts=30, statistic=statistic)
    assert perm.observed == pytest.approx(compute_metrics(result).value(statistic), rel=1e-12)
    assert perm.statistic == statistic and len(perm.null) == len(perm.shifts) == perm.n_shifts == 30


@pytest.mark.parametrize("statistic", ["sharpe", "cagr", "total_return"])
def test_constant_target_strategy_gets_p_value_one(statistic):
    prices = random_walk(n=900, seed=2)
    result = _run(prices, key="buy_and_hold")
    perm = timing_permutation_test(prices, result, CFG, statistic=statistic)
    assert perm.n_shifts == 200
    assert (perm.null == perm.observed).all()  # every shift of a constant path is the same path
    assert perm.p_value == 1.0  # ties count as ">= observed"


def test_never_invested_strategy_returns_no_p_value_and_a_cash_benchmark():
    prices = random_walk(n=700, seed=6)
    never = {"lookback": 63, "threshold": 0.5}  # a 50% quarterly gain never happens here
    for config in (CFG, CFG.model_copy(update={"cash_interest_rate": 0.04})):
        result = _run(prices, key="ts_momentum", params=never, config=config)
        assert result.ledger.empty
        for statistic in ("sharpe", "cagr", "total_return"):
            perm = timing_permutation_test(prices, result, config, statistic=statistic)
            assert perm.p_value is None and "never invested" in perm.note
            assert perm.n_shifts == 0 and len(perm.null) == 0 and len(perm.shifts) == 0
        bench = exposure_matched_benchmark(prices, result, config)
        assert bench.is_valid and bench.ledger.empty
        assert bench.strategy_label == "Constant 0% exposure (exposure-matched)"
        assert "never invested" in bench.meta["note"]
        pd.testing.assert_series_equal(bench.equity, result.equity)  # cash, with the same interest


def test_short_window_or_bad_arguments():
    prices = random_walk(n=400, seed=9)
    result = _run(prices, start=260)  # 140 decision sessions: only 15 shifts of >= 63 sessions both ways
    perm = timing_permutation_test(prices, result, CFG)
    assert perm.p_value is None and "at least" in perm.note
    with pytest.raises(ValueError):
        timing_permutation_test(prices, result, CFG, statistic="sortino")
    with pytest.raises(ValueError):
        timing_permutation_test(prices, result, CFG, n_shifts=0)


def test_all_admissible_shifts_are_used_when_there_are_few():
    prices = random_walk(n=400, seed=12)
    result = _run(prices, start=180)
    n_decisions = len(result.daily) - 1
    perm = timing_permutation_test(prices, result, CFG, n_shifts=200)
    expected = admissible_shifts(n_decisions, 63)
    assert MIN_VALID_SHIFTS <= len(expected) < 200
    np.testing.assert_array_equal(perm.shifts, expected)
    assert perm.p_value is not None and "exact" in perm.note
    assert "window is short" in perm.note  # ~220 sessions: the over-rejection warning is attached


def test_reproducible_with_a_seed_and_seed_dependent():
    prices = random_walk(n=900, seed=14)
    result = _run(prices)
    a = timing_permutation_test(prices, result, CFG, n_shifts=40, seed=1)
    b = timing_permutation_test(prices, result, CFG, n_shifts=40, seed=1)
    c = timing_permutation_test(prices, result, CFG, n_shifts=40, seed=2)
    np.testing.assert_array_equal(a.shifts, b.shifts)
    np.testing.assert_array_equal(a.null, b.null)
    assert a.p_value == b.p_value and a.note == b.note
    assert set(a.shifts) != set(c.shifts)
    n_decisions = len(result.daily) - 1
    for perm in (a, c):
        assert len(set(perm.shifts)) == len(perm.shifts) == 40  # sampled without replacement
        assert perm.shifts.min() >= 63 and perm.shifts.max() <= n_decisions - 63
    assert isinstance(a, PermutationResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.p_value = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        a.null[0] = 0.0  # arrays are read-only


@pytest.fixture(scope="module")
def noise_p_values() -> np.ndarray:
    """MA-crossover p-values on 40 zero-arithmetic-drift noise markets (49 shifts each, for speed).

    Windows of ~500 sessions: in much shorter windows the shift null over-rejects (see the
    module docstring of quantlab.nulls), which this test is not meant to exercise.
    """
    p_values = []
    for seed in range(1000, 1040):
        prices = _noise_market(600, seed)
        result = _run(prices, start=100)
        perm = timing_permutation_test(prices, result, CFG, n_shifts=49)
        assert perm.p_value is not None
        p_values.append(perm.p_value)
    return np.array(p_values)


def test_calibrated_under_the_null(noise_p_values):
    # Nominal size of p < 0.05 with 49 shifts is 2/50 = 4%; allow generous sampling noise.
    assert (noise_p_values < 0.05).mean() <= 0.15
    # Under the null p is roughly uniform: neither anti-conservative nor stuck near 1.
    assert 0.35 <= noise_p_values.mean() <= 0.65


def test_planted_momentum_gives_smaller_p_values_than_noise(noise_p_values):
    momentum = []
    for seed in range(500, 506):
        prices = _momentum_market(1000, seed)
        result = _run(prices, start=100)
        momentum.append(timing_permutation_test(prices, result, CFG, n_shifts=49).p_value)
    momentum = np.array(momentum, dtype=float)
    assert np.median(momentum) < np.median(noise_p_values)
    assert momentum.mean() < noise_p_values.mean()
