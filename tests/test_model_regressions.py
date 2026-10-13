"""Regression tests for model and strategy defects found in an adversarial audit (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import ml
from quantlab.backtest import run_backtest
from quantlab.config import BacktestConfig
from quantlab.strategies import (
    STRATEGIES,
    AdaptiveTrendEnsemble,
    StrategyParameterError,
    WalkForwardLogistic,
    _deadband_steps,
    create_strategy,
)
from tests.helpers import make_prices, random_walk


@pytest.mark.parametrize("step", AdaptiveTrendEnsemble.param_specs[-1].choices)
def test_every_exposure_step_can_reach_fully_invested_and_flat(step):
    """Steps such as 0.4 used to round 1.0 down to 0.8 (banker's rounding), capping exposure."""
    ramp = np.linspace(0, 1, 1001)
    assert np.nanmax(_deadband_steps(ramp, step)) == 1.0
    assert _deadband_steps(np.array([1.0]), step)[0] == 1.0
    assert _deadband_steps(np.r_[ramp, ramp[::-1]], step)[-1] == 0.0


def test_ensemble_reaches_full_exposure_in_a_calm_uptrend():
    close = 100 * np.exp(np.cumsum(np.full(700, 0.0008) + np.random.default_rng(1).normal(0, 0.002, 700)))
    signals = AdaptiveTrendEnsemble(step=0.25).generate_signals(make_prices(close))
    assert signals["target"].max() == 1.0


@pytest.mark.parametrize("train_window", [126, 147, 231, 251])
def test_ml_training_windows_too_short_to_fit_are_rejected(train_window):
    with pytest.raises(StrategyParameterError):
        create_strategy("ml_logistic", {"train_window": train_window})


def test_ml_keeps_its_last_model_when_a_refit_window_has_one_class():
    """A one-class training window used to drop the target to NaN (forced liquidation)."""
    rng = np.random.default_rng(3)
    rets = np.r_[rng.normal(0.0004, 0.01, 700), rng.normal(0.004, 0.003, 500), rng.normal(0, 0.012, 600)]
    prices = make_prices(100 * np.exp(np.cumsum(rets)))
    model = WalkForwardLogistic(horizon=21, train_window=252)
    signals = model.generate_signals(prices)
    first = signals["target"].first_valid_index()
    assert signals.loc[first:, "target"].notna().all()
    fits = model.walk_forward_fit(prices).fits
    assert fits["skipped"].any()  # the scenario does exercise the skipped-refit path
    assert not signals["reason"].fillna("").str.contains("nan%").any()


def test_ml_labels_measure_the_tradeable_move_from_the_next_open():
    close = pd.Series([100.0, 110.0, 105.0, 108.0, 120.0])
    opens = pd.Series([100.0, 112.0, 104.0, 109.0, 118.0])
    labels = ml.forward_labels(close, 2, opens)
    # t=0: enter at open[1]=112, exit at close[2]=105 -> down, although close[2] > close[0].
    assert labels.tolist()[:3] == [0.0, 1.0, 1.0] and labels.iloc[3:].isna().all()
    assert ml.forward_labels(close, 2).iloc[0] == 1.0  # close-to-close would have said "up"


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_labels_distinguish_every_non_default_parameter(key):
    cls = STRATEGIES[key]
    base = cls().label
    for spec in cls.param_specs:
        if spec.kind == "choice":
            value = next(c for c in spec.choices if c != spec.default)
        else:
            value = spec.default + (spec.step or 1)
            if spec.max is not None and value > spec.max:
                value = spec.default - (spec.step or 1)
        params = {spec.name: value}
        if not cls.is_valid(params):
            continue
        assert cls(**params).label != base, (key, spec.name)


@pytest.mark.parametrize("params", [{"skip": True}, {"threshold": False}])
def test_booleans_are_not_accepted_as_numbers(params):
    with pytest.raises(StrategyParameterError):
        create_strategy("ts_momentum", params)


@pytest.mark.parametrize(("key", "params"), [
    ("ma_crossover", {}), ("ma_crossover", {"ma_type": "EMA", "short_window": 50, "long_window": 200}),
    ("ts_momentum", {}), ("rsi_reversion", {}), ("bollinger_reversion", {}), ("adaptive_ensemble", {}),
])
def test_results_do_not_depend_on_how_much_history_was_loaded(key, params):
    """Regression: a recursive EMA and Wilder's RSI are seeded at the first loaded close, and the
    mean-reversion and ensemble strategies enter the evaluation window with a position state built from
    whatever history was loaded. Changing only the warm-up history changed signals on 6-18 of 60 random
    walks (final value by up to 8.6%). With ``history_sessions`` of history, more history changes nothing."""
    strategy = create_strategy(key, params)
    for seed in range(12):
        prices = random_walk(1500, seed=seed)
        start = prices.index[900]
        full = run_backtest(prices, strategy, BacktestConfig(), start)
        trimmed = run_backtest(prices.iloc[900 - strategy.history_sessions - 1:], strategy, BacktestConfig(), start)
        np.testing.assert_array_equal(full.signals["target"].to_numpy(),
                                      trimmed.signals["target"].reindex(full.signals.index).to_numpy())
        assert full.equity.iloc[-1] == pytest.approx(trimmed.equity.iloc[-1], rel=1e-9)


def test_ema_is_a_finite_exponential_kernel():
    from quantlab import indicators as ind

    x = pd.Series(np.random.default_rng(3).normal(100, 5, 80))
    out = ind.ema(x, 10)
    k, alpha = ind.EMA_SPANS * 10, 2 / 11
    assert out.iloc[:k - 1].isna().all() and out.iloc[k - 1:].notna().all()
    w = (1 - alpha) ** np.arange(k)
    t = 60
    assert out.iloc[t] == pytest.approx(float(np.dot(w, x.iloc[t - k + 1:t + 1].to_numpy()[::-1]) / w.sum()), rel=1e-12)
    # Close to the recursive EMA once the recursion has converged (truncated weight < 0.25%).
    recursive = x.ewm(span=10, adjust=False).mean()
    assert np.abs(out.iloc[k:] - recursive.iloc[k:]).max() < 0.01 * x.std()


def test_ml_declares_that_its_results_depend_on_the_data_start():
    assert create_strategy("ml_logistic").history_sessions is None
