"""Signal correctness, warm-up handling, causality and parameter validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import indicators as ind
from quantlab.strategies import (
    STRATEGY_CLASSES,
    BollingerMeanReversion,
    BuyAndHold,
    MovingAverageCrossover,
    RSIMeanReversion,
    StrategyParameterError,
    TimeSeriesMomentum,
    create_strategy,
)
from tests.helpers import make_prices, random_walk

FAST_PARAMS = {"ts_momentum": {"lookback": 40}}


def _all_strategies():
    return [cls(**FAST_PARAMS.get(cls.key, {})) for cls in STRATEGY_CLASSES]


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #


def test_sma_hand_values_and_warmup():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = ind.sma(s, 3)
    assert out.isna().sum() == 2
    assert out.tolist()[2:] == [2.0, 3.0, 4.0]


def _reference_rsi(closes, n):
    """Independent, deliberately plain implementation of Wilder's RSI."""
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    out = [None] * len(closes)
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    out[n] = 100 - 100 / (1 + ag / al) if al else 100.0
    for i in range(n, len(deltas)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al) if al else 100.0
    return out


def test_rsi_matches_independent_reference():
    closes = random_walk(n=120, seed=4)["close"]
    ours = ind.rsi_wilder(closes, 14)
    ref = _reference_rsi(closes.tolist(), 14)
    assert ours.iloc[:14].isna().all()
    np.testing.assert_allclose(ours.iloc[14:].to_numpy(), np.array(ref[14:], dtype=float), rtol=1e-12)


def test_rsi_edge_cases():
    rising = pd.Series(np.arange(1.0, 31.0))
    assert ind.rsi_wilder(rising, 14).iloc[-1] == 100.0
    flat = pd.Series([50.0] * 30)
    assert ind.rsi_wilder(flat, 14).iloc[-1] == 50.0


def test_trailing_return_and_skip():
    close = pd.Series([100.0, 110, 121, 100, 90])
    np.testing.assert_allclose(ind.trailing_return(close, 2).iloc[2:], [0.21, 100 / 110 - 1, 90 / 121 - 1])
    skipped = ind.trailing_return(close, 2, skip=1)
    assert skipped.iloc[:3].isna().all()
    np.testing.assert_allclose(skipped.iloc[3:], [0.21, 100 / 110 - 1])


def test_bollinger_uses_population_std():
    close = pd.Series([1.0, 2, 3, 4])
    bands = ind.bollinger_bands(close, 4, 2.0)
    std = np.std([1, 2, 3, 4])  # ddof=0
    assert bands["bb_middle"].iloc[-1] == 2.5
    assert bands["bb_upper"].iloc[-1] == pytest.approx(2.5 + 2 * std)


# --------------------------------------------------------------------------- #
# Strategy signals (hand-computed)
# --------------------------------------------------------------------------- #


def test_ma_crossover_targets_hand_computed():
    prices = make_prices(closes=[5, 4, 3, 2, 1, 2, 3, 4, 5, 6])
    signals = MovingAverageCrossover(short_window=2, long_window=3).generate_signals(prices)
    # SMA2 > SMA3 first at index 6 (2.5 > 2.0); indices 0-1 are warm-up.
    expected = [np.nan, np.nan, 0, 0, 0, 0, 1, 1, 1, 1]
    np.testing.assert_array_equal(signals["target"].to_numpy(), np.array(expected, dtype=float))
    assert signals["reason"].iloc[6].startswith("SMA2")
    assert (signals["reason"].iloc[7:] == "").all()  # no new order while the signal persists


def test_momentum_targets_hand_computed():
    prices = make_prices(closes=[100, 110, 121, 100, 90])
    t = TimeSeriesMomentum(lookback=2, skip=0, threshold=0.0).generate_signals(prices)["target"]
    np.testing.assert_array_equal(t.to_numpy(), [np.nan, np.nan, 1, 0, 0])
    t2 = TimeSeriesMomentum(lookback=2, skip=1, threshold=0.0).generate_signals(prices)["target"]
    np.testing.assert_array_equal(t2.to_numpy(), [np.nan, np.nan, np.nan, 1, 0])


def test_rsi_persistent_oversold_produces_single_entry():
    # Steady decline keeps RSI pinned near zero for many sessions, then a rally.
    closes = list(np.linspace(100, 60, 40)) + list(np.linspace(61, 90, 30))
    signals = RSIMeanReversion(window=5, entry_threshold=30, exit_threshold=60).generate_signals(make_prices(closes))
    target = signals["target"].fillna(0)
    entries = int(((target == 1) & (target.shift(1, fill_value=0) == 0)).sum())
    exits = int(((target == 0) & (target.shift(1, fill_value=0) == 1)).sum())
    assert entries == 1 and exits == 1
    rsi = signals["rsi"]
    assert (rsi[target.diff() == 1] < 30).all()


def _check_hysteresis(target, enter, exit_):
    prev = 0.0
    for t, e, x in zip(target, enter, exit_):
        if np.isnan(t):
            continue
        expected = 1.0 if (prev == 0 and e) else 0.0 if (prev == 1 and x) else prev
        assert t == expected
        prev = t


def test_rsi_state_machine_property(walk):
    strat = RSIMeanReversion(window=14, entry_threshold=35, exit_threshold=55)
    s = strat.generate_signals(walk)
    _check_hysteresis(s["target"].to_numpy(), (s["rsi"] < 35).to_numpy(), (s["rsi"] > 55).to_numpy())
    assert s["target"].dropna().diff().abs().sum() > 2  # it actually traded


def test_bollinger_state_machine_property(walk):
    for exit_band in ("middle", "upper"):
        strat = BollingerMeanReversion(window=20, num_std=1.5, exit_band=exit_band)
        s = strat.generate_signals(walk)
        level = s["bb_middle"] if exit_band == "middle" else s["bb_upper"]
        _check_hysteresis(s["target"].to_numpy(), (walk["close"] < s["bb_lower"]).to_numpy(),
                          (walk["close"] > level).to_numpy())


def test_buy_and_hold_is_always_long(walk):
    assert (BuyAndHold().generate_signals(walk)["target"] == 1).all()


# --------------------------------------------------------------------------- #
# Warm-up and look-ahead
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", _all_strategies(), ids=lambda s: s.key)
def test_warmup_length_matches_declaration(strategy, walk):
    target = strategy.generate_signals(walk)["target"]
    assert int(target.isna().sum()) == strategy.warmup_sessions
    assert target.iloc[strategy.warmup_sessions:].notna().all()


@pytest.mark.parametrize("strategy", _all_strategies(), ids=lambda s: s.key)
def test_changing_future_prices_cannot_change_earlier_signals(strategy):
    """Regression test for look-ahead bias, run for every strategy."""
    prices = random_walk(n=900, seed=13)  # long enough for the ML model's ~524-session warm-up
    k = 700
    altered = prices.copy()
    altered.iloc[k:, :4] = altered.iloc[k:, :4].to_numpy() * np.linspace(0.5, 2.0, len(prices) - k)[:, None]
    base = strategy.generate_signals(prices)
    alt = strategy.generate_signals(altered)
    pd.testing.assert_frame_equal(base.iloc[:k], alt.iloc[:k])
    if strategy.key != "buy_and_hold":
        numeric = base.columns.drop(["target", "reason"])
        assert not base[numeric].iloc[k:].equals(alt[numeric].iloc[k:])  # not vacuous


@pytest.mark.parametrize("strategy", _all_strategies(), ids=lambda s: s.key)
def test_signals_only_take_allowed_values(strategy, walk):
    target = strategy.generate_signals(walk)["target"].dropna()
    assert ((target >= 0) & (target <= 1)).all()


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key,params",
    [
        ("ma_crossover", {"short_window": 50, "long_window": 20}),
        ("ma_crossover", {"short_window": 20, "long_window": 20}),
        ("ma_crossover", {"short_window": 2.5}),
        ("ma_crossover", {"short_window": 1}),
        ("ma_crossover", {"ma_type": "WMA"}),
        ("ts_momentum", {"lookback": 20, "skip": 20}),
        ("ts_momentum", {"threshold": float("nan")}),
        ("rsi_reversion", {"entry_threshold": 70, "exit_threshold": 30}),
        ("rsi_reversion", {"window": 0}),
        ("bollinger_reversion", {"num_std": 10}),
        ("bollinger_reversion", {"unknown": 1}),
    ],
)
def test_nonsensical_parameters_are_rejected(key, params):
    with pytest.raises(StrategyParameterError):
        create_strategy(key, params)


def test_parameter_grid_excludes_invalid_combinations():
    grid = MovingAverageCrossover.parameter_grid({"short_window": (10, 50), "long_window": (20, 50)})
    pairs = {(g["short_window"], g["long_window"]) for g in grid}
    assert pairs == {(10, 20), (10, 50)}


def test_defaults_are_recognised():
    assert MovingAverageCrossover().uses_default_parameters()
    assert not MovingAverageCrossover(short_window=10).uses_default_parameters()
    assert MovingAverageCrossover().params == {"short_window": 20, "long_window": 50, "ma_type": "SMA"}


def test_every_strategy_is_documented():
    for cls in STRATEGY_CLASSES:
        assert cls.doc.summary and cls.doc.entry and cls.doc.exit
        assert cls.doc.failure_modes and cls.doc.assumptions
