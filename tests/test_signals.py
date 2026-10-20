"""Signal specifications for every strategy, and the current signal state computed by the strategy itself."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import signals
from quantlab.signals import (
    COMMON_LIMITATIONS,
    NON_DEFAULT_WARNING,
    SignalSpec,
    all_signal_specs,
    current_signal,
    signal_spec,
)
from quantlab.strategies import STRATEGIES, StrategyParameterError, create_strategy
from tests.helpers import random_walk
from tests.technical_fixtures import ohlcv, trend

SPEC_FIELDS = ("key", "name", "hypothesis", "inputs", "calculation", "entry", "exit", "holding_period",
               "invalidation", "benchmark", "limitations")

# --------------------------------------------------------------------------- #
# Specifications
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_every_strategy_has_a_complete_signal_spec(key):
    """Every registered strategy states all of the agent's mandatory signal fields, none of them empty."""
    spec = signal_spec(key)
    strategy = create_strategy(key)
    for name in SPEC_FIELDS:
        value = getattr(spec, name)
        assert value and all(value) if isinstance(value, tuple) else value, name
    assert spec.key == key and spec.name == strategy.label
    assert spec.hypothesis == strategy.doc.summary
    assert spec.limitations[:len(strategy.doc.failure_modes)] == strategy.doc.failure_modes
    assert set(COMMON_LIMITATIONS) <= set(spec.limitations)
    assert spec.assumptions == strategy.doc.assumptions and spec.references == strategy.doc.references
    assert spec.entry.startswith(strategy.doc.entry) and spec.exit.startswith(strategy.doc.exit)
    assert spec.params == strategy.params and spec.warmup_sessions == strategy.warmup_sessions
    assert NON_DEFAULT_WARNING not in spec.limitations


def test_every_registered_strategy_has_a_dedicated_builder():
    """A new strategy must get its own parameter-aware specification, not just the generic fallback."""
    assert set(signals._BUILDERS) == set(STRATEGIES)


def test_generic_fallback_still_produces_a_complete_spec(monkeypatch):
    """If a builder were missing, the spec would still be complete, built from the strategy's documentation."""
    monkeypatch.delitem(signals._BUILDERS, "ma_crossover")
    spec = signal_spec("ma_crossover")
    assert all(getattr(spec, name) for name in SPEC_FIELDS)
    assert "short_window = 20" in spec.calculation


@pytest.mark.parametrize("key,params,expected", [
    ("ma_crossover", {"short_window": 10, "long_window": 100, "ma_type": "EMA"},
     ("EMA10 closes above EMA100", "EMA10 closes at or below EMA100", "finite kernel")),
    ("ts_momentum", {"lookback": 126, "skip": 21, "threshold": 0.05},
     ("126-session trailing return, skipping the latest 21 sessions is above +5.00%", "close(t-21) / close(t-147)")),
    ("rsi_reversion", {"window": 7, "entry_threshold": 25.0, "exit_threshold": 60.0},
     ("RSI(7) closes below 25", "RSI(7) closes above 60", "no price stop")),
    ("bollinger_reversion", {"window": 30, "num_std": 2.5, "exit_band": "upper"},
     ("30-session, 2.5-sigma lower band", "above the upper band")),
    ("adaptive_ensemble", {"target_vol": 0.10, "step": 0.25},
     ("10% / EWM volatility", "25% steps")),
    ("ml_logistic", {"horizon": 10, "full_confidence": 0.70},
     ("higher in 10 sessions", "100% at 70%", "(P - 0.5) / (0.70 - 0.5)")),
])
def test_spec_text_states_the_parameters_actually_used(key, params, expected):
    """The specification quotes the validated parameter values, so text and code cannot disagree."""
    spec = signal_spec(key, params)
    text = " ".join([spec.calculation, spec.entry, spec.exit, spec.holding_period, spec.invalidation])
    for fragment in expected:
        assert fragment in text, fragment
    assert NON_DEFAULT_WARNING in spec.limitations  # non-default parameters may have been searched for
    assert spec.params == create_strategy(key, params).params


def test_mean_reversion_specs_disclose_the_missing_stop():
    """Strategies without a stop say so in their invalidation text instead of implying bounded losses."""
    for key in ("rsi_reversion", "bollinger_reversion"):
        assert "no stop" in signal_spec(key).invalidation


@pytest.mark.parametrize("key,params", [("ma_crossover", {"short_window": 50, "long_window": 20}),
                                        ("rsi_reversion", {"window": 0}), ("nope", None),
                                        ("ts_momentum", {"unknown": 1})])
def test_invalid_parameters_or_unknown_strategy_raise(key, params):
    """A specification is never written for a signal that cannot run."""
    with pytest.raises(StrategyParameterError):
        signal_spec(key, params)


def test_spec_to_dict_is_json_and_all_specs_follow_the_registry():
    """Specs serialise to JSON for the agent's tools; all_signal_specs lists every strategy in registry order."""
    specs = all_signal_specs()
    assert [s.key for s in specs] == list(STRATEGIES)
    for spec in specs:
        assert isinstance(spec, SignalSpec)
        payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
        assert set(SPEC_FIELDS) <= set(payload) and isinstance(payload["limitations"], list)


# --------------------------------------------------------------------------- #
# Current signal
# --------------------------------------------------------------------------- #


FAST = {"ts_momentum": {"lookback": 40}}


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_current_signal_agrees_with_generate_signals(key):
    """The reported state is read from Strategy.generate_signals, the same code the backtester runs."""
    prices = random_walk(n=900, seed=13)
    strategy = create_strategy(key, FAST.get(key))
    frame = strategy.generate_signals(prices)
    state = current_signal(strategy, prices)
    assert state["status"] == "ok" and state["as_of"] == prices.index[-1].date().isoformat()
    assert state["latest_target"] == pytest.approx(frame["target"].iloc[-1])
    changed = frame.index[frame["reason"] != ""]
    assert state["last_change_date"] == changed[-1].date().isoformat()
    assert state["last_change_reason"] == frame.loc[changed[-1], "reason"]
    assert state["sessions_since_change"] == len(frame) - 1 - frame.index.get_loc(changed[-1])
    indicator_columns = [c for c in frame.columns if c not in ("target", "reason")]
    assert list(state["conditions"]) == indicator_columns
    for column in indicator_columns:
        expected, got = frame[column].iloc[-1], state["conditions"][column]
        if pd.isna(expected):
            assert got is None  # missing stays missing in the JSON
        else:
            assert got == pytest.approx(expected)
    json.dumps(state, allow_nan=False)


def _cross_on_last_session() -> pd.DataFrame:
    # SMA2 vs SMA3 on a falling series that jumps on the last session: 2-session average 6 > 3-session average 5.
    return ohlcv([10.0, 9, 8, 7, 6, 5, 4, 3, 2, 10])


def test_a_change_at_the_latest_close_is_reported_as_unexecuted():
    """A target change on the last session has not traded yet: its order would fill at the next open."""
    strategy = create_strategy("ma_crossover", {"short_window": 2, "long_window": 3})
    prices = _cross_on_last_session()
    state = current_signal(strategy, prices)
    assert state["unexecuted"] is True and state["sessions_since_change"] == 0
    assert state["latest_target"] == 1.0 and state["previous_target"] == 0.0 and state["position"] == "long"
    assert state["last_change_from"] == 0.0 and state["last_change_to"] == 1.0
    assert state["last_change_date"] == prices.index[-1].date().isoformat()
    assert state["conditions"] == {"ma_short": 6.0, "ma_long": 5.0}
    assert any("next session's open" in n for n in state["notes"])


def test_an_older_change_is_reported_as_executed():
    """One session later the same change has executed at that session's open."""
    strategy = create_strategy("ma_crossover", {"short_window": 2, "long_window": 3})
    prices = ohlcv([10.0, 9, 8, 7, 6, 5, 4, 3, 2, 10, 11])
    state = current_signal(strategy, prices)
    assert state["unexecuted"] is False and state["sessions_since_change"] == 1
    assert state["last_change_date"] == prices.index[-2].date().isoformat()


def test_warm_up_has_no_target_and_says_so():
    """Before the first valid signal the target is None (treated as cash), never a made-up value."""
    state = current_signal("ts_momentum", trend(100, 0.001))
    assert state["status"] == "warm_up" and state["latest_target"] is None and state["position"] is None
    assert state["last_change_date"] is None and state["unexecuted"] is False
    assert state["conditions"] == {"trailing_return": None}
    assert any("first signal on session 253" in n for n in state["notes"])


def test_a_strategy_that_never_changed_reports_no_change():
    """A trend strategy on a steadily falling series stays in cash; no change date is invented."""
    state = current_signal(create_strategy("ma_crossover", {"short_window": 5, "long_window": 20}),
                           trend(80, -0.002))
    assert state["status"] == "ok" and state["latest_target"] == 0.0 and state["position"] == "cash"
    assert state["last_change_date"] is None and state["sessions_since_change"] is None
    assert any("has not changed" in n for n in state["notes"])


def test_partial_targets_and_history_dependence_are_disclosed():
    """Partial exposure mentions the rebalance band; a short history for a stateful strategy is flagged."""
    ensemble = current_signal("adaptive_ensemble", random_walk(n=600, seed=3))
    assert any("rebalance band" in n for n in ensemble["notes"])
    rsi = current_signal("rsi_reversion", random_walk(n=200, seed=3))
    assert any("can depend on where the data starts" in n for n in rsi["notes"])


def test_current_signal_is_causal():
    """The state as of session k equals the full-history signal at k: later prices are never used."""
    prices = random_walk(n=500, seed=4)
    strategy = create_strategy("bollinger_reversion")
    full = strategy.generate_signals(prices)
    for k in (100, 250, 499):
        state = current_signal(strategy, prices.iloc[:k])
        assert state["latest_target"] == pytest.approx(full["target"].iloc[k - 1])
        assert state["conditions"]["bb_lower"] == pytest.approx(full["bb_lower"].iloc[k - 1])


@pytest.mark.parametrize("bad", ["empty", "range_index", "unsorted", "duplicates"])
def test_current_signal_requires_validated_prices(bad):
    """Empty, undated, unsorted or duplicated sessions are refused."""
    prices = random_walk(n=120)
    if bad == "empty":
        prices = prices.iloc[:0]
    elif bad == "range_index":
        prices = prices.reset_index(drop=True)
    elif bad == "unsorted":
        prices = prices.iloc[::-1]
    else:
        prices = pd.concat([prices.iloc[:60], prices.iloc[59:]])
    with pytest.raises(ValueError):
        current_signal("ma_crossover", prices)


def test_current_signal_accepts_a_key_and_uses_default_parameters():
    """A strategy key means the default configuration."""
    prices = random_walk(n=300)
    assert current_signal("ma_crossover", prices) == current_signal(create_strategy("ma_crossover"), prices)
    assert current_signal("ma_crossover", prices)["params"] == {"short_window": 20, "long_window": 50,
                                                                "ma_type": "SMA"}
    assert np.isfinite(current_signal("ma_crossover", prices)["conditions"]["ma_long"])
