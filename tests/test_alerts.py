"""ALERTS mode: conditions evaluated on the latest validated data; data failures never count as conditions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantlab.alerts import ALERT_KINDS, AlertRule, describe_alert_kinds, evaluate_alerts, triggered_events
from quantlab.data import ProviderConnectionError, TickerNotFoundError
from tests.technical_fixtures import dataset, loader, ohlcv, trend


def _one(rule, frame=None, entry=None):
    entry = entry if entry is not None else dataset(rule.ticker, frame)
    return evaluate_alerts([rule], loader({rule.ticker: entry}))[0]


def test_every_kind_is_documented_with_its_parameters():
    lines = describe_alert_kinds()
    assert len(lines) == len(ALERT_KINDS) and all("Parameters:" in line and "Value reported:" in line for line in lines)


def test_price_levels_are_strict_comparisons_on_the_latest_close():
    frame = trend(60, 0.001, level=100)
    close = float(frame["close"].iloc[-1])
    above = _one(AlertRule("UP", "price_above", {"level": close - 0.01}), frame)
    at = _one(AlertRule("UP", "price_above", {"level": close}), frame)
    below = _one(AlertRule("UP", "price_below", {"level": close + 0.01}), frame)
    assert above.triggered and not at.triggered and below.triggered
    assert above.value == pytest.approx(close) and above.as_of == frame.index[-1].date()
    assert above.data_status == "ok" and above.provenance is not None


def test_rsi_alerts_use_wilder_rsi_and_need_enough_history():
    rising = trend(60, 0.002)                     # no down days: RSI is exactly 100
    assert _one(AlertRule("UP", "rsi_above", {"level": 70}), rising).triggered
    assert not _one(AlertRule("UP", "rsi_below", {"level": 30}), rising).triggered
    short = _one(AlertRule("UP", "rsi_above", {"level": 70}), trend(10, 0.002))
    assert not short.triggered and short.data_status in {"insufficient_history", "unusable"}


def test_moving_average_cross_is_detected_only_when_it_happened_and_still_holds():
    """A fall then a rise: SMA5 crosses above SMA20 near the end; an old cross that reversed does not trigger."""
    closes = np.r_[np.linspace(120, 80, 60), np.linspace(80, 100, 15)]
    frame = ohlcv(closes)
    fast, slow = pd.Series(closes).rolling(5).mean(), pd.Series(closes).rolling(20).mean()
    cross_idx = [i for i in range(20, len(closes)) if fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]]
    assert cross_idx, "fixture must contain a cross"
    within = len(closes) - cross_idx[-1]
    hit = _one(AlertRule("X", "sma_cross_up", {"fast": 5, "slow": 20, "within": within}), frame)
    assert hit.triggered and "Crossed above" in hit.detail
    missed = _one(AlertRule("X", "sma_cross_up", {"fast": 5, "slow": 20, "within": max(1, within - 1)}), frame)
    assert not missed.triggered
    assert not _one(AlertRule("X", "sma_cross_down", {"fast": 5, "slow": 20, "within": within}), frame).triggered


def test_volume_spike_and_new_52_week_extremes():
    frame = trend(300, 0.001, volume=1e6)
    noisy = frame["volume"].to_numpy() * np.exp(np.random.default_rng(2).normal(0, 0.1, len(frame)))
    noisy[-1] = 8e6
    frame["volume"] = noisy
    spike = _one(AlertRule("V", "volume_z_above", {"level": 3}), frame)
    assert spike.triggered and spike.value > 3
    assert _one(AlertRule("V", "new_52w_high"), frame).triggered          # a steady rise makes new highs
    assert not _one(AlertRule("V", "new_52w_low"), frame).triggered
    assert _one(AlertRule("V", "new_52w_high", {"window": 400}), frame).data_status == "insufficient_history"


def test_signal_change_triggers_only_when_the_target_changes_at_the_latest_close():
    closes = np.r_[np.linspace(120, 80, 80), np.linspace(80, 82, 3)]
    frame = ohlcv(closes)
    rule = AlertRule("S", "signal_change", {"strategy": "ma_crossover",
                                             "strategy_params": {"short_window": 2, "long_window": 5}})
    event = _one(rule, frame)
    flat = _one(rule, trend(80, 0.002))
    assert isinstance(event.triggered, bool) and not flat.triggered
    assert "no change" in flat.detail and flat.value == pytest.approx(1.0)


def test_data_failures_are_reported_and_never_trigger():
    rule = AlertRule("GONE", "price_above", {"level": 1})
    missing = evaluate_alerts([rule], loader({"GONE": TickerNotFoundError("GONE", "unknown ticker")}))[0]
    offline = evaluate_alerts([rule], loader({"GONE": ProviderConnectionError("GONE", "offline")}))[0]
    for event in (missing, offline):
        assert not event.triggered and event.data_status == "load_failed" and event.as_of is None
    assert triggered_events([missing, offline]) == []


def test_each_ticker_is_loaded_once_and_rules_keep_their_order():
    calls = []
    frame = trend(300, 0.001)

    def load(ticker):
        calls.append(ticker)
        return dataset(ticker, frame)

    rules = [AlertRule("A", "price_above", {"level": 1}), AlertRule("B", "price_below", {"level": 1}),
             AlertRule("A", "new_52w_high")]
    events = evaluate_alerts(rules, load)
    assert calls == ["A", "B"] and [e.rule for e in events] == rules
    assert [e.triggered for e in events] == [True, False, True]
    json.dumps([e.to_dict() for e in events])  # JSON-serialisable for the agent and the audit log


@pytest.mark.parametrize(("kind", "params"), [
    ("teleport", {}), ("price_above", {}), ("price_above", {"level": -1}), ("price_above", {"level": True}),
    ("rsi_below", {"level": 150}), ("sma_cross_up", {"fast": 50, "slow": 20}), ("sma_cross_up", {"within": 0}),
    ("signal_change", {"strategy": "nope"}), ("price_above", {"level": 1, "colour": "red"}),
])
def test_invalid_rules_are_rejected_when_created(kind, params):
    with pytest.raises(ValueError):
        AlertRule("SPY", kind, params)


def test_52_week_extremes_compare_printed_prices_on_total_return_data():
    """Dividend back-adjustment lowers old highs; a 'new high' must beat the price that actually printed."""
    from quantlab.data import AdjustmentMode, CSVProvider, load_market_data
    from tests.technical_fixtures import NOW, sessions

    idx = sessions(300)
    close = np.full(300, 100.0)
    high = close * 1.01
    high[50] = 110.55                                    # the printed 52-week high
    high[-1] = 108.54                                    # today's high: below it
    factor = np.where(np.arange(300) < 250, 0.94, 1.0)   # dividends after session 250 lowered earlier prices
    raw = pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"), "Open": close, "High": high, "Low": close * 0.99,
                        "Close": close, "Adj Close": close * factor, "Volume": 1e6})
    ds = load_market_data("DIV", idx[0], idx[-1], CSVProvider(raw.to_csv(index=False).encode(), "div.csv"),
                          adjustment=AdjustmentMode.TOTAL_RETURN, now=NOW, today=idx[-1].date())
    event = evaluate_alerts([AlertRule("DIV", "new_52w_high")], loader({"DIV": ds}))[0]
    assert not event.triggered and "printed" in event.detail and "110.55" in event.detail


def test_infinite_parameters_are_invalid_rules_not_crashes():
    with pytest.raises(ValueError):
        AlertRule("SPY", "new_52w_high", json.loads('{"window": Infinity}'))
