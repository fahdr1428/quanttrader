"""Input validation and market-data quality checks."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.calendar import expected_sessions, session_gaps
from quantlab.validation import (
    InputValidationError,
    QualityReport,
    Severity,
    check_staleness,
    clean_price_frame,
    validate_date_range,
    validate_ticker,
)
from tests.helpers import make_prices


def _codes(report, severity=None):
    return {i.code for i in report.issues if severity is None or i.severity is severity}


@pytest.mark.parametrize("raw,expected", [("spy", "SPY"), (" brk-b ", "BRK-B"), ("^GSPC", "^GSPC"), ("EURUSD=X", "EURUSD=X")])
def test_valid_tickers(raw, expected):
    assert validate_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "SP Y", "$$$", "A" * 16, None, "SPY;DROP"])
def test_invalid_tickers(raw):
    with pytest.raises(InputValidationError):
        validate_ticker(raw)


def test_date_range_rules():
    today = date(2024, 6, 1)
    start, end = validate_date_range("2020-01-01", "2024-05-31", today=today)
    assert start == pd.Timestamp("2020-01-01") and end == pd.Timestamp("2024-05-31")
    with pytest.raises(InputValidationError, match="future"):
        validate_date_range("2020-01-01", "2024-07-01", today=today)
    with pytest.raises(InputValidationError, match="after the start"):
        validate_date_range("2024-01-01", "2023-01-01", today=today)
    with pytest.raises(InputValidationError, match="at least"):
        validate_date_range("2024-01-01", "2024-01-20", today=today)


def test_clean_frame_passes_through_good_data():
    prices = make_prices(np.linspace(100, 110, 60), start="2024-01-02")
    clean, report = clean_price_frame(prices)
    assert len(clean) == 60 and not report.is_blocking
    assert list(clean.columns[:5]) == ["open", "high", "low", "close", "volume"]


def test_missing_prices_are_removed_and_reported_never_filled():
    prices = make_prices(np.linspace(100, 110, 60))
    prices.iloc[10, prices.columns.get_loc("close")] = np.nan
    clean, report = clean_price_frame(prices)
    assert prices.index[10] not in clean.index
    assert len(clean) == 59
    issue = next(i for i in report.warnings if i.code == "missing_prices")
    assert issue.dates == (prices.index[10].strftime("%Y-%m-%d"),)
    assert list(report.removed["reason"]) == ["missing price field"]


def test_non_positive_prices_are_removed():
    prices = make_prices(np.linspace(100, 110, 60))
    prices.iloc[5, prices.columns.get_loc("open")] = 0.0
    prices.iloc[6, prices.columns.get_loc("low")] = -1.0
    clean, report = clean_price_frame(prices)
    assert len(clean) == 58
    assert "non_positive_prices" in _codes(report, Severity.WARNING)


def test_duplicates_identical_kept_once_conflicting_removed():
    prices = make_prices(np.linspace(100, 110, 60))
    dup_same = prices.iloc[[5]]
    dup_diff = prices.iloc[[8]].copy()
    dup_diff["close"] += 1
    frame = pd.concat([prices, dup_same, dup_diff])
    clean, report = clean_price_frame(frame)
    assert clean.index.is_unique and clean.index.is_monotonic_increasing
    assert prices.index[5] in clean.index
    assert prices.index[8] not in clean.index
    assert {"duplicate_timestamps", "conflicting_duplicates", "unsorted"} <= _codes(report)


def test_malformed_columns_and_empty_frames_block():
    prices = make_prices(np.linspace(100, 110, 60)).drop(columns=["volume"])
    _, report = clean_price_frame(prices)
    assert report.is_blocking and "malformed_columns" in _codes(report, Severity.ERROR)
    _, report = clean_price_frame(pd.DataFrame())
    assert report.is_blocking


def test_non_numeric_values_are_flagged():
    prices = make_prices(np.linspace(100, 110, 60)).astype({"close": object})
    prices.iloc[3, prices.columns.get_loc("close")] = "n/a"
    clean, report = clean_price_frame(prices)
    assert "malformed_values" in _codes(report) and len(clean) == 59


def test_too_few_observations_block():
    _, report = clean_price_frame(make_prices(np.linspace(100, 110, 20)))
    assert "insufficient_observations" in _codes(report, Severity.ERROR)


def test_extreme_moves_and_ohlc_inconsistency_are_flagged_but_kept():
    closes = np.linspace(100, 110, 60)
    closes[30] = closes[29] * 0.5  # a -50% day, e.g. an unadjusted 2:1 split
    prices = make_prices(closes)
    prices.iloc[40, prices.columns.get_loc("high")] = prices.iloc[40]["close"] * 0.9
    clean, report = clean_price_frame(prices)
    assert len(clean) == 60
    assert {"extreme_returns", "ohlc_inconsistent"} <= _codes(report, Severity.WARNING)


def test_holidays_are_not_reported_as_missing():
    sessions = expected_sessions("2024-01-01", "2024-12-31")
    assert pd.Timestamp("2024-07-04") not in sessions  # Independence Day
    assert pd.Timestamp("2024-03-29") not in sessions  # Good Friday
    assert len(sessions) == 252  # the actual 2024 NYSE session count
    missing, unexpected = session_gaps(sessions)
    assert missing.empty and unexpected.empty


def test_genuinely_missing_weekday_is_reported_as_info():
    sessions = expected_sessions("2024-01-01", "2024-06-28")
    observed = sessions.drop(pd.Timestamp("2024-03-12"))
    prices = make_prices(np.linspace(100, 110, len(observed)), index=observed)
    _, report = clean_price_frame(prices)
    issue = next(i for i in report.issues if i.code == "missing_sessions")
    assert issue.severity is Severity.INFO and issue.dates == ("2024-03-12",)


def test_weekend_observation_is_flagged():
    idx = pd.bdate_range("2024-01-02", periods=60).append(pd.DatetimeIndex(["2024-01-06"])).sort_values()
    prices = make_prices(np.linspace(100, 110, 61), index=idx)
    _, report = clean_price_frame(prices)
    assert "unexpected_sessions" in _codes(report, Severity.WARNING)


def test_staleness_check():
    report = QualityReport()
    check_staleness(pd.Timestamp("2024-01-31"), pd.Timestamp("2024-03-29"), report, today=date(2024, 6, 1))
    assert "stale_data" in _codes(report)
    fresh = QualityReport()
    check_staleness(pd.Timestamp("2024-03-28"), pd.Timestamp("2024-03-29"), fresh, today=date(2024, 6, 1))
    assert not fresh.issues


def test_timezone_aware_index_is_normalised():
    prices = make_prices(np.linspace(100, 110, 60))
    prices.index = prices.index.tz_localize("America/New_York")
    clean, _ = clean_price_frame(prices)
    assert clean.index.tz is None
    assert clean.index[0] == pd.Timestamp("2024-01-02")
