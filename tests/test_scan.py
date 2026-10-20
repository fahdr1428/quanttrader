"""SCAN mode: documented criteria, pass/fail per criterion, and failures that are listed, never dropped."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantlab.data import AdjustmentMode, CacheStatus, ProviderConnectionError, SyntheticProvider, TickerNotFoundError
from quantlab.data import load_market_data
from quantlab.scan import CRITERIA_ORDER, SURVIVORSHIP_NOTE, ScanCriteria, ScanResult, scan_universe
from quantlab.technicals import relative_strength
from tests.technical_fixtures import csv_dataset, dataset, loader, trend, walk


def _universe() -> dict[str, object]:
    return {
        "LEAD": dataset("LEAD", trend(300, 0.002, level=50)),            # ~91, liquid, steady uptrend
        "PENNY": dataset("PENNY", trend(300, 0.002, level=2, volume=5_000_000)),  # close below 5, still liquid
        "THIN": dataset("THIN", trend(300, 0.002, level=50, volume=1_000)),  # ~0.09m a day traded
        "YOUNG": dataset("YOUNG", trend(100, 0.002, level=50)),          # 100 sessions only
        "FALL": dataset("FALL", trend(300, -0.002, level=200)),          # steady downtrend
    }


def _bench() -> pd.DataFrame:
    return trend(300, 0.001, level=100)


def _row(result: ScanResult, ticker: str) -> pd.Series:
    return result.rows.set_index("ticker").loc[ticker]


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #


def test_default_criteria_and_their_plain_english_description():
    """Defaults screen price, liquidity and history; describe() states each and the combination rule."""
    criteria = ScanCriteria()
    assert criteria.active() == ("min_price", "min_avg_dollar_volume", "min_history_sessions")
    text = criteria.describe()
    assert text[0].startswith("Price: the latest close is at least 5.00")
    assert text[1].startswith("Liquidity: average daily dollar volume") and "5,000,000" in text[1]
    assert text[2].startswith("History: at least 252 validated daily sessions")
    assert any("counts as failed, never as passed" in line for line in text)
    assert any("listed as failures" in line for line in text)


def test_every_criterion_is_described_when_active():
    """Each optional criterion, once switched on, appears in the description with its threshold."""
    criteria = ScanCriteria(min_price=None, min_avg_dollar_volume=None, min_history_sessions=100, trend="uptrend",
                            rsi_below=70, rsi_above=40, min_rs_126=0.05, max_atr_pct=0.03)
    assert criteria.active() == tuple(c for c in CRITERIA_ORDER if c not in ("min_price", "min_avg_dollar_volume"))
    text = " ".join(criteria.describe())
    for fragment in ("at least 100 validated", "'uptrend', meaning latest close > 50-session SMA > 200-session SMA",
                     "RSI(14) of daily closes is below 70", "RSI(14) of daily closes is above 40",
                     "at least +5.00%", "at most 3.00% of the latest close"):
        assert fragment in text, fragment
    assert "Price:" not in text and "Liquidity:" not in text
    payload = criteria.to_dict()
    assert payload["active"] == list(criteria.active()) and payload["description"] == criteria.describe()


@pytest.mark.parametrize("kwargs", [
    {"min_price": -1}, {"min_price": float("nan")}, {"min_avg_dollar_volume": float("inf")},
    {"min_history_sessions": 0}, {"min_history_sessions": 2.5}, {"min_history_sessions": True},
    {"trend": "sideways"}, {"trend": "mixed"}, {"rsi_below": 150}, {"rsi_above": -5},
    {"rsi_below": 30, "rsi_above": 70}, {"rsi_below": 50, "rsi_above": 50}, {"max_atr_pct": 0},
    {"min_rs_126": "high"}, {"min_price": True},
])
def test_nonsensical_criteria_are_rejected(kwargs):
    """Impossible or contradictory criteria fail loudly instead of silently screening everything out."""
    with pytest.raises(ValueError):
        ScanCriteria(**kwargs)


# --------------------------------------------------------------------------- #
# Pass / fail
# --------------------------------------------------------------------------- #


def test_each_criterion_passes_or_fails_on_hand_built_tickers():
    """Price, liquidity and history each fail exactly the ticker built to fail them."""
    result = scan_universe(list(_universe()), loader(_universe()))
    assert list(result.rows["ticker"]) == ["LEAD", "PENNY", "THIN", "YOUNG", "FALL"]
    assert result.passed == ["LEAD", "FALL"]
    penny, thin, young = _row(result, "PENNY"), _row(result, "THIN"), _row(result, "YOUNG")
    assert not penny["pass_min_price"] and penny["pass_min_avg_dollar_volume"] and not penny["passed"]
    assert thin["pass_min_price"] and not thin["pass_min_avg_dollar_volume"]
    assert not young["pass_min_history_sessions"] and young["history_sessions"] == 100
    lead = _row(result, "LEAD")
    assert lead["close"] == pytest.approx(50 * 1.002**299) and lead["trend"] == "uptrend"
    assert lead["avg_dollar_volume_20"] == pytest.approx(float(np.mean(50 * 1.002 ** np.arange(280, 300) * 1e6)))


def test_trend_rsi_and_volatility_criteria():
    """Trend, RSI and ATR criteria use the snapshot's definitions; strict and inclusive bounds as documented."""
    universe = _universe()
    up = scan_universe(["LEAD", "FALL"], loader(universe), ScanCriteria(trend="uptrend"))
    assert up.passed == ["LEAD"] and not _row(up, "FALL")["pass_trend"]
    down = scan_universe(["LEAD", "FALL"], loader(universe), ScanCriteria(trend="downtrend"))
    assert down.passed == ["FALL"]
    assert any("never to short" in n for n in down.notes)
    # A steady rise has no down days, so RSI(14) is exactly 100: "below 100" is false (strict), "above 70" is true.
    rsi = scan_universe(["LEAD"], loader(universe), ScanCriteria(rsi_below=100))
    assert _row(rsi, "LEAD")["rsi_14"] == 100.0 and rsi.passed == []
    assert scan_universe(["LEAD"], loader(universe), ScanCriteria(rsi_above=70)).passed == ["LEAD"]
    atr_pct = _row(rsi, "LEAD")["atr_pct"]
    assert scan_universe(["LEAD"], loader(universe), ScanCriteria(max_atr_pct=atr_pct)).passed == ["LEAD"]
    assert scan_universe(["LEAD"], loader(universe), ScanCriteria(max_atr_pct=atr_pct * 0.99)).passed == []


def test_minimum_thresholds_are_inclusive():
    """A close exactly at the minimum price passes; one cent above it does not."""
    universe = _universe()
    close = _row(scan_universe(["LEAD"], loader(universe)), "LEAD")["close"]
    assert scan_universe(["LEAD"], loader(universe), ScanCriteria(min_price=close)).passed == ["LEAD"]
    assert scan_universe(["LEAD"], loader(universe), ScanCriteria(min_price=close + 0.01)).passed == []


def test_unavailable_values_count_as_failed_with_the_reason():
    """Without a benchmark, or with too little history for a trend, the criterion fails and says why."""
    result = scan_universe(["LEAD", "YOUNG"], loader(_universe()),
                           ScanCriteria(min_history_sessions=50, trend="uptrend", min_rs_126=-1.0))
    lead, young = _row(result, "LEAD"), _row(result, "YOUNG")
    assert lead["rs_126"] is None and not lead["pass_min_rs_126"] and not lead["passed"]
    assert "min_rs_126: no benchmark supplied" in lead["unavailable"]
    assert young["trend"] == "undetermined" and not young["pass_trend"]
    assert "trend: fewer than 200 sessions" in young["unavailable"]
    assert result.passed == []
    assert any("needs benchmark prices" in n for n in result.notes)


@pytest.mark.parametrize("form", ["frame", "series", "dataset"])
def test_relative_strength_criterion_with_a_benchmark(form):
    """The benchmark may be a frame, a close series or a dataset; RS equals relative_strength() over 126 sessions."""
    bench = _bench()
    given = {"frame": bench, "series": bench["close"], "dataset": dataset("BENCH", bench)}[form]
    universe = _universe()
    result = scan_universe(["LEAD", "FALL"], loader(universe), ScanCriteria(min_rs_126=0.05), bench_prices=given)
    expected = relative_strength(universe["LEAD"].prices["close"], bench["close"], (126,))[126]
    assert _row(result, "LEAD")["rs_126"] == pytest.approx(expected) and expected > 0.05
    assert result.passed == ["LEAD"] and _row(result, "FALL")["rs_126"] < 0


def test_an_unusable_benchmark_is_disclosed_not_used():
    """A benchmark that failed validation is not used, and the scan says so."""
    result = scan_universe(["LEAD"], loader(_universe()), ScanCriteria(min_rs_126=0.0),
                           bench_prices=dataset("BENCH", walk(20)))
    assert result.passed == [] and any("benchmark data (BENCH) failed validation" in n for n in result.notes)
    with pytest.raises(TypeError):
        scan_universe(["LEAD"], loader(_universe()), bench_prices=[1, 2, 3])


# --------------------------------------------------------------------------- #
# Failures, universe handling, notes
# --------------------------------------------------------------------------- #


def test_load_and_validation_failures_are_listed_never_dropped_or_passed():
    """Every ticker that cannot be scanned appears in failures with its reason, in universe order."""
    entries = {**_universe(),
               "GONE": TickerNotFoundError("No price history was found for 'GONE'."),
               "DOWN": ProviderConnectionError("Could not reach the provider."),
               "OOPS": RuntimeError("unexpected"),
               "WEIRD": "not a dataset",
               "SHORT": dataset("SHORT", walk(20))}
    universe = ["GONE", "LEAD", "DOWN", "OOPS", "WEIRD", "SHORT", "MISSING"]
    result = scan_universe(universe, loader(entries))
    assert list(result.failures) == ["GONE", "DOWN", "OOPS", "WEIRD", "SHORT", "MISSING"]
    assert result.failures["GONE"].startswith("not_found:")
    assert result.failures["DOWN"].startswith("connection:")
    assert result.failures["OOPS"] == "RuntimeError: unexpected"
    assert "not a validated MarketDataset" in result.failures["WEIRD"]
    assert result.failures["SHORT"].startswith("failed validation:") and "30" in result.failures["SHORT"]
    assert list(result.rows["ticker"]) == ["LEAD"] and result.passed == ["LEAD"]
    assert set(result.failures).isdisjoint(result.passed)
    assert any("6 ticker(s) could not be scanned" in n for n in result.notes)


def test_tickers_are_normalised_deduplicated_and_validated():
    """'lead ' and 'LEAD' are one ticker, loaded once; an invalid symbol is a failure, not a crash."""
    calls: list[str] = []
    result = scan_universe(["lead ", "LEAD", "not a ticker!", "FALL"], loader(_universe(), calls))
    assert calls == ["LEAD", "FALL"]
    assert list(result.rows["ticker"]) == ["LEAD", "FALL"]
    assert "invalid ticker" in result.failures["not a ticker!"]
    assert any("scanned once: LEAD" in n for n in result.notes)


def test_universe_must_be_a_non_empty_list():
    """An empty universe or a bare string (which would be scanned letter by letter) is refused."""
    with pytest.raises(ValueError):
        scan_universe([], loader({}))
    with pytest.raises(TypeError):
        scan_universe("LEAD", loader(_universe()))


def test_survivorship_caveat_is_always_present_even_when_nothing_loads():
    """The survivorship-bias note is part of every result; so is 'no qualifying candidates' when none pass."""
    result = scan_universe(["GONE"], loader({}))
    assert SURVIVORSHIP_NOTE in result.notes and result.rows.empty and result.passed == []
    assert list(result.rows.columns)[:3] == ["ticker", "as_of", "close"]
    none_pass = scan_universe(["PENNY"], loader(_universe()))
    assert SURVIVORSHIP_NOTE in none_pass.notes
    assert any("No ticker met every criterion" in n for n in none_pass.notes)


def test_data_quality_flags_and_differing_dates_are_reported():
    """Usable data with warnings or from a stale cache is scanned but flagged; differing as-of dates are noted."""
    warned = trend(300, 0.002, level=50)
    warned.iloc[150, warned.columns.get_loc("close")] *= 1.5  # an extreme move -> validation warning
    entries = {**_universe(),
               "ODD": dataset("ODD", warned),
               "OLD": dataset("OLD", trend(300, 0.002, level=50), cache_status=CacheStatus.STALE_CACHE)}
    result = scan_universe(["LEAD", "ODD", "OLD", "YOUNG"], loader(entries))
    assert _row(result, "LEAD")["data_status"] == "ok" and _row(result, "LEAD")["data_warnings"] == ""
    assert _row(result, "ODD")["data_status"] == "ok_with_warnings"
    assert "extreme_returns" in _row(result, "ODD")["data_warnings"]
    assert _row(result, "OLD")["data_status"] == "stale_cache" and _row(result, "OLD")["cache_status"] == "stale_cache"
    assert any("Data-quality flags on ODD, OLD" in n for n in result.notes)
    assert any("Latest sessions differ across tickers" in n for n in result.notes)
    assert set(result.provenance) == {"LEAD", "ODD", "OLD", "YOUNG"}


def test_scan_through_the_real_loading_pipeline():
    """Synthetic-provider and CSV datasets from load_market_data scan like any other; simulated data is labelled."""
    entries = {
        "SIM-TREND-1": load_market_data("SIM-TREND-1", "2022-01-01", "2024-12-31", SyntheticProvider(),
                                        adjustment=AdjustmentMode.PRICE_ONLY),
        "CSVCO": csv_dataset("CSVCO", trend(300, 0.002, level=50)),
    }
    result = scan_universe(["SIM-TREND-1", "CSVCO"], loader(entries), ScanCriteria(min_avg_dollar_volume=None))
    assert list(result.rows["ticker"]) == ["SIM-TREND-1", "CSVCO"] and not result.failures
    assert _row(result, "SIM-TREND-1")["cache_status"] == "synthetic"
    assert _row(result, "CSVCO")["cache_status"] == "uploaded" and _row(result, "CSVCO")["passed"]
    assert any("SIMULATED data (not market prices) for: SIM-TREND-1" in n for n in result.notes)
    assert result.provenance["SIM-TREND-1"].timeliness.value == "simulated"


def test_scan_result_to_dict_is_json_with_nulls_for_missing_values():
    """The serialised result is valid JSON; unavailable values are null, never NaN or zero."""
    result = scan_universe(["LEAD", "YOUNG", "GONE"], loader(_universe()), ScanCriteria(min_rs_126=0.0))
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    young = next(r for r in payload["rows"] if r["ticker"] == "YOUNG")
    assert young["rs_126"] is None and young["passed"] is False
    assert payload["failures"] == {"GONE": result.failures["GONE"]}
    assert SURVIVORSHIP_NOTE in payload["notes"] and payload["criteria"]["min_rs_126"] == 0.0
    assert set(payload["provenance"]) == {"LEAD", "YOUNG"}
