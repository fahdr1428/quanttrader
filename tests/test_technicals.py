"""Technical indicators and snapshots: hand-checked values, causality, honest warm-up and provenance."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantlab import indicators as ind
from quantlab.data import CacheStatus, DataProviderError, MalformedDataError, TickerNotFoundError
from quantlab.provenance import Adjustment, Timeliness
from quantlab.technicals import (
    DATA_STATUSES,
    SUPPORT_RESISTANCE_METHOD,
    TREND_LABELS,
    atr,
    average_dollar_volume,
    classify_trend,
    dataset_provenance,
    dataset_status,
    load_validated,
    macd,
    overnight_gaps,
    range_52w,
    realised_volatility,
    relative_strength,
    snapshot_from_dataset,
    support_resistance,
    technical_snapshot,
    true_range,
    volume_zscore,
)
from quantlab.validation import InputValidationError
from tests.technical_fixtures import NOW, csv_dataset, dataset, loader, ohlcv, prov, trend, walk

# --------------------------------------------------------------------------- #
# MACD
# --------------------------------------------------------------------------- #


def _kernel_lag(window: int) -> float:
    """Mean age (in sessions) of the finite EMA kernel: sum_j j * w_j with w_j proportional to (1 - alpha)^j."""
    alpha = 2.0 / (window + 1)
    j = np.arange(ind.EMA_SPANS * window)
    w = (1 - alpha) ** j
    return float((j * w).sum() / w.sum())


def test_macd_warmup_is_nan_and_never_backfilled():
    """MACD needs a full slow kernel and the signal a further signal kernel; earlier values stay NaN."""
    out = macd(walk(200)["close"])
    assert list(out.columns) == ["macd", "signal", "histogram"]
    assert int(out["macd"].isna().sum()) == ind.EMA_SPANS * 26 - 1
    assert int(out["signal"].isna().sum()) == ind.EMA_SPANS * (26 + 9) - 2
    assert out["histogram"].isna().sum() == out["signal"].isna().sum()
    assert out.iloc[ind.EMA_SPANS * (26 + 9) - 2:].notna().all().all()


def test_macd_on_a_linear_trend_equals_the_hand_computed_lag_difference():
    """For close = a + b t, EMA_t = close_t - b * lag, so MACD = b * (lag_slow - lag_fast) and the histogram is 0."""
    b = 0.5
    close = pd.Series(100 + b * np.arange(300.0))
    out = macd(close).dropna()
    expected = b * (_kernel_lag(26) - _kernel_lag(12))
    np.testing.assert_allclose(out["macd"], expected, rtol=1e-9)
    np.testing.assert_allclose(out["signal"], expected, rtol=1e-9)
    np.testing.assert_allclose(out["histogram"], 0.0, atol=1e-9)


def test_macd_of_constant_prices_is_zero():
    """No trend, no divergence: a flat series has MACD, signal and histogram of exactly zero."""
    out = macd(pd.Series(np.full(150, 42.0))).dropna()
    assert len(out) and (out.abs() < 1e-12).all().all()


@pytest.mark.parametrize("kwargs", [{"fast": 26, "slow": 12}, {"fast": 12, "slow": 12}, {"fast": 0},
                                    {"signal": 2.5}, {"slow": True}])
def test_macd_rejects_nonsensical_windows(kwargs):
    """A fast window at or above the slow one, zero, fractional or boolean windows are refused."""
    with pytest.raises(ValueError):
        macd(walk(120)["close"], **kwargs)


# --------------------------------------------------------------------------- #
# True range and ATR
# --------------------------------------------------------------------------- #


def _atr_frame() -> pd.DataFrame:
    return pd.DataFrame({"high": [10.0, 11, 12, 11, 13], "low": [9.0, 10, 10, 9, 11],
                         "close": [9.5, 10.5, 11, 10, 12]}, index=pd.bdate_range("2024-01-02", periods=5))


def test_true_range_hand_values_and_undefined_first_session():
    """TR = max(high - low, |high - prev close|, |low - prev close|); the first session has no previous close."""
    tr = true_range(_atr_frame())
    assert math.isnan(tr.iloc[0])
    assert tr.iloc[1:].tolist() == [1.5, 2.0, 2.0, 3.0]


def test_atr_uses_wilder_smoothing_seeded_by_a_simple_mean():
    """ATR(2): first value = mean(TR1, TR2); then (ATR_prev * 1 + TR) / 2 — hand-computed."""
    out = atr(_atr_frame(), window=2)
    assert out.iloc[:2].isna().all()
    np.testing.assert_allclose(out.iloc[2:].to_numpy(), [1.75, 1.875, 2.4375])


def test_atr_warmup_and_short_series():
    """The first ``window`` values are NaN; a series no longer than the window has no ATR at all."""
    prices = walk(60)
    out = atr(prices, 14)
    assert int(out.isna().sum()) == 14 and out.iloc[14:].notna().all()
    assert atr(prices.iloc[:14], 14).isna().all()


def test_atr_refuses_missing_prices_instead_of_filling_them():
    """A gap in the highs is never filled or skipped silently: the caller must validate first."""
    prices = walk(40)
    prices.iloc[20, prices.columns.get_loc("high")] = np.nan
    with pytest.raises(ValueError, match="missing"):
        atr(prices, 14)


# --------------------------------------------------------------------------- #
# Volume, volatility, gaps
# --------------------------------------------------------------------------- #


def test_volume_zscore_hand_value_excludes_the_current_session():
    """z_t compares ln v_t with the mean and sample std of ln v over the *previous* window, not including t."""
    volume = pd.Series([100.0, 200, 400, 800, 1600, 100])
    z = volume_zscore(volume, window=4)
    assert z.iloc[:4].isna().all()
    prior = np.log([100.0, 200, 400, 800])
    assert z.iloc[4] == pytest.approx((math.log(1600) - prior.mean()) / prior.std(ddof=1))
    prior = np.log([200.0, 400, 800, 1600])
    assert z.iloc[5] == pytest.approx((math.log(100) - prior.mean()) / prior.std(ddof=1))


def test_volume_zscore_treats_missing_and_zero_volume_as_missing_never_zero():
    """A zero or missing volume gets no z-score; one gap in the window is tolerated (80% rule), two are not."""
    rng = np.random.default_rng(1)
    base = pd.Series(np.exp(rng.normal(13, 0.3, 12)))
    zero_today = base.copy()
    zero_today.iloc[-1] = 0.0
    assert math.isnan(volume_zscore(zero_today, 5).iloc[-1])
    missing_today = base.copy()
    missing_today.iloc[-1] = np.nan
    assert math.isnan(volume_zscore(missing_today, 5).iloc[-1])
    one_gap = base.copy()
    one_gap.iloc[-3] = 0.0  # 4 of the previous 5 sessions report a volume
    assert math.isfinite(volume_zscore(one_gap, 5).iloc[-1])
    two_gaps = one_gap.copy()
    two_gaps.iloc[-4] = np.nan  # only 3 of 5
    assert math.isnan(volume_zscore(two_gaps, 5).iloc[-1])


def test_volume_zscore_is_undefined_for_constant_volume():
    """With no variation there is no 'unusual': the z-score is NaN rather than an infinity or zero."""
    assert volume_zscore(pd.Series(np.full(80, 1e6)), 50).isna().all()


def test_realised_volatility_hand_value():
    """Annualised sample std of the last ``window`` close-to-close returns; NaN until ``window`` returns exist."""
    close = pd.Series([100.0, 101, 99, 102, 100, 103])
    out = realised_volatility(close, window=3, periods_per_year=252)
    assert out.iloc[:3].isna().all()
    rets = np.array([102 / 99 - 1, 100 / 102 - 1, 103 / 100 - 1])
    assert out.iloc[-1] == pytest.approx(rets.std(ddof=1) * math.sqrt(252))


def test_average_dollar_volume_skips_missing_but_counts_reported_zero():
    """Missing volume is left out of the mean; a reported zero is a real zero; too many gaps give NaN."""
    prices = ohlcv([10.0] * 4, volumes=[100.0, np.nan, 300, 0])
    out = average_dollar_volume(prices, window=4)
    assert out.iloc[:3].isna().all()
    assert out.iloc[3] == pytest.approx((1000 + 3000 + 0) / 3)
    sparse = ohlcv([10.0] * 4, volumes=[100.0, np.nan, np.nan, 50])
    assert math.isnan(average_dollar_volume(sparse, window=4).iloc[-1])


def test_overnight_gaps_hand_values():
    """gap_t = open_t / close_{t-1} - 1; undefined on the first session."""
    prices = ohlcv([10.0, 10.0, 10.0], opens=[10.0, 11.0, 9.0])
    gaps = overnight_gaps(prices)
    assert math.isnan(gaps.iloc[0])
    np.testing.assert_allclose(gaps.iloc[1:], [0.1, -0.1])


# --------------------------------------------------------------------------- #
# Causality and ordering, for every series indicator
# --------------------------------------------------------------------------- #


SERIES_INDICATORS = {
    "macd": lambda p: macd(p["close"]),
    "atr": lambda p: atr(p, 14),
    "true_range": true_range,
    "volume_zscore": lambda p: volume_zscore(p["volume"], 50),
    "realised_volatility": lambda p: realised_volatility(p["close"], 63),
    "average_dollar_volume": lambda p: average_dollar_volume(p, 20),
    "overnight_gaps": overnight_gaps,
}


@pytest.mark.parametrize("name", list(SERIES_INDICATORS))
def test_changing_future_prices_never_changes_earlier_values(name):
    """Look-ahead regression test: perturb every price and volume after session k; values up to k are identical."""
    func = SERIES_INDICATORS[name]
    prices = walk(400, seed=21)
    k = 300
    altered = prices.copy()
    scale = np.linspace(0.5, 2.0, len(prices) - k)[:, None]
    altered.iloc[k:] = altered.iloc[k:].to_numpy() * scale
    base, alt = func(prices), func(altered)
    pd.testing.assert_frame_equal(pd.DataFrame(base).iloc[:k], pd.DataFrame(alt).iloc[:k])
    assert not pd.DataFrame(base).iloc[k:].equals(pd.DataFrame(alt).iloc[k:])  # the test is not vacuous


@pytest.mark.parametrize("name", list(SERIES_INDICATORS))
def test_unsorted_or_duplicated_sessions_are_refused(name):
    """Rolling windows count rows, so out-of-order or repeated dates would silently corrupt every value."""
    func = SERIES_INDICATORS[name]
    prices = walk(120)
    with pytest.raises(ValueError, match="chronological"):
        func(prices.iloc[::-1])
    with pytest.raises(ValueError, match="chronological"):
        func(pd.concat([prices.iloc[:60], prices.iloc[59:]]))


# --------------------------------------------------------------------------- #
# Relative strength, 52-week range, support and resistance
# --------------------------------------------------------------------------- #


def test_relative_strength_hand_values_and_insufficient_history():
    """RS = asset return - benchmark return over each lookback; None when the lookback exceeds the history."""
    n = 200
    asset = ohlcv(100 * 1.01 ** np.arange(n))["close"]
    bench = ohlcv(100 * 1.005 ** np.arange(n))["close"]
    out = relative_strength(asset, bench, (63, 199, 252))
    assert out[63] == pytest.approx((1.01**63 - 1) - (1.005**63 - 1))
    assert out[199] == pytest.approx((1.01**199 - 1) - (1.005**199 - 1))
    assert out[252] is None


def test_relative_strength_uses_common_sessions_without_forward_filling():
    """A session missing from the benchmark is dropped from both series; lookbacks count common sessions."""
    asset_close = np.array([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    bench_close = np.array([50.0, 51, 52, 53, 54, 55, 56, 57, 58, 59])
    asset = ohlcv(asset_close)["close"]
    bench = ohlcv(bench_close)["close"].drop(asset.index[7])  # the benchmark has no bar on session 7
    out = relative_strength(asset, bench, (3,))
    # Common sessions: 0..6, 8, 9 -> three common sessions back from 9 is session 5.
    assert out[3] == pytest.approx((19 / 15 - 1) - (59 / 55 - 1))


def test_relative_strength_accepts_a_price_frame_as_benchmark():
    """The benchmark can be given as its OHLCV frame (the close column is used)."""
    a, b = walk(100, seed=1), walk(100, seed=2)
    assert relative_strength(a["close"], b, (20,)) == relative_strength(a["close"], b["close"], (20,))


def test_range_52w_values_dates_and_distance():
    """High/low of the last ``window`` sessions (older sessions ignored) with the latest close's distance."""
    prices = ohlcv([50.0, 60, 40, 45, 47, 44, 46], spread=0.0)
    prices.loc[prices.index[0], "high"] = 500.0  # outside the 5-session window
    out = range_52w(prices, window=5)
    assert out["high"] == 47.0 and out["high_date"] == prices.index[4].date().isoformat()
    assert out["low"] == 40.0 and out["low_date"] == prices.index[2].date().isoformat()
    assert out["pct_from_high"] == pytest.approx(46 / 47 - 1) and out["pct_from_low"] == pytest.approx(46 / 40 - 1)
    assert out["basis"] == "high/low" and out["reason"] == ""


def test_range_52w_with_short_history_is_none_with_a_reason():
    """Fewer sessions than the window is not a 52-week range: every value is None and the reason is given."""
    out = range_52w(walk(100))
    assert out["high"] is None and out["low"] is None and out["pct_from_high"] is None
    assert "252" in out["reason"] and "100" in out["reason"]


def test_range_52w_falls_back_to_closes_without_high_low():
    """Without high/low columns the range is of closes, and the basis says so."""
    prices = walk(30)[["close"]]
    out = range_52w(prices, window=10)
    assert out["basis"] == "close" and out["high"] == prices["close"].iloc[-10:].max()


def test_support_resistance_is_labelled_a_heuristic_with_hand_values():
    """Support = lowest low and resistance = highest high of the last N sessions; always marked heuristic."""
    prices = ohlcv([50.0, 60, 40, 45, 47, 44, 46], spread=0.0)
    out = support_resistance(prices, windows=(3, 10))
    assert out["heuristic"] is True and out["method"] == SUPPORT_RESISTANCE_METHOD and "not forecasts" in out["method"]
    level = out["levels"][3]
    assert level["support"] == 44.0 and level["resistance"] == 47.0
    assert level["pct_to_support"] == pytest.approx(44 / 46 - 1) and level["pct_to_resistance"] == pytest.approx(47 / 46 - 1)
    assert out["levels"][10]["support"] is None and "10" in out["levels"][10]["reason"]


# --------------------------------------------------------------------------- #
# Trend rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("close,sma50,sma200,expected", [
    (110, 105, 100, "uptrend"), (90, 95, 100, "downtrend"), (104, 106, 100, "mixed"), (100, 100, 90, "mixed"),
    (100, 105, None, "undetermined"), (None, 1, 1, "undetermined"),
])
def test_trend_rule(close, sma50, sma200, expected):
    """uptrend: close > SMA50 > SMA200; downtrend: the reverse; ties and other orders are mixed; gaps undetermined."""
    assert classify_trend(close, sma50, sma200) == expected
    assert expected in TREND_LABELS


def test_snapshot_trend_on_clean_trends():
    """A steadily rising series is an uptrend, a falling one a downtrend, and a short one undetermined."""
    assert technical_snapshot("UP", trend(260, 0.002), prov()).trend == "uptrend"
    assert technical_snapshot("DN", trend(260, -0.002), prov()).trend == "downtrend"
    short = technical_snapshot("NEW", trend(120, 0.002), prov())
    assert short.trend == "undetermined" and any("Trend undetermined" in n for n in short.notes)


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #


def test_snapshot_point_values_equal_the_causal_series_at_that_date():
    """A snapshot as of session k uses only data up to k: it equals the full-history series values at k."""
    prices = walk(420, seed=5)
    full_rsi = ind.rsi_wilder(prices["close"], 14)
    full_macd = macd(prices["close"])
    full_atr = atr(prices, 14)
    full_z = volume_zscore(prices["volume"], 50)
    full_vol = realised_volatility(prices["close"], 63)
    for k in (130, 300, 420):
        snap = technical_snapshot("X", prices.iloc[:k], prov())
        i = k - 1
        assert snap.as_of == prices.index[i].date()
        assert snap.value("sma_50") == pytest.approx(prices["close"].iloc[k - 50:k].mean())
        assert snap.value("rsi_14") == pytest.approx(full_rsi.iloc[i])
        assert snap.value("macd") == pytest.approx(full_macd["macd"].iloc[i])
        assert snap.value("macd_hist") == pytest.approx(full_macd["histogram"].iloc[i])
        assert snap.value("atr_14") == pytest.approx(full_atr.iloc[i])
        assert snap.value("volume_z") == pytest.approx(full_z.iloc[i])
        assert snap.value("realised_vol_63") == pytest.approx(full_vol.iloc[i])


def test_snapshot_derived_values_follow_their_formulas():
    """Ratios are computed from the reported inputs exactly as their formula strings say."""
    snap = technical_snapshot("X", walk(300, seed=8), prov())
    close = snap.close.value
    assert snap.value("pct_vs_sma_50") == pytest.approx(close / snap.value("sma_50") - 1)
    assert snap.value("pct_vs_sma_200") == pytest.approx(close / snap.value("sma_200") - 1)
    assert snap.value("atr_pct") == pytest.approx(snap.value("atr_14") / close)
    assert snap.value("pct_from_high_52w") == pytest.approx(close / snap.value("high_52w") - 1)
    assert snap.value("pct_from_high_52w") <= 0 <= snap.value("pct_from_low_52w")
    assert snap.value("support_20") <= close <= snap.value("resistance_20")
    assert snap.values["pct_vs_sma_50"].formula == "close / sma_50 - 1"
    assert {i.name for i in snap.values["pct_vs_sma_50"].inputs} == {"close", "sma_50"}


def test_short_history_gives_none_with_reasons_never_zero():
    """With 40 sessions the long-window values are None with the reason; nothing is estimated from less data."""
    snap = technical_snapshot("NEW", walk(40), prov())
    for name in ("sma_50", "sma_200", "macd", "macd_signal", "high_52w", "realised_vol_63", "volume_z",
                 "support_60", "pct_vs_sma_50"):
        calc = snap.values[name]
        assert calc.value is None and calc.reason, name
    assert "needs 50 sessions" in snap.values["sma_50"].reason
    assert snap.values["pct_vs_sma_50"].reason == "sma_50 unavailable"
    assert snap.value("rsi_14") is not None and snap.value("support_20") is not None
    assert any("Only 40 sessions" in n for n in snap.notes)


def test_snapshot_without_benchmark_volume_or_high_low():
    """Missing inputs are named: no benchmark, no volume column, no high/low — never zeros."""
    prices = walk(300)[["open", "close"]]
    snap = technical_snapshot("X", prices, prov())
    for lb in (63, 126, 252):
        assert snap.value(f"rs_{lb}") is None and snap.values[f"rs_{lb}"].reason == "no benchmark prices supplied"
    assert snap.values["volume_z"].reason == "volume column not available"
    assert snap.values["avg_dollar_volume_20"].reason == "volume column not available"
    assert snap.values["atr_14"].reason == "high/low columns not available"
    assert snap.values["atr_pct"].value is None
    assert any("no benchmark" in n for n in snap.notes)


def test_snapshot_relative_strength_and_benchmark_lag_note():
    """RS values equal relative_strength(); a benchmark ending earlier is disclosed with its last common date."""
    asset, bench = walk(300, seed=1), walk(300, seed=2)
    snap = technical_snapshot("X", asset, prov(), bench_prices=bench)
    expected = relative_strength(asset["close"], bench["close"])
    assert snap.value("rs_63") == pytest.approx(expected[63]) and snap.value("rs_126") == pytest.approx(expected[126])
    assert snap.value("rs_252") == pytest.approx(expected[252])
    lagging = technical_snapshot("X", asset, prov(), bench_prices=bench.iloc[:-3])
    assert any(f"as of {bench.index[-4].date()}" in n for n in lagging.notes)


@pytest.mark.parametrize("bad", ["empty", "nan_close", "zero_close", "range_index", "unsorted"])
def test_snapshot_requires_validated_prices(bad):
    """Unvalidated input is refused rather than producing numbers from broken data."""
    prices = walk(80)
    if bad == "empty":
        prices = prices.iloc[:0]
    elif bad == "nan_close":
        prices.iloc[10, prices.columns.get_loc("close")] = np.nan
    elif bad == "zero_close":
        prices.iloc[10, prices.columns.get_loc("close")] = 0.0
    elif bad == "range_index":
        prices = prices.reset_index(drop=True)
    else:
        prices = prices.iloc[::-1]
    with pytest.raises(ValueError):
        technical_snapshot("X", prices, prov())


def test_snapshot_as_of_is_the_exchange_local_session_date():
    """A tz-aware index keeps the exchange's calendar date; converting to UTC would move Asian sessions back a day."""
    prices = walk(60)
    prices.index = pd.DatetimeIndex(prices.index.tz_localize("Asia/Tokyo"), name="date")
    snap = technical_snapshot("7203.T", prices, prov())
    assert snap.as_of == prices.index[-1].date()
    assert snap.as_of != prices.index[-1].tz_convert("UTC").date()


def test_snapshot_to_dict_is_json_and_carries_provenance_and_formulas():
    """The serialised snapshot has no NaN, a fact with provenance for the close, and formulas for every value."""
    snap = technical_snapshot("X", walk(100), prov(currency="USD"))
    payload = snap.to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text
    assert payload["close"]["kind"] == "fact" and payload["close"]["provenance"]["source"] == "Synthetic test frame"
    assert payload["close"]["unit"] == "USD per share"
    assert all(v["kind"] == "calculated" and v["formula"] for v in payload["values"].values())
    assert all(v["reason"] for v in payload["values"].values() if v["value"] is None)
    assert payload["trend_rule"].startswith("uptrend:")
    assert snap.provenance is snap.close.provenance


def test_snapshot_notes_adjustment_and_simulated_data():
    """Total-return levels and simulated prices are disclosed in the notes."""
    adjusted = technical_snapshot("X", walk(60), prov(adjustment=Adjustment.SPLITS_AND_DIVIDENDS))
    assert any("dividend-adjusted" in n for n in adjusted.notes)
    simulated = technical_snapshot("X", walk(60), prov(timeliness=Timeliness.SIMULATED))
    assert any("SIMULATED" in n for n in simulated.notes)
    assert any("not live" in n for n in simulated.notes)


# --------------------------------------------------------------------------- #
# Datasets: provenance, status, loading
# --------------------------------------------------------------------------- #


def test_dataset_provenance_for_an_uploaded_csv():
    """A CSV loaded through load_market_data is user-supplied, adjustment as provided, period from the data."""
    frame = walk(120)
    ds = csv_dataset("ABC", frame)
    p = dataset_provenance(ds, currency="EUR")
    assert p.timeliness is Timeliness.USER_SUPPLIED and p.adjustment is Adjustment.AS_PROVIDED
    assert p.period_start == frame.index[0].date() and p.period_end == frame.index[-1].date()
    assert p.currency == "EUR" and p.units == "EUR per share" and p.cache_status == "uploaded"
    assert p.retrieved_at == NOW and p.warnings == ()


def test_dataset_provenance_for_provider_and_synthetic_data():
    """Provider data is end-of-day with its adjustment; synthetic data is simulated; warnings travel along."""
    frame = walk(120)
    frame.iloc[50, frame.columns.get_loc("close")] *= 1.6  # an extreme move -> a validation warning
    yahoo = dataset("SPY", frame, cache_status=CacheStatus.DOWNLOADED, provider_key="yahoo",
                    adjustment="total_return", fetched_at=datetime(2024, 6, 3, 21, 0))  # naive -> UTC
    p = dataset_provenance(yahoo)
    assert p.timeliness is Timeliness.END_OF_DAY and p.adjustment is Adjustment.SPLITS_AND_DIVIDENDS
    assert p.retrieved_at.tzinfo is not None and p.currency is None
    assert any("extreme_returns" in w for w in p.warnings)
    synthetic = dataset("SIM-RANDOM-1", walk(120), cache_status=CacheStatus.SYNTHETIC, provider_key="synthetic")
    assert dataset_provenance(synthetic).timeliness is Timeliness.SIMULATED
    assert dataset_provenance(synthetic).adjustment is Adjustment.NOT_APPLICABLE


def test_dataset_status_ranks_problems():
    """Unusable, stale cache, stale data, warnings and clean data map to distinct documented statuses."""
    assert dataset_status(dataset("A", walk(120))) == ("ok", ())
    assert dataset_status(dataset("A", walk(20)))[0] == "unusable"  # fewer than 30 observations
    assert dataset_status(dataset("A", walk(120), cache_status=CacheStatus.STALE_CACHE))[0] == "stale_cache"
    warned = walk(120)
    warned.iloc[50, warned.columns.get_loc("close")] *= 1.6
    status, codes = dataset_status(dataset("A", warned))
    assert status == "ok_with_warnings" and set(codes) == {"extreme_returns", "ohlc_inconsistent"}
    stale = dataset("A", walk(120))
    from quantlab.validation import Severity
    stale.quality.add(Severity.WARNING, "stale_data", "old", count=9)
    assert dataset_status(stale)[0] == "stale"
    assert set(DATA_STATUSES) >= {"ok", "ok_with_warnings", "stale", "stale_cache", "unusable", "load_failed"}


@pytest.mark.parametrize("entry,status,fragment", [
    (TickerNotFoundError("No data for 'ZZZ'."), "load_failed", "not_found: No data"),
    (MalformedDataError("Bad file."), "load_failed", "malformed: Bad file."),
    (DataProviderError("Provider down."), "load_failed", "provider_error: Provider down."),
    (InputValidationError("bad ticker"), "load_failed", "invalid input: bad ticker"),
    (RuntimeError("boom"), "load_failed", "RuntimeError: boom"),
    ("not a dataset", "load_failed", "not a validated MarketDataset"),
    ("short", "unusable", "failed validation"),
])
def test_load_validated_turns_every_failure_into_a_reason(entry, status, fragment):
    """Loader exceptions, wrong return types and unusable data become failures with reasons — never a crash."""
    if entry == "short":
        entry = dataset("ZZZ", walk(20))
    outcome = load_validated("ZZZ", loader({"ZZZ": entry}))
    assert outcome.dataset is None and not outcome.ok
    assert outcome.status == status and fragment in outcome.reason


def test_load_validated_passes_usable_data_through_with_its_status():
    """A usable dataset is returned with its status and warning codes."""
    ds = dataset("ABC", walk(120), cache_status=CacheStatus.STALE_CACHE)
    outcome = load_validated("ABC", loader({"ABC": ds}))
    assert outcome.ok and outcome.dataset is ds and outcome.status == "stale_cache"


def test_snapshot_from_dataset_refuses_unusable_data():
    """A dataset that failed validation never produces a snapshot."""
    with pytest.raises(ValueError, match="failed validation"):
        snapshot_from_dataset(dataset("X", walk(20)))
    snap = snapshot_from_dataset(csv_dataset("X", walk(80)), currency="USD")
    assert snap.ticker == "X" and snap.provenance.timeliness is Timeliness.USER_SUPPLIED


def test_relative_strength_is_refused_across_different_price_bases():
    from quantlab.technicals import snapshot_from_dataset

    frame = trend(300, 0.001)
    asset = dataset("ACME", frame)                                     # as provided (an upload)
    bench = dataset("SPY", frame, provider_key="yahoo", adjustment="total_return")
    snap = snapshot_from_dataset(asset, bench)
    assert snap.value("rs_63") is None and "different price bases" in snap.values["rs_63"].reason
    assert snapshot_from_dataset(asset, dataset("SPY", frame)).value("rs_63") == pytest.approx(0.0)
