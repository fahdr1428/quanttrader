"""Data providers, price adjustment, caching and provider failures (no network used)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab.data import (
    AdjustmentMode,
    CacheStatus,
    CSVProvider,
    DataProviderError,
    DiskCache,
    MalformedDataError,
    ProviderConnectionError,
    RateLimitError,
    TickerNotFoundError,
    YahooFinanceProvider,
    align_benchmark,
    fingerprint_prices,
    load_market_data,
    parse_price_csv,
)
from quantlab.validation import InputValidationError

TODAY = date(2024, 6, 1)


def yahoo_frame(n: int = 80, start: str = "2024-01-02", factor: float = 0.98) -> pd.DataFrame:
    """Frame shaped like yfinance's ``history(auto_adjust=False)`` output (synthetic values)."""
    idx = pd.bdate_range(start, periods=n, tz="America/New_York")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * factor,
            "Volume": 1_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


class FakeTicker:
    def __init__(self, frame=None, exc=None):
        self.frame, self.exc, self.calls = frame, exc, 0

    def history(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.frame


def provider_returning(frame=None, exc=None):
    fake = FakeTicker(frame, exc)
    return YahooFinanceProvider(ticker_factory=lambda symbol: fake), fake


def test_yahoo_request_is_unadjusted_with_inclusive_end():
    provider, fake = provider_returning(yahoo_frame())
    provider.fetch("SPY", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-30"))
    assert fake.kwargs["auto_adjust"] is False
    assert fake.kwargs["end"] == "2024-05-01"  # yfinance's end is exclusive
    assert fake.kwargs["raise_errors"] is True


def test_total_return_adjustment_scales_ohlc_by_adj_factor():
    provider, _ = provider_returning(yahoo_frame(factor=0.98))
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, today=TODAY)
    raw = ds.raw
    np.testing.assert_allclose(ds.prices["close"], raw["adj_close"])
    np.testing.assert_allclose(ds.prices["open"], raw["open"] * 0.98)
    assert ds.adjustment == "total_return"
    assert ds.prices.index.tz is None


def test_price_only_mode_leaves_prices_unchanged():
    provider, _ = provider_returning(yahoo_frame())
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", provider,
                          adjustment=AdjustmentMode.PRICE_ONLY, today=TODAY)
    np.testing.assert_allclose(ds.prices["close"], ds.raw["close"])
    assert ds.adjustment == "price_only"


@pytest.mark.parametrize(
    "message,expected",
    [
        ("YFRateLimitError: Too Many Requests. Rate limited.", RateLimitError),
        ("YFTzMissingError: $ZZZZ: possibly delisted; no timezone found", TickerNotFoundError),
        ("YFPricesMissingError: no price data found", TickerNotFoundError),
        ("ConnectionError: Failed to resolve query1.finance.yahoo.com", ProviderConnectionError),
        ("curl: (56) CONNECT tunnel failed, response 403", ProviderConnectionError),
        ("Something unexpected", DataProviderError),
    ],
)
def test_provider_errors_are_mapped_to_typed_errors(message, expected):
    provider, _ = provider_returning(exc=RuntimeError(message))
    with pytest.raises(expected) as info:
        load_market_data("SPY", "2024-01-01", "2024-04-30", provider, today=TODAY)
    assert info.value.user_message  # always a displayable explanation


def test_empty_provider_response_is_not_found():
    provider, _ = provider_returning(pd.DataFrame())
    with pytest.raises(TickerNotFoundError):
        load_market_data("SPY", "2024-01-01", "2024-04-30", provider, today=TODAY)


def test_invalid_ticker_never_reaches_provider():
    provider, fake = provider_returning(yahoo_frame())
    with pytest.raises(InputValidationError):
        load_market_data("not a ticker!", "2024-01-01", "2024-04-30", provider, today=TODAY)
    assert fake.calls == 0


def test_cache_hit_avoids_second_request(tmp_path):
    provider, fake = provider_returning(yahoo_frame())
    cache = DiskCache(tmp_path)
    first = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, cache=cache, today=TODAY)
    second = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, cache=cache, today=TODAY)
    assert fake.calls == 1
    assert first.cache_status is CacheStatus.DOWNLOADED and second.cache_status is CacheStatus.CACHED
    assert first.fingerprint == second.fingerprint
    pd.testing.assert_frame_equal(first.prices, second.prices, check_freq=False)


def test_expired_cache_is_refreshed(tmp_path):
    provider, fake = provider_returning(yahoo_frame())
    cache = DiskCache(tmp_path)
    now = datetime(2024, 6, 1, tzinfo=timezone.utc)
    load_market_data("SPY", "2024-01-01", "2024-04-30", provider, cache=cache, now=now, today=TODAY)
    later = now + timedelta(hours=30)
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, cache=cache, now=later, today=TODAY)
    assert fake.calls == 2 and ds.cache_status is CacheStatus.DOWNLOADED


def test_stale_cache_is_used_and_labelled_when_provider_fails(tmp_path):
    good, _ = provider_returning(yahoo_frame())
    cache = DiskCache(tmp_path)
    load_market_data("SPY", "2024-01-01", "2024-04-30", good, cache=cache, today=TODAY)
    failing, _ = provider_returning(exc=RuntimeError("YFRateLimitError: Too Many Requests"))
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", failing, cache=cache,
                          force_refresh=True, today=TODAY)
    assert ds.cache_status is CacheStatus.STALE_CACHE
    assert "rate-limiting" in ds.provider_error
    assert any(i.code == "stale_cache" for i in ds.quality.warnings)


def test_invalid_data_is_not_cached(tmp_path):
    bad = yahoo_frame(n=10)  # too few observations -> blocking error
    provider, _ = provider_returning(bad)
    cache = DiskCache(tmp_path)
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, cache=cache, today=TODAY)
    assert ds.quality.is_blocking and not ds.is_usable
    assert not list(tmp_path.glob("*.csv"))


def test_corrupted_cache_entry_is_ignored(tmp_path):
    provider, fake = provider_returning(yahoo_frame())
    cache = DiskCache(tmp_path)
    load_market_data("SPY", "2024-01-01", "2024-04-30", provider, cache=cache, today=TODAY)
    csv_file = next(tmp_path.glob("*.csv"))
    csv_file.write_text(csv_file.read_text().replace("100", "999", 1))  # checksum now fails
    load_market_data("SPY", "2024-01-01", "2024-04-30", provider, cache=cache, today=TODAY)
    assert fake.calls == 2


def test_csv_provider_round_trip():
    content = yahoo_frame().tz_localize(None).to_csv(index_label="Date").encode()
    provider = CSVProvider(content, "spy.csv")
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, today=TODAY)
    assert ds.cache_status is CacheStatus.UPLOADED and ds.n_obs == 80
    assert "spy.csv" in ds.notes[0]


@pytest.mark.parametrize(
    "content",
    [b"", b"Date,Open,High\n2024-01-02,1,2\n", b"Date,Open,High,Low,Close,Volume\nnot-a-date,1,2,0.5,1.5,10\n"],
)
def test_malformed_csv_is_rejected(content):
    with pytest.raises(MalformedDataError):
        parse_price_csv(content)


def test_fingerprint_detects_any_price_revision():
    provider, _ = provider_returning(yahoo_frame())
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, today=TODAY)
    revised = ds.prices.copy()
    revised.iloc[10, revised.columns.get_loc("close")] *= 1.001
    assert fingerprint_prices(revised) != ds.fingerprint
    assert fingerprint_prices(ds.prices.copy()) == ds.fingerprint


def test_benchmark_alignment_drops_rather_than_fills():
    idx = pd.bdate_range("2024-01-01", periods=10)
    a = pd.DataFrame({"close": range(10)}, index=idx)
    b = a.drop(idx[[3, 7]])
    a2, b2, dropped = align_benchmark(a, b)
    assert len(a2) == len(b2) == 8
    assert list(dropped) == list(idx[[3, 7]])
