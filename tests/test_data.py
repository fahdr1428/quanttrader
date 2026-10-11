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


def provider_returning(frame=None, exc=None, retries=0):
    fake = FakeTicker(frame, exc)
    return YahooFinanceProvider(ticker_factory=lambda symbol: fake, retries=retries, sleep=lambda s: None), fake


class FlakyTicker:
    """Fails with the given errors first, then succeeds."""

    def __init__(self, errors, frame):
        self.errors, self.frame, self.calls = list(errors), frame, 0

    def history(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.frame


def test_transient_errors_are_retried_with_backoff():
    flaky = FlakyTicker([RuntimeError("YFRateLimitError: Too Many Requests"),
                         RuntimeError("ConnectionError: timed out")], yahoo_frame())
    waits = []
    provider = YahooFinanceProvider(ticker_factory=lambda s: flaky, retries=2, backoff_seconds=1.0,
                                    sleep=waits.append)
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", provider, today=TODAY)
    assert ds.n_obs == 80 and flaky.calls == 3
    assert waits == [1.0, 2.0]


def test_retries_give_up_and_unknown_tickers_are_not_retried():
    flaky = FlakyTicker([RuntimeError("YFRateLimitError: Too Many Requests")] * 5, yahoo_frame())
    provider = YahooFinanceProvider(ticker_factory=lambda s: flaky, retries=2, sleep=lambda s: None)
    with pytest.raises(RateLimitError):
        load_market_data("SPY", "2024-01-01", "2024-04-30", provider, today=TODAY)
    assert flaky.calls == 3
    missing = FlakyTicker([RuntimeError("YFTzMissingError: $ZZZ: possibly delisted; no timezone found")] * 5,
                          yahoo_frame())
    provider = YahooFinanceProvider(ticker_factory=lambda s: missing, retries=2, sleep=lambda s: None)
    with pytest.raises(TickerNotFoundError):
        load_market_data("ZZZ", "2024-01-01", "2024-04-30", provider, today=TODAY)
    assert missing.calls == 1


def test_yahoo_request_is_unadjusted_with_inclusive_end():
    provider, fake = provider_returning(yahoo_frame())
    provider.fetch("SPY", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-30"))
    assert fake.kwargs["auto_adjust"] is False
    assert fake.kwargs["end"] == "2024-05-01"  # yfinance's end is exclusive
    assert "raise_errors" not in fake.kwargs  # deprecated in yfinance 1.x; errors are raised via config


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


def _csv_text(frame: pd.DataFrame, date_format: str | None = None, sep: str = ",") -> bytes:
    f = frame.tz_localize(None).drop(columns=["Dividends", "Stock Splits"])
    return f.to_csv(index_label="Date", date_format=date_format, sep=sep).encode()


@pytest.mark.parametrize("date_format", ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y%m%d"])
def test_csv_date_formats_are_detected_from_the_whole_column(date_format):
    """Regression: DD/MM files were read month-first and rows with day > 12 silently dropped."""
    content = _csv_text(yahoo_frame(), date_format)
    parsed = parse_price_csv(content)
    assert len(parsed) == 80 and parsed.index.notna().all()
    assert parsed.index[0] == pd.Timestamp("2024-01-02") and parsed.index.is_monotonic_increasing


def test_csv_with_utc_offsets_keeps_the_local_session_date():
    """Regression: a yfinance ``.to_csv()`` export (dates with -05:00/-04:00) was rejected."""
    content = yahoo_frame().to_csv(index_label="Date").encode()
    assert b"-05:00" in content and b"-04:00" in content
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", CSVProvider(content), today=TODAY)
    assert ds.is_usable and ds.n_obs == 80
    tokyo = yahoo_frame().tz_convert(None).tz_localize("Asia/Tokyo").to_csv(index_label="Date").encode()
    assert parse_price_csv(tokyo).index[0] == pd.Timestamp("2024-01-02")  # not shifted to UTC


def test_csv_thousands_separators_and_european_dialect():
    """Regression: quoted "1,234.50" prices became missing and every row was removed."""
    frame = yahoo_frame().tz_localize(None)[["Open", "High", "Low", "Close", "Volume"]]
    frame[["Open", "High", "Low", "Close"]] *= 25  # prices above 1,000
    rows = [(d, *(f"{v:,.2f}" for v in r[:4]), f"{int(r[4]):,}") for d, r in zip(frame.index, frame.to_numpy())]
    us = "Date,Open,High,Low,Close,Volume\n" + "\n".join(
        f"{d:%Y-%m-%d}," + ",".join(f'"{v}"' for v in vals) for d, *vals in rows)
    parsed = parse_price_csv(us.encode())
    assert pd.to_numeric(parsed["close"]).round(2).tolist() == frame["Close"].round(2).tolist()

    def euro(v: str) -> str:  # 1,234.50 -> 1.234,50
        return v.replace(",", "_").replace(".", ",").replace("_", ".")

    eu = "Datum;Open;High;Low;Close;Volume\n" + "\n".join(
        f"{d:%d.%m.%Y};" + ";".join(euro(v) for v in vals) for d, *vals in rows)
    parsed = parse_price_csv(("\ufeff" + eu).encode("utf-8"))
    assert parsed.index[0] == pd.Timestamp("2024-01-02")
    assert pd.to_numeric(parsed["close"]).round(2).tolist() == frame["Close"].round(2).tolist()
    assert pd.to_numeric(parsed["volume"]).tolist() == frame["Volume"].tolist()


def test_ambiguous_day_month_order_is_refused():
    content = b"Date,Open,High,Low,Close,Volume\n" + b"\n".join(
        f"{m:02d}/{d:02d}/2024,1,2,0.5,1.5,10".encode() for m, d in [(1, 2), (2, 1), (3, 4), (5, 2), (1, 9)]
    )
    with pytest.raises(MalformedDataError, match="ambiguous"):
        parse_price_csv(content)


def test_unreadable_csv_dates_are_reported_not_silently_dropped():
    content = _csv_text(yahoo_frame()) + b"N/A,1,2,0.5,1.5,1.5,10\nSource: example vendor,,,,,,\n"
    ds = load_market_data("SPY", "2024-01-01", "2024-04-30", CSVProvider(content), today=TODAY)
    assert ds.n_obs == 80
    issue = next(i for i in ds.quality.issues if i.code == "malformed_dates")
    assert issue.count == 2  # the "N/A" row and the "Source" footer are both disclosed
    assert "Dates read as YYYY-MM-DD" in ds.notes


@pytest.mark.parametrize("freq,code", [("W-FRI", "not_daily"), ("D", "seven_day_calendar")])
def test_non_daily_calendars_are_refused(freq, code):
    """Regression: weekly and seven-day data were accepted and annualised at 252 sessions a year."""
    frame = yahoo_frame(n=200)
    frame.index = pd.date_range("2020-01-03", periods=200, freq=freq)
    content = _csv_text(frame)
    ds = load_market_data("BTC", "2020-01-01", "2024-05-31", CSVProvider(content), today=TODAY)
    assert not ds.is_usable and code in {i.code for i in ds.quality.errors}


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


def test_synthetic_demo_data_is_deterministic_labelled_and_never_cached(tmp_path):
    from quantlab.data import SyntheticProvider

    kwargs = dict(adjustment=AdjustmentMode.PRICE_ONLY, today=date(2024, 12, 31))
    first = load_market_data("SIM-RANDOM-1", "2015-01-01", "2020-12-31", SyntheticProvider(),
                             cache=DiskCache(tmp_path), **kwargs)
    again = load_market_data("SIM-RANDOM-1", "2015-01-01", "2020-12-31", SyntheticProvider(), **kwargs)
    other = load_market_data("SIM-RANDOM-2", "2015-01-01", "2020-12-31", SyntheticProvider(), **kwargs)
    assert first.cache_status is CacheStatus.SYNTHETIC and "not market" in first.source_name
    assert first.fingerprint == again.fingerprint != other.fingerprint
    assert first.is_usable and not first.quality.issues  # passes every data check on the NYSE calendar
    assert not list(tmp_path.iterdir())  # simulated prices are never written to the market-data cache
    with pytest.raises(TickerNotFoundError):
        load_market_data("SPY", "2015-01-01", "2020-12-31", SyntheticProvider(), **kwargs)


def test_planted_momentum_market_has_momentum_and_random_market_does_not():
    from quantlab.simulation import simulate_market

    def sign_persistence(close):
        q = np.log(close).diff(63).iloc[63::63].dropna()
        nxt = q.shift(-1).dropna()
        return float(np.mean(np.sign(q.loc[nxt.index]) == np.sign(nxt)))

    trend = np.mean([sign_persistence(simulate_market("trend", s)["Close"]) for s in (1, 2, 3)])
    noise = np.mean([sign_persistence(simulate_market("random", s)["Close"]) for s in (1, 2, 3)])
    assert trend > noise + 0.05
