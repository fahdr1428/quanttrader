"""Regression tests for data-layer defects found in an adversarial audit (no network; synthetic data)."""

from __future__ import annotations

import copy
import os
import time
from datetime import date, timedelta

import pandas as pd
import pytest

from quantlab.data import (
    AdjustmentMode,
    CSVProvider,
    DiskCache,
    RateLimitError,
    TickerNotFoundError,
    YahooFinanceProvider,
    _map_yahoo_exception,
    apply_adjustment,
    load_market_data,
)
from quantlab.validation import QualityReport
from tests.test_data import FakeTicker, yahoo_frame

TODAY = date(2024, 6, 1)


def _provider_at(frame, now_utc):
    fake = FakeTicker(frame)
    return YahooFinanceProvider(ticker_factory=lambda s: fake, sleep=lambda s: None, clock=lambda: now_utc), fake


def test_unfinished_session_is_dropped_during_trading_hours_and_disclosed():
    """Today's bar used to be treated as a completed session (its 'close' was the latest price)."""
    frame = yahoo_frame(n=40, start="2024-04-01")
    last = frame.index[-1]  # a weekday, 00:00 America/New_York
    during = (last + pd.Timedelta(hours=11)).tz_convert("UTC").to_pydatetime()
    after = (last + pd.Timedelta(hours=17)).tz_convert("UTC").to_pydatetime()
    provider, _ = _provider_at(frame, during)
    ds = load_market_data("SPY", "2024-04-01", last.date(), provider, today=last.date())
    assert ds.last_date < last.tz_localize(None) and any("had not closed" in n for n in ds.notes)
    provider, _ = _provider_at(frame, after)
    ds = load_market_data("SPY", "2024-04-01", last.date(), provider, today=last.date())
    assert ds.last_date == last.tz_localize(None) and not any("had not closed" in n for n in ds.notes)


@pytest.mark.parametrize("formatted", [True, False])
def test_unfinished_session_uses_the_exchange_close_time_from_yfinance_metadata(formatted):
    """Regression: yfinance gives the close as a Timestamp, int() failed silently and every exchange closed at 16:00."""
    from yfinance.utils import format_history_metadata

    frame = yahoo_frame(n=40, start="2024-02-12").tz_localize(None).tz_localize("Europe/Berlin")
    last = frame.index[-1]
    regular = {"start": int((last + pd.Timedelta(hours=9)).timestamp()),
               "end": int((last + pd.Timedelta(hours=17, minutes=30)).timestamp()), "timezone": "CET", "gmtoffset": 3600}
    meta = {"exchangeTimezoneName": "Europe/Berlin", "currentTradingPeriod": {"regular": regular}}
    for hour, dropped in ((16.5, True), (18, False)):  # XETRA closes at 17:30: still open at 16:30, closed at 18:00
        provider, fake = _provider_at(frame, (last + pd.Timedelta(hours=hour)).tz_convert("UTC").to_pydatetime())
        fake.history_metadata = format_history_metadata(copy.deepcopy(meta)) if formatted else meta
        out = provider.fetch("SAP.DE", pd.Timestamp("2024-02-01"), last.tz_localize(None))
        assert (out.index[-1] != last) is dropped and any("had not closed" in n for n in provider.fetch_notes()) is dropped


def test_download_missing_an_unfinished_session_expires_from_the_cache_when_that_session_closes(tmp_path, monkeypatch):
    """Regression: the incomplete download was cached for 24 h, so after the close the session stayed missing unnoticed."""
    from quantlab.data import CacheStatus
    from ui import state

    frame = yahoo_frame(n=60, start="2024-02-01")
    last = frame.index[-1]
    cache = DiskCache(tmp_path)
    loads = []
    for hour in (11, 14, 18):  # New York: two loads while the session is open, one after it closed
        now = (last + pd.Timedelta(hours=hour)).tz_convert("UTC").to_pydatetime()
        provider, fake = _provider_at(frame, now)
        loads.append((load_market_data("SPY", "2024-02-01", last.date(), provider, cache=cache, now=now,
                                       today=last.date()), fake.calls))
    (morning, _), (afternoon, afternoon_calls), (evening, evening_calls) = loads
    assert morning.cache_status is CacheStatus.DOWNLOADED and morning.complete_after is not None
    assert afternoon.cache_status is CacheStatus.CACHED and afternoon_calls == 0  # still open: the cache is fine
    assert any("had not closed" in n for n in afternoon.notes)  # ... and the cached copy keeps its disclosure
    assert evening.cache_status is CacheStatus.DOWNLOADED and evening_calls == 1
    assert evening.last_date == last.tz_localize(None) and evening.complete_after is None

    calls = []

    def fake_load(*args, **kwargs):
        calls.append(args)
        return types_ns(cache_status=CacheStatus.DOWNLOADED, complete_after=morning.complete_after)

    monkeypatch.setattr(state, "load_market_data", fake_load)
    state._fetch_yahoo.clear()
    state._yahoo("ZZZ", "2024-02-01", str(last.date()), "total_return")
    state._yahoo("ZZZ", "2024-02-01", str(last.date()), "total_return")
    assert len(calls) == 2  # an incomplete download is not kept in the shared memory cache
    state._fetch_yahoo.clear()


def test_yahoo_errors_are_classified_by_type_not_by_digits_in_the_message():
    from yfinance.exceptions import YFRateLimitError, YFTzMissingError

    assert isinstance(_map_yahoo_exception("4295.T", YFTzMissingError("4295.T")), TickerNotFoundError)
    assert isinstance(_map_yahoo_exception("SPY", YFRateLimitError()), RateLimitError)
    assert isinstance(_map_yahoo_exception("SPY", RuntimeError("HTTP Error 429: Too Many Requests")), RateLimitError)


def test_invalidate_clears_the_yfinance_response_cache():
    from yfinance.data import YfData

    YahooFinanceProvider().invalidate()
    assert YfData().cache_get.cache_info().currsize == 0


def test_cache_keys_do_not_collide_and_metadata_is_checked(tmp_path):
    cache = DiskCache(tmp_path)
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-30")
    assert DiskCache.make_key("yahoo", "GC=F", start, end) != DiskCache.make_key("yahoo", "GC.F", start, end)
    gold = yahoo_frame().tz_localize(None)
    other = gold * 2
    for ticker, frame in (("GC=F", gold), ("GC.F", other)):
        fake = FakeTicker(frame.tz_localize("America/New_York"))
        provider = YahooFinanceProvider(ticker_factory=lambda s, f=fake: f, sleep=lambda s: None)
        load_market_data(ticker, start, end, provider, cache=cache, today=TODAY)
    key = DiskCache.make_key("yahoo", "GC=F", start, end)
    assert cache.load(key, {"ticker": "GC.F"}) is None  # a mismatching request is refused
    assert cache.load(key, {"ticker": "GC=F"}) is not None


def test_cache_prunes_old_and_excess_entries(tmp_path):
    cache = DiskCache(tmp_path)
    frame = yahoo_frame(n=5).tz_localize(None)
    for i in range(6):
        cache.save(f"k{i}", frame, {"ticker": "X", "fetched_at": "2024-01-01T00:00:00+00:00"})
    old = time.time() - 30 * 86400
    os.utime(tmp_path / "k0.json", (old, old))
    removed = cache.prune(max_age=timedelta(days=14), max_entries=4)
    assert removed == 2 and len(list(tmp_path.glob("*.json"))) == 4 and not (tmp_path / "k0.csv").exists()


def _with_dividends(adj_factor_steps: bool, dividend: float = 1.0) -> pd.DataFrame:
    frame = yahoo_frame(n=60).tz_localize(None)
    frame.columns = [c.lower().replace(" ", "_") for c in frame.columns]
    ex = frame.index[30]
    frame.loc[ex, "dividends"] = dividend
    if adj_factor_steps:
        prev_close = frame["close"].shift(1).loc[ex]
        factor = pd.Series(1.0, index=frame.index)
        factor[frame.index < ex] = 1 - dividend / prev_close
        frame["adj_close"] = frame["close"] * factor
    else:
        frame["adj_close"] = frame["close"] * 0.98
    return frame


def test_unadjusted_adj_close_with_reported_dividends_is_flagged():
    report = QualityReport()
    apply_adjustment(_with_dividends(adj_factor_steps=False), AdjustmentMode.TOTAL_RETURN, report)
    assert "adjustment_missing" in {i.code for i in report.issues}


def test_consistent_dividend_adjustment_is_not_flagged_but_inconsistent_is():
    report = QualityReport()
    apply_adjustment(_with_dividends(adj_factor_steps=True), AdjustmentMode.TOTAL_RETURN, report)
    assert not {"adjustment_missing", "adjustment_inconsistent"} & {i.code for i in report.issues}
    frame = _with_dividends(adj_factor_steps=True)
    frame.loc[frame.index[30], "dividends"] = 5.0  # the factor step implies about 1.0
    report = QualityReport()
    apply_adjustment(frame, AdjustmentMode.TOTAL_RETURN, report)
    assert "adjustment_inconsistent" in {i.code for i in report.issues}


def test_uploaded_price_only_data_is_not_claimed_to_be_split_adjusted():
    content = yahoo_frame().tz_localize(None).to_csv(index_label="Date").encode()
    ds = load_market_data("X", "2024-01-01", "2024-04-30", CSVProvider(content), AdjustmentMode.PRICE_ONLY,
                          today=TODAY)
    assert ds.adjustment == "as_provided"


# ---- research-data assembly (ui.state) ------------------------------------------------


def _csv_bytes(n=400, start="2021-01-04"):
    frame = yahoo_frame(n=n, start=start).tz_localize(None)
    return frame.to_csv(index_label="Date").encode()


def test_minimum_sample_applies_to_the_evaluation_window_not_the_warm_up():
    from ui import state
    from quantlab.validation import InputValidationError

    with pytest.raises(InputValidationError):
        state.load_research_data(source=state.SOURCE_CSV, ticker="X", benchmark_mode=state.BENCH_SAME,
                                 benchmark_ticker="", start=date(2022, 6, 6), end=date(2022, 6, 9),
                                 adjustment="total_return", warmup_days=800, csv_content=_csv_bytes())


def test_uploaded_file_is_labelled_from_its_name_not_the_default_ticker():
    from ui import state

    data = state.load_research_data(source=state.SOURCE_CSV, ticker=state.DEFAULT_TICKER,
                                    benchmark_mode=state.BENCH_SAME, benchmark_ticker="", start=date(2021, 6, 1),
                                    end=date(2022, 6, 1), adjustment="total_return", warmup_days=100,
                                    csv_content=_csv_bytes(), csv_name="my_fund-prices.csv")
    assert data.asset.ticker == "MYFUND-PRICES" and any("file name" in n for n in data.notes)


def test_benchmark_failure_names_the_benchmark(monkeypatch):
    from ui import state

    def fail(*args, **kwargs):
        raise RateLimitError("Yahoo Finance is rate-limiting requests.")

    monkeypatch.setattr(state, "_yahoo", fail)
    with pytest.raises(RateLimitError, match="benchmark 'QQQ'"):
        state.load_research_data(source=state.SOURCE_CSV, ticker="X", benchmark_mode=state.BENCH_TICKER,
                                 benchmark_ticker="QQQ", start=date(2021, 6, 1), end=date(2022, 6, 1),
                                 adjustment="total_return", warmup_days=100, csv_content=_csv_bytes())


def test_benchmark_alignment_notes_separate_range_from_gaps_and_flag_price_basis(monkeypatch):
    from ui import state

    bench_frame = yahoo_frame(n=700, start="2019-06-03")
    bench_frame = bench_frame.drop(bench_frame.index[500])  # one mid-sample gap
    provider_bench = load_market_data("QQQ", "2019-06-01", "2022-06-01",
                                      YahooFinanceProvider(ticker_factory=lambda s: FakeTicker(bench_frame),
                                                           sleep=lambda s: None),
                                      today=date(2024, 1, 1))
    monkeypatch.setattr(state, "_yahoo", lambda *a, **k: provider_bench)
    data = state.load_research_data(source=state.SOURCE_CSV, ticker="X", benchmark_mode=state.BENCH_TICKER,
                                    benchmark_ticker="QQQ", start=date(2021, 6, 1), end=date(2022, 6, 1),
                                    adjustment="total_return", warmup_days=200, csv_content=_csv_bytes())
    notes = " ".join(data.notes)
    assert "limited to the dates both series cover" in notes
    assert "1 session(s) inside the common range" in notes


def test_forced_refresh_replaces_the_shared_entry_and_stale_results_are_not_memoised(monkeypatch):
    from ui import state

    calls = []

    def fake_load(ticker, start, end, provider, adjustment, cache, force_refresh):
        calls.append(force_refresh)
        return types_ns(cache_status=state.CacheStatus.DOWNLOADED, n=len(calls), complete_after=None)

    monkeypatch.setattr(state, "load_market_data", fake_load)
    state._fetch_yahoo.clear()
    a = state._yahoo("ZZZ", "2020-01-01", "2021-01-01", "total_return")
    b = state._yahoo("ZZZ", "2020-01-01", "2021-01-01", "total_return")
    assert a.n == b.n == 1  # memoised
    c = state._yahoo("ZZZ", "2020-01-01", "2021-01-01", "total_return", force=True)
    d = state._yahoo("ZZZ", "2020-01-01", "2021-01-01", "total_return")
    assert c.n == d.n == 2 and calls == [False, True]  # the refreshed value is what later loads see
    state._fetch_yahoo.clear()


def types_ns(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**kw)


def test_stale_fallback_uses_the_latest_entry_for_the_ticker_when_the_end_date_moved(tmp_path):
    """The exact-key fallback almost never fired: the default end date changes every day."""
    from quantlab.data import CacheStatus, ProviderConnectionError

    cache = DiskCache(tmp_path)
    good = YahooFinanceProvider(ticker_factory=lambda s: FakeTicker(yahoo_frame()), sleep=lambda s: None)
    load_market_data("SPY", "2024-01-01", "2024-04-30", good, cache=cache, today=TODAY)
    down = YahooFinanceProvider(ticker_factory=lambda s: FakeTicker(exc=ProviderConnectionError("offline")),
                                retries=0, sleep=lambda s: None)
    ds = load_market_data("SPY", "2024-01-01", "2024-05-15", down, cache=cache, today=TODAY)
    assert ds.cache_status is CacheStatus.STALE_CACHE and ds.n_obs > 0
    assert "stale_cache" in {i.code for i in ds.quality.issues}
    with pytest.raises(ProviderConnectionError):  # another ticker's entry is never used
        load_market_data("QQQ", "2024-01-01", "2024-05-15", down, cache=cache, today=TODAY)


def test_historical_nyse_closures_are_not_expected_sessions():
    from quantlab.calendar import expected_sessions

    sessions = expected_sessions(pd.Timestamp("1970-01-01"), pd.Timestamp("1990-01-01"))
    for day in ("1972-11-07", "1972-12-28", "1973-01-25", "1976-11-02", "1977-07-14", "1980-11-04", "1985-09-27"):
        assert pd.Timestamp(day) not in sessions


def test_non_numeric_price_cells_are_disclosed():
    frame = yahoo_frame(n=40).tz_localize(None)
    frame = frame.astype({"Close": object})
    frame.iloc[5, frame.columns.get_loc("Close")] = "n/a"
    report = QualityReport()
    apply_adjustment(frame, AdjustmentMode.PRICE_ONLY, report)
    issue = next(i for i in report.issues if i.code == "malformed_values")
    assert issue.count == 1 and "close" in issue.message


def test_manifest_records_benchmark_alignment():
    from quantlab.config import BacktestConfig
    from quantlab.reporting import build_manifest

    m = build_manifest(ticker="X", benchmark="Y buy & hold", eval_start=pd.Timestamp("2020-01-02"),
                       eval_end=pd.Timestamp("2021-01-04"), data_start=pd.Timestamp("2019-01-02"), provider_key="yahoo",
                       source_name="Yahoo", adjustment="total_return", fingerprint="0" * 64,
                       first_observation=pd.Timestamp("2019-01-02"), last_observation=pd.Timestamp("2021-01-04"),
                       strategy_key="buy_and_hold", params={}, config=BacktestConfig(), benchmark_ticker="Y",
                       dropped_dates=["2020-07-03"])
    assert m["data"]["benchmark_ticker"] == "Y" and m["data"]["dropped_dates"] == ["2020-07-03"]


def test_sidebar_starts_collapsed_on_phones():
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parent.parent / "app.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "set_page_config")
    value = next(k.value.value for k in call.keywords if k.arg == "initial_sidebar_state")
    assert value == "auto"


def test_rows_without_a_date_are_counted_but_never_listed_as_nan_dates():
    """Regression: two exports concatenated (a second header row) or a 'Total' footer put NaN among an issue's
    example dates, and the Data page crashed joining them."""
    csv = yahoo_frame(n=60).tz_localize(None).to_csv(index_label="Date")
    content = (csv + csv.splitlines()[0] + "\nTotal,abc,abc,abc,abc,abc,0,0,0\n").encode()
    ds = load_market_data("SPY", "2024-01-01", "2024-05-31", CSVProvider(content), AdjustmentMode.TOTAL_RETURN,
                          today=TODAY)
    assert ds.is_usable and ds.n_obs == 60
    malformed = [i for i in ds.quality.issues if i.code == "malformed_values"]
    assert malformed and malformed[0].count >= 2  # the undated rows are still counted
    for issue in ds.quality.issues:
        assert all(isinstance(d, str) for d in issue.dates), issue
        ", ".join(issue.dates)  # what the Data page renders
    ds.quality.to_frame()


def test_stale_fallback_uses_the_entry_that_covers_the_request_and_skips_unreadable_files(tmp_path):
    """Regression: the newest cache entry was used even when it did not cover the request (so the load failed
    although an older entry covered it), and one unreadable file disabled the fallback for every ticker."""
    from datetime import datetime, timezone

    from quantlab.data import CacheStatus, ProviderConnectionError

    cache = DiskCache(tmp_path)
    frame = yahoo_frame(n=120)
    t0 = datetime(2024, 6, 3, 12, tzinfo=timezone.utc)
    for rows, end, now in ((frame, "2024-06-14", t0), (frame.iloc[:40], "2024-02-26", t0 + timedelta(hours=1))):
        provider = YahooFinanceProvider(ticker_factory=lambda s, f=rows: FakeTicker(f), sleep=lambda s: None)
        load_market_data("SPY", "2024-01-01", end, provider, cache=cache, now=now, today=date(2024, 7, 1))
    (tmp_path / "yahoo_QQQ_bad.json").write_text("{truncated")
    (tmp_path / "yahoo_QQQ_list.json").write_text("[]")
    down = YahooFinanceProvider(ticker_factory=lambda s: FakeTicker(exc=ProviderConnectionError("offline")),
                                retries=0, sleep=lambda s: None)
    ds = load_market_data("SPY", "2024-01-01", "2024-06-30", down, cache=cache, now=t0 + timedelta(hours=2),
                          today=date(2024, 7, 1))
    assert ds.cache_status is CacheStatus.STALE_CACHE and ds.n_obs == 120  # the covering entry, not the newest


def test_newest_first_files_with_correct_dividend_adjustment_are_not_flagged():
    """Regression: the adjustment check compared each row with the next one in file order, so a correctly
    adjusted newest-first CSV was reported as 'The provider's adjustment may be wrong'."""
    frame = yahoo_frame(n=80).tz_localize(None)
    close = frame["Close"]
    dividends = pd.Series(0.0, index=frame.index)
    dividends.iloc[30], dividends.iloc[60] = 1.0, 1.2
    factor = pd.Series(1.0, index=frame.index)
    for i in (30, 60):
        factor.iloc[:i] *= 1 - dividends.iloc[i] / close.iloc[i - 1]
    frame["Adj Close"], frame["Dividends"] = close * factor, dividends
    codes = {}
    for order in ("oldest", "newest"):
        f = frame if order == "oldest" else frame.iloc[::-1]
        content = f.drop(columns=["Stock Splits"]).to_csv(index_label="Date").encode()
        ds = load_market_data("SPY", "2024-01-01", "2024-05-31", CSVProvider(content), AdjustmentMode.TOTAL_RETURN,
                              today=TODAY)
        codes[order] = {i.code for i in ds.quality.issues}
    assert "adjustment_inconsistent" not in codes["oldest"] and "adjustment_inconsistent" not in codes["newest"]


def test_an_uploaded_file_keeps_the_label_the_user_typed_even_spy():
    """Regression: a CSV labelled 'SPY' was always relabelled from its file name, because the label could not be
    told apart from the untouched default ticker."""
    from ui import state

    common = dict(source=state.SOURCE_CSV, benchmark_mode=state.BENCH_SAME, benchmark_ticker="",
                  start=date(2021, 6, 1), end=date(2022, 6, 1), adjustment="total_return", warmup_days=100,
                  csv_content=_csv_bytes(), csv_name="export2024.csv")
    typed = state.load_research_data(ticker="SPY", label_from_file=False, **common)
    assert typed.asset.ticker == "SPY" and not any("file name" in n for n in typed.notes)
    blank = state.load_research_data(ticker="", label_from_file=True, **common)
    assert blank.asset.ticker == "EXPORT2024" and any("file name" in n for n in blank.notes)


@pytest.mark.parametrize(("start", "end"), [(date(2000, 6, 1), date(2001, 12, 31)),
                                            (date(2025, 3, 3), date(2026, 10, 13))])
def test_demo_windows_outside_the_simulated_period_get_a_specific_message(start, end):
    """Regression: after the window was clipped to the simulated period, a valid-looking choice failed with the
    misleading 'End date must be after the start date'."""
    from quantlab.validation import InputValidationError
    from ui import state

    with pytest.raises(InputValidationError, match="synthetic series covers 2000-01-03 to 2024-12-31"):
        state.load_research_data(source=state.SOURCE_DEMO, ticker="", benchmark_mode=state.BENCH_SAME,
                                 benchmark_ticker="", start=start, end=end, adjustment="price_only",
                                 warmup_days=800)
