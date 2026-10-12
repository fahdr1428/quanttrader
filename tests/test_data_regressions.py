"""Regression tests for data-layer defects found in an adversarial audit (no network; synthetic data)."""

from __future__ import annotations

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
        return types_ns(cache_status=state.CacheStatus.DOWNLOADED, n=len(calls))

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
