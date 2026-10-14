"""Metric definitions, annualisation conventions and degenerate cases."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics import monthly_return_table, rolling_sharpe, rolling_volatility
from quantlab.backtest import run_backtest, run_buy_and_hold, simulate
from quantlab.config import BacktestConfig
from quantlab.metrics import (
    annualised_return,
    annualised_turnover,
    annualised_volatility,
    compute_metrics,
    daily_risk_free,
    drawdown_episodes,
    max_drawdown,
    period_returns,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from quantlab.strategies import create_strategy
from tests.helpers import make_prices, random_walk


def _equity(values, start="2024-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


def test_total_return_and_returns():
    eq = _equity([100, 110, 99, 108.9])
    assert total_return(eq) == pytest.approx(0.089)
    np.testing.assert_allclose(eq.pct_change().iloc[1:], [0.1, -0.1, 0.1])


def test_cagr_uses_252_session_convention():
    one_year = _equity(100 * 2 ** (np.arange(253) / 252))
    assert annualised_return(one_year) == pytest.approx(1.0)
    two_years = _equity(np.linspace(100, 200, 505))
    assert annualised_return(two_years) == pytest.approx(math.sqrt(2) - 1)


def test_sharpe_and_sortino_hand_computed():
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0])
    std = math.sqrt((0 + 0.01**2 + 0.02**2 + 0.02**2 + 0.01**2) / 4)
    assert sharpe_ratio(r) == pytest.approx(0.01 / std * math.sqrt(252))
    downside = math.sqrt(0.01**2 / 5)
    assert sortino_ratio(r) == pytest.approx(0.01 / downside * math.sqrt(252))
    assert annualised_volatility(r) == pytest.approx(std * math.sqrt(252))


def test_risk_free_rate_conversion():
    rf_d = daily_risk_free(0.05)
    assert (1 + rf_d) ** 252 == pytest.approx(1.05)
    r = pd.Series([0.001, 0.002, 0.0, 0.003])
    excess = r - rf_d
    assert sharpe_ratio(r, 0.05) == pytest.approx(excess.mean() / excess.std(ddof=1) * math.sqrt(252))


def test_undefined_ratios_return_none():
    flat = pd.Series([0.0] * 50)
    assert sharpe_ratio(flat) is None
    assert sortino_ratio(flat) is None
    assert sortino_ratio(pd.Series([0.01, 0.02, 0.03])) is None  # no downside
    assert sharpe_ratio(pd.Series([0.01])) is None
    assert annualised_volatility(pd.Series([0.01])) is None


def test_max_drawdown_and_episodes():
    eq = _equity([100, 120, 90, 110, 130, 100])
    assert max_drawdown(eq) == pytest.approx(90 / 120 - 1)
    episodes = drawdown_episodes(eq)
    assert len(episodes) == 2
    first, second = episodes.iloc[0], episodes.iloc[1]
    assert first["peak"] == eq.index[1] and first["trough"] == eq.index[2]
    assert first["recovery"] == eq.index[4] and first["sessions"] == 3 and not first["ongoing"]
    assert second["ongoing"] and pd.isna(second["recovery"])
    assert second["depth"] == pytest.approx(100 / 130 - 1)


def test_no_drawdown_for_monotonic_equity():
    eq = _equity([100, 101, 102, 103])
    assert max_drawdown(eq) == 0
    assert drawdown_episodes(eq).empty


def test_period_returns_chain_from_anchor():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])
    eq = pd.Series([100.0, 110.0, 121.0, 99.0, 108.9], index=idx)
    monthly = period_returns(eq, "ME")
    np.testing.assert_allclose(monthly.to_numpy(), [0.10, 99 / 110 - 1, 108.9 / 99 - 1])
    compounded = (1 + monthly).prod() - 1
    assert compounded == pytest.approx(total_return(eq))


def test_anchor_in_previous_year_does_not_create_a_stub_period():
    """Regression: an anchor dated 31 Dec produced a bogus 0.00% year (and month)."""
    idx = pd.to_datetime(["2014-12-31", "2015-01-02", "2015-06-30", "2015-12-31", "2016-01-04"])
    eq = pd.Series([100.0, 101.0, 90.0, 95.0, 96.0], index=idx)
    yearly = period_returns(eq, "YE")
    assert [d.year for d in yearly.index] == [2015, 2016]
    np.testing.assert_allclose(yearly.to_numpy(), [-0.05, 96 / 95 - 1])
    assert period_returns(eq, "ME").index[0] == pd.Timestamp("2015-01-31")


def test_entries_and_exits_count_flat_invested_transitions():
    """Regression: 'position changes' equalled fills, overstating it for rebalancing strategies."""
    from quantlab.backtest import backtest_from_signals

    prices = make_prices(np.linspace(100, 110, 12))
    targets = [1.0, 0.5, 1.0, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    signals = pd.DataFrame({"target": targets}, index=prices.index)
    result = backtest_from_signals(prices, signals, BacktestConfig(fee_rate=0, slippage_rate=0, rebalance_threshold=0),
                                   "scripted", "Scripted", {})
    report = compute_metrics(result)
    assert report.value("fills") > report.value("position_changes")
    assert report.value("position_changes") == 3  # enter, exit, re-enter
    assert report.value("round_trips") == 1


def test_monthly_table_compounds_to_annual():
    result = run_buy_and_hold(random_walk(n=520, seed=2), BacktestConfig())
    table = monthly_return_table(result.equity)
    for _year, row in table.iterrows():
        months = row.drop("Full year").dropna()
        assert (1 + months).prod() - 1 == pytest.approx(row["Full year"], rel=1e-9)


def test_turnover_hand_computed():
    eq = _equity([10_000.0] * 253)
    ledger = pd.DataFrame({"gross_notional": [10_000.0, 10_000.0]})
    assert annualised_turnover(ledger, eq) == pytest.approx(2.0)


def test_rolling_statistics_respect_window():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))
    vol = rolling_volatility(r, 63)
    assert vol.iloc[:62].isna().all() and vol.iloc[62:].notna().all()
    assert vol.iloc[62] == pytest.approx(r.iloc[:63].std(ddof=1) * math.sqrt(252))
    rs = rolling_sharpe(pd.Series([0.0] * 150), 126)
    assert rs.isna().all()  # zero volatility windows are undefined, not infinite


def test_never_invested_strategy_reports_without_crashing():
    prices = random_walk(n=300, seed=1)
    result = run_backtest(prices, create_strategy("rsi_reversion", {"entry_threshold": 1, "exit_threshold": 99}),
                          BacktestConfig())
    report = compute_metrics(result)
    assert report.value("total_return") == 0
    assert report.value("sharpe") is None
    assert "zero volatility" in report["sharpe"].note
    assert report.value("win_rate") is None and not report["win_rate"].reliable
    assert report.value("fills") == 0


def test_small_samples_are_flagged_unreliable():
    prices = random_walk(n=40, seed=3)
    report = compute_metrics(run_buy_and_hold(prices, BacktestConfig()))
    assert not report["cagr"].reliable and "extrapolation" in report["cagr"].note
    assert not report["sharpe"].reliable
    assert report.value("sharpe") is not None  # computed, but labelled


def test_benchmark_relative_metrics_against_itself():
    prices = random_walk(n=400, seed=4)
    a = run_buy_and_hold(prices, BacktestConfig())
    report = compute_metrics(a, benchmark=a)
    assert report.value("excess_return") == pytest.approx(0)
    assert report.value("beta") == pytest.approx(1)
    assert report.value("correlation") == pytest.approx(1)
    assert report.value("tracking_error") == pytest.approx(0, abs=1e-12)
    assert report.value("information_ratio") is None


def test_metrics_derive_from_equity_curve(walk):
    result = run_backtest(walk, create_strategy("ma_crossover"), BacktestConfig())
    report = compute_metrics(result)
    eq = result.equity
    assert report.value("final_value") == eq.iloc[-1]
    assert report.value("total_return") == pytest.approx(eq.iloc[-1] / eq.iloc[0] - 1)
    assert report.value("max_drawdown") == pytest.approx((eq / eq.cummax() - 1).min())
    assert report.value("total_fees") == pytest.approx(result.ledger["fee"].sum())


def test_trade_statistics_hand_computed():
    prices = make_prices(closes=[10, 10, 10, 12, 12, 12, 9, 9], opens=[10, 10, 10, 12, 12, 12, 9, 9])
    targets = pd.Series([1, 1, 0, 1, 1, 0, 0, 0], index=prices.index, dtype=float)
    daily, ledger, _, _ = simulate(prices, targets, BacktestConfig(fee_rate=0, slippage_rate=0))
    # Trade 1: buy 10 -> sell 12 (+20%). Trade 2: buy 12 -> sell 9 (-25%).
    from quantlab.backtest import build_round_trips

    trades = build_round_trips(ledger, daily)
    np.testing.assert_allclose(trades["net_return"], [0.2, -0.25])
    np.testing.assert_allclose(trades["holding_sessions"], [2, 2])


@pytest.mark.parametrize("rf", [0.0, 0.04])
def test_inference_metrics_are_consistent_with_the_headline_sharpe(walk, rf):
    from quantlab.research import probabilistic_sharpe_ratio

    config = BacktestConfig(risk_free_rate=rf)
    bench = run_buy_and_hold(walk, config, walk.index[300])
    result = run_backtest(walk, create_strategy("ma_crossover"), config, walk.index[300])
    report = compute_metrics(result, benchmark=bench)
    sharpe, lo, hi = (report.value(k) for k in ("sharpe", "sharpe_ci_low", "sharpe_ci_high"))
    assert lo < sharpe < hi and (lo + hi) / 2 == pytest.approx(sharpe, rel=1e-9)
    assert report.value("sharpe_se") == pytest.approx((hi - lo) / (2 * 1.959963984540054), rel=1e-6)
    assert report.value("psr") == pytest.approx(
        probabilistic_sharpe_ratio(result.returns, 0.0, rf, config.periods_per_year), rel=1e-9)
    assert report.value("var_95") >= report.value("cvar_95")
    assert report.get("alpha_annual") is not None and report.get("alpha_tstat") is not None
    self_report = compute_metrics(bench, benchmark=bench)
    assert self_report.value("alpha_annual") == pytest.approx(0.0, abs=1e-10)


def test_sharpe_inference_is_flagged_when_few_sessions_carry_market_risk():
    """Regression: an RSI rule invested on 1% of sessions showed a 'reliable' PSR of 98% and a Sharpe interval
    excluding zero on a driftless random walk, because reliability was judged by the window length."""
    from quantlab.analytics import run_strategy
    from quantlab.evidence import evidence_checklist
    from quantlab.metrics import exposed_sessions

    prices = random_walk(1400, seed=16, drift=0.0)
    start, end = prices.index[260], prices.index[-1]
    run = run_strategy(prices, create_strategy("rsi_reversion", {"window": 14, "entry_threshold": 15, "exit_threshold": 50}),
                       BacktestConfig(), start, end)
    assert exposed_sessions(run.net) < 252 and run.report["observations"].value > 1000
    for key in ("sharpe_se", "sharpe_ci_low", "sharpe_ci_high", "psr", "min_track_record_years"):
        metric = run.report.get(key)
        assert metric is None or not metric.reliable, key
    items = evidence_checklist(report=run.report, accounting_valid=True, data_warnings=0, synthetic=True,
                               parameters_prespecified=True, n_trials=1, dsr=None)
    assert next(i for i in items if i.key == "sharpe_ci").status != "pass"

    always = run_strategy(prices, create_strategy("buy_and_hold"), BacktestConfig(), start, end)
    assert always.report["psr"].reliable  # a fully invested strategy over the same window is unaffected
