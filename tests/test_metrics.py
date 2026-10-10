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
