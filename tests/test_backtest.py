"""Engine tests with hand-calculated portfolio values.

Each test states the arithmetic it expects so a reader can verify it with a
calculator. Timing convention under test: signal at close t -> fill at open t+1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import (
    reconcile,
    run_backtest,
    run_buy_and_hold,
    simulate,
)
from quantlab.config import BacktestConfig
from quantlab.strategies import BuyAndHold, MovingAverageCrossover, create_strategy
from tests.helpers import make_prices, random_walk

NO_COSTS = BacktestConfig(fee_rate=0.0, slippage_rate=0.0)


def _targets(prices: pd.DataFrame, values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype="float64"), index=prices.index)


# --------------------------------------------------------------------------- #
# Hand-calculated accounting
# --------------------------------------------------------------------------- #


def test_buy_and_hold_matches_hand_calculation():
    prices = make_prices(closes=[10.5, 11.5, 12.5, 13.5], opens=[10, 11, 12, 13])
    cfg = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_rate=0.0005)
    result = run_buy_and_hold(prices, cfg)

    exec_price = 11 * 1.0005  # bought at session 1's open plus slippage
    qty = 10_000 / (exec_price * 1.001)  # notional + 0.1% fee = all cash
    notional = qty * exec_price
    assert len(result.ledger) == 1
    fill = result.ledger.iloc[0]
    assert fill["signal_date"] == prices.index[0]
    assert fill["execution_date"] == prices.index[1]
    assert fill["reference_price"] == pytest.approx(11.0)
    assert fill["execution_price"] == pytest.approx(exec_price)
    assert fill["quantity"] == pytest.approx(qty)
    assert fill["gross_notional"] == pytest.approx(10_000 / 1.001)
    assert fill["fee"] == pytest.approx(0.001 * notional)
    assert fill["slippage_cost"] == pytest.approx(qty * 11 * 0.0005)
    assert fill["cash_after"] == pytest.approx(0.0, abs=1e-9)

    expected_equity = [10_000.0, qty * 11.5, qty * 12.5, qty * 13.5]
    np.testing.assert_allclose(result.equity.to_numpy(), expected_equity, rtol=1e-12)
    assert result.liquidation_value == pytest.approx(qty * 13.5 * (1 - 0.0005) * (1 - 0.001))
    assert result.is_valid


def test_round_trip_without_costs():
    # Buy at session 1 open (10), sell at session 3 open (15).
    prices = make_prices(closes=[10, 11, 13, 14, 16], opens=[10, 10, 12, 15, 15])
    daily, ledger, _, _ = simulate(prices, _targets(prices, [1, 1, 0, 0, 0]), NO_COSTS)
    np.testing.assert_allclose(daily["equity"], [10_000, 11_000, 13_000, 15_000, 15_000])
    assert list(ledger["side"]) == ["BUY", "SELL"]
    assert ledger["quantity"].tolist() == pytest.approx([1000.0, 1000.0])
    assert daily["cash"].iloc[-1] == pytest.approx(15_000)


def test_round_trip_with_costs_matches_formula():
    prices = make_prices(closes=[100, 100, 100, 110, 110], opens=[100, 100, 100, 110, 110])
    cfg = BacktestConfig(initial_capital=10_000, fee_rate=0.01, slippage_rate=0.01)
    result = run_backtest(prices, _FixedTargets([1, 1, 0, 0, 0]), cfg)

    qty = 10_000 / (101 * 1.01)  # buy at 100 * 1.01, 1% fee
    sell_price = 110 * 0.99  # sell slippage is adverse: below the open
    expected_final = qty * sell_price * 0.99  # minus 1% sell fee
    assert result.final_value == pytest.approx(expected_final, rel=1e-12)
    trade = result.trades.iloc[0]
    assert trade["entry_cost"] == pytest.approx(10_000)
    assert trade["exit_value"] == pytest.approx(expected_final)
    assert trade["net_return"] == pytest.approx(expected_final / 10_000 - 1)
    assert trade["gross_return"] == pytest.approx(0.10)
    assert trade["slippage_cost"] == pytest.approx(qty * 1.0 + qty * 1.1)
    assert result.ledger["fee"].sum() == pytest.approx(0.01 * qty * 101 + 0.01 * qty * sell_price)
    assert result.is_valid


def test_whole_share_sizing_leaves_residual_cash():
    prices = make_prices(closes=[300] * 4, opens=[300] * 4)
    cfg = BacktestConfig(initial_capital=1_000, fee_rate=0.001, slippage_rate=0.0, fractional_shares=False)
    result = run_buy_and_hold(prices, cfg)
    assert result.ledger["quantity"].iloc[0] == 3  # floor(1000 / 300.3)
    assert result.daily["cash"].iloc[-1] == pytest.approx(1_000 - 900 - 0.9)
    assert result.daily["exposure"].iloc[-1] == pytest.approx(900 / (900 + 99.1))


def test_insufficient_cash_rejects_once_per_signal_episode():
    prices = make_prices(closes=[300] * 10, opens=[300] * 10)
    cfg = BacktestConfig(initial_capital=100, fee_rate=0.0, slippage_rate=0.0, fractional_shares=False)
    targets = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    daily, ledger, rejected, _ = simulate(prices, _targets(prices, targets), cfg)
    assert ledger.empty
    # One rejection for the first episode, one for the second — not one per session.
    assert len(rejected) == 2
    assert (daily["cash"] == 100).all()


# --------------------------------------------------------------------------- #
# Timing and look-ahead prevention
# --------------------------------------------------------------------------- #


def test_fill_uses_next_open_not_signal_close():
    # Signal at close of session 0 (close 100); session 1 gaps down to open at 80.
    prices = make_prices(closes=[100, 100, 100], opens=[100, 80, 80])
    daily, ledger, _, _ = simulate(prices, _targets(prices, [1, 1, 1]), NO_COSTS)
    assert ledger["execution_price"].iloc[0] == 80
    assert ledger["quantity"].iloc[0] == pytest.approx(10_000 / 80)
    assert ledger["execution_date"].iloc[0] == prices.index[1]
    assert daily["equity"].iloc[1] == pytest.approx(10_000 / 80 * 100)


def test_persistent_signal_does_not_generate_repeated_orders():
    prices = random_walk(n=60, seed=3)
    daily, ledger, _, _ = simulate(prices, _targets(prices, [1] * 60), BacktestConfig())
    assert len(ledger) == 1
    assert (daily["shares"].iloc[1:] == ledger["quantity"].iloc[0]).all()


def test_warmup_nan_is_treated_as_cash():
    prices = make_prices(closes=[10, 10, 10, 10, 10])
    daily, ledger, _, _ = simulate(prices, _targets(prices, [np.nan, np.nan, 1, 1, 1]), NO_COSTS)
    assert ledger["signal_date"].iloc[0] == prices.index[2]
    assert ledger["execution_date"].iloc[0] == prices.index[3]
    assert (daily["shares"].iloc[:3] == 0).all()


def test_final_session_signal_is_reported_not_executed():
    prices = make_prices(closes=[10, 10, 10, 10])
    daily, ledger, _, unexecuted = simulate(prices, _targets(prices, [0, 0, 0, 1]), NO_COSTS)
    assert ledger.empty
    assert unexecuted is not None and unexecuted["side"] == "BUY"
    assert unexecuted["signal_date"] == prices.index[-1]
    assert daily["equity"].iloc[-1] == 10_000


def test_open_position_is_marked_to_market_at_final_close():
    prices = make_prices(closes=[10, 12, 15], opens=[10, 10, 14])
    result = run_backtest(prices, _FixedTargets([1, 1, 1]), NO_COSTS)
    trade = result.trades.iloc[-1]
    assert trade["status"].startswith("open")
    assert trade["exit_price"] == 15
    assert result.final_value == pytest.approx(1_000 * 15)


def test_future_prices_cannot_change_past_decisions():
    """Regression test: altering prices from session k onward leaves every
    decision and valuation before k untouched."""
    prices = random_walk(n=400, seed=21)
    k = 250
    altered = prices.copy()
    rng = np.random.default_rng(99)
    shock = np.exp(rng.normal(0, 0.05, len(prices) - k))
    for col in ("open", "high", "low", "close"):
        altered.iloc[k:, altered.columns.get_loc(col)] *= shock

    strategy = MovingAverageCrossover(short_window=10, long_window=30)
    base = run_backtest(prices, strategy, BacktestConfig())
    alt = run_backtest(altered, strategy, BacktestConfig())

    cutoff = prices.index[k]
    pd.testing.assert_series_equal(base.equity.loc[: prices.index[k - 1]], alt.equity.loc[: prices.index[k - 1]])
    base_fills = base.ledger[base.ledger["execution_date"] < cutoff].reset_index(drop=True)
    alt_fills = alt.ledger[alt.ledger["execution_date"] < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_fills, alt_fills)
    # And the alteration really did matter afterwards (the test is not vacuous).
    assert not np.isclose(base.final_value, alt.final_value)


def test_data_after_evaluation_end_is_ignored():
    prices = random_walk(n=500, seed=5)
    end = prices.index[399]
    strategy = create_strategy("bollinger_reversion")
    full = run_backtest(prices, strategy, BacktestConfig(), eval_end=end)
    truncated = run_backtest(prices.loc[:end], strategy, BacktestConfig())
    pd.testing.assert_frame_equal(full.daily, truncated.daily)
    pd.testing.assert_frame_equal(full.ledger, truncated.ledger)


def test_evaluation_start_anchors_on_previous_session():
    prices = random_walk(n=300, seed=8)
    start = prices.index[100]
    result = run_backtest(prices, BuyAndHold(), BacktestConfig(), eval_start=start)
    assert result.daily.index[0] == prices.index[99]  # anchor
    assert result.daily.index[1] == start
    assert result.equity.iloc[0] == 10_000
    assert result.ledger["execution_date"].iloc[0] == start
    assert result.meta["evaluation_start"] == start


def test_indicator_warmup_uses_history_before_evaluation_start():
    prices = random_walk(n=300, seed=9)
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    with_history = run_backtest(prices, strategy, BacktestConfig(), eval_start=prices.index[100])
    # The long average was already valid at the anchor, so no session is lost to warm-up.
    assert with_history.first_signal_date <= with_history.daily.index[0]
    assert with_history.signals["target"].notna().all()


# --------------------------------------------------------------------------- #
# Reconciliation and integrity
# --------------------------------------------------------------------------- #


def test_reconciliation_passes_for_all_strategies(walk):
    for key in ("buy_and_hold", "ma_crossover", "ts_momentum", "rsi_reversion", "bollinger_reversion"):
        params = {"lookback": 63} if key == "ts_momentum" else {}
        result = run_backtest(walk, create_strategy(key, params), BacktestConfig())
        assert result.is_valid, (key, result.reconciliation_issues)


def test_reconciliation_detects_corrupted_equity(walk):
    result = run_backtest(walk, create_strategy("ma_crossover"), BacktestConfig())
    daily = result.daily.copy()
    daily.iloc[100, daily.columns.get_loc("equity")] += 1.0
    issues = reconcile(daily, result.ledger, result.config)
    assert any("Equity differs" in i for i in issues)


def test_reconciliation_detects_lookahead_fill(walk):
    result = run_backtest(walk, create_strategy("ma_crossover"), BacktestConfig())
    ledger = result.ledger.copy()
    ledger.loc[0, "execution_date"] = ledger.loc[0, "signal_date"]
    issues = reconcile(result.daily, ledger, result.config)
    assert any("session after its signal" in i or "look-ahead" in i for i in issues)


def test_costs_only_reduce_value(walk):
    strategy = create_strategy("bollinger_reversion")
    net = run_backtest(walk, strategy, BacktestConfig())
    gross = run_backtest(walk, strategy, BacktestConfig().without_costs())
    assert len(net.ledger) == len(gross.ledger)  # costs do not change signal timing
    assert gross.final_value > net.final_value
    assert (net.ledger["fee"] > 0).all() and (gross.ledger["fee"] == 0).all()


def test_backtest_is_deterministic(walk):
    strategy = create_strategy("rsi_reversion")
    a = run_backtest(walk, strategy, BacktestConfig())
    b = run_backtest(walk, strategy, BacktestConfig())
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.ledger, b.ledger)


def test_simulate_rejects_bad_inputs():
    prices = make_prices(closes=[10, 11, 12])
    with pytest.raises(ValueError):
        simulate(prices, pd.Series([1.0, 1.0]), NO_COSTS)
    broken = prices.copy()
    broken.iloc[1, broken.columns.get_loc("open")] = np.nan
    with pytest.raises(ValueError):
        simulate(broken, _targets(broken, [1, 1, 1]), NO_COSTS)


def test_never_invested_strategy_keeps_capital():
    prices = make_prices(closes=[10, 9, 8, 7, 8])
    daily, ledger, _, _ = simulate(prices, _targets(prices, [0] * 5), BacktestConfig())
    assert ledger.empty
    assert (daily["equity"] == 10_000).all()
    assert math.isclose(daily["exposure"].sum(), 0.0)


# --------------------------------------------------------------------------- #
# Test double
# --------------------------------------------------------------------------- #


class _FixedTargets(BuyAndHold):
    """A strategy that emits a predetermined target sequence (test only)."""

    key = "fixed_targets"
    name = "Fixed targets"

    def __init__(self, values):
        super().__init__()
        self._values = np.asarray(values, dtype="float64")

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"target": self._values[: len(prices)]}, index=prices.index)
