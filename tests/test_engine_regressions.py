"""Regression tests for engine defects found in an adversarial audit.

Each case is the minimal reproduction that exposed the defect; all data is synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import backtest_from_signals, build_round_trips, run_backtest, simulate
from quantlab.config import BacktestConfig
from quantlab.portfolio import Portfolio
from quantlab.execution import Fill, Side
from quantlab.strategies import BuyAndHold, create_strategy
from tests.helpers import make_prices, random_walk

NO_COSTS = BacktestConfig(fee_rate=0.0, slippage_rate=0.0)


def _t(prices, values):
    return pd.Series(np.asarray(values, dtype="float64"), index=prices.index)


def test_rebalancing_does_not_dilute_round_trip_returns():
    """Capital recycled by rebalancing was counted again on every top-up, so a trade that
    doubled the money showed +16.7%."""
    prices = make_prices([100, 100, 200] + [200] * 12)
    targets = [1, 1, 1] + [0.5, 1.0] * 5 + [0, 0]
    daily, ledger, _, _ = simulate(prices, _t(prices, targets), NO_COSTS.model_copy(update={"rebalance_threshold": 0}))
    trades = build_round_trips(ledger, daily)
    assert len(trades) == 1 and trades.iloc[0]["fills"] == 12
    trade = trades.iloc[0]
    assert trade["capital_committed"] == pytest.approx(10_000)
    assert trade["pnl"] == pytest.approx(10_000)
    assert trade["net_return"] == pytest.approx(1.0)
    assert trade["gross_return"] == pytest.approx(1.0)  # 100 -> 200
    assert trade["quantity"] == pytest.approx(100)  # peak shares held, not the sum of buys


def test_rejected_whole_share_entry_retries_once_affordable():
    """A rejected entry used to block a partial-exposure strategy until its target hit exactly 0."""
    prices = make_prices([3000.0] * 10)
    config = NO_COSTS.model_copy(update={"initial_capital": 10_000, "fractional_shares": False})
    targets = [0.2, 0.2, 0.2, 0.5, 0.5, 1, 1, 1, 1, 1]
    daily, ledger, rejected, _ = simulate(prices, _t(prices, targets), config)
    assert len(rejected) == 1 and rejected.iloc[0]["reason"].startswith("Target value $2,000.00")
    assert (ledger["side"] == "BUY").any()
    assert daily["shares"].iloc[-1] == 3


def test_falling_price_unblocks_a_rejected_entry():
    prices = make_prices(np.linspace(1200, 120, 30))
    config = NO_COSTS.model_copy(update={"initial_capital": 1_000, "fractional_shares": False})
    daily, ledger, rejected, _ = simulate(prices, _t(prices, np.full(30, 0.6)), config)
    assert len(rejected) >= 1 and len(ledger) > 0


def test_whole_share_top_up_on_a_persistent_signal_is_labelled_a_rebalance():
    prices = make_prices([900, 900, 850, 700, 600, 480, 470, 460, 450, 440])
    config = NO_COSTS.model_copy(update={"initial_capital": 5_000, "fractional_shares": False})
    _, ledger, _, _ = simulate(prices, _t(prices, np.ones(10)), config)
    assert len(ledger) == 2 and ledger.iloc[1]["reason"].startswith("Rebalance")
    _, ledger_frac, _, _ = simulate(prices, _t(prices, np.ones(10)), NO_COSTS.model_copy(update={"initial_capital": 5_000}))
    assert len(ledger_frac) == 1  # fractional shares: a persistent 0/1 signal never re-trades


def test_share_count_reconciliation_scales_with_position_size():
    """Huge share counts at tiny prices produced false 'Accounting FAILED' banners."""
    walk = random_walk(200, seed=1)
    prices = walk.assign(**{c: walk[c] * 0.0002 for c in ("open", "high", "low", "close")})
    targets = np.round(np.random.default_rng(0).uniform(0.05, 1, 200), 2)
    targets[-3:] = 0
    signals = pd.DataFrame({"target": targets}, index=prices.index)
    result = backtest_from_signals(prices, signals, BacktestConfig(initial_capital=1e7, rebalance_threshold=0.01),
                                   "scripted", "Scripted", {})
    assert result.is_valid, result.reconciliation_issues


def test_small_genuine_cash_balances_are_not_snapped_to_zero():
    config = NO_COSTS.model_copy(update={"initial_capital": 1e12, "fractional_shares": False})
    result = run_backtest(make_prices([700.0] * 5), BuyAndHold(), config)
    assert result.daily["cash"].iloc[-1] == pytest.approx(1e12 - 700 * (1e12 // 700))
    book = Portfolio(cash=1e-7)
    book.apply(Fill(side=Side.BUY, quantity=5e-10, reference_price=200.0, execution_price=200.0,
                    gross_notional=1e-7, fee=0.0, slippage_cost=0.0))
    assert book.shares == pytest.approx(5e-10)


def test_rebalance_reason_describes_the_open_after_an_overnight_gap():
    prices = make_prices(closes=[100, 100, 70, 130, 130], opens=[100, 100, 100, 130, 130])
    daily, ledger, _, _ = simulate(prices, _t(prices, [0.5] * 5), NO_COSTS)
    rebalance = ledger[ledger["reason"].str.startswith("Rebalance")]
    assert len(rebalance) == 1
    fill = rebalance.iloc[0]
    assert fill["side"] == "SELL" and "at the open" in fill["reason"]


@pytest.mark.parametrize("case", ["empty", "end_before_data", "reversed", "duplicates", "short_signals"])
def test_degenerate_inputs_raise_clear_value_errors(case):
    prices = random_walk(300, seed=3)
    strategy = create_strategy("ma_crossover")
    kwargs = {}
    if case == "empty":
        prices = prices.iloc[:0]
    elif case == "end_before_data":
        kwargs["eval_end"] = prices.index[0] - pd.Timedelta(days=30)
    elif case == "reversed":
        prices = prices.iloc[::-1]
    elif case == "duplicates":
        prices = pd.concat([prices, prices.iloc[[10]]]).sort_index()
    elif case == "short_signals":
        kwargs["signals"] = strategy.generate_signals(prices).iloc[100:]
    with pytest.raises(ValueError):
        run_backtest(prices, strategy, NO_COSTS, **kwargs)


def test_whole_share_sizing_terminates_beyond_float_integer_precision():
    """Above 2**53 shares ``q - 1.0 == q`` in float64, so stepping the quantity down by one share never
    made progress: a micro-priced CSV with large capital hung the app at 100% CPU."""
    import threading

    from quantlab.execution import ExecutionModel

    price, cash = 6.630037871702824e-08, 1e9
    result: dict = {}
    model = ExecutionModel(fee_rate=0.001, slippage_rate=0.0005, fractional_shares=False)
    worker = threading.Thread(target=lambda: result.update(q=model.buy_quantity(cash, price)), daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "buy_quantity did not terminate"
    q = result["q"]
    assert q > 2**53 and q * model.execution_price(Side.BUY, price) * (1 + model.fee_rate) <= cash

    run = run_backtest(make_prices([price] * 5), BuyAndHold(),
                       BacktestConfig(initial_capital=cash, fractional_shares=False))
    assert run.is_valid, run.reconciliation_issues
