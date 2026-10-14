"""Execution prices, costs, sizing and portfolio constraints."""

from __future__ import annotations

import pytest

from quantlab.execution import ExecutionModel, Side
from quantlab.portfolio import AccountingError, Portfolio


def test_slippage_is_adverse_for_buys_and_sells():
    model = ExecutionModel(fee_rate=0.0, slippage_rate=0.001)
    assert model.execution_price(Side.BUY, 100.0) == pytest.approx(100.1)
    assert model.execution_price(Side.SELL, 100.0) == pytest.approx(99.9)


def test_fee_is_charged_on_actual_traded_notional():
    model = ExecutionModel(fee_rate=0.002, slippage_rate=0.001)
    fill = model.fill(Side.BUY, 10, 50.0)
    assert fill.execution_price == pytest.approx(50.05)
    assert fill.gross_notional == pytest.approx(500.5)
    assert fill.fee == pytest.approx(0.002 * 500.5)
    assert fill.slippage_cost == pytest.approx(10 * 0.05)
    assert fill.cash_flow == pytest.approx(-(500.5 + 1.001))

    sell = model.fill(Side.SELL, 10, 50.0)
    assert sell.execution_price == pytest.approx(49.95)
    assert sell.cash_flow == pytest.approx(499.5 - 0.002 * 499.5)


def test_fractional_buy_spends_exactly_the_available_cash():
    model = ExecutionModel(fee_rate=0.001, slippage_rate=0.0005)
    qty = model.buy_quantity(10_000.0, 123.45)
    fill = model.fill(Side.BUY, qty, 123.45)
    assert fill.gross_notional + fill.fee == pytest.approx(10_000.0, rel=1e-12)


def test_whole_share_buy_rounds_down_and_fits():
    model = ExecutionModel(fee_rate=0.001, slippage_rate=0.0, fractional_shares=False)
    qty = model.buy_quantity(1_000.0, 300.0)
    assert qty == 3
    fill = model.fill(Side.BUY, qty, 300.0)
    assert fill.gross_notional + fill.fee <= 1_000.0


def test_exact_whole_share_boundary_is_not_floored_away():
    model = ExecutionModel(fee_rate=0.0, slippage_rate=0.0, fractional_shares=False)
    assert model.buy_quantity(1_000.0, 10.0) == 100  # 99.9999 must not become 99


def test_invalid_reference_price_is_rejected():
    model = ExecutionModel(0.001, 0.0005)
    with pytest.raises(ValueError):
        model.execution_price(Side.BUY, 0.0)
    with pytest.raises(ValueError):
        model.execution_price(Side.BUY, float("nan"))


def test_portfolio_refuses_leverage_and_short_selling():
    model = ExecutionModel(0.0, 0.0)
    book = Portfolio(cash=1_000.0)
    with pytest.raises(AccountingError):
        book.apply(model.fill(Side.BUY, 11, 100.0))  # costs 1,100 > 1,000
    with pytest.raises(AccountingError):
        book.apply(model.fill(Side.SELL, 1, 100.0))  # nothing to sell


def test_portfolio_apply_updates_cash_and_shares():
    model = ExecutionModel(0.001, 0.0)
    book = Portfolio(cash=1_000.0)
    before_after = book.apply(model.fill(Side.BUY, 5, 100.0))
    assert before_after == pytest.approx((1_000.0, 1_000.0 - 500 - 0.5, 0.0, 5.0))
    book.apply(model.fill(Side.SELL, 5, 110.0))
    assert book.shares == 0
    assert book.cash == pytest.approx(499.5 + 550 - 0.55)


def test_rates_must_be_sensible():
    with pytest.raises(ValueError):
        ExecutionModel(fee_rate=-0.01, slippage_rate=0.0)
    with pytest.raises(ValueError):
        ExecutionModel(fee_rate=0.0, slippage_rate=1.5)
