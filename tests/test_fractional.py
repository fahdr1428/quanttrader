"""Fractional target weights, rebalance bands, position episodes and cash interest.

Hand-calculated expectations; zero costs unless stated so the arithmetic is easy
to follow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import build_round_trips, reconcile, simulate
from quantlab.config import BacktestConfig
from tests.helpers import make_prices, random_walk

NO_COSTS = BacktestConfig(fee_rate=0.0, slippage_rate=0.0)


def _t(prices, values):
    return pd.Series(np.asarray(values, dtype="float64"), index=prices.index)


def test_half_position_from_cash():
    prices = make_prices(closes=[100, 100, 100], opens=[100, 100, 100])
    daily, ledger, _, _ = simulate(prices, _t(prices, [0.5, 0.5, 0.5]), NO_COSTS)
    assert len(ledger) == 1
    assert ledger["quantity"].iloc[0] == pytest.approx(50)  # 0.5 x 10,000 / 100
    assert daily["cash"].iloc[-1] == pytest.approx(5_000)
    assert daily["exposure"].iloc[-1] == pytest.approx(0.5)


def test_drift_beyond_band_triggers_rebalance_to_target():
    # Price doubles at session 2's close: weight 10,000 / 15,000 = 66.7% > 50% + 5% band.
    prices = make_prices(closes=[100, 100, 200, 200, 200], opens=[100, 100, 100, 200, 200])
    daily, ledger, _, _ = simulate(prices, _t(prices, [0.5] * 5), NO_COSTS)
    assert list(ledger["side"]) == ["BUY", "SELL"]
    sell = ledger.iloc[1]
    assert sell["signal_date"] == prices.index[2] and sell["execution_date"] == prices.index[3]
    # At session 3's open: holdings 50 x 200 = 10,000, equity 15,000, target 7,500 -> sell 12.5 shares.
    assert sell["quantity"] == pytest.approx(12.5)
    assert daily["exposure"].iloc[3] == pytest.approx(0.5)
    assert "Rebalance" in sell["reason"]


def test_drift_within_band_does_not_trade():
    prices = make_prices(closes=[100, 100, 104, 104], opens=[100, 100, 100, 104])
    _, ledger, _, _ = simulate(prices, _t(prices, [0.5] * 4), NO_COSTS)
    # Weight after the move: 5,200 / 10,200 = 51.0%, inside the 5-point band.
    assert len(ledger) == 1


def test_zero_band_rebalances_on_any_drift():
    prices = make_prices(closes=[100, 100, 104, 104], opens=[100, 100, 100, 104])
    cfg = NO_COSTS.model_copy(update={"rebalance_threshold": 0.0})
    _, ledger, _, _ = simulate(prices, _t(prices, [0.5] * 4), cfg)
    assert len(ledger) == 2


def test_increase_to_full_spends_all_cash_and_partial_exit():
    prices = make_prices(closes=[100] * 6, opens=[100] * 6)
    targets = [0.4, 1.0, 1.0, 0.3, 0.3, 0.3]
    daily, ledger, _, _ = simulate(prices, _t(prices, targets), NO_COSTS)
    assert list(ledger["side"]) == ["BUY", "BUY", "SELL"]
    assert daily["cash"].iloc[2] == pytest.approx(0.0, abs=1e-9)
    assert daily["exposure"].iloc[4] == pytest.approx(0.3)
    assert ledger["target_weight"].tolist() == [0.4, 1.0, 0.3]


def test_full_exit_sells_everything_regardless_of_band():
    prices = make_prices(closes=[100] * 4, opens=[100] * 4)
    daily, ledger, _, _ = simulate(prices, _t(prices, [0.02, 0.02, 0.0, 0.0]), NO_COSTS)
    assert list(ledger["side"]) == ["BUY", "SELL"]
    assert daily["shares"].iloc[-1] == 0


def test_targets_outside_unit_interval_are_rejected():
    prices = make_prices(closes=[100] * 3)
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate(prices, _t(prices, [1.5, 1.0, 1.0]), NO_COSTS)
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate(prices, _t(prices, [-0.2, 0.0, 0.0]), NO_COSTS)


def test_whole_shares_skip_infeasible_rebalances_without_rejections():
    prices = make_prices(closes=[300, 300, 310, 320, 330], opens=[300] * 5)
    cfg = NO_COSTS.model_copy(update={"fractional_shares": False, "initial_capital": 1_000.0})
    _, ledger, rejected, _ = simulate(prices, _t(prices, [1.0] * 5), cfg)
    assert len(ledger) == 1 and ledger["quantity"].iloc[0] == 3
    assert rejected.empty  # residual $100 cannot buy a share; no order spam


def test_episode_round_trip_aggregates_partial_fills():
    prices = make_prices(closes=[100, 100, 100, 100, 110, 110], opens=[100, 100, 100, 100, 110, 110])
    targets = [0.5, 1.0, 1.0, 0.0, 0.0, 0.0]
    daily, ledger, _, _ = simulate(prices, _t(prices, targets), NO_COSTS)
    trades = build_round_trips(ledger, daily)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["fills"] == 3 and trade["status"] == "closed"
    # Bought 50 then 50 shares at 100 (entry cost 10,000); sold 100 at 110 on session 4's open.
    assert trade["entry_cost"] == pytest.approx(10_000)
    assert trade["exit_value"] == pytest.approx(11_000)
    assert trade["net_return"] == pytest.approx(0.10)


def test_cash_interest_compounds_per_session():
    prices = make_prices(closes=[100] * 253)
    cfg = NO_COSTS.model_copy(update={"cash_interest_rate": 0.05})
    daily, ledger, _, _ = simulate(prices, _t(prices, [0.0] * 253), cfg)
    assert daily["equity"].iloc[-1] == pytest.approx(10_500, rel=1e-10)  # 252 accrual periods
    assert reconcile(daily, ledger, cfg) == []


def test_interest_only_on_uninvested_cash():
    prices = make_prices(closes=[100] * 253)
    cfg = NO_COSTS.model_copy(update={"cash_interest_rate": 0.05})
    daily, _, _, _ = simulate(prices, _t(prices, [0.5] * 253), cfg)
    # Half invested from session 1: interest accrues on ~5,000 of cash, not 10,000.
    assert 10_240 < daily["equity"].iloc[-1] < 10_260


@pytest.mark.parametrize("fractional", [True, False])
def test_fractional_targets_reconcile_with_costs_and_interest(fractional):
    prices = random_walk(n=500, seed=31)
    rng = np.random.default_rng(5)
    targets = np.clip(np.round(rng.uniform(-0.2, 1.2, len(prices)), 2), 0, 1)
    targets[rng.uniform(size=len(prices)) < 0.6] = np.nan  # persistent stretches of 'no change'
    targets = pd.Series(targets, index=prices.index).ffill().fillna(0.0)
    cfg = BacktestConfig(cash_interest_rate=0.03, fractional_shares=fractional, rebalance_threshold=0.1)
    daily, ledger, _, _ = simulate(prices, targets, cfg)
    assert reconcile(daily, ledger, cfg) == []
    assert (daily["cash"] >= -1e-6).all() and (daily["exposure"] <= 1 + 1e-9).all()
