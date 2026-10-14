"""Randomised stress tests of the engine's accounting invariants.

Thousands of seeded random scenarios — random price paths with gaps, random
target sequences (binary and fractional, with long persistent stretches),
random costs, capital, share sizing, rebalance bands and cash interest — must
all satisfy every accounting identity. Seeds make failures reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import build_round_trips, reconciliation_report, simulate
from quantlab.config import BacktestConfig
from tests.helpers import make_prices


def _scenario(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 260))
    vol = rng.uniform(0.002, 0.05)
    closes = 50 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    opens = closes * np.exp(rng.normal(0, vol / 2, n))  # overnight gaps
    prices = make_prices(closes, opens)
    if rng.random() < 0.5:
        raw = rng.choice([0.0, 1.0], size=n)
    else:
        raw = np.round(rng.uniform(0, 1, n), 2)
    persist = rng.random(n) < rng.uniform(0, 0.9)  # stretches where the target does not change
    targets = pd.Series(np.where(persist, np.nan, raw), index=prices.index).ffill()
    targets.iloc[: int(rng.integers(0, 5))] = np.nan  # warm-up
    cfg = BacktestConfig(
        initial_capital=float(rng.choice([500.0, 10_000.0, 2_500_000.0])),
        fee_rate=float(rng.choice([0.0, 0.001, 0.01])),
        slippage_rate=float(rng.choice([0.0, 0.0005, 0.005])),
        fractional_shares=bool(rng.random() < 0.7),
        rebalance_threshold=float(rng.choice([0.0, 0.02, 0.05, 0.2])),
        cash_interest_rate=float(rng.choice([0.0, 0.04])),
    )
    return prices, targets, cfg


@pytest.mark.parametrize("block", range(10))
def test_accounting_invariants_hold_in_random_scenarios(block):
    for seed in range(block * 150, block * 150 + 150):
        prices, targets, cfg = _scenario(seed)
        daily, ledger, rejected, _ = simulate(prices, targets, cfg)
        failed = [r for r in reconciliation_report(daily, ledger, cfg) if not r["passed"]]
        assert not failed, (seed, failed)
        assert (daily["cash"] >= -1e-6 * cfg.initial_capital).all(), seed
        assert (daily["shares"] >= 0).all(), seed
        assert (daily["exposure"] <= 1 + 1e-9).all(), seed
        if not cfg.fractional_shares and len(ledger):
            assert np.allclose(ledger["quantity"], np.round(ledger["quantity"])), seed
        trades = build_round_trips(ledger, daily)
        if len(trades):
            assert (trades["holding_sessions"] >= 0).all(), seed
            closed = trades[trades["status"] == "closed"]
            assert np.isclose(closed["pnl"].sum() + trades.loc[trades["status"] != "closed", "pnl"].sum(),
                              trades["exit_value"].sum() - trades["entry_cost"].sum()), seed


def test_costs_never_increase_final_value():
    for seed in range(200):
        prices, targets, cfg = _scenario(seed)
        net, _, _, _ = simulate(prices, targets, cfg)
        gross, _, _, _ = simulate(prices, targets, cfg.without_costs())
        if cfg.fractional_shares and cfg.rebalance_threshold == 0.0:
            # With identical trade timing, costs can only lower the final value.
            assert net["equity"].iloc[-1] <= gross["equity"].iloc[-1] + 1e-6, seed
