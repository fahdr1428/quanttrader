"""Exports are consistent with results, and manifests reproduce runs exactly."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from quantlab.analytics import compare_strategies, cost_sensitivity, parameter_sensitivity, run_strategy
from quantlab.backtest import run_buy_and_hold
from quantlab.config import BacktestConfig
from quantlab.data import fingerprint_prices
from quantlab.reporting import (
    build_manifest,
    equity_curve_csv,
    ledger_csv,
    manifest_to_json,
    metrics_csv,
    prices_csv,
    reproduce_from_manifest,
    trades_csv,
)
from quantlab.strategies import create_strategy
from quantlab.validation import clean_price_frame


@pytest.fixture
def run(walk):
    cfg = BacktestConfig()
    start = walk.index[200]
    bench = run_buy_and_hold(walk, cfg, start)
    return run_strategy(walk, create_strategy("ma_crossover"), cfg, start, benchmark=bench), bench


def _manifest(walk, run_obj, prices=None):
    prices = walk if prices is None else prices
    return build_manifest(
        ticker="SYN", benchmark="SYN", eval_start=walk.index[200], eval_end=walk.index[-1],
        data_start=walk.index[0], provider_key="csv", source_name="synthetic", adjustment="as_provided",
        fingerprint=fingerprint_prices(prices), first_observation=walk.index[0],
        last_observation=walk.index[-1], strategy_key="ma_crossover",
        params=run_obj.strategy.params, config=run_obj.net.config, report=run_obj.report,
    )


def test_manifest_reproduces_exactly_from_exported_prices(walk, run):
    run_obj, _ = run
    manifest = json.loads(manifest_to_json(_manifest(walk, run_obj)))
    reloaded = pd.read_csv(io.StringIO(prices_csv(walk)), index_col=0, parse_dates=[0], float_precision="round_trip")
    clean, _ = clean_price_frame(reloaded)
    outcome = reproduce_from_manifest(manifest, clean, fingerprint_prices(clean))
    assert outcome["fingerprint_match"] and outcome["all_metrics_match"] and outcome["accounting_valid"]


def test_revised_data_is_detected(walk, run):
    run_obj, _ = run
    manifest = _manifest(walk, run_obj)
    revised = walk.copy()
    revised.iloc[300:, :4] *= 1.01
    outcome = reproduce_from_manifest(manifest, revised, fingerprint_prices(revised))
    assert not outcome["fingerprint_match"]


def test_equity_csv_matches_result(run):
    run_obj, bench = run
    frame = pd.read_csv(io.StringIO(equity_curve_csv(run_obj.net, bench, run_obj.gross)), index_col=0, parse_dates=[0])
    pd.testing.assert_series_equal(
        frame["equity"], run_obj.net.equity, check_names=False, check_freq=False, check_index_type=False, rtol=1e-9
    )
    assert {"benchmark_equity", "gross_equity", "drawdown", "daily_return"} <= set(frame.columns)


def test_ledger_trades_and_metrics_csv_are_complete(run):
    run_obj, _ = run
    ledger = pd.read_csv(io.StringIO(ledger_csv(run_obj.net)))
    assert len(ledger) == len(run_obj.net.ledger)
    assert ledger["fee"].sum() == pytest.approx(run_obj.net.ledger["fee"].sum())
    trades = pd.read_csv(io.StringIO(trades_csv(run_obj.net)))
    assert len(trades) == len(run_obj.net.trades)
    metrics = pd.read_csv(io.StringIO(metrics_csv(run_obj.report)))
    assert set(metrics["key"]) == set(run_obj.report.metrics)


def test_manifest_is_strict_json(walk, run):
    run_obj, _ = run
    text = manifest_to_json(_manifest(walk, run_obj))
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


def test_comparison_requires_identical_sessions_and_costs(walk):
    cfg = BacktestConfig()
    a = run_strategy(walk, create_strategy("buy_and_hold"), cfg, walk.index[100])
    b = run_strategy(walk, create_strategy("rsi_reversion"), cfg, walk.index[100])
    table = compare_strategies([a, b])
    assert len(table) == 2 and table["Accounting reconciled"].all()
    c = run_strategy(walk, create_strategy("rsi_reversion"), cfg, walk.index[150])
    with pytest.raises(ValueError, match="identical sessions"):
        compare_strategies([a, c])
    d = run_strategy(walk, create_strategy("rsi_reversion"), cfg.model_copy(update={"fee_rate": 0.01}), walk.index[100])
    with pytest.raises(ValueError, match="same capital and cost"):
        compare_strategies([a, d])


def test_sensitivity_grid_records_invalid_combinations(walk):
    grid = {"short_window": [10, 50], "long_window": [30, 100]}
    frame = parameter_sensitivity(walk, "ma_crossover", {}, grid, BacktestConfig())
    assert len(frame) == 4
    invalid = frame[~frame["valid"]]
    assert list(zip(invalid["short_window"], invalid["long_window"])) == [(50, 30)]
    with pytest.raises(ValueError, match="limit"):
        parameter_sensitivity(walk, "ma_crossover", {}, {"short_window": list(range(2, 40)), "long_window": list(range(41, 50))}, BacktestConfig())


def test_cost_sensitivity_is_monotonic(walk):
    frame = cost_sensitivity(walk, create_strategy("bollinger_reversion"), BacktestConfig(), multipliers=(0, 1, 5))
    assert frame["total_return"].is_monotonic_decreasing
    assert frame.loc[0, "total_costs"] == 0
