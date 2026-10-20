"""The engine tools the agent calls: validated data in, provenance-tagged JSON out (synthetic data only)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.agent.catalog import engine_tools
from quantlab.agent.guardrails import is_execution_tool
from quantlab.agent.tools import Mode, ToolContext, ToolRegistry
from quantlab.data import AdjustmentMode, CSVProvider, SyntheticProvider, load_market_data
from quantlab.simulation import SIM_END
from tests.helpers import random_walk


@pytest.fixture(scope="module")
def ctx():
    ds = load_market_data("SIM-TREND-1", "2014-01-01", "2024-12-31", SyntheticProvider(),
                          adjustment=AdjustmentMode.PRICE_ONLY, today=SIM_END.date())
    return ToolContext(load_prices=lambda t: ds)


REG = ToolRegistry(engine_tools())
BT = {"ticker": "SIM-TREND-1", "strategy": "ma_crossover", "params_json": None, "start": None, "end": None,
      "fee_rate": None, "slippage_rate": None}


def test_price_summary_carries_provenance_and_labels_simulated_data(ctx):
    out = REG.execute("get_price_summary", {"ticker": "sim-trend-1"}, ctx, Mode.RESEARCH)
    assert out.ok, out.error
    r = out.result
    assert r["ticker"] == "SIM-TREND-1" and r["last_session"] == "2024-12-31" and r["last_close"] > 0
    prov = r["provenance"]
    assert prov["timeliness"] == "simulated" and prov["period_end"] == "2024-12-31" and prov["retrieved_at"]
    assert "not market" in prov["source"]


def test_data_that_fails_validation_is_refused_rather_than_summarised():
    bad = random_walk(30)
    bad.iloc[5:, bad.columns.get_loc("close")] = -1.0   # most closes impossible
    content = bad.rename(columns=str.title).to_csv(index_label="Date").encode()
    ds = load_market_data("BAD", "2018-01-01", "2018-03-31", CSVProvider(content), today=pd.Timestamp("2024-01-01").date())
    out = REG.execute("get_price_summary", {"ticker": "BAD"}, ToolContext(load_prices=lambda t: ds), Mode.RESEARCH)
    assert not out.ok and "failed validation" in out.error and out.result is None


def test_backtests_are_counted_as_trials_and_deflated(ctx):
    ctx.extras.clear()
    first = REG.execute("run_backtest", BT, ctx, Mode.BACKTEST)
    assert first.ok, first.error
    sel = first.result["selection"]
    assert sel["configurations_tried_this_run"] == 1 and sel["deflated_sharpe"] is None
    second = REG.execute("run_backtest", {**BT, "params_json": '{"short_window": 10, "long_window": 100}'}, ctx,
                         Mode.BACKTEST)
    assert second.ok and second.result["selection"]["configurations_tried_this_run"] == 2
    assert second.result["selection"]["deflated_sharpe"] is not None
    assert any("possibly tuned" in w for w in second.result["warnings"])
    assert first.result["accounting_reconciled"] and "sharpe" in first.result["metrics"]
    assert first.result["costs"]["execution"].startswith("signal at close")


@pytest.mark.parametrize(("params", "message"), [("{not json", "not valid JSON"), ('[1, 2]', "JSON object"),
                                                  ('{"short_window": 80, "long_window": 50}', "shorter")])
def test_invalid_parameters_come_back_as_errors(ctx, params, message):
    out = REG.execute("run_backtest", {**BT, "params_json": params}, ctx, Mode.BACKTEST)
    assert not out.ok and message in out.error


def test_walk_forward_reports_out_of_sample_results_and_parameter_stability(ctx):
    out = REG.execute("run_walk_forward", {"ticker": "SIM-TREND-1", "strategy": "ma_crossover",
                                           "grid_json": '{"short_window": [10, 20], "long_window": [50, 100]}',
                                           "train_sessions": None, "test_sessions": None}, ctx, Mode.BACKTEST)
    assert out.ok, out.error
    r = out.result
    assert r["folds"] > 5 and r["combinations"] == 4 and r["most_selected_parameters"]
    assert r["mean_in_sample_sharpe"] is not None and "sharpe" in r["out_of_sample_metrics"]
    none = REG.execute("run_walk_forward", {"ticker": "SIM-TREND-1", "strategy": "buy_and_hold", "grid_json": None,
                                            "train_sessions": None, "test_sessions": None}, ctx, Mode.BACKTEST)
    assert not none.ok and "no parameters" in none.error


def test_engine_tools_are_research_only():
    assert not any(is_execution_tool(t.name, t.description, t.simulated) for t in engine_tools())
    assert all(Mode.PAPER_TRADE not in t.modes for t in engine_tools() if t.name.startswith("run_"))
