"""The research engine exposed as agent tools.

Each handler is a thin adapter: it loads validated data through the context,
calls the tested engine, and returns plain JSON with provenance attached. No
handler computes anything the engine does not already compute and test, and
no handler accepts the user's risk limits, holdings or account as input — those
come from the context, so the model cannot change them.

Data-quality gating happens here: a dataset that failed validation raises a
:class:`ToolError`, which reaches the model as an error result, never as numbers.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from quantlab.agent.tools import Mode, Tool, ToolContext, ToolError, obj, optional
from quantlab.provenance import Provenance

ALL_MODES = frozenset(Mode)
ANALYSIS_MODES = frozenset({Mode.RESEARCH, Mode.BACKTEST, Mode.SCAN, Mode.ALERTS, Mode.PAPER_TRADE,
                            Mode.PORTFOLIO_REVIEW})

def dataset_provenance(ds: Any) -> Provenance:
    """Provenance of a validated price dataset (one definition, shared with the technicals module)."""
    from quantlab.technicals import dataset_provenance as _provenance

    prov = _provenance(ds)
    return prov if not ds.notes else replace(prov, warnings=prov.warnings + tuple(ds.notes))


def load_validated(ctx: ToolContext, ticker: str) -> Any:
    """Load a ticker through the context; refuse data that failed validation."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ToolError("A ticker is required.")
    ds = ctx.load_prices(ticker)
    if not ds.is_usable:
        errors = "; ".join(i.message for i in ds.quality.errors) or "no usable rows"
        raise ToolError(f"Price data for {ticker} failed validation ({errors}). Analysis that depends on it is "
                        "withheld.")
    return ds


def _window_return(close: pd.Series, n: int) -> float | None:
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n else None


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #


def price_summary(ctx: ToolContext, inputs: dict[str, Any]) -> dict[str, Any]:
    from quantlab.metrics import max_drawdown

    ds = load_validated(ctx, inputs["ticker"])
    prices = ds.prices
    close = prices["close"]
    rets = close.pct_change().dropna()
    last_year = close.iloc[-253:]
    dollar_volume = (prices["close"] * prices["volume"]).iloc[-20:] if "volume" in prices else None
    return {
        "ticker": ds.ticker,
        "last_close": float(close.iloc[-1]),
        "last_session": ds.last_date.date().isoformat(),
        "sessions": int(len(prices)),
        "returns": {f"{n}_sessions": _window_return(close, n) for n in (21, 63, 126, 252)},
        "realised_vol_63_annualised": float(rets.iloc[-63:].std(ddof=1) * np.sqrt(252)) if len(rets) >= 63 else None,
        "max_drawdown_252": float(max_drawdown(last_year)) if len(last_year) > 1 else None,
        "avg_dollar_volume_20": float(dollar_volume.mean()) if dollar_volume is not None and dollar_volume.notna().any()
        else None,
        "data_quality": {"usable": True, "warnings": [i.message for i in ds.quality.warnings],
                         "fingerprint": ds.fingerprint[:16]},
        "provenance": dataset_provenance(ds).to_dict(),
        "definitions": {"returns": "close_t / close_(t-n) - 1 over n sessions",
                        "realised_vol_63_annualised": "std of daily returns over 63 sessions × sqrt(252)",
                        "max_drawdown_252": "largest peak-to-trough fall of the close over the last 252 sessions"},
    }


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #

_METRIC_KEYS = ("total_return", "cagr", "volatility", "sharpe", "sharpe_ci_low", "sharpe_ci_high", "psr",
                "sortino", "max_drawdown", "calmar", "time_invested", "turnover", "round_trips", "win_rate",
                "profit_factor", "avg_win", "avg_loss", "total_fees", "total_slippage", "excess_return",
                "benchmark_total_return", "benchmark_cagr", "benchmark_sharpe", "benchmark_max_drawdown")


def _parse_params(strategy_key: str, raw: str | None) -> dict[str, Any]:
    from quantlab.strategies import STRATEGIES, StrategyParameterError

    try:
        params = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ToolError(f"params_json is not valid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise ToolError("params_json must be a JSON object of parameter names to values.")
    try:
        return STRATEGIES[strategy_key].validate_params({**STRATEGIES[strategy_key].defaults(), **params})
    except StrategyParameterError as exc:
        raise ToolError(str(exc)) from exc


def _config(inputs: dict[str, Any]) -> Any:
    from quantlab.config import BacktestConfig

    overrides = {k: inputs[k] for k in ("fee_rate", "slippage_rate") if inputs.get(k) is not None}
    try:
        return BacktestConfig(**overrides)
    except ValueError as exc:
        raise ToolError(f"Invalid cost settings: {exc}") from exc


def _metrics(report: Any) -> dict[str, Any]:
    out = {}
    for key in _METRIC_KEYS:
        metric = report.get(key)
        if metric is not None:
            out[key] = {"value": metric.value, "reliable": metric.reliable, **({"note": metric.note} if metric.note else {})}
    return out


def _record_trial(ctx: ToolContext, returns: pd.Series, config: Any) -> dict[str, Any]:
    """Every backtest the agent runs is a trial; report the deflated Sharpe across them."""
    from quantlab.research import deflated_sharpe_ratio, per_period_sharpe

    trials = ctx.extras.setdefault("trial_sharpes", [])
    sharpe = per_period_sharpe(returns, config.risk_free_rate, config.periods_per_year)
    if sharpe is not None:
        trials.append(sharpe)
    dsr = (deflated_sharpe_ratio(returns, trials, config.risk_free_rate, config.periods_per_year)
           if len(trials) >= 2 else None)
    return {"configurations_tried_this_run": len(trials), "deflated_sharpe": dsr,
            "note": "Each backtest in this run is a trial; the more configurations tried, the more the best one "
                    "overstates what to expect. Deflated Sharpe needs at least two trials."}


def backtest(ctx: ToolContext, inputs: dict[str, Any]) -> dict[str, Any]:
    from quantlab.analytics import run_strategy
    from quantlab.backtest import run_buy_and_hold
    from quantlab.strategies import STRATEGIES

    ds = load_validated(ctx, inputs["ticker"])
    key = inputs["strategy"]
    params = _parse_params(key, inputs.get("params_json"))
    config = _config(inputs)
    start = pd.Timestamp(inputs["start"]) if inputs.get("start") else ds.prices.index[min(252, len(ds.prices) - 2)]
    end = pd.Timestamp(inputs["end"]) if inputs.get("end") else None
    strategy = STRATEGIES[key](**params)
    try:
        bench = run_buy_and_hold(ds.prices, config, start, end)
        run = run_strategy(ds.prices, strategy, config, start, end, benchmark=bench)
    except ValueError as exc:
        raise ToolError(f"Backtest could not run: {exc}") from exc
    report = run.report
    trades = int(report.value("round_trips") or 0)
    warnings = ["Simulated, in-sample result: a profitable backtest is not evidence of future profitability."]
    if trades < 10:
        warnings.append(f"Only {trades} completed round trips: trade statistics are not meaningful.")
    if not strategy.uses_default_parameters():
        warnings.append("Parameters differ from the pre-specified defaults: treat as possibly tuned (in-sample).")
    if not run.net.is_valid:
        raise ToolError("The backtest's accounting did not reconcile; its results are withheld: "
                        + "; ".join(run.net.reconciliation_issues))
    return {
        "ticker": ds.ticker, "strategy": key, "label": strategy.label, "params": params,
        "evaluation": {"start": run.net.start_date.date().isoformat(), "end": run.net.end_date.date().isoformat(),
                       "sessions": int(len(run.net.daily) - 1)},
        "costs": {"fee_rate": config.fee_rate, "slippage_rate": config.slippage_rate,
                  "execution": "signal at close t, fill at the open of t+1"},
        "metrics": _metrics(report),
        "accounting_reconciled": True,
        "selection": _record_trial(ctx, run.net.returns, config),
        "warnings": warnings,
        "provenance": dataset_provenance(ds).to_dict(),
    }


def walk_forward_test(ctx: ToolContext, inputs: dict[str, Any]) -> dict[str, Any]:
    from quantlab.config import BacktestConfig
    from quantlab.research import WalkForwardConfig, walk_forward
    from quantlab.metrics import compute_metrics
    from quantlab.strategies import STRATEGIES

    ds = load_validated(ctx, inputs["ticker"])
    key = inputs["strategy"]
    cls = STRATEGIES[key]
    if not cls.search_space:
        raise ToolError(f"{cls.name} has no parameters to select, so walk-forward selection does not apply.")
    try:
        grid = json.loads(inputs["grid_json"]) if inputs.get("grid_json") else {k: list(v) for k, v in
                                                                                 cls.search_space.items()}
        wf_config = WalkForwardConfig(train_sessions=inputs.get("train_sessions") or 504,
                                      test_sessions=inputs.get("test_sessions") or 126)
        result = walk_forward(ds.prices, key, grid, BacktestConfig(), wf_config)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ToolError(f"Walk-forward could not run: {exc}") from exc
    report = compute_metrics(result.oos, benchmark=result.oos_benchmark)
    folds = result.fold_table
    chosen = folds["Selected parameters"].value_counts() if "Selected parameters" in folds else pd.Series(dtype=int)
    is_mean = float(folds["IS Sharpe"].astype(float).mean()) if "IS Sharpe" in folds else None
    oos_mean = float(folds["OOS Sharpe"].astype(float).mean()) if "OOS Sharpe" in folds else None
    return {
        "ticker": ds.ticker, "strategy": key, "grid": grid, "combinations": result.n_combinations,
        "folds": int(len(result.folds)), "in_sample_backtests": int(result.total_backtests),
        "train_sessions": wf_config.train_sessions, "test_sessions": wf_config.test_sessions,
        "out_of_sample_metrics": _metrics(report),
        "mean_in_sample_sharpe": is_mean, "mean_out_of_sample_sharpe": oos_mean,
        "most_selected_parameters": {str(k): int(v) for k, v in chosen.head(3).items()},
        "notes": list(result.notes) + ["Out-of-sample periods are stitched; parameters were chosen on prior data only."],
        "provenance": dataset_provenance(ds).to_dict(),
    }


def strategies_overview(ctx: ToolContext, inputs: dict[str, Any]) -> dict[str, Any]:
    from quantlab.strategies import STRATEGIES

    return {"strategies": [{"key": k, "name": c.name, "family": c.family, "summary": c.doc.summary,
                            "entry": c.doc.entry, "exit": c.doc.exit, "defaults": c.defaults(),
                            "search_space": {p: list(v) for p, v in c.search_space.items()},
                            "failure_modes": list(c.doc.failure_modes)} for k, c in STRATEGIES.items()]}


def _strategy_enum() -> list[str]:
    from quantlab.strategies import STRATEGIES

    return list(STRATEGIES)


def engine_tools() -> list[Tool]:
    date = optional({"type": "string", "description": "YYYY-MM-DD"})
    ticker = {"type": "string", "description": "Ticker symbol, e.g. SPY"}
    strategy = {"type": "string", "enum": _strategy_enum()}
    return [
        Tool("get_price_summary",
             "Latest validated end-of-day price data for one ticker: last close and session, trailing returns, "
             "realised volatility, recent drawdown, average dollar volume, data-quality warnings and provenance. "
             "Fails with an error if the data did not pass validation.",
             obj({"ticker": ticker}), price_summary, ANALYSIS_MODES),
        Tool("list_strategies",
             "The strategies the backtest engine supports, with rules, default parameters, parameter search space "
             "and known failure modes.",
             obj({}), strategies_overview, ALL_MODES),
        Tool("run_backtest",
             "Backtest one fully specified strategy on one ticker with realistic costs (signal at close, fill at next "
             "open) against buy & hold over the same sessions. Returns metrics with reliability flags, the number of "
             "round trips, and the deflated Sharpe across all backtests run so far in this session.",
             obj({"ticker": ticker, "strategy": strategy,
                  "params_json": optional({"type": "string", "description": "JSON object of non-default parameters"}),
                  "start": date, "end": date,
                  "fee_rate": optional({"type": "number", "minimum": 0, "maximum": 0.05}),
                  "slippage_rate": optional({"type": "number", "minimum": 0, "maximum": 0.05})}),
             backtest, frozenset({Mode.RESEARCH, Mode.BACKTEST})),
        Tool("run_walk_forward",
             "Chronological out-of-sample test: parameters chosen on each training window only, applied to the "
             "following unseen window. Reports out-of-sample metrics, in- vs out-of-sample Sharpe and parameter "
             "stability.",
             obj({"ticker": ticker, "strategy": strategy,
                  "grid_json": optional({"type": "string", "description": "JSON object: parameter -> list of values"}),
                  "train_sessions": optional({"type": "integer", "minimum": 60, "maximum": 2520}),
                  "test_sessions": optional({"type": "integer", "minimum": 20, "maximum": 1260})}),
             walk_forward_test, frozenset({Mode.RESEARCH, Mode.BACKTEST})),
    ]
