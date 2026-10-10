"""Exports and reproducibility.

A *run manifest* is a JSON document containing every input that determines a
backtest: ticker, sample period, data source, price convention, data
fingerprint, strategy parameters, cost assumptions and methodology version.
Feeding a manifest to :func:`reproduce_from_manifest` (or to
``scripts/reproduce_backtest.py``) re-runs the backtest and reports whether the
data and the headline metrics match.

Reproducibility limit: free data providers revise history (late corrections,
and dividend-adjusted prices rescale every time a new dividend is paid). The
data fingerprint detects this; when it differs, identical code can legitimately
produce slightly different numbers. Save the price CSV alongside the manifest
for an exact, offline reproduction.
"""

from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from quantlab.analytics import run_strategy
from quantlab.backtest import BacktestResult, run_buy_and_hold
from quantlab.config import APP_VERSION, METHODOLOGY_VERSION, BacktestConfig
from quantlab.metrics import MetricReport
from quantlab.serialization import clean_json
from quantlab.strategies import create_strategy

MANIFEST_FORMAT = "quant-lab-pro/run-manifest"

HEADLINE_METRICS = ("final_value", "total_return", "cagr", "volatility", "sharpe", "max_drawdown", "fills")


def build_manifest(
    *,
    ticker: str,
    benchmark: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    data_start: pd.Timestamp,
    provider_key: str,
    source_name: str,
    adjustment: str,
    fingerprint: str,
    first_observation: pd.Timestamp,
    last_observation: pd.Timestamp,
    strategy_key: str,
    params: dict[str, Any],
    config: BacktestConfig,
    report: MetricReport | None = None,
    benchmark_fingerprint: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "format": MANIFEST_FORMAT,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "app_version": APP_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "data": {
            "ticker": ticker,
            "benchmark": benchmark,
            "provider": provider_key,
            "source": source_name,
            "price_adjustment": adjustment,
            "data_start": str(pd.Timestamp(data_start).date()),
            "first_observation": str(pd.Timestamp(first_observation).date()),
            "last_observation": str(pd.Timestamp(last_observation).date()),
            "fingerprint_sha256": fingerprint,
            "benchmark_fingerprint_sha256": benchmark_fingerprint,
        },
        "evaluation": {
            "start": str(pd.Timestamp(eval_start).date()),
            "end": str(pd.Timestamp(eval_end).date()),
        },
        "strategy": {"key": strategy_key, "params": params},
        "backtest_config": config.model_dump(),
        "random_seed": None,
        "notes": (
            "The engine is fully deterministic and uses no randomness. Re-running with the "
            "same price data (same fingerprint) reproduces every number exactly."
        ),
    }
    if report is not None:
        manifest["headline_metrics"] = {k: report.value(k) for k in HEADLINE_METRICS}
    return clean_json(manifest)


def manifest_to_json(manifest: dict[str, Any]) -> str:
    return json.dumps(clean_json(manifest), indent=2, sort_keys=False)


def reproduce_from_manifest(
    manifest: dict[str, Any], prices: pd.DataFrame, fingerprint: str
) -> dict[str, Any]:
    """Re-run a manifest on supplied (validated) prices and compare with the record."""
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Not a Quant Lab Pro run manifest.")
    config = BacktestConfig(**manifest["backtest_config"])
    strategy = create_strategy(manifest["strategy"]["key"], manifest["strategy"]["params"])
    start = pd.Timestamp(manifest["evaluation"]["start"])
    end = pd.Timestamp(manifest["evaluation"]["end"])
    benchmark = run_buy_and_hold(prices, config, start, end)
    run = run_strategy(prices, strategy, config, start, end, benchmark=benchmark)
    recorded = manifest.get("headline_metrics", {})
    comparison = []
    for key in HEADLINE_METRICS:
        new = run.report.value(key)
        old = recorded.get(key)
        same = (old is None and new is None) or (
            old is not None and new is not None and math.isclose(old, new, rel_tol=1e-9, abs_tol=1e-9)
        )
        comparison.append({"metric": key, "recorded": old, "reproduced": new, "match": same})
    return {
        "fingerprint_match": fingerprint == manifest["data"]["fingerprint_sha256"],
        "methodology_match": manifest.get("methodology_version") == METHODOLOGY_VERSION,
        "metrics": comparison,
        "all_metrics_match": all(row["match"] for row in comparison),
        "accounting_valid": run.net.is_valid,
        "run": run,
    }


# --------------------------------------------------------------------------- #
# CSV exports
# --------------------------------------------------------------------------- #


def _to_csv(frame: pd.DataFrame, index: bool = True, float_format: str | None = "%.10g") -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=index, date_format="%Y-%m-%d", float_format=float_format)
    return buffer.getvalue()


def prices_csv(prices: pd.DataFrame) -> str:
    """Full-precision prices, so the file reproduces a run exactly."""
    return _to_csv(prices, float_format=None)


def equity_curve_frame(
    result: BacktestResult, benchmark: BacktestResult | None = None, gross: BacktestResult | None = None
) -> pd.DataFrame:
    frame = result.daily[
        ["open", "close", "target", "shares", "cash", "holdings_value", "equity", "exposure", "fees", "slippage_cost"]
    ].copy()
    frame["daily_return"] = frame["equity"].pct_change()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1
    if gross is not None:
        frame["gross_equity"] = gross.equity.reindex(frame.index)
    if benchmark is not None:
        frame["benchmark_equity"] = benchmark.equity.reindex(frame.index)
    return frame


def equity_curve_csv(result: BacktestResult, benchmark: BacktestResult | None = None,
                     gross: BacktestResult | None = None) -> str:
    return _to_csv(equity_curve_frame(result, benchmark, gross))


def ledger_csv(result: BacktestResult) -> str:
    return _to_csv(result.ledger, index=False)


def trades_csv(result: BacktestResult) -> str:
    return _to_csv(result.trades, index=False)


def metrics_csv(report: MetricReport) -> str:
    frame = report.to_frame()[["key", "Metric", "raw_value", "Value", "Reliable", "Note", "Definition"]]
    return _to_csv(frame.rename(columns={"raw_value": "Raw value", "Value": "Formatted"}), index=False)
