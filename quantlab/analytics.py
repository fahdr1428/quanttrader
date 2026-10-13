"""Higher-level analytics built on the engine: rolling statistics, period tables,
strategy comparison and sensitivity analysis.

Sensitivity analysis is *in-sample* by construction: it shows how fragile a
result is to parameter and cost choices, not how a strategy would perform on
unseen data. The walk-forward module handles out-of-sample evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest import BacktestResult, as_index_time, check_price_index, run_backtest
from quantlab.config import TRADING_DAYS_PER_YEAR, BacktestConfig
from quantlab.metrics import MetricReport, compute_metrics, period_returns
from quantlab.strategies import STRATEGIES, Strategy, create_strategy

MAX_SENSITIVITY_COMBINATIONS = 150

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- #
# Rolling statistics and period tables
# --------------------------------------------------------------------------- #


def rolling_volatility(
    returns: pd.Series, window: int = 63, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualised rolling volatility; NaN until ``window`` returns are available."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualised rolling Sharpe ratio; NaN during warm-up or when the window has no volatility."""
    rf_d = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_d
    mean = excess.rolling(window, min_periods=window).mean()
    std = excess.rolling(window, min_periods=window).std(ddof=1)
    return (mean / std.where(std > 1e-12)) * math.sqrt(periods_per_year)


def monthly_return_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month grid of calendar-month returns plus a full-year column."""
    monthly = period_returns(equity, "ME")
    table = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.to_numpy()}
    ).pivot(index="year", columns="month", values="ret")
    table = table.reindex(columns=range(1, 13))
    table.columns = MONTH_LABELS
    yearly = period_returns(equity, "YE")
    table["Full year"] = pd.Series(yearly.to_numpy(), index=yearly.index.year)
    table.index.name = "Year"
    return table


# --------------------------------------------------------------------------- #
# Running a strategy together with its gross twin and benchmark
# --------------------------------------------------------------------------- #


@dataclass
class StrategyRun:
    """Net result, zero-cost twin and metrics for one strategy configuration."""

    strategy: Strategy
    net: BacktestResult
    gross: BacktestResult
    report: MetricReport


def run_strategy(
    prices: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    benchmark: BacktestResult | None = None,
    meta: dict[str, Any] | None = None,
) -> StrategyRun:
    # Signals are computed once (on data truncated at eval_end, exactly as run_backtest
    # would) and shared by the net run and its zero-cost twin.
    check_price_index(prices)
    history = prices.loc[prices.index <= as_index_time(eval_end, prices.index)] if eval_end is not None else prices
    signals = strategy.generate_signals(history)
    net = run_backtest(history, strategy, config, eval_start, eval_end, signals=signals, meta=meta)
    gross = run_backtest(history, strategy, config.without_costs(), eval_start, eval_end, signals=signals, meta=meta)
    report = compute_metrics(net, benchmark=benchmark, gross=gross)
    return StrategyRun(strategy=strategy, net=net, gross=gross, report=report)


COMPARISON_COLUMNS = {
    "total_return": "Net total return",
    "gross_total_return": "Gross total return",
    "cagr": "CAGR (net)",
    "volatility": "Volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar",
    "time_invested": "Time invested",
    "avg_exposure": "Average exposure",
    "fills": "Fills",
    "round_trips": "Round trips",
    "turnover": "Turnover (ann.)",
    "total_fees": "Fees ($)",
    "total_slippage": "Slippage ($)",
    "cost_drag": "Cost drag",
    "excess_return": "Excess vs benchmark",
}


def compare_strategies(runs: list[StrategyRun]) -> pd.DataFrame:
    """One row per strategy with identical capital, costs, sessions and benchmark."""
    if not runs:
        return pd.DataFrame()
    first = runs[0].net
    for run in runs[1:]:
        if not run.net.daily.index.equals(first.daily.index):
            raise ValueError("strategies must be evaluated over identical sessions to be compared")
        if run.net.config != first.config:
            raise ValueError("strategies must share the same capital and cost configuration")
    rows = []
    for run in runs:
        row = {"Strategy": run.net.strategy_label, "key": run.strategy.key}
        for key, label in COMPARISON_COLUMNS.items():
            row[label] = run.report.value(key)
        row["Default parameters"] = run.strategy.uses_default_parameters()
        row["Accounting reconciled"] = run.net.is_valid
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #


def parameter_sensitivity(
    prices: pd.DataFrame,
    strategy_key: str,
    base_params: dict[str, Any],
    grid: dict[str, list[Any]],
    config: BacktestConfig,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    return_matrix: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a (small) grid of parameter values on the same sample.

    Invalid combinations (e.g. short window >= long window) are recorded with
    ``valid = False`` rather than silently dropped. With ``return_matrix`` the
    daily net returns of every valid combination are returned too (sessions x
    combinations), for selection-bias tests that need the whole grid.
    """
    cls = STRATEGIES[strategy_key]
    names = list(grid)
    sizes = [len(grid[n]) for n in names]
    total = int(np.prod(sizes)) if sizes else 0
    if total > MAX_SENSITIVITY_COMBINATIONS:
        raise ValueError(
            f"{total} combinations requested; the limit is {MAX_SENSITIVITY_COMBINATIONS} to keep "
            "the search space (and the multiple-testing problem) constrained."
        )
    rows = []
    columns: dict[str, pd.Series] = {}
    index = pd.MultiIndex.from_product([grid[n] for n in names], names=names)
    for combo in index:
        combo = combo if isinstance(combo, tuple) else (combo,)
        params = {**base_params, **dict(zip(names, combo))}
        row: dict[str, Any] = dict(zip(names, combo))
        if not cls.is_valid(params):
            row.update(valid=False, total_return=np.nan, cagr=np.nan, sharpe=np.nan, max_drawdown=np.nan,
                       fills=np.nan, time_invested=np.nan)
            rows.append(row)
            continue
        strategy = cls(**params)
        result = run_backtest(prices, strategy, config, eval_start, eval_end)
        report = compute_metrics(result)
        columns[", ".join(f"{n}={v:g}" if isinstance(v, (int, float)) else f"{n}={v}"
                          for n, v in zip(names, combo))] = result.returns
        row.update(
            valid=True,
            total_return=report.value("total_return"),
            cagr=report.value("cagr"),
            sharpe=report.value("sharpe"),
            max_drawdown=report.value("max_drawdown"),
            fills=report.value("fills"),
            time_invested=report.value("time_invested"),
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if return_matrix:
        return frame, pd.DataFrame(columns)
    return frame


@dataclass
class SelectionTests:
    """Selection-bias diagnostics for a parameter grid (see :mod:`quantlab.overfitting`,
    :mod:`quantlab.significance`)."""

    n_configs: int
    n_obs: int
    pbo: Any  # PBOResult | None
    n_blocks: int | None
    spa_vs_benchmark: Any  # SPAResult
    spa_vs_cash: Any  # SPAResult
    stepdown_vs_benchmark: Any  # RomanoWolfResult
    pbo_note: str = ""  # why PBO was not computed, when it was not


def selection_tests(returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, config: BacktestConfig,
                    n_boot: int = 1000, mean_block: float = 21.0) -> SelectionTests | None:
    """Was the grid's best configuration picked by luck? Needs at least two configurations.

    * PBO (combinatorially symmetric cross-validation) with the largest even
      number of blocks up to 12 that leaves at least 20 sessions per block.
    * Hansen's SPA test of the best configuration against the benchmark and
      against cash, and Romano-Wolf stepdown p-values per configuration, all
      with one shared stationary bootstrap so correlation across the grid is kept.
    """
    from quantlab.overfitting import probability_of_backtest_overfitting
    from quantlab.significance import hansen_spa, loss_differentials, romano_wolf_stepdown

    matrix = returns_matrix.dropna()
    if matrix.shape[1] < 2 or len(matrix) < 60:
        return None
    rf, ppy = config.risk_free_rate, config.periods_per_year
    blocks = next((s for s in range(12, 1, -2) if len(matrix) // s >= 20), None)
    values = matrix.to_numpy(dtype="float64")
    identical = bool(np.allclose(values, values[:, :1], rtol=0.0, atol=1e-12))
    pbo, pbo_note = None, ""
    if identical:  # e.g. no combination ever took a position: there is nothing to choose between
        pbo_note = "Every combination produced the same returns, so there is nothing to select between."
    elif not blocks or blocks < 4:
        pbo_note = "Too few sessions to split the sample into at least four blocks of 20."
    else:
        pbo = probability_of_backtest_overfitting(matrix, blocks, rf, ppy)
    bench = benchmark_returns.reindex(matrix.index)
    vs_bench = loss_differentials(matrix, bench, rf, ppy)
    vs_cash = loss_differentials(matrix, None, rf, ppy)
    block = min(float(mean_block), len(matrix) / 4)
    return SelectionTests(
        n_configs=matrix.shape[1],
        n_obs=len(matrix),
        pbo=pbo,
        n_blocks=blocks if pbo is not None else None,
        spa_vs_benchmark=hansen_spa(vs_bench, n_boot=n_boot, mean_block=block),
        spa_vs_cash=hansen_spa(vs_cash, n_boot=n_boot, mean_block=block),
        stepdown_vs_benchmark=romano_wolf_stepdown(vs_bench, n_boot=n_boot, mean_block=block),
        pbo_note=pbo_note,
    )


def cost_sensitivity(
    prices: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-run the same signals with fees and slippage scaled by each multiplier.

    The benchmark is re-run at the same cost level, so 'excess return' always
    compares like with like.
    """
    from quantlab.backtest import run_buy_and_hold

    bench_prices = benchmark_prices if benchmark_prices is not None else prices
    check_price_index(prices)
    signals = strategy.generate_signals(
        prices.loc[prices.index <= as_index_time(eval_end, prices.index)] if eval_end is not None else prices
    )
    rows = []
    for m in multipliers:
        cfg = config.model_copy(
            update={
                "fee_rate": min(config.fee_rate * m, 0.05),
                "slippage_rate": min(config.slippage_rate * m, 0.05),
            }
        )
        result = run_backtest(prices, strategy, cfg, eval_start, eval_end, signals=signals)
        bench = run_buy_and_hold(bench_prices, cfg, eval_start, eval_end)
        report = compute_metrics(result, benchmark=bench)
        rows.append(
            {
                "multiplier": m,
                "fee_rate": cfg.fee_rate,
                "slippage_rate": cfg.slippage_rate,
                "cost_per_side_bps": (cfg.fee_rate + cfg.slippage_rate) * 1e4,
                "total_return": report.value("total_return"),
                "sharpe": report.value("sharpe"),
                "benchmark_total_return": report.value("benchmark_total_return"),
                "excess_return": report.value("excess_return"),
                "total_costs": report.value("total_fees") + report.value("total_slippage"),
            }
        )
    return pd.DataFrame(rows)


def default_strategy_specs() -> list[tuple[str, dict[str, Any]]]:
    return [(key, cls.defaults()) for key, cls in STRATEGIES.items()]


def build_strategies(specs: list[tuple[str, dict[str, Any]]]) -> list[Strategy]:
    return [create_strategy(key, params) for key, params in specs]
