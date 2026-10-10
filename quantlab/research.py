"""Research-integrity tooling: chronological walk-forward evaluation and
multiple-testing adjustments for Sharpe ratios.

Walk-forward procedure
----------------------
1. Split the evaluation sessions into consecutive folds. Each fold has a
   *training* window followed immediately by a *test* window. Nothing is ever
   shuffled; test windows never overlap and always come after their training data.
2. For each fold, every parameter combination in a deliberately small grid is
   backtested on the training window only. The price history is truncated at
   the training end *before* any signal is computed, so data from the test
   period cannot influence the choice — even through a bug in a strategy.
3. The best combination (by the chosen objective, ties broken by grid order) is
   then applied to the test window. Indicators may use history from before the
   test start (that is information a real trader would have), never after.
4. The test-window targets of all folds are stitched into one continuous
   out-of-sample portfolio, simulated with the normal engine, costs and timing.

The out-of-sample curve is the honest estimate. In-sample (training) figures are
reported next to it only to show how much performance decays out of sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest import (
    BacktestResult,
    backtest_from_signals,
    run_backtest,
    run_buy_and_hold,
)
from quantlab.config import BacktestConfig
from quantlab.metrics import (
    annualised_return,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from quantlab.strategies import STRATEGIES

OBJECTIVES = {
    "sharpe": "Sharpe ratio",
    "cagr": "Annualised return (CAGR)",
    "calmar": "Calmar ratio",
    "total_return": "Total return",
}

MAX_WALK_FORWARD_COMBINATIONS = 100
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WalkForwardConfig:
    train_sessions: int = 504
    test_sessions: int = 126
    anchored: bool = False
    objective: str = "sharpe"
    min_test_sessions: int = 20

    def __post_init__(self) -> None:
        if self.train_sessions < 60:
            raise ValueError("Training windows need at least 60 sessions.")
        if self.test_sessions < 20:
            raise ValueError("Test windows need at least 20 sessions.")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{self.objective}'.")


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_sessions: int
    test_sessions: int


def make_folds(sessions: pd.DatetimeIndex, cfg: WalkForwardConfig) -> list[Fold]:
    """Chronological train/test splits over ``sessions``.

    Rolling mode moves a fixed-length training window forward by one test
    window each fold. Anchored (expanding) mode always trains from the first
    session. A final, shorter test window is kept if it has at least
    ``min_test_sessions`` sessions.
    """
    n = len(sessions)
    folds: list[Fold] = []
    k = 0
    while True:
        train_start_pos = 0 if cfg.anchored else k * cfg.test_sessions
        train_end_pos = k * cfg.test_sessions + cfg.train_sessions - 1
        test_start_pos = train_end_pos + 1
        test_end_pos = min(test_start_pos + cfg.test_sessions - 1, n - 1)
        if test_start_pos >= n or test_end_pos - test_start_pos + 1 < cfg.min_test_sessions:
            break
        folds.append(
            Fold(
                number=k + 1,
                train_start=sessions[train_start_pos],
                train_end=sessions[train_end_pos],
                test_start=sessions[test_start_pos],
                test_end=sessions[test_end_pos],
                train_sessions=train_end_pos - train_start_pos + 1,
                test_sessions=test_end_pos - test_start_pos + 1,
            )
        )
        k += 1
    return folds


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #


def _objective_value(result: BacktestResult, objective: str) -> float:
    equity = result.equity
    if objective == "sharpe":
        value = sharpe_ratio(result.returns, result.config.risk_free_rate, result.config.periods_per_year)
    elif objective == "cagr":
        value = annualised_return(equity, result.config.periods_per_year)
    elif objective == "calmar":
        cagr = annualised_return(equity, result.config.periods_per_year)
        mdd = abs(max_drawdown(equity))
        value = None if (cagr is None or mdd < 1e-12) else cagr / mdd
    else:
        value = total_return(equity)
    return -math.inf if value is None or not math.isfinite(value) else float(value)


@dataclass
class WalkForwardResult:
    strategy_key: str
    config: WalkForwardConfig
    grid: dict[str, list[Any]]
    n_combinations: int
    folds: list[Fold]
    fold_table: pd.DataFrame
    oos: BacktestResult
    oos_benchmark: BacktestResult
    in_sample_trials: pd.DataFrame  # every (fold, params) training score
    notes: list[str] = field(default_factory=list)

    @property
    def total_backtests(self) -> int:
        return len(self.in_sample_trials)


def walk_forward(
    prices: pd.DataFrame,
    strategy_key: str,
    grid: dict[str, list[Any]],
    backtest_config: BacktestConfig,
    wf_config: WalkForwardConfig,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> WalkForwardResult:
    cls = STRATEGIES[strategy_key]
    combos = cls.parameter_grid({k: tuple(v) for k, v in grid.items()} if grid else None)
    if not combos:
        raise ValueError("The parameter grid contains no valid combinations.")
    if len(combos) > MAX_WALK_FORWARD_COMBINATIONS:
        raise ValueError(
            f"{len(combos)} combinations exceed the limit of {MAX_WALK_FORWARD_COMBINATIONS}. "
            "Constrain the search space: every extra combination raises the chance of "
            "selecting a configuration that only fit noise."
        )

    end = pd.Timestamp(eval_end) if eval_end is not None else prices.index[-1]
    start = pd.Timestamp(eval_start) if eval_start is not None else prices.index[0]
    sessions = prices.index[(prices.index >= start) & (prices.index <= end)]
    folds = make_folds(sessions, wf_config)
    if not folds:
        raise ValueError(
            f"The evaluation period has {len(sessions)} sessions, which is not enough for one "
            f"fold of {wf_config.train_sessions} training + {wf_config.min_test_sessions}+ test sessions."
        )

    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    stitched_parts: list[pd.DataFrame] = []

    for fold in folds:
        # --- in-sample selection: data truncated at the training end ----------
        train_prices = prices.loc[prices.index <= fold.train_end]
        if train_prices.index.max() != fold.train_end:  # structural leakage guard
            raise RuntimeError("training data extends beyond the training window")
        best_score, best_params, best_result = -math.inf, None, None
        for params in combos:
            strategy = cls(**params)
            result = run_backtest(train_prices, strategy, backtest_config, eval_start=fold.train_start)
            score = _objective_value(result, wf_config.objective)
            trial_rows.append(
                {"fold": fold.number, **params, "objective": score if math.isfinite(score) else np.nan}
            )
            if score > best_score or best_params is None:
                best_score, best_params, best_result = score, params, result

        # --- out-of-sample targets: data truncated at the test end --------------
        test_hist = prices.loc[prices.index <= fold.test_end]
        signals = cls(**best_params).generate_signals(test_hist)
        if fold.number == 1:
            # The anchor is the final training session; its signal is known before
            # the first test open and uses fold-1 parameters.
            part = signals.loc[(signals.index >= fold.train_end) & (signals.index <= fold.test_end)]
        else:
            part = signals.loc[(signals.index >= fold.test_start) & (signals.index <= fold.test_end)]
        part = part[["target", "reason"]].copy()
        part["fold"] = fold.number
        stitched_parts.append(part)

        is_report = compute_metrics(best_result)
        fold_rows.append(
            {
                "Fold": fold.number,
                "Train start": fold.train_start.date(),
                "Train end": fold.train_end.date(),
                "Test start": fold.test_start.date(),
                "Test end": fold.test_end.date(),
                "Train sessions": fold.train_sessions,
                "Test sessions": fold.test_sessions,
                "Selected parameters": ", ".join(f"{k}={v}" for k, v in best_params.items()),
                "params": best_params,
                "Selection data ends": train_prices.index.max().date(),
                "IS objective": best_score if math.isfinite(best_score) else np.nan,
                "IS total return": is_report.value("total_return"),
                "IS Sharpe": is_report.value("sharpe"),
            }
        )

    stitched = pd.concat(stitched_parts)
    # Where a new fold's parameters flip the target on its first session, say so.
    effective = stitched["target"].fillna(0.0)
    flipped = effective.ne(effective.shift(1)) & stitched["fold"].ne(stitched["fold"].shift(1))
    flipped.iloc[0] = False
    for ts in stitched.index[flipped & (stitched["reason"] == "")]:
        fold_no = int(stitched.at[ts, "fold"])
        chosen = fold_rows[fold_no - 1]["Selected parameters"]
        stitched.at[ts, "reason"] = f"Fold {fold_no} re-selected parameters ({chosen}); their signal differs"
    window_prices = prices.loc[stitched.index]
    oos = backtest_from_signals(
        window_prices=window_prices,
        window_signals=stitched,
        config=backtest_config,
        strategy_key=strategy_key,
        strategy_label=f"{cls.name} — walk-forward out-of-sample",
        params={"selection": "re-optimised each fold"},
        meta={"walk_forward": True},
    )
    bench_src = benchmark_prices if benchmark_prices is not None else prices
    oos_benchmark = run_buy_and_hold(
        bench_src, backtest_config, eval_start=folds[0].test_start, eval_end=folds[-1].test_end
    )

    # --- per-fold out-of-sample statistics from the stitched portfolio ---------
    equity = oos.equity
    bench_equity = oos_benchmark.equity.reindex(equity.index)
    for row, fold in zip(fold_rows, folds):
        base = equity.index[equity.index < fold.test_start][-1]
        seg = equity.loc[base:fold.test_end]
        bseg = bench_equity.loc[base:fold.test_end]
        seg_rets = seg.pct_change().iloc[1:]
        row["OOS total return"] = total_return(seg)
        row["OOS Sharpe"] = sharpe_ratio(seg_rets, backtest_config.risk_free_rate, backtest_config.periods_per_year)
        row["OOS max drawdown"] = max_drawdown(seg)
        row["Benchmark OOS return"] = total_return(bseg) if bseg.notna().all() else np.nan
        row["OOS excess return"] = row["OOS total return"] - row["Benchmark OOS return"]

    fold_table = pd.DataFrame(fold_rows)
    notes = [
        f"{len(combos)} parameter combinations x {len(folds)} folds = {len(trial_rows)} in-sample backtests.",
        "Out-of-sample periods are contiguous and never used for parameter selection.",
    ]
    changes = (fold_table["Selected parameters"] != fold_table["Selected parameters"].shift()).iloc[1:].sum()
    notes.append(f"Selected parameters changed in {int(changes)} of {max(len(folds) - 1, 0)} fold transitions.")
    return WalkForwardResult(
        strategy_key=strategy_key,
        config=wf_config,
        grid={k: list(v) for k, v in grid.items()},
        n_combinations=len(combos),
        folds=folds,
        fold_table=fold_table,
        oos=oos,
        oos_benchmark=oos_benchmark,
        in_sample_trials=pd.DataFrame(trial_rows),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Sharpe-ratio inference under multiple testing
# --------------------------------------------------------------------------- #


def _moments(returns: pd.Series) -> tuple[float, float, float, int] | None:
    x = returns.dropna().to_numpy(dtype="float64")
    n = len(x)
    if n < 3:
        return None
    mean = x.mean()
    sd = x.std(ddof=0)
    if sd < 1e-12:
        return None
    z = (x - mean) / sd
    return float(mean / x.std(ddof=1)), float(np.mean(z**3)), float(np.mean(z**4)), n


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float | None:
    """Probability that the true (per-period) Sharpe ratio exceeds ``benchmark_sharpe``.

    Bailey & López de Prado (2012). Adjusts for sample length, skewness and
    fat tails: ``PSR = Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2))``.
    Sharpe ratios here are *per period* (daily), not annualised.
    """
    m = _moments(returns)
    if m is None:
        return None
    sr, skew, kurt, n = m
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return None
    z = (sr - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denom)
    return NormalDist().cdf(z)


def expected_max_sharpe(trial_sharpes: list[float] | np.ndarray) -> float | None:
    """Expected maximum per-period Sharpe among N trials if all true Sharpes were zero.

    ``SR0 = sqrt(Var[SR_n]) * ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))``.
    """
    values = np.asarray([s for s in trial_sharpes if s is not None and np.isfinite(s)], dtype="float64")
    n = len(values)
    if n < 2:
        return None
    var = float(values.var(ddof=1))
    if var <= 0:
        return 0.0
    nd = NormalDist()
    return math.sqrt(var) * (
        (1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / n) + EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e))
    )


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: list[float] | np.ndarray) -> float | None:
    """PSR measured against the Sharpe you would expect from the best of N useless trials.

    A value near 1 means the observed Sharpe is unlikely to be explained by
    selection among the trials alone; near 0.5 or below means it is consistent
    with luck. This is a heuristic that assumes the trials are independent; real
    parameter variants are correlated, so treat it as indicative, not exact.
    """
    sr0 = expected_max_sharpe(trial_sharpes)
    if sr0 is None:
        return None
    return probabilistic_sharpe_ratio(returns, sr0)


def per_period_sharpe(returns: pd.Series) -> float | None:
    m = _moments(returns)
    return None if m is None else m[0]
