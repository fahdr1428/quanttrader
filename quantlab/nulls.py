"""Honest null benchmarks for long-only timing strategies.

A long-only timing strategy is usually compared with a 100%-invested
buy-and-hold. That comparison confounds two things: *how much* the strategy is
invested and *when*. Lower average exposure alone produces smaller drawdowns,
lower volatility and — in falling markets — a better return or Sharpe ratio,
none of which is timing skill. This module separates the two.

Exposure-matched benchmark
--------------------------
:func:`exposure_matched_benchmark` holds a **constant** target weight equal to
the strategy's average exposure, simulated through the same engine, costs,
rebalance band, execution timing and evaluation window. It answers: "would a
portfolio carrying the same average amount of market risk, but with no timing
at all, have done as well?"

*Average exposure* is defined as the mean of the strategy's **realised** daily
exposure (holdings value / equity at each close) over the evaluation window,
excluding the anchor session — the ``avg_exposure`` metric of
:func:`quantlab.metrics.compute_metrics`. The realised figure is used rather than
the mean target because it is what the portfolio actually carried: it includes
the one-session execution lag, drift inside the rebalance band, whole-share
cash drag and rejected entries. The benchmark then hovers around that weight
(within the rebalance band), so its realised average exposure matches the
strategy's to within ``config.rebalance_threshold``.

With a zero risk-free rate and no cash interest, a constant-exposure portfolio
has (almost exactly) the same Sharpe ratio as buy-and-hold, because scaling
returns does not change their Sharpe ratio. The benchmark is therefore the
right yardstick for return, volatility and drawdown; for risk-adjusted
performance use the permutation test below.

Timing permutation test
-----------------------
:func:`timing_permutation_test` circularly shifts the strategy's target path
inside the evaluation window and re-simulates every shifted path. A circular
shift keeps the exposure distribution, the number and size of target changes
and the holding periods (up to the wrap-around boundary), but destroys the
alignment between the positions and the returns. If the strategy has no timing
ability, its observed statistic is a typical draw from the shifted
distribution, and

    p = (1 + #{shifted statistic >= observed}) / (number of shifts + 1)

is a valid (slightly conservative) p-value. Ties count against the strategy, so
a constant target — every shift identical — gets ``p = 1``.

Caveats
-------
* Shifts shorter than ``min_shift`` sessions in either direction are excluded:
  a trend signal shifted by a few days still lines up with the trends it
  caught and would make the test needlessly conservative.
* Positions that wrap around the end of the window use signals computed from
  *later* prices. When ``min_shift`` is shorter than the strategy's look-back,
  such a signal can contain the very return it is credited with, which inflates
  the null for trend-following rules (a conservative p-value) and could deflate
  it for mean-reversion rules. Choose ``min_shift`` above the strategy's memory
  when that matters; the result's ``note`` flags it.
* The test treats warm-up sessions without a signal as cash and shifts them
  like any other session.
* The shift null is approximate, not exact: the observed positions are a
  function of the same prices they are evaluated on. In short windows, with
  few independent position episodes, this makes the test over-reject. In
  seeded noise simulations (zero-drift random walks, MA 20/50, 49 shifts,
  nominal size 4% at p < 0.05) the false-positive rate was about 10% with
  ~300 sessions, 6% with ~500 and 5% with ~700. The result's ``note`` flags
  windows shorter than ``SHORT_WINDOW_SESSIONS``.
* Power is modest: a few years of daily data detect only a strong timing
  effect. A large p-value means "no evidence of timing skill", not "no skill".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantlab.backtest import BacktestResult, backtest_from_signals, evaluation_window, simulate
from quantlab.config import BacktestConfig
from quantlab.metrics import annualised_return, sharpe_ratio, total_return
from quantlab.robustness import DEFAULT_SEED

Statistic = Literal["sharpe", "cagr", "total_return"]

STATISTICS: dict[str, str] = {
    "sharpe": "Sharpe ratio",
    "cagr": "Annualised return (CAGR)",
    "total_return": "Total return",
}

# Below this many usable shifted paths the p-value is too coarse to report.
MIN_VALID_SHIFTS = 20

# Below this many decision sessions (about two years) the shift null tends to over-reject.
SHORT_WINDOW_SESSIONS = 504

EXPOSURE_MATCHED_KEY = "exposure_matched"

EXPOSURE_DEFINITION = (
    "Mean of the strategy's realised daily exposure (holdings value / equity at each close) "
    "over the evaluation window, excluding the anchor session."
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def average_exposure(result: BacktestResult) -> float:
    """The strategy's average exposure as defined in the module docstring (0.0 if never invested)."""
    exposure = result.daily["exposure"].iloc[1:]
    if exposure.empty:
        return 0.0
    return float(min(max(exposure.mean(), 0.0), 1.0))


def _window_prices(prices: pd.DataFrame, window: pd.DatetimeIndex, result: BacktestResult) -> pd.DataFrame:
    """Prices for ``window``, checked against the prices the result was simulated on."""
    missing = window.difference(prices.index)
    if len(missing):
        raise ValueError(
            f"prices do not cover the strategy's evaluation window ({len(missing)} session(s) missing, "
            f"first {missing[0].date()})"
        )
    window_prices = prices.loc[window]
    if window.equals(result.daily.index):
        for column in ("open", "close"):
            if not np.allclose(
                window_prices[column].to_numpy(dtype="float64"),
                result.daily[column].to_numpy(dtype="float64"),
                rtol=1e-10,
                atol=0.0,
            ):
                raise ValueError(
                    f"'{column}' prices differ from those the strategy result was simulated on; "
                    "pass the same price data that produced the result"
                )
    return window_prices


def _pct(weight: float) -> str:
    """Percentage for labels: whole percent unless that would round a non-trivial weight to 0%/100%."""
    if weight in (0.0, 1.0) or 0.01 <= weight <= 0.99:
        return f"{weight:.0%}"
    return f"{weight:.1%}"


# --------------------------------------------------------------------------- #
# 1. Exposure-matched benchmark
# --------------------------------------------------------------------------- #


def exposure_matched_benchmark(
    prices: pd.DataFrame,
    result: BacktestResult,
    config: BacktestConfig,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
) -> BacktestResult:
    """Constant-exposure benchmark with the strategy's average exposure.

    The benchmark holds a constant target weight ``w`` = :func:`average_exposure`
    of ``result`` from the anchor session onward, so it buys at the first
    evaluation open exactly like buy-and-hold. It is simulated with
    :func:`quantlab.backtest.backtest_from_signals` — the same engine, ``config``
    (costs, rebalance band, share sizing, cash interest) and sessions as the
    strategy — so the full reconciliation report runs and ``is_valid`` is
    meaningful. Pass ``config = result.config`` for a like-for-like comparison.

    ``eval_start`` / ``eval_end`` are optional and interpreted exactly as in
    :func:`quantlab.backtest.run_backtest`; the window they define must be the
    strategy's own window (a ``ValueError`` is raised otherwise). When both are
    omitted the strategy's window is used.

    A strategy that was never invested yields a cash-only benchmark (``w = 0``,
    no trades; cash still earns ``config.cash_interest_rate``), labelled
    "Constant 0% exposure (exposure-matched)" with an explanatory
    ``meta["note"]``.
    """
    window = result.daily.index
    if eval_start is not None or eval_end is not None:
        index = prices.index
        if eval_end is not None:
            index = index[index <= pd.Timestamp(eval_end)]
        requested = evaluation_window(index, eval_start, None)
        if not requested.equals(window):
            raise ValueError(
                "eval_start/eval_end define a different evaluation window from the strategy result's "
                f"({len(requested)} vs {len(window)} sessions); an exposure-matched benchmark must cover "
                "exactly the strategy's sessions"
            )
    window_prices = _window_prices(prices, window, result)

    avg = average_exposure(result)
    weight = avg
    reasons = np.full(len(window), "", dtype=object)
    meta: dict[str, object] = {
        "evaluation_start": result.meta.get("evaluation_start", window[0]),
        "benchmark_type": EXPOSURE_MATCHED_KEY,
        "matched_strategy": result.strategy_label,
        "strategy_average_exposure": avg,
        "exposure_definition": EXPOSURE_DEFINITION,
    }
    if weight > 0.0:
        reasons[0] = (
            f"Exposure-matched benchmark: hold a constant {weight:.1%} of equity (the strategy's average "
            f"exposure), rebalanced only when the weight drifts more than {config.rebalance_threshold:.0%}"
        )
    else:
        meta["note"] = (
            "The strategy was never invested in the evaluation window, so the exposure-matched "
            "benchmark holds cash only."
        )
    signals = pd.DataFrame({"target": weight, "reason": reasons}, index=window)
    return backtest_from_signals(
        window_prices=window_prices,
        window_signals=signals,
        config=config,
        strategy_key=EXPOSURE_MATCHED_KEY,
        strategy_label=f"Constant {_pct(weight)} exposure (exposure-matched)",
        params={"weight": weight},
        warmup_sessions=0,
        first_signal_date=window[0],
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# 2. Timing permutation test (circular shifts)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class PermutationResult:
    """Outcome of :func:`timing_permutation_test`.

    ``null`` and ``shifts`` are aligned (one statistic per shift; NaN where the
    statistic is undefined for that path) and read-only. ``n_shifts`` counts
    the finite null values the p-value is based on. ``p_value`` is None when the
    test is not applicable; ``note`` then says why. ``eq=False`` because the
    array fields have no single truth value under ``==``.
    """

    observed: float | None
    null: np.ndarray
    p_value: float | None
    n_shifts: int
    shifts: np.ndarray
    statistic: str
    note: str

    @property
    def label(self) -> str:
        return STATISTICS.get(self.statistic, self.statistic)


def circular_shift(targets: pd.Series | np.ndarray, shift: int) -> np.ndarray:
    """``np.roll`` of a target path by ``shift`` sessions, with NaN (no signal) treated as cash (0)."""
    values = np.nan_to_num(np.asarray(targets, dtype="float64"), nan=0.0)
    return np.roll(values, int(shift))


def admissible_shifts(n_sessions: int, min_shift: int) -> np.ndarray:
    """Shifts of a circular path of ``n_sessions`` that move it at least ``min_shift`` sessions both ways."""
    if n_sessions - min_shift < min_shift:
        return np.empty(0, dtype=np.int64)
    return np.arange(min_shift, n_sessions - min_shift + 1, dtype=np.int64)


def sample_shifts(n_sessions: int, n_shifts: int, min_shift: int, seed: int) -> np.ndarray:
    """Deterministic, sorted sample of distinct admissible shifts (all of them if there are too few)."""
    candidates = admissible_shifts(n_sessions, min_shift)
    if len(candidates) <= n_shifts:
        return candidates
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=n_shifts, replace=False))


def _statistic(equity: pd.Series, statistic: str, config: BacktestConfig) -> float:
    if statistic == "sharpe":
        value = sharpe_ratio(equity.pct_change().iloc[1:], config.risk_free_rate, config.periods_per_year)
    elif statistic == "cagr":
        value = annualised_return(equity, config.periods_per_year)
    else:
        value = total_return(equity)
    return float("nan") if value is None else float(value)


def _frozen(values: np.ndarray) -> np.ndarray:
    values = np.array(values, copy=True)
    values.setflags(write=False)
    return values


def timing_permutation_test(
    prices: pd.DataFrame,
    result: BacktestResult,
    config: BacktestConfig,
    n_shifts: int = 200,
    min_shift: int = 63,
    seed: int = DEFAULT_SEED,
    statistic: Statistic = "sharpe",
) -> PermutationResult:
    """Does the strategy's timing matter? A circular-shift permutation test.

    The strategy's target path over the evaluation window (warm-up NaN = cash)
    is split into the *decision* sessions — the anchor through the penultimate
    session, whose targets are executed at the next open — and the final
    session, whose target can never execute. The decision path is rotated by
    each sampled shift ``k`` (``min_shift <= k <= n - min_shift``, distinct,
    drawn deterministically from ``seed``; every admissible shift is used when
    there are at most ``n_shifts`` of them) and re-simulated with
    :func:`quantlab.backtest.simulate` under ``config`` — same prices, costs,
    rebalance band and timing. ``observed`` is computed through the same path
    for the unshifted targets, so it equals the result's own statistic when
    ``config == result.config``.

    ``statistic`` is ``"sharpe"`` (:func:`quantlab.metrics.sharpe_ratio` with
    ``config.risk_free_rate`` and ``config.periods_per_year``), ``"cagr"`` or
    ``"total_return"``. The p-value is ``(1 + #{null >= observed}) / (n + 1)``
    over the ``n`` finite null values, with ties counted as ``>=``.

    ``p_value`` is None (and ``note`` explains) when the strategy was never
    invested, when the observed statistic is undefined, or when fewer than
    ``MIN_VALID_SHIFTS`` usable shifts exist.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {', '.join(STATISTICS)}; got {statistic!r}")
    if int(n_shifts) < 1:
        raise ValueError("n_shifts must be at least 1")
    if int(min_shift) < 1:
        raise ValueError("min_shift must be at least 1 session")
    n_shifts, min_shift = int(n_shifts), int(min_shift)

    window = result.daily.index
    window_prices = _window_prices(prices, window, result)[["open", "close"]]
    targets = np.nan_to_num(result.daily["target"].to_numpy(dtype="float64"), nan=0.0)
    decisions, final = targets[:-1], targets[-1:]
    n_decisions = len(decisions)

    cache: dict[bytes, float] = {}

    def evaluate(path: np.ndarray) -> float:
        key = path.tobytes()
        if key not in cache:
            series = pd.Series(np.concatenate([path, final]), index=window)
            daily, _, _, _ = simulate(window_prices, series, config)
            cache[key] = _statistic(daily["equity"], statistic, config)
        return cache[key]

    observed_value = evaluate(decisions)
    observed = observed_value if math.isfinite(observed_value) else None
    empty_null, empty_shifts = _frozen(np.empty(0)), _frozen(np.empty(0, dtype=np.int64))

    def not_applicable(note: str) -> PermutationResult:
        return PermutationResult(observed, empty_null, None, 0, empty_shifts, statistic, note)

    if not (decisions > 0).any() or not (result.daily["shares"] > 0).any():
        return not_applicable(
            "The strategy was never invested during the evaluation window, so it has no timing to test."
        )
    if observed is None:
        return not_applicable(f"The observed {STATISTICS[statistic].lower()} is undefined for this run.")
    available = len(admissible_shifts(n_decisions, min_shift))
    if available < MIN_VALID_SHIFTS:
        return not_applicable(
            f"Only {available} circular shift(s) of at least {min_shift} sessions exist in a window of "
            f"{n_decisions} decision sessions; at least {MIN_VALID_SHIFTS} are needed. Use a longer "
            "evaluation window or a smaller minimum shift."
        )

    shifts = sample_shifts(n_decisions, n_shifts, min_shift, seed)
    null = np.array([evaluate(np.roll(decisions, k)) for k in shifts], dtype="float64")
    finite = null[np.isfinite(null)]
    used = len(finite)

    sampling = (
        f"all {len(shifts)} admissible shifts (exact circular-shift null)"
        if len(shifts) == available
        else f"{len(shifts)} distinct shifts sampled with seed {seed}"
    )
    notes = [
        f"Target path circularly shifted by {sampling}, each at least {min_shift} sessions in both directions "
        f"within {n_decisions} decision sessions, and re-simulated with the same costs, rebalance band and timing."
    ]
    if used < len(shifts):
        notes.append(f"{len(shifts) - used} shifted path(s) had an undefined statistic and were left out.")
    if n_decisions < SHORT_WINDOW_SESSIONS:
        notes.append(
            f"The window is short ({n_decisions} sessions, under about two years): with few independent "
            "position episodes the shift null is approximate and tends to over-reject, so treat a small "
            "p-value with extra caution."
        )
    if result.warmup_sessions >= min_shift:
        notes.append(
            f"The minimum shift ({min_shift}) does not exceed the strategy's look-back "
            f"({result.warmup_sessions} sessions): wrapped segments of some shifted paths use signals built "
            "from later prices, which can bias the null (conservatively for trend-following rules)."
        )
    if used < MIN_VALID_SHIFTS:
        notes.append(f"Fewer than {MIN_VALID_SHIFTS} usable shifts; no p-value is reported.")
        p_value = None
    else:
        tolerance = 1e-12 * max(1.0, abs(observed))
        p_value = (1 + int(np.count_nonzero(finite >= observed - tolerance))) / (used + 1)
    return PermutationResult(
        observed=observed,
        null=_frozen(null),
        p_value=p_value,
        n_shifts=used,
        shifts=_frozen(shifts),
        statistic=statistic,
        note=" ".join(notes),
    )
