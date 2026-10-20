"""Signal specifications and the current state of every strategy's signal.

The agent's rules require every proposed signal to state its hypothesis,
required inputs, calculation, entry and exit conditions, expected holding
period, invalidation condition, benchmark and known limitations.
:func:`signal_spec` builds that statement for every strategy in
:data:`quantlab.strategies.STRATEGIES` from the strategy's own documentation
(:class:`quantlab.strategies.StrategyDoc`) and its validated parameters, so the
text always matches the code that produces the signal and the numbers in it are
the parameters actually used.

:func:`current_signal` reports where a strategy stands on the latest validated
session, computed with :meth:`Strategy.generate_signals` — the same code the
backtester runs — so the state shown is exactly what a backtest would do next.

Nothing here is a recommendation. Signals are research hypotheses; the system
is long-only and unleveraged, so a bearish reading means "cash", never a short.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab import indicators as ind
from quantlab import ml
from quantlab.config import TRADING_DAYS_PER_YEAR
from quantlab.serialization import clean_json
from quantlab.strategies import (
    STRATEGIES,
    AdaptiveTrendEnsemble,
    BollingerMeanReversion,
    BuyAndHold,
    MovingAverageCrossover,
    RSIMeanReversion,
    Strategy,
    TimeSeriesMomentum,
    WalkForwardLogistic,
    create_strategy,
)

TIMING = ("Timing: the condition is evaluated on the close of session t and the order executes at the open of "
          "session t+1, so the fill can gap away from the signal close.")
COMMON_LIMITATIONS = (
    "Backtested and simulated results are not realised results; a profitable history does not establish future "
    "returns.",
    "End-of-day signals: orders execute at the next session's open, and overnight gaps can make the fill much "
    "worse than the signal-day close.",
    "Long-only and unleveraged: a bearish reading means holding cash (avoid or reduce), never a short position.",
)
BENCHMARK_SAME_ASSET = ("Buy & Hold of the same asset over the same period, with the same costs and next-open "
                        "execution: the strategy must beat it after costs to justify its trading.")
NON_DEFAULT_WARNING = ("These parameters differ from the defaults. If they were chosen by comparing alternatives on "
                       "the same data, the backtest overstates performance unless the number of trials is accounted "
                       "for (deflated Sharpe ratio, probability of backtest overfitting).")


@dataclass(frozen=True)
class SignalSpec:
    """Everything a reader needs to test a signal: the agent's mandatory signal fields.

    ``assumptions``, ``references``, ``params`` and ``warmup_sessions`` are additions to the
    documented contract; they come straight from the strategy.
    """

    key: str
    name: str
    hypothesis: str
    inputs: tuple[str, ...]
    calculation: str
    entry: str
    exit: str
    holding_period: str
    invalidation: str
    benchmark: str
    limitations: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    warmup_sessions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return clean_json({
            "key": self.key, "name": self.name, "hypothesis": self.hypothesis, "inputs": list(self.inputs),
            "calculation": self.calculation, "entry": self.entry, "exit": self.exit,
            "holding_period": self.holding_period, "invalidation": self.invalidation, "benchmark": self.benchmark,
            "limitations": list(self.limitations), "assumptions": list(self.assumptions),
            "references": list(self.references), "params": dict(self.params),
            "warmup_sessions": self.warmup_sessions,
        })


def _months(sessions: int) -> str:
    return f"about {sessions / (TRADING_DAYS_PER_YEAR / 12):.1f} months"


def _open_ended(condition: str, horizon: int, what: str) -> str:
    return (f"Not fixed: a position is held until {condition}. The signal looks back {horizon} sessions "
            f"({_months(horizon)}) {what}; the realised holding periods are in a backtest's trade list, and "
            "nothing guarantees they repeat.")


# --------------------------------------------------------------------------- #
# Per-strategy fields. Each builder returns the parameter-specific parts; the
# hypothesis, limitations and assumptions come from the strategy's StrategyDoc.
# --------------------------------------------------------------------------- #


def _buy_and_hold(s: BuyAndHold) -> dict[str, Any]:
    return {
        "inputs": ("daily close (only to know that a session exists)",),
        "calculation": "Target exposure is 100% on every session; there is no indicator.",
        "entry_detail": "Here: the first session of the sample sets the target to 100%.",
        "exit_detail": "",
        "holding_period": "Indefinite: held to the end of the sample (or of the research horizon).",
        "invalidation": ("There is no exit signal, so nothing invalidates a position. The passive thesis is "
                         "invalidated for this asset if it trails cash (or a broad index) over a full market "
                         "cycle, or if a drawdown exceeds the loss the investor can tolerate."),
        "benchmark": ("It is itself the benchmark for the timing strategies; compare it with cash and a broad "
                      "market index over the same period."),
    }


def _ma_crossover(s: MovingAverageCrossover) -> dict[str, Any]:
    p = s.params
    kind, short, long = p["ma_type"], p["short_window"], p["long_window"]
    fast, slow = f"{kind}{short}", f"{kind}{long}"
    weights = ("equal weights" if kind == "SMA" else
               f"exponential weights, alpha = 2 / (N + 1), over a finite kernel of {ind.EMA_SPANS}N sessions")
    return {
        "inputs": ("daily close",),
        "calculation": (f"{fast} and {slow}: {kind} of daily closes over {short} and {long} sessions ({weights}). "
                        f"Target = 100% when {fast} > {slow}, else 0%. No signal for the first "
                        f"{s.warmup_sessions} sessions."),
        "entry_detail": f"Here: long when {fast} closes above {slow}.",
        "exit_detail": f"Here: cash when {fast} closes at or below {slow}.",
        "holding_period": _open_ended(f"{fast} closes at or below {slow}", long, "through its slow average"),
        "invalidation": (f"Per position: the exit crossover ({fast} at or below {slow}). For the hypothesis: no "
                         "reduction in drawdown or improvement in risk-adjusted return versus buy-and-hold after "
                         "costs out of sample, or many short round trips (whipsaws) in sideways markets."),
        "benchmark": BENCHMARK_SAME_ASSET,
    }


def _ts_momentum(s: TimeSeriesMomentum) -> dict[str, Any]:
    p = s.params
    lookback, skip, threshold = p["lookback"], p["skip"], p["threshold"]
    skipped = f", skipping the latest {skip} sessions" if skip else ""
    ret = f"close(t-{skip}) / close(t-{skip + lookback}) - 1" if skip else f"close(t) / close(t-{lookback}) - 1"
    return {
        "inputs": ("daily close",),
        "calculation": (f"Trailing return r = {ret} ({lookback} sessions{skipped}). Target = 100% when "
                        f"r > {threshold:+.2%}, else 0%. No signal for the first {s.warmup_sessions} sessions."),
        "entry_detail": f"Here: long when the {lookback}-session trailing return{skipped} is above {threshold:+.2%}.",
        "exit_detail": f"Here: cash when that return is at or below {threshold:+.2%}.",
        "holding_period": _open_ended(f"the trailing return falls to {threshold:+.2%} or below", lookback + skip,
                                      "for its trailing return"),
        "invalidation": (f"Per position: the {lookback}-session trailing return at or below {threshold:+.2%}. For "
                         "the hypothesis: returns after positive trailing windows no better than after negative "
                         "ones out of sample, or sharp reversals (momentum crashes) dominating the results."),
        "benchmark": BENCHMARK_SAME_ASSET,
    }


def _rsi_reversion(s: RSIMeanReversion) -> dict[str, Any]:
    p = s.params
    window, entry, exit_ = p["window"], p["entry_threshold"], p["exit_threshold"]
    return {
        "inputs": ("daily close",),
        "calculation": (f"Wilder RSI({window}) of daily closes. A long/cash switch: from cash, go long when RSI < "
                        f"{entry:g}; when long, go to cash when RSI > {exit_:g}; otherwise keep the current state. "
                        f"No signal for the first {s.warmup_sessions} sessions."),
        "entry_detail": f"Here: long when RSI({window}) closes below {entry:g} while flat.",
        "exit_detail": f"Here: cash when RSI({window}) closes above {exit_:g} while invested.",
        "holding_period": (f"Not fixed: held until RSI({window}) closes above {exit_:g}. There is no time stop and "
                           "no price stop, so a position can last as long as the asset stays weak."),
        "invalidation": (f"Per position the strategy has no stop: it exits only when RSI({window}) rises above "
                         f"{exit_:g}, and a position can keep losing while RSI stays low. Set a price stop with the "
                         "risk tools before acting on it. For the hypothesis: oversold entries that do not rebound "
                         "more often than chance out of sample, or losses concentrated in sustained declines."),
        "benchmark": BENCHMARK_SAME_ASSET,
    }


def _bollinger(s: BollingerMeanReversion) -> dict[str, Any]:
    p = s.params
    window, k, band = p["window"], p["num_std"], p["exit_band"]
    return {
        "inputs": ("daily close",),
        "calculation": (f"Middle band = SMA{window} of closes; upper/lower = middle +/- {k:g} x the {window}-session "
                        f"population standard deviation. From cash, go long when the close < lower band; when "
                        f"long, go to cash when the close > the {band} band. No signal for the first "
                        f"{s.warmup_sessions} sessions."),
        "entry_detail": f"Here: long when the close is below the {window}-session, {k:g}-sigma lower band while flat.",
        "exit_detail": f"Here: cash when the close rises above the {band} band while invested.",
        "holding_period": (f"Not fixed: held until the close rises above the {band} band. There is no time stop and "
                           "no price stop."),
        "invalidation": (f"Per position the strategy has no stop: it exits only above the {band} band, and the "
                         "price can 'walk the lower band' down for weeks. Set a price stop with the risk tools. For "
                         "the hypothesis: closes below the lower band that do not revert more often than chance out "
                         "of sample."),
        "benchmark": BENCHMARK_SAME_ASSET,
    }


def _ensemble(s: AdaptiveTrendEnsemble) -> dict[str, Any]:
    p = s.params
    horizons = ", ".join(str(h) for h in s.HORIZONS)
    return {
        "inputs": ("daily close",),
        "calculation": (f"Trend score = share of five positive votes: trailing returns over {horizons} sessions "
                        f"and close > SMA200. Volatility scale = min(1, {p['target_vol']:.0%} / EWM volatility of "
                        f"daily returns (span {p['vol_span']}, annualised)). Pullback = 1 when RSI(3) < 20 and close "
                        f"> SMA200. Raw target = min(1, scale x (trend score + {p['pullback_boost']:g} x pullback)), "
                        f"moved in {p['step']:.0%} steps only when a full step away. No signal for the first "
                        f"{s.warmup_sessions} sessions."),
        "entry_detail": (f"Here: exposure rises in {p['step']:.0%} steps as trend votes turn positive and recent "
                         f"volatility falls below {p['target_vol']:.0%} a year."),
        "exit_detail": (f"Here: exposure falls in {p['step']:.0%} steps as votes turn negative or volatility rises; "
                        "0% when every vote is negative."),
        "holding_period": (f"Continuous: exposure is resized in {p['step']:.0%} steps whenever the raw target moves "
                           f"a full step; trend horizons span {min(s.HORIZONS)} to {max(s.HORIZONS)} sessions "
                           f"({_months(min(s.HORIZONS))} to {_months(max(s.HORIZONS))})."),
        "invalidation": ("Per position: exposure reaches 0% when every trend vote is negative. For the hypothesis: "
                         "no improvement in volatility-adjusted return or drawdown versus buy-and-hold after costs "
                         "out of sample, or repeated cuts at lows followed by late re-entry after rebounds."),
        "benchmark": BENCHMARK_SAME_ASSET,
    }


def _ml(s: WalkForwardLogistic) -> dict[str, Any]:
    p = s.params
    features = ", ".join(ml.FEATURE_SHORT[name] for name in ml.FEATURE_NAMES)
    return {
        "inputs": ("daily close (features and labels)",
                   "daily open (training labels: close(t+h) / open(t+1) - 1 > 0, the move a trade could capture)"),
        "calculation": (f"L2-regularised logistic regression (ridge {p['ridge']:g}) of 'higher in {p['horizon']} "
                        f"sessions' on twelve causal features ({features}). Refit every {p['retrain_every']} "
                        f"sessions on the latest {p['train_window']} labelled sessions, using only labels known at "
                        f"the refit (purged). Probability smoothed by an EWM of span {p['smoothing']}; target = "
                        f"clip((P - 0.5) / ({p['full_confidence']:.2f} - 0.5), 0, 1) in 20% steps. No signal for "
                        f"the first {s.warmup_sessions} sessions."),
        "entry_detail": (f"Here: exposure is positive once the smoothed P(up over {p['horizon']} sessions) exceeds "
                         f"50%, and 100% at {p['full_confidence']:.0%} or more."),
        "exit_detail": f"Here: 0% whenever the smoothed P(up over {p['horizon']} sessions) is 50% or lower.",
        "holding_period": (f"Continuous: the forecast horizon is {p['horizon']} sessions ({_months(p['horizon'])}); "
                           "exposure is resized in 20% steps whenever the smoothed probability moves a full step."),
        "invalidation": ("Per position: the smoothed probability at or below 50%. For the hypothesis: an "
                         "out-of-sample hit rate no better than the 'always up' base rate, or an AUC near 0.5."),
        "benchmark": BENCHMARK_SAME_ASSET + " Also compare the hit rate with the 'always up' base rate.",
    }


_BUILDERS: dict[str, Callable[[Any], dict[str, Any]]] = {
    BuyAndHold.key: _buy_and_hold,
    MovingAverageCrossover.key: _ma_crossover,
    TimeSeriesMomentum.key: _ts_momentum,
    RSIMeanReversion.key: _rsi_reversion,
    BollingerMeanReversion.key: _bollinger,
    AdaptiveTrendEnsemble.key: _ensemble,
    WalkForwardLogistic.key: _ml,
}


def _generic(s: Strategy) -> dict[str, Any]:
    """Fallback for a strategy without a dedicated builder: its documentation and parameters only."""
    shown = ", ".join(f"{k} = {v}" for k, v in s.params.items()) or "no parameters"
    return {
        "inputs": ("daily OHLCV",),
        "calculation": f"{s.doc.entry} Parameters: {shown}. No signal for the first {s.warmup_sessions} sessions.",
        "entry_detail": "", "exit_detail": "",
        "holding_period": "Not fixed: a position is held until the exit condition is met.",
        "invalidation": f"Per position: the exit condition. For the hypothesis: {s.doc.failure_modes[0]}",
        "benchmark": BENCHMARK_SAME_ASSET,
    }


def signal_spec(strategy_key: str, params: dict[str, Any] | None = None) -> SignalSpec:
    """The full signal statement for ``strategy_key`` with ``params`` (defaults when omitted).

    Parameters are validated by the strategy itself, so an invalid combination raises
    :class:`quantlab.strategies.StrategyParameterError` instead of describing a signal that
    cannot run. The text is derived from the strategy's documentation and parameters.
    """
    strategy = create_strategy(strategy_key, params)
    doc = strategy.doc
    parts = _BUILDERS.get(strategy.key, _generic)(strategy)
    entry = " ".join(x for x in (doc.entry, parts["entry_detail"], TIMING) if x)
    exit_ = " ".join(x for x in (doc.exit, parts["exit_detail"], TIMING if parts["exit_detail"] else "") if x)
    limitations = (*doc.failure_modes, *COMMON_LIMITATIONS)
    if not strategy.uses_default_parameters():
        limitations = (*limitations, NON_DEFAULT_WARNING)
    return SignalSpec(
        key=strategy.key,
        name=strategy.label,
        hypothesis=doc.summary,
        inputs=parts["inputs"],
        calculation=parts["calculation"],
        entry=entry,
        exit=exit_,
        holding_period=parts["holding_period"],
        invalidation=parts["invalidation"],
        benchmark=parts["benchmark"],
        limitations=limitations,
        assumptions=doc.assumptions,
        references=doc.references,
        params=dict(strategy.params),
        warmup_sessions=strategy.warmup_sessions,
    )


def all_signal_specs() -> list[SignalSpec]:
    """The default-parameter specification of every registered strategy, in registry order."""
    return [signal_spec(key) for key in STRATEGIES]


# --------------------------------------------------------------------------- #
# Current state
# --------------------------------------------------------------------------- #


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _position(target: float | None) -> str | None:
    if target is None:
        return None
    if target >= 1.0 - 1e-12:
        return "long"
    if target <= 1e-12:
        return "cash"
    return "partial"


def current_signal(strategy: Strategy | str, prices: pd.DataFrame) -> dict[str, Any]:
    """Where ``strategy`` stands on the latest session of ``prices`` (validated daily bars).

    Returns a JSON-serialisable dict:

    * ``latest_target`` — target exposure after the latest close (``None`` during warm-up)
      and ``position`` (``long`` / ``partial`` / ``cash``);
    * ``last_change_date``, ``last_change_from``, ``last_change_to``, ``last_change_reason``
      and ``sessions_since_change`` — the latest session on which the target changed
      (warm-up counts as cash, exactly as in the backtester); ``None`` if it never changed;
    * ``conditions`` — the strategy's indicator values on the latest session;
    * ``unexecuted`` — True when the target changed at the latest close: the order it implies
      would execute at the next session's open, which is not in the data yet. Earlier
      changes would already have executed (at the open after the change).

    ``strategy`` may be a :class:`Strategy` or a strategy key (default parameters).
    """
    if isinstance(strategy, str):
        strategy = create_strategy(strategy)
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("no prices: the current signal needs at least one validated session")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must be indexed by session date")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices must be in chronological order with one row per session; validate them first")

    frame = strategy.generate_signals(prices)
    dates = frame.index
    n = len(frame)
    target = frame["target"]
    effective = target.fillna(0.0).to_numpy(dtype="float64")
    previous = np.r_[0.0, effective[:-1]]
    changes = np.flatnonzero(effective != previous)
    latest = _number(target.iloc[-1])
    status = "ok" if latest is not None else "warm_up"

    out: dict[str, Any] = {
        "strategy": strategy.key,
        "label": strategy.label,
        "params": dict(strategy.params),
        "as_of": dates[-1].date(),
        "status": status,
        "latest_target": latest,
        "position": _position(latest),
        "previous_target": _number(target.iloc[-2]) if n > 1 else None,
        "last_change_date": None,
        "last_change_from": None,
        "last_change_to": None,
        "last_change_reason": None,
        "sessions_since_change": None,
        "unexecuted": False,
        "conditions": {c: _number(frame[c].iloc[-1]) for c in frame.columns if c not in ("target", "reason")},
        "warmup_sessions": strategy.warmup_sessions,
        "sessions_available": n,
        "notes": [],
    }
    notes: list[str] = out["notes"]
    if status == "warm_up":
        notes.append(f"No valid signal yet: {strategy.label} gives its first signal on session "
                     f"{strategy.warmup_sessions + 1} and only {n} are available. Treated as cash.")
    if len(changes):
        i = int(changes[-1])
        out.update(
            last_change_date=dates[i].date(),
            last_change_from=float(previous[i]),
            last_change_to=float(effective[i]),
            last_change_reason=str(frame["reason"].iloc[i]) or None,
            sessions_since_change=n - 1 - i,
            unexecuted=i == n - 1,
        )
    elif status == "ok":
        notes.append("The target has not changed in the loaded history: the strategy has been in cash throughout.")
    if out["unexecuted"]:
        notes.append(f"The target changed at the close of {dates[-1].date()}: the implied order would execute at "
                     "the next session's open, which is not in the data yet.")
    elif len(changes):
        notes.append("The latest change would already have executed, at the open of the session after it.")
    if strategy.continuous_exposure:
        notes.append("Partial targets: the backtester trades only when the held weight drifts outside the "
                     "rebalance band, so a small change may not produce an order.")
    history = strategy.history_sessions
    if history is None:
        notes.append("This strategy's signals depend on where the loaded data starts by design (its refit "
                     "schedule is anchored to the first complete session).")
    elif n < history:
        notes.append(f"Only {n} sessions are loaded; with fewer than {history} the current state can depend on "
                     "where the data starts.")
    return clean_json(out)
