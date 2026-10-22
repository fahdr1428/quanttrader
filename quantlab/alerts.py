"""ALERTS mode: check user-defined conditions against the latest validated end-of-day data.

Alerts are evaluated on demand. Nothing here runs in the background, sends a
notification or places an order: :func:`evaluate_alerts` returns one
:class:`AlertEvent` per rule describing whether its condition holds on the
latest validated session, with the measured value and the data status.

Rules are validated when they are created (unknown kinds, missing or
nonsensical parameters raise ``ValueError``), and every kind is documented in
plain English in :data:`ALERT_KINDS`. A rule whose data cannot be loaded, fails
validation or is too short for the condition is reported with
``triggered=False`` and a ``data_status`` that says why — a data failure is
never reported as a condition that holds or does not hold.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from quantlab import indicators as ind
from quantlab.data import MarketDataset
from quantlab.provenance import Provenance
from quantlab.serialization import clean_json
from quantlab.signals import current_signal
from quantlab.strategies import STRATEGIES, create_strategy
from quantlab.technicals import DATA_STATUSES, LoadOutcome, dataset_provenance, load_validated, volume_zscore
from quantlab.validation import validate_ticker

REQUIRED = object()  # marks a parameter without a default


@dataclass(frozen=True)
class AlertKind:
    """One kind of alert: its parameters (with defaults) and what it checks, in plain English."""

    key: str
    summary: str
    params: dict[str, Any]
    value: str  # what AlertEvent.value reports for this kind


ALERT_KINDS: dict[str, AlertKind] = {kind.key: kind for kind in (
    AlertKind("price_above", "The latest daily close is above `level`.", {"level": REQUIRED}, "latest close"),
    AlertKind("price_below", "The latest daily close is below `level`.", {"level": REQUIRED}, "latest close"),
    AlertKind("rsi_below", "Wilder RSI(`window`) of daily closes is below `level` (0-100) on the latest session.",
              {"level": REQUIRED, "window": 14}, "latest RSI"),
    AlertKind("rsi_above", "Wilder RSI(`window`) of daily closes is above `level` (0-100) on the latest session.",
              {"level": REQUIRED, "window": 14}, "latest RSI"),
    AlertKind("sma_cross_up",
              "The `fast`-session simple moving average of closes crossed from at-or-below to above the `slow`-session "
              "one within the last `within` sessions (1 = on the latest session) and is still above it. `fast` = 1 "
              "means the close itself (price crossing above its average).",
              {"fast": 50, "slow": 200, "within": 1}, "fast SMA / slow SMA - 1 on the latest session"),
    AlertKind("sma_cross_down",
              "The `fast`-session simple moving average of closes crossed from at-or-above to below the `slow`-session "
              "one within the last `within` sessions (1 = on the latest session) and is still below it. `fast` = 1 "
              "means the close itself.",
              {"fast": 50, "slow": 200, "within": 1}, "fast SMA / slow SMA - 1 on the latest session"),
    AlertKind("volume_z_above",
              "The latest session's volume is more than `level` standard deviations above normal, measured on log "
              "volume against the previous `window` sessions.",
              {"level": REQUIRED, "window": 50}, "latest volume z-score"),
    AlertKind("new_52w_high",
              "The latest session's high is above the highest high of the previous `window` - 1 sessions (a new "
              "52-week high with the default `window` of 252 sessions).",
              {"window": 252}, "latest high"),
    AlertKind("new_52w_low",
              "The latest session's low is below the lowest low of the previous `window` - 1 sessions (a new 52-week "
              "low with the default `window` of 252 sessions).",
              {"window": 252}, "latest low"),
    AlertKind("signal_change",
              "The target exposure of strategy `strategy` (with `strategy_params`) changed at the latest close, so "
              "a backtest would trade at the next session's open.",
              {"strategy": REQUIRED, "strategy_params": {}}, "new target exposure (0 to 1)"),
)}


def describe_alert_kinds() -> list[str]:
    """Every alert kind in plain English, with its parameters and defaults."""
    lines = []
    for kind in ALERT_KINDS.values():
        params = ", ".join(f"{name} (required)" if default is REQUIRED else f"{name} = {default!r}"
                           for name, default in kind.params.items())
        lines.append(f"{kind.key}: {kind.summary} Parameters: {params}. Value reported: {kind.value}.")
    return lines


def _number(params: dict[str, Any], name: str, low: float | None = None, high: float | None = None,
            positive: bool = False) -> float:
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    if (low is not None and value <= low) or (high is not None and value >= high):
        raise ValueError(f"{name} must be strictly between {low:g} and {high:g}, got {value!r}")
    return float(value)


def _integer(params: dict[str, Any], name: str, minimum: int) -> int:
    value = params[name]
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value != int(value) or value < minimum):
        raise ValueError(f"{name} must be a whole number >= {minimum}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class AlertRule:
    """A condition on one ticker. ``params`` are validated and completed with defaults on creation."""

    ticker: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ALERT_KINDS:
            raise ValueError(f"unknown alert kind {self.kind!r}; choose one of {', '.join(ALERT_KINDS)}")
        spec = ALERT_KINDS[self.kind]
        given = dict(self.params or {})
        unknown = set(given) - set(spec.params)
        if unknown:
            raise ValueError(f"{self.kind} does not take parameter(s) {', '.join(sorted(unknown))}; "
                             f"it takes {', '.join(spec.params)}")
        missing = [name for name, default in spec.params.items() if default is REQUIRED and name not in given]
        if missing:
            raise ValueError(f"{self.kind} needs parameter(s) {', '.join(missing)}")
        params = {name: given.get(name, default) for name, default in spec.params.items()}

        if self.kind in ("price_above", "price_below"):
            params["level"] = _number(params, "level", positive=True)
        elif self.kind in ("rsi_below", "rsi_above"):
            params["level"] = _number(params, "level", 0.0, 100.0)
            params["window"] = _integer(params, "window", 2)
        elif self.kind in ("sma_cross_up", "sma_cross_down"):
            params["fast"] = _integer(params, "fast", 1)
            params["slow"] = _integer(params, "slow", 2)
            params["within"] = _integer(params, "within", 1)
            if params["fast"] >= params["slow"]:
                raise ValueError(f"fast ({params['fast']}) must be shorter than slow ({params['slow']})")
        elif self.kind == "volume_z_above":
            params["level"] = _number(params, "level")
            params["window"] = _integer(params, "window", 2)
        elif self.kind in ("new_52w_high", "new_52w_low"):
            params["window"] = _integer(params, "window", 2)
        elif self.kind == "signal_change":
            if params["strategy"] not in STRATEGIES:
                raise ValueError(f"unknown strategy {params['strategy']!r}; choose one of {', '.join(STRATEGIES)}")
            if not isinstance(params["strategy_params"], dict):
                raise ValueError("strategy_params must be a dict")
            # Validated (and completed with defaults) by the strategy itself.
            params["strategy_params"] = dict(create_strategy(params["strategy"], params["strategy_params"]).params)
        object.__setattr__(self, "ticker", validate_ticker(self.ticker))
        object.__setattr__(self, "params", params)

    def describe(self) -> str:
        """The rule in plain English."""
        p = self.params
        text = {
            "price_above": lambda: f"the latest daily close is above {p['level']:g}",
            "price_below": lambda: f"the latest daily close is below {p['level']:g}",
            "rsi_below": lambda: f"RSI({p['window']}) is below {p['level']:g}",
            "rsi_above": lambda: f"RSI({p['window']}) is above {p['level']:g}",
            "sma_cross_up": lambda: f"{_sma_name(p['fast'])} crosses above SMA{p['slow']}{_within(p['within'])}",
            "sma_cross_down": lambda: f"{_sma_name(p['fast'])} crosses below SMA{p['slow']}{_within(p['within'])}",
            "volume_z_above": lambda: (f"volume is more than {p['level']:g} standard deviations above its "
                                       f"{p['window']}-session norm"),
            "new_52w_high": lambda: f"the latest high is above the previous {p['window'] - 1} sessions' highs",
            "new_52w_low": lambda: f"the latest low is below the previous {p['window'] - 1} sessions' lows",
            "signal_change": lambda: (f"the {create_strategy(p['strategy'], p['strategy_params']).label} target "
                                      "changes at the latest close"),
        }[self.kind]()
        return f"{self.ticker}: alert when {text}."

    def to_dict(self) -> dict[str, Any]:
        return clean_json({"ticker": self.ticker, "kind": self.kind, "params": dict(self.params),
                           "description": self.describe()})


def _sma_name(window: int) -> str:
    return "the close" if window == 1 else f"SMA{window}"


def _within(within: int) -> str:
    return " on the latest session" if within == 1 else f" within the last {within} sessions"


@dataclass(frozen=True)
class AlertEvent:
    """The outcome of one rule on the latest validated session.

    ``triggered`` is True only when the data was usable and the condition holds.
    ``as_of`` is the session evaluated (``None`` when no data could be loaded); ``value``
    is the measured quantity (see :attr:`AlertKind.value`), ``None`` when unavailable;
    ``data_status`` is one of :data:`quantlab.technicals.DATA_STATUSES` or
    ``insufficient_history`` / ``evaluation_failed``.
    """

    rule: AlertRule
    triggered: bool
    as_of: date | None
    value: float | None
    detail: str
    data_status: str
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return clean_json({
            "rule": self.rule.to_dict(), "triggered": self.triggered,
            "as_of": self.as_of.isoformat() if self.as_of else None, "value": self.value, "detail": self.detail,
            "data_status": self.data_status,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        })


ALERT_DATA_STATUSES = {
    **DATA_STATUSES,
    "insufficient_history": "The data loaded but is too short for this condition (e.g. a 200-session average "
                            "with 150 sessions); the condition was not evaluated.",
    "evaluation_failed": "The condition could not be computed from the data (the reason is in the detail).",
}


class _InsufficientHistory(Exception):
    """The data is valid but too short for the rule."""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _price(value: float) -> str:
    """Two decimals for ordinary prices; significant digits for sub-unit prices so they do not round to 0.00."""
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"


def _fmt_date(stamp: pd.Timestamp) -> str:
    return pd.Timestamp(stamp).date().isoformat()


def _evaluate(rule: AlertRule, prices: pd.DataFrame) -> tuple[bool, float | None, str]:
    """``(triggered, value, detail)`` for a rule on validated prices; raises _InsufficientHistory."""
    p, kind = rule.params, rule.kind
    close = prices["close"].astype("float64")
    n, last = len(prices), _fmt_date(prices.index[-1])

    if kind in ("price_above", "price_below"):
        value = float(close.iloc[-1])
        hit = value > p["level"] if kind == "price_above" else value < p["level"]
        side = "above" if kind == "price_above" else "below"
        return hit, value, f"Close {_price(value)} on {last} is {'' if hit else 'not '}{side} {p['level']:g}."

    if kind in ("rsi_below", "rsi_above"):
        value = _finite(ind.rsi_wilder(close, p["window"]).iloc[-1])
        if value is None:
            raise _InsufficientHistory(f"RSI({p['window']}) needs {p['window'] + 1} sessions, only {n} available")
        hit = value < p["level"] if kind == "rsi_below" else value > p["level"]
        side = "below" if kind == "rsi_below" else "above"
        return hit, value, f"RSI({p['window']}) is {value:.1f} on {last}: {'' if hit else 'not '}{side} {p['level']:g}."

    if kind in ("sma_cross_up", "sma_cross_down"):
        fast_w, slow_w, within = p["fast"], p["slow"], p["within"]
        if n < slow_w + within:
            raise _InsufficientHistory(f"needs {slow_w + within} sessions (SMA{slow_w} on the last {within + 1} "
                                       f"sessions), only {n} available")
        fast, slow = ind.sma(close, fast_w).to_numpy(), ind.sma(close, slow_w).to_numpy()
        up = kind == "sma_cross_up"

        def crossed(i: int) -> bool:
            if up:
                return fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]
            return fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]

        crosses = [i for i in range(n - within, n) if crossed(i)]
        still = fast[-1] > slow[-1] if up else fast[-1] < slow[-1]
        hit = bool(crosses) and still
        value = float(fast[-1] / slow[-1] - 1.0)
        names = f"{_sma_name(fast_w)} ({_price(fast[-1])}) vs SMA{slow_w} ({_price(slow[-1])})"
        side = "above" if up else "below"
        if hit:
            return True, value, f"Crossed {side} on {_fmt_date(prices.index[crosses[-1]])}; on {last}: {names}."
        if crosses:
            return False, value, (f"Crossed {side} on {_fmt_date(prices.index[crosses[-1]])} but no longer {side} "
                                  f"on {last}: {names}.")
        return False, value, f"No cross {side}{_within(within)}; on {last}: {names}."

    if kind == "volume_z_above":
        if "volume" not in prices.columns:
            raise _InsufficientHistory("the data has no volume column")
        volume = prices["volume"].astype("float64")
        value = _finite(volume_zscore(volume, p["window"]).iloc[-1])
        if value is None:
            if n <= p["window"]:
                raise _InsufficientHistory(f"needs {p['window'] + 1} sessions, only {n} available")
            latest = _finite(volume.iloc[-1])
            if latest is None or latest <= 0:
                raise _InsufficientHistory(f"the volume on {last} is missing or zero")
            raise _InsufficientHistory(f"too few of the previous {p['window']} sessions report a volume, or volume "
                                       "did not vary")
        hit = value > p["level"]
        return hit, value, f"Volume z-score {value:+.2f} on {last}: {'' if hit else 'not '}above {p['level']:g}."

    if kind in ("new_52w_high", "new_52w_low"):
        window = p["window"]
        if n < window:
            raise _InsufficientHistory(f"needs {window} sessions, only {n} available")
        high = kind == "new_52w_high"
        column = ("high" if high else "low") if {"high", "low"} <= set(prices.columns) else "close"
        series = prices[column].astype("float64")
        basis = ""
        if "adjustment_factor" in prices.columns:
            # Total-return data lowers every past price by later dividends: compare the prices that printed.
            factor = prices["adjustment_factor"].astype("float64")
            if factor.notna().all() and (factor > 0).all():
                series = series / factor
                basis = " (printed, unadjusted prices)"
        prior = series.iloc[-window:-1]
        value = float(series.iloc[-1])
        extreme = float(prior.max() if high else prior.min())
        when = _fmt_date(prior.idxmax() if high else prior.idxmin())
        hit = value > extreme if high else value < extreme
        side = "above the highest" if high else "below the lowest"
        return hit, value, (f"{column.capitalize()} {_price(value)} on {last} is {'' if hit else 'not '}{side} "
                            f"{column} of the previous {window - 1} sessions ({_price(extreme)} on {when}){basis}.")

    if kind == "signal_change":
        strategy = create_strategy(p["strategy"], p["strategy_params"])
        state = current_signal(strategy, prices)
        if state["status"] != "ok":
            raise _InsufficientHistory(f"{strategy.label} gives its first signal on session "
                                       f"{strategy.warmup_sessions + 1}; only {n} available")
        value = state["latest_target"]
        if state["unexecuted"]:
            trade = ("a backtest trades at the next open only if the held weight is outside the rebalance band"
                     if strategy.continuous_exposure else "a backtest would trade at the next open")
            return True, value, (f"{strategy.label}: target changed from {state['last_change_from']:.0%} to "
                                 f"{state['last_change_to']:.0%} at the close of {last} "
                                 f"({state['last_change_reason']}); {trade}.")
        since = state["last_change_date"] or "the start of the data"
        return False, value, f"{strategy.label}: no change at the close of {last}; target {value:.0%} since {since}."

    raise AssertionError(f"unhandled alert kind {kind}")  # pragma: no cover - kinds are validated on creation


def _status_suffix(outcome: LoadOutcome) -> str:
    if outcome.status == "ok":
        return ""
    codes = f" ({', '.join(outcome.warnings)})" if outcome.warnings else ""
    return f" Data status {outcome.status}{codes}: {DATA_STATUSES[outcome.status]}"


def evaluate_alerts(rules: Iterable[AlertRule], load_prices: Callable[[str], MarketDataset]) -> list[AlertEvent]:
    """One :class:`AlertEvent` per rule, in the order given.

    ``load_prices(ticker)`` must return a validated :class:`quantlab.data.MarketDataset`;
    it is called once per distinct ticker. Load failures, validation failures and
    too-short histories produce ``triggered=False`` with the reason in ``detail`` and
    ``data_status``. Usable data with quality warnings, or stale data, is evaluated and
    flagged in ``data_status``.
    """
    rules = list(rules)
    for rule in rules:
        if not isinstance(rule, AlertRule):
            raise TypeError(f"expected AlertRule objects, got {type(rule).__name__}")
    outcomes: dict[str, LoadOutcome] = {}
    events: list[AlertEvent] = []
    for rule in rules:
        if rule.ticker not in outcomes:
            outcomes[rule.ticker] = load_validated(rule.ticker, load_prices)
        outcome = outcomes[rule.ticker]
        dataset = outcome.dataset
        if dataset is None:
            events.append(AlertEvent(rule, False, None, None, f"Not evaluated: {outcome.reason}", outcome.status))
            continue
        provenance = dataset_provenance(dataset)
        as_of = dataset.prices.index[-1].date()
        try:
            triggered, value, detail = _evaluate(rule, dataset.prices)
        except _InsufficientHistory as exc:
            events.append(AlertEvent(rule, False, as_of, None, f"Not evaluated: {exc}.", "insufficient_history",
                                     provenance))
            continue
        except Exception as exc:  # noqa: BLE001 - one rule must not stop the others; the reason is reported
            events.append(AlertEvent(rule, False, as_of, None, f"Not evaluated: {type(exc).__name__}: {exc}",
                                     "evaluation_failed", provenance))
            continue
        events.append(AlertEvent(rule, bool(triggered), as_of, _finite(value), detail + _status_suffix(outcome),
                                 outcome.status, provenance))
    return events


def triggered_events(events: Iterable[AlertEvent]) -> list[AlertEvent]:
    """The events whose condition holds (data failures are never among them)."""
    return [e for e in events if e.triggered]

