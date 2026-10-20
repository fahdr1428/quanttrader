"""SCAN mode: screen a user-defined universe with documented, testable criteria.

A scan answers one narrow question — *which of these tickers meet these
conditions on their latest validated session?* — and is a starting point for
research, never a recommendation. Its rules:

* **Every criterion is written down.** :meth:`ScanCriteria.describe` states each
  active criterion in plain English, and the values it uses come from
  :func:`quantlab.technicals.technical_snapshot`, so a scan and a research
  report never disagree about what "RSI" or "uptrend" means.
* **Missing is not passing.** A criterion whose value cannot be computed (too
  little history, no benchmark, no volume) counts as failed, and the reason is
  listed in the ``unavailable`` column.
* **Nothing disappears.** A ticker that cannot be loaded, or whose data fails
  validation, is listed in ``failures`` with the reason; it is neither dropped
  silently nor counted as passing.
* **Survivorship bias is disclosed.** The universe is supplied by the user and is
  usually today's constituents of something, which excludes everything that was
  delisted or dropped. Every result carries that caveat.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from quantlab.data import MarketDataset, SyntheticProvider
from quantlab.provenance import Provenance
from quantlab.serialization import clean_json
from quantlab.technicals import (
    TREND_RULE,
    TechnicalSnapshot,
    dataset_provenance,
    load_validated,
    technical_snapshot,
)
from quantlab.validation import InputValidationError, validate_ticker

# Order in which criteria are described and reported.
CRITERIA_ORDER = ("min_price", "min_avg_dollar_volume", "min_history_sessions", "trend", "rsi_below",
                  "rsi_above", "min_rs_126", "max_atr_pct")
TREND_CHOICES = ("uptrend", "downtrend")
# Which snapshot value each criterion reads.
_CRITERION_VALUE = {
    "min_price": "close", "min_avg_dollar_volume": "avg_dollar_volume_20", "min_history_sessions": "history_sessions",
    "trend": "trend", "rsi_below": "rsi_14", "rsi_above": "rsi_14", "min_rs_126": "rs_126", "max_atr_pct": "atr_pct",
}
VALUE_COLUMNS = ("close", "avg_dollar_volume_20", "history_sessions", "trend", "rsi_14", "rs_126", "atr_pct")

SURVIVORSHIP_NOTE = (
    "Survivorship bias: the universe is user-supplied. If it is today's list of index constituents, it leaves out "
    "every company that was delisted, acquired or dropped, so anything learned from it describes survivors only."
)
SCREEN_NOTE = ("A scan is a screen for further research, not a recommendation: meeting every criterion says nothing "
               "about future returns.")
TIMING_NOTE = "Values are end-of-day, as of each ticker's latest validated session — not live prices."


@dataclass(frozen=True)
class ScanCriteria:
    """Conditions a ticker must meet on its latest validated session; ``None`` switches a criterion off.

    The defaults screen out penny stocks, thinly traded names and short histories:
    a close of at least 5, at least 5 million a day of average dollar volume over 20
    sessions, and a year (252 sessions) of history. Prices and dollar volumes are in
    the instrument's own currency. See :meth:`describe` for every criterion in words.
    """

    min_price: float | None = 5.0
    min_avg_dollar_volume: float | None = 5e6
    min_history_sessions: int = 252
    trend: str | None = None            # "uptrend" | "downtrend" | None
    rsi_below: float | None = None
    rsi_above: float | None = None
    min_rs_126: float | None = None     # relative strength vs benchmark over 126 sessions
    max_atr_pct: float | None = None

    def __post_init__(self) -> None:
        def check(name: str, low: float | None = None, high: float | None = None, positive: bool = False) -> None:
            value = getattr(self, name)
            if value is None:
                return
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number or None, got {value!r}")
            if positive and value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            if low is not None and value < low:
                raise ValueError(f"{name} must be at least {low:g}, got {value!r}")
            if high is not None and value > high:
                raise ValueError(f"{name} must be at most {high:g}, got {value!r}")

        check("min_price", low=0.0)
        check("min_avg_dollar_volume", low=0.0)
        check("rsi_below", 0.0, 100.0)
        check("rsi_above", 0.0, 100.0)
        check("min_rs_126")
        check("max_atr_pct", positive=True)
        sessions = self.min_history_sessions
        if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
            raise ValueError(f"min_history_sessions must be a whole number >= 1, got {sessions!r}")
        if self.trend is not None and self.trend not in TREND_CHOICES:
            raise ValueError(f"trend must be one of {', '.join(TREND_CHOICES)} or None, got {self.trend!r}")
        if self.rsi_below is not None and self.rsi_above is not None and self.rsi_above >= self.rsi_below:
            raise ValueError(f"no RSI can be above {self.rsi_above:g} and below {self.rsi_below:g} at once")

    def active(self) -> tuple[str, ...]:
        """Names of the criteria this scan applies, in :data:`CRITERIA_ORDER`."""
        return tuple(name for name in CRITERIA_ORDER if getattr(self, name) is not None)

    def describe(self) -> list[str]:
        """Every active criterion, and the rules for combining them, in plain English."""
        lines = []
        if self.min_price is not None:
            lines.append(f"Price: the latest close is at least {self.min_price:,.2f} (instrument currency).")
        if self.min_avg_dollar_volume is not None:
            lines.append(f"Liquidity: average daily dollar volume (close x volume) over the last 20 sessions is at "
                         f"least {self.min_avg_dollar_volume:,.0f} (instrument currency).")
        lines.append(f"History: at least {self.min_history_sessions} validated daily sessions "
                     f"(about {self.min_history_sessions / 252:.1f} years).")
        if self.trend is not None:
            rule = TREND_RULE.split("; ")[0 if self.trend == "uptrend" else 1].split(": ", 1)[1]
            lines.append(f"Trend: '{self.trend}', meaning {rule}.")
        if self.rsi_below is not None:
            lines.append(f"Momentum: Wilder RSI(14) of daily closes is below {self.rsi_below:g}.")
        if self.rsi_above is not None:
            lines.append(f"Momentum: Wilder RSI(14) of daily closes is above {self.rsi_above:g}.")
        if self.min_rs_126 is not None:
            lines.append(f"Relative strength: the 126-session (about six months) return minus the benchmark's return "
                         f"over the same sessions is at least {self.min_rs_126:+.2%} (percentage points).")
        if self.max_atr_pct is not None:
            lines.append(f"Volatility: the 14-session average true range (Wilder) is at most {self.max_atr_pct:.2%} "
                         "of the latest close.")
        lines.append("A ticker passes only if it meets every criterion above. A criterion whose value cannot be "
                     "computed counts as failed, never as passed.")
        lines.append("Tickers whose data cannot be loaded or fails validation are listed as failures with the reason.")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "active": list(self.active()), "description": self.describe()}


@dataclass(frozen=True)
class ScanResult:
    """One row per ticker that loaded (``rows``), the tickers that did not (``failures``), and caveats (``notes``).

    ``rows`` columns: ``ticker``, ``as_of``, the criterion values (``close``,
    ``avg_dollar_volume_20``, ``history_sessions``, ``trend``, ``rsi_14``, ``rs_126``,
    ``atr_pct``; ``None`` when unavailable), ``pass_<criterion>`` for each active
    criterion, ``passed``, ``unavailable`` (criteria that could not be evaluated, with
    reasons), ``data_status`` and ``data_warnings`` (see
    :data:`quantlab.technicals.DATA_STATUSES`), ``source`` and ``cache_status``.
    """

    rows: pd.DataFrame
    criteria: ScanCriteria
    failures: dict[str, str]
    notes: tuple[str, ...]
    provenance: dict[str, Provenance] = field(default_factory=dict)

    @property
    def passed(self) -> list[str]:
        """Tickers that met every criterion, in universe order."""
        if self.rows.empty:
            return []
        return [str(t) for t in self.rows.loc[self.rows["passed"].astype(bool), "ticker"]]

    def to_dict(self) -> dict[str, Any]:
        return clean_json({
            "criteria": self.criteria.to_dict(),
            "rows": self.rows.to_dict(orient="records"),
            "passed": self.passed,
            "failures": dict(self.failures),
            "notes": list(self.notes),
            "provenance": {t: p.to_dict() for t, p in self.provenance.items()},
        })


def _columns(criteria: ScanCriteria) -> list[str]:
    return ["ticker", "as_of", *VALUE_COLUMNS, *(f"pass_{c}" for c in criteria.active()),
            "passed", "unavailable", "data_status", "data_warnings", "source", "cache_status"]


def _evaluate(name: str, criteria: ScanCriteria, value: Any) -> bool:
    threshold = getattr(criteria, name)
    if name == "trend":
        return value == threshold
    if name == "rsi_below":
        return value < threshold
    if name == "rsi_above":
        return value > threshold
    if name == "max_atr_pct":
        return value <= threshold
    return value >= threshold  # min_price, min_avg_dollar_volume, min_history_sessions, min_rs_126


def _row(snapshot: TechnicalSnapshot, dataset: MarketDataset, criteria: ScanCriteria, status: str,
         warnings: tuple[str, ...], has_bench: bool) -> dict[str, Any]:
    values: dict[str, Any] = {name: snapshot.value(name) for name in VALUE_COLUMNS if name not in
                              ("close", "history_sessions", "trend")}
    values.update(close=snapshot.close.value, history_sessions=dataset.n_obs, trend=snapshot.trend)
    row: dict[str, Any] = {"ticker": dataset.ticker, "as_of": snapshot.as_of.isoformat(), **values}
    unavailable: list[str] = []
    passed = True
    for name in criteria.active():
        key = _CRITERION_VALUE[name]
        value = values[key]
        if value is None or value == "undetermined":
            if key == "trend":
                reason = "fewer than 200 sessions, so the 200-session average does not exist"
            elif key == "rs_126" and not has_bench:
                reason = "no benchmark supplied"
            else:
                reason = snapshot.values[key].reason or "not available"
            unavailable.append(f"{name}: {reason}")
            ok = False
        else:
            ok = bool(_evaluate(name, criteria, value))
        row[f"pass_{name}"] = ok
        passed = passed and ok
    row.update(passed=passed, unavailable="; ".join(unavailable), data_status=status,
               data_warnings=", ".join(warnings), source=dataset.source_name,
               cache_status=getattr(dataset.cache_status, "value", str(dataset.cache_status)))
    return row


def _benchmark(bench_prices: Any, notes: list[str]) -> pd.DataFrame | pd.Series | None:
    if bench_prices is None:
        return None
    if isinstance(bench_prices, MarketDataset):
        if not bench_prices.is_usable:
            notes.append(f"The benchmark data ({bench_prices.ticker}) failed validation, so relative strength was "
                         "not computed.")
            return None
        return bench_prices.prices
    if isinstance(bench_prices, (pd.DataFrame, pd.Series)):
        if bench_prices.empty:
            notes.append("The benchmark prices are empty, so relative strength was not computed.")
            return None
        return bench_prices
    raise TypeError("bench_prices must be a price DataFrame, a close Series, a MarketDataset or None")


def scan_universe(universe: Sequence[str], load_prices: Callable[[str], MarketDataset],
                  criteria: ScanCriteria | None = None,
                  bench_prices: pd.DataFrame | pd.Series | MarketDataset | None = None) -> ScanResult:
    """Apply ``criteria`` (defaults when omitted) to every ticker in ``universe``.

    ``load_prices(ticker)`` must return a validated :class:`quantlab.data.MarketDataset`
    (e.g. a wrapper around :func:`quantlab.data.load_market_data`); it is called once per
    distinct ticker. Tickers are normalised (``spy`` -> ``SPY``); duplicates are scanned
    once and noted. ``bench_prices`` enables the relative-strength criterion. Rows follow
    the universe order.
    """
    if isinstance(universe, str):
        raise TypeError("universe must be a list of tickers, not a single string")
    criteria = criteria or ScanCriteria()
    if not universe:
        raise ValueError("the universe is empty: supply at least one ticker to scan")

    notes: list[str] = [SURVIVORSHIP_NOTE, SCREEN_NOTE, TIMING_NOTE]
    bench = _benchmark(bench_prices, notes)
    if criteria.min_rs_126 is not None and bench is None:
        notes.append("The relative-strength criterion needs benchmark prices and none were usable, so no ticker can "
                     "pass it.")
    if criteria.trend == "downtrend":
        notes.append("Long-only system: tickers in a downtrend are candidates to avoid or reduce, never to short.")

    failures: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Provenance] = {}
    seen: set[str] = set()
    duplicates: list[str] = []
    simulated: list[str] = []
    for raw in universe:
        try:
            ticker = validate_ticker(raw)
        except InputValidationError as exc:
            failures[str(raw)] = f"invalid ticker: {exc}"
            continue
        if ticker in seen:
            duplicates.append(ticker)
            continue
        seen.add(ticker)
        outcome = load_validated(ticker, load_prices)
        if outcome.dataset is None:
            failures[ticker] = outcome.reason
            continue
        dataset = outcome.dataset
        try:
            prov = dataset_provenance(dataset)
            snapshot = technical_snapshot(ticker, dataset.prices, prov, bench)
        except Exception as exc:  # noqa: BLE001 - one bad series must not stop the scan; the reason is reported
            failures[ticker] = f"analysis failed: {type(exc).__name__}: {exc}"
            continue
        provenance[ticker] = prov
        if dataset.provider_key == SyntheticProvider.key:
            simulated.append(ticker)
        rows.append(_row(snapshot, dataset, criteria, outcome.status, outcome.warnings, bench is not None))
    if duplicates:
        notes.append(f"Listed more than once and scanned once: {', '.join(dict.fromkeys(duplicates))}.")

    frame = pd.DataFrame(rows, columns=_columns(criteria))
    if failures:
        notes.append(f"{len(failures)} ticker(s) could not be scanned and are listed in failures with the reason; "
                     "they are neither passing nor failing.")
    if rows:
        dates = sorted({r["as_of"] for r in rows})
        if len(dates) > 1:
            notes.append(f"Latest sessions differ across tickers ({dates[0]} to {dates[-1]}); each row is as of its "
                         "own latest session.")
        flagged = [r["ticker"] for r in rows if r["data_status"] != "ok"]
        if flagged:
            notes.append(f"Data-quality flags on {', '.join(flagged)} (see data_status and data_warnings).")
        if not frame["passed"].any():
            notes.append("No ticker met every criterion: no qualifying candidates in this universe.")
    if simulated:
        notes.append(f"SIMULATED data (not market prices) for: {', '.join(simulated)}.")
    return ScanResult(rows=frame, criteria=criteria, failures=failures, notes=tuple(notes), provenance=provenance)
