"""Technical indicators and a point-in-time technical snapshot for the research agent.

Everything here works on *validated* daily OHLCV (the output of
:func:`quantlab.data.load_market_data`) and follows the rules of
:mod:`quantlab.indicators`:

* **Causal.** The value at session ``t`` uses only sessions at or before ``t``;
  changing later prices never changes it (``tests/test_technicals.py`` perturbs
  the future to check every series function).
* **Honest warm-up.** Until an indicator has the history it needs it is ``NaN``
  (series) or ``None`` with a reason (snapshot values) — never computed from a
  shorter window, never back-filled, never zero.
* **Missing stays missing.** A missing or zero volume is not a number of shares
  traded; prices with gaps are refused rather than filled.

Levels such as moving averages, the 52-week range and support/resistance are on
the dataset's price basis. With total-return adjustment, levels before the most
recent dividend are lower than the prices that actually printed; the snapshot
says so in its notes.

Support and resistance are a *descriptive heuristic* (rolling lowest low and
highest high). They are labelled as such everywhere: a price level has no
guaranteed significance and is not a forecast.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timezone
from typing import Any

import numpy as np
import pandas as pd

from quantlab import indicators as ind
from quantlab.config import TRADING_DAYS_PER_YEAR
from quantlab.data import CacheStatus, DataProviderError, MarketDataset
from quantlab.provenance import Adjustment, Calculated, Fact, Provenance, Timeliness
from quantlab.serialization import clean_json
from quantlab.validation import InputValidationError

# 52 weeks of NYSE sessions.
SESSIONS_52W = 252
# Fraction of a volume window that must hold a reported (positive) volume for a z-score.
VOLUME_MIN_VALID_FRACTION = 0.8
# Fraction of a dollar-volume window that must hold a reported volume (zero counts as reported).
DOLLAR_VOLUME_MIN_VALID_FRACTION = 0.75

TREND_RULE = (
    "uptrend: latest close > 50-session SMA > 200-session SMA; downtrend: latest close < 50-session SMA < "
    "200-session SMA; mixed: any other ordering; undetermined: fewer than 200 sessions, so the 200-session "
    "SMA does not exist yet."
)
SUPPORT_RESISTANCE_METHOD = (
    "Heuristic: support = lowest low and resistance = highest high over the last N sessions, including the "
    "latest. Descriptive only: these levels are not forecasts and price has no obligation to respect them."
)
TREND_LABELS = ("uptrend", "downtrend", "mixed", "undetermined")


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #


def _window(value: int, name: str = "window", minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _check_order(obj: pd.Series | pd.DataFrame, what: str) -> None:
    """Rolling windows count rows, so rows must be sessions in chronological order, once each."""
    if not obj.index.is_monotonic_increasing or not obj.index.is_unique:
        raise ValueError(
            f"{what} must be in chronological order with one row per session; validate it first "
            "(quantlab.validation.clean_price_frame sorts it and resolves duplicate dates)."
        )


def _column(prices: pd.DataFrame, name: str) -> pd.Series:
    if name not in prices.columns:
        raise ValueError(f"prices must contain a '{name}' column")
    return pd.to_numeric(prices[name], errors="coerce").astype("float64")


def _require_complete(series: pd.Series, what: str) -> None:
    if series.isna().any():
        raise ValueError(f"{what} has missing values; prices are never filled in, so validate the data first")


def _finite(value: Any) -> float | None:
    """A plain float, or None for NaN, infinities and missing values (never 0)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------- #
# Indicator series (causal, NaN during warm-up)
# --------------------------------------------------------------------------- #


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving-average convergence/divergence (Appel): ``macd``, ``signal`` and ``histogram`` columns.

    ``macd = EMA(fast) - EMA(slow)`` of closes, ``signal = EMA(signal)`` of ``macd`` and
    ``histogram = macd - signal``. The averages are :func:`quantlab.indicators.ema`, whose
    finite kernel (``EMA_SPANS`` spans) makes every value independent of where the data
    starts; values therefore differ marginally from a recursive EMA seeded at the first
    close. Warm-up: ``macd`` is NaN for the first ``EMA_SPANS * slow - 1`` sessions and
    ``signal``/``histogram`` for ``EMA_SPANS * (slow + signal) - 2``.
    """
    fast, slow, signal = _window(fast, "fast"), _window(slow, "slow", 2), _window(signal, "signal")
    if fast >= slow:
        raise ValueError(f"the fast window must be shorter than the slow window (got {fast} and {slow})")
    _check_order(close, "close")
    close = pd.to_numeric(close, errors="coerce").astype("float64")
    line = ind.ema(close, fast) - ind.ema(close, slow)
    sig = ind.ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig}, index=close.index)


def true_range(prices: pd.DataFrame) -> pd.Series:
    """``max(high - low, |high - close_prev|, |low - close_prev|)``; NaN on the first session.

    The first session has no previous close, so its true range is undefined rather than
    replaced by ``high - low``.
    """
    _check_order(prices, "prices")
    high, low, close = (_column(prices, c) for c in ("high", "low", "close"))
    prev = close.shift(1)
    parts = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1)
    return parts.max(axis=1, skipna=False).rename("true_range")


def atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range with Wilder's smoothing.

    The first value (at session ``window``) is the simple mean of the first ``window``
    true ranges; afterwards ``ATR_t = (ATR_{t-1} * (window - 1) + TR_t) / window``.
    The first ``window`` values are NaN. Like :func:`quantlab.indicators.rsi_wilder`, the
    recursion is seeded at the first loaded session; the seed's weight decays by a factor
    ``(window - 1) / window`` per session. Missing high/low/close values raise ``ValueError``.
    """
    window = _window(window)
    tr = true_range(prices)
    _require_complete(tr.iloc[1:], "high/low/close")
    values = tr.to_numpy()
    out = np.full(len(values), np.nan)
    if len(values) > window:
        avg = float(values[1:window + 1].mean())
        out[window] = avg
        for i in range(window + 1, len(values)):
            avg = (avg * (window - 1) + values[i]) / window
            out[i] = avg
    return pd.Series(out, index=prices.index, name=f"atr_{window}")


def realised_volatility(close: pd.Series, window: int = 63,
                        periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Annualised standard deviation (ddof=1) of the last ``window`` daily close-to-close returns.

    ``window`` returns need ``window + 1`` closes, so the first ``window`` values are NaN.
    """
    window = _window(window, minimum=2)
    _check_order(close, "close")
    rets = pd.to_numeric(close, errors="coerce").astype("float64").pct_change(fill_method=None)
    vol = rets.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(periods_per_year)
    return vol.rename(f"realised_vol_{window}")


def volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series:
    """How unusual each session's volume is: ``(ln v_t - mean) / std`` of ``ln v`` over the previous ``window`` sessions.

    Log volume, because volume is right-skewed and a doubling should count the same at any
    level. The window *excludes* session ``t`` so that a spike does not inflate its own
    baseline. Missing or zero volumes are not trading information: they are NaN, never
    zero. A value needs at least ``VOLUME_MIN_VALID_FRACTION`` of the previous ``window``
    sessions to have a reported volume and a non-zero spread; the first ``window`` values
    are always NaN.
    """
    window = _window(window, minimum=2)
    _check_order(volume, "volume")
    v = pd.to_numeric(volume, errors="coerce").astype("float64")
    log_v = np.log(v.where(v > 0))
    prior = log_v.shift(1).rolling(window, min_periods=max(2, math.ceil(VOLUME_MIN_VALID_FRACTION * window)))
    mean, std = prior.mean(), prior.std(ddof=1)
    z = (log_v - mean) / std.where(std > 1e-12)
    z.iloc[:window] = np.nan
    return z.rename(f"volume_z_{window}")


def average_dollar_volume(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Mean of ``close * volume`` over the last ``window`` sessions (a liquidity measure).

    Sessions with missing volume are left out of the mean, and at least
    ``DOLLAR_VOLUME_MIN_VALID_FRACTION`` of the window must report a volume. A reported
    zero volume is a real zero (nothing traded) and counts. The first ``window - 1``
    values are NaN.
    """
    window = _window(window)
    _check_order(prices, "prices")
    dollars = _column(prices, "close") * _column(prices, "volume")
    out = dollars.rolling(window, min_periods=max(1, math.ceil(DOLLAR_VOLUME_MIN_VALID_FRACTION * window))).mean()
    out.iloc[:window - 1] = np.nan
    return out.rename(f"avg_dollar_volume_{window}")


def overnight_gaps(prices: pd.DataFrame) -> pd.Series:
    """``open_t / close_{t-1} - 1``: the move between one session's close and the next open.

    NaN on the first session. A gap spans whatever lies between two consecutive rows, so
    after a removed or missing session it covers more than one night.
    """
    _check_order(prices, "prices")
    gaps = _column(prices, "open") / _column(prices, "close").shift(1) - 1.0
    return gaps.rename("overnight_gap")


# --------------------------------------------------------------------------- #
# Point-in-time measures (as of the last row)
# --------------------------------------------------------------------------- #


def _bench_close(bench: pd.Series | pd.DataFrame | MarketDataset) -> pd.Series:
    prices = getattr(bench, "prices", bench)
    series = prices["close"] if isinstance(prices, pd.DataFrame) else prices
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _common_closes(close: pd.Series, bench_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Both closes on the sessions where each has a positive price (never forward-filled across calendars)."""
    _check_order(close, "close")
    _check_order(bench_close, "benchmark close")
    a = pd.to_numeric(close, errors="coerce").astype("float64")
    b = pd.to_numeric(bench_close, errors="coerce").astype("float64")
    a, b = a[np.isfinite(a) & (a > 0)], b[np.isfinite(b) & (b > 0)]
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def relative_strength(close: pd.Series, bench_close: pd.Series | pd.DataFrame,
                      lookbacks: tuple[int, ...] = (63, 126, 252)) -> dict[int, float | None]:
    """Asset return minus benchmark return over each lookback, as of the last common session.

    Both series are restricted to the sessions they share (no forward-filling), so
    ``lookback`` counts common sessions. ``+0.05`` means the asset returned five
    percentage points more than the benchmark. ``None`` when fewer than ``lookback + 1``
    common sessions exist.
    """
    a, b = _common_closes(close, _bench_close(bench_close))
    out: dict[int, float | None] = {}
    for lookback in lookbacks:
        lookback = _window(lookback, "lookback")
        if len(a) <= lookback:
            out[lookback] = None
            continue
        asset_ret = a.iloc[-1] / a.iloc[-1 - lookback] - 1.0
        bench_ret = b.iloc[-1] / b.iloc[-1 - lookback] - 1.0
        out[lookback] = _finite(asset_ret - bench_ret)
    return out


def _iso(stamp: Any) -> str | None:
    if stamp is None or stamp is pd.NaT:
        return None
    return pd.Timestamp(stamp).date().isoformat()


def range_52w(prices: pd.DataFrame, window: int = SESSIONS_52W) -> dict[str, Any]:
    """Highest high and lowest low of the last ``window`` sessions (252 = 52 weeks), and the latest close's distance from each.

    ``pct_from_high = close / high - 1`` (zero or negative) and ``pct_from_low = close / low - 1``
    (zero or positive). Uses the high and low columns when present, otherwise closes (``basis``
    says which). With fewer than ``window`` sessions every value is ``None`` with a ``reason``:
    a shorter range would be mislabelled as a 52-week one.
    """
    window = _window(window)
    _check_order(prices, "prices")
    close = _column(prices, "close")
    has_hl = {"high", "low"} <= set(prices.columns)
    high = _column(prices, "high") if has_hl else close
    low = _column(prices, "low") if has_hl else close
    out: dict[str, Any] = {
        "window_sessions": window, "sessions_available": len(prices), "basis": "high/low" if has_hl else "close",
        "close": _finite(close.iloc[-1]) if len(close) else None,
        "high": None, "high_date": None, "low": None, "low_date": None,
        "pct_from_high": None, "pct_from_low": None, "reason": "",
    }
    if len(prices) < window:
        out["reason"] = f"needs {window} sessions, only {len(prices)} available"
        return out
    hi, lo = high.iloc[-window:], low.iloc[-window:]
    if hi.isna().any() or lo.isna().any() or out["close"] is None:
        out["reason"] = "missing prices inside the window"
        return out
    out.update(high=float(hi.max()), high_date=_iso(hi.idxmax()), low=float(lo.min()), low_date=_iso(lo.idxmin()))
    if out["high"] > 0:
        out["pct_from_high"] = out["close"] / out["high"] - 1.0
    if out["low"] > 0:
        out["pct_from_low"] = out["close"] / out["low"] - 1.0
    return out


def support_resistance(prices: pd.DataFrame, windows: tuple[int, ...] = (20, 60)) -> dict[str, Any]:
    """Rolling support (lowest low) and resistance (highest high) for each window — a labelled heuristic.

    Returns ``{"method", "heuristic": True, "close", "levels": {window: {...}}}``; each level
    holds ``support``, ``resistance``, their dates, ``pct_to_support = support / close - 1``
    (zero or negative), ``pct_to_resistance = resistance / close - 1`` (zero or positive) and a
    ``reason`` when the window is longer than the history.
    """
    _check_order(prices, "prices")
    close = _column(prices, "close")
    has_hl = {"high", "low"} <= set(prices.columns)
    high = _column(prices, "high") if has_hl else close
    low = _column(prices, "low") if has_hl else close
    last = _finite(close.iloc[-1]) if len(close) else None
    levels: dict[int, dict[str, Any]] = {}
    for window in windows:
        window = _window(window)
        level: dict[str, Any] = {"support": None, "support_date": None, "resistance": None,
                                 "resistance_date": None, "pct_to_support": None, "pct_to_resistance": None,
                                 "reason": ""}
        hi, lo = high.iloc[-window:], low.iloc[-window:]
        if len(prices) < window:
            level["reason"] = f"needs {window} sessions, only {len(prices)} available"
        elif hi.isna().any() or lo.isna().any() or last is None:
            level["reason"] = "missing prices inside the window"
        else:
            level.update(support=float(lo.min()), support_date=_iso(lo.idxmin()),
                         resistance=float(hi.max()), resistance_date=_iso(hi.idxmax()))
            level["pct_to_support"] = level["support"] / last - 1.0
            level["pct_to_resistance"] = level["resistance"] / last - 1.0
        levels[window] = level
    return {"method": SUPPORT_RESISTANCE_METHOD, "heuristic": True, "basis": "high/low" if has_hl else "close",
            "close": last, "levels": levels}


def classify_trend(close: float | None, sma_50: float | None, sma_200: float | None) -> str:
    """Apply :data:`TREND_RULE`: ``uptrend``, ``downtrend``, ``mixed`` or ``undetermined``."""
    if close is None or sma_50 is None or sma_200 is None:
        return "undetermined"
    if close > sma_50 > sma_200:
        return "uptrend"
    if close < sma_50 < sma_200:
        return "downtrend"
    return "mixed"


# --------------------------------------------------------------------------- #
# Provenance of a loaded dataset
# --------------------------------------------------------------------------- #


_ADJUSTMENTS = {
    "total_return": Adjustment.SPLITS_AND_DIVIDENDS,
    "price_only": Adjustment.SPLITS_ONLY,
    "as_provided": Adjustment.AS_PROVIDED,
}


def dataset_provenance(dataset: MarketDataset, currency: str | None = None) -> Provenance:
    """Describe a :class:`quantlab.data.MarketDataset` as a :class:`Provenance`.

    Market data from a provider is end-of-day (never live); uploads are user-supplied and
    synthetic series are labelled simulated. Validation warnings and errors travel with the
    provenance. The currency is not recorded by the price providers, so it is ``None``
    unless the caller knows it.
    """
    retrieved = dataset.fetched_at
    if retrieved.tzinfo is None:  # load_market_data records UTC; be explicit for hand-built datasets
        retrieved = retrieved.replace(tzinfo=timezone.utc)
    timeliness = {CacheStatus.SYNTHETIC: Timeliness.SIMULATED,
                  CacheStatus.UPLOADED: Timeliness.USER_SUPPLIED}.get(dataset.cache_status, Timeliness.END_OF_DAY)
    adjustment = Adjustment.NOT_APPLICABLE if timeliness is Timeliness.SIMULATED else _ADJUSTMENTS.get(
        str(dataset.adjustment), Adjustment.AS_PROVIDED)
    warnings = tuple(f"{i.severity.value}: {i.code}: {i.message}"
                     for i in (*dataset.quality.errors, *dataset.quality.warnings))
    if dataset.provider_error:
        warnings += (f"provider error: {dataset.provider_error}",)
    first, last = dataset.first_date, dataset.last_date
    return Provenance(
        source=dataset.source_name,
        retrieved_at=retrieved,
        timeliness=timeliness,
        period_start=first.date() if first is not None else None,
        period_end=last.date() if last is not None else None,
        currency=currency,
        units=f"{currency} per share" if currency else "price per share (instrument currency)",
        adjustment=adjustment,
        cache_status=getattr(dataset.cache_status, "value", str(dataset.cache_status)),
        warnings=warnings,
    )


# Data statuses shared by SCAN and ALERTS, from best to worst.
DATA_STATUSES = {
    "ok": "Validated end-of-day data with no quality warnings.",
    "ok_with_warnings": "Validated and usable, but with data-quality warnings (listed); results are flagged.",
    "stale": "The last observation is well before the requested end date (delisted, suspended or lagging "
             "provider); results describe that older session.",
    "stale_cache": "The provider could not be reached; previously cached data was used and may be out of date.",
    "unusable": "The data failed validation (e.g. too few observations); nothing was evaluated.",
    "load_failed": "The data could not be loaded (unknown ticker, provider or network error); nothing was evaluated.",
}


@dataclass(frozen=True)
class LoadOutcome:
    """The result of asking a loader for one ticker: a usable dataset, or the reason there is none."""

    ticker: str
    dataset: MarketDataset | None
    status: str                      # a key of DATA_STATUSES
    reason: str = ""                 # why there is no dataset (failures only)
    warnings: tuple[str, ...] = ()   # data-quality warning codes of a usable dataset

    @property
    def ok(self) -> bool:
        return self.dataset is not None


def dataset_status(dataset: MarketDataset) -> tuple[str, tuple[str, ...]]:
    """``(status, warning codes)`` for a loaded dataset, using the keys of :data:`DATA_STATUSES`."""
    codes = tuple(dict.fromkeys(i.code for i in dataset.quality.warnings))
    if not dataset.is_usable:
        return "unusable", tuple(dict.fromkeys(i.code for i in dataset.quality.errors)) + codes
    if dataset.cache_status is CacheStatus.STALE_CACHE:
        return "stale_cache", codes
    if "stale_data" in codes:
        return "stale", codes
    return ("ok_with_warnings" if codes else "ok"), codes


def load_validated(ticker: str, load_prices: Callable[[str], MarketDataset]) -> LoadOutcome:
    """Call ``load_prices(ticker)`` and classify the result; never raises for a data problem.

    A loader exception or a dataset that failed validation becomes a failure with its
    reason, so callers can list it instead of silently dropping the ticker (or, worse,
    treating it as passing).
    """
    try:
        dataset = load_prices(ticker)
    except DataProviderError as exc:
        return LoadOutcome(ticker, None, "load_failed", f"{exc.kind}: {exc.user_message}")
    except InputValidationError as exc:
        return LoadOutcome(ticker, None, "load_failed", f"invalid input: {exc}")
    except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop a scan; the reason is reported
        return LoadOutcome(ticker, None, "load_failed", f"{type(exc).__name__}: {exc}")
    if not isinstance(dataset, MarketDataset):
        return LoadOutcome(ticker, None, "load_failed",
                           f"the loader returned {type(dataset).__name__}, not a validated MarketDataset")
    status, codes = dataset_status(dataset)
    if status == "unusable":
        reasons = "; ".join(i.message for i in dataset.quality.errors) or "no validated observations"
        return LoadOutcome(ticker, None, "unusable", f"failed validation: {reasons}", codes)
    return LoadOutcome(ticker, dataset, status, "", codes)


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TechnicalSnapshot:
    """The technical state of one ticker as of its latest validated session.

    ``close`` is a fact (read from the data); every entry of ``values`` is a calculated
    metric with its formula, and ``None`` with a reason when it cannot be computed.
    ``trend`` follows :data:`TREND_RULE`. ``notes`` carry the caveats a reader needs.
    """

    ticker: str
    as_of: date
    close: Fact
    values: dict[str, Calculated]
    trend: str
    notes: tuple[str, ...]

    @property
    def provenance(self) -> Provenance:
        return self.close.provenance

    def value(self, name: str) -> float | None:
        """The number for ``name``, or None when it is unavailable (or unknown)."""
        calc = self.values.get(name)
        return None if calc is None else calc.value

    def to_dict(self) -> dict[str, Any]:
        return clean_json({
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat(),
            "close": self.close.to_dict(),
            "values": {name: calc.to_dict() for name, calc in self.values.items()},
            "trend": self.trend,
            "trend_rule": TREND_RULE,
            "notes": list(self.notes),
        })


def _last(series: pd.Series) -> float | None:
    return _finite(series.iloc[-1]) if len(series) else None


def technical_snapshot(ticker: str, prices: pd.DataFrame, provenance: Provenance,
                       bench_prices: pd.DataFrame | pd.Series | MarketDataset | None = None) -> TechnicalSnapshot:
    """Compute the standard technical values for ``ticker`` as of the last row of ``prices``.

    ``prices`` must be validated daily bars (chronological, one row per session, with a
    positive ``close`` on every row; ``open``, ``high``, ``low`` and ``volume`` are used when
    present). ``provenance`` describes them (see :func:`dataset_provenance`).
    ``bench_prices`` enables relative strength. Values that need more history, or a column
    that is missing, are ``None`` with the reason — never estimated from less data.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError(f"no prices for {ticker}: a snapshot needs at least one validated session")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must be indexed by session date")
    _check_order(prices, "prices")
    close = _column(prices, "close")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("close has missing or non-positive values; validate the data first")

    n = len(prices)
    as_of = prices.index[-1].date()  # the exchange-local session date, whatever the index timezone
    last_close = float(close.iloc[-1])
    price_unit = f"{provenance.currency} per share" if provenance.currency else "price (instrument currency)"
    history = Fact("price_history", f"{n} validated daily sessions, {prices.index[0].date()} to {as_of}",
                   provenance, note="Daily OHLCV used for every calculated value")
    close_fact = Fact("close", last_close, provenance, unit=price_unit, note=f"Latest validated daily close ({as_of})")
    values: dict[str, Calculated] = {}
    notes: list[str] = []

    def put(name: str, value: Any, formula: str, unit: str | None, reason: str,
            inputs: tuple[Fact | Calculated, ...] = (history,)) -> Calculated:
        number = _finite(value)
        calc = Calculated(name, number, formula, inputs, unit, reason="" if number is not None else reason)
        values[name] = calc
        return calc

    def need(sessions: int) -> str:
        return f"needs {sessions} sessions, only {n} available"

    def ratio(name: str, numerator: Calculated | Fact, denominator: Calculated, formula: str,
              unit: str = "fraction", minus_one: bool = True) -> None:
        """``numerator / denominator`` (minus one for a distance); None with a reason for a missing or
        non-positive denominator."""
        num, den = numerator.value, denominator.value
        if num is None or den is None:
            missing = denominator if den is None else numerator
            put(name, None, formula, unit, f"{missing.name} unavailable", (numerator, denominator))
        elif den <= 0:
            put(name, None, formula, unit, f"{denominator.name} is not positive", (numerator, denominator))
        else:
            put(name, num / den - (1.0 if minus_one else 0.0), formula, unit, "", (numerator, denominator))

    # Trend: moving averages.
    sma_50 = put("sma_50", _last(ind.sma(close, 50)), "mean of the last 50 daily closes", price_unit, need(50))
    sma_200 = put("sma_200", _last(ind.sma(close, 200)), "mean of the last 200 daily closes", price_unit, need(200))
    ratio("pct_vs_sma_50", close_fact, sma_50, "close / sma_50 - 1")
    ratio("pct_vs_sma_200", close_fact, sma_200, "close / sma_200 - 1")

    # Momentum oscillators.
    put("rsi_14", _last(ind.rsi_wilder(close, 14)),
        "Wilder RSI(14) of daily closes: 100 - 100 / (1 + avg gain / avg loss), Wilder-smoothed",
        "index points (0-100)", need(15))
    m = macd(close)
    macd_reason = need(ind.EMA_SPANS * 26)
    signal_reason = need(ind.EMA_SPANS * (26 + 9) - 1)
    put("macd", _last(m["macd"]), "EMA(12) - EMA(26) of daily closes (finite-kernel EMA)", price_unit, macd_reason)
    put("macd_signal", _last(m["signal"]), "EMA(9) of the MACD line", price_unit, signal_reason)
    put("macd_hist", _last(m["histogram"]), "MACD line - signal line", price_unit, signal_reason)

    # Volatility.
    has_hl = {"high", "low"} <= set(prices.columns)
    if has_hl:
        try:
            atr_value, atr_reason = _last(atr(prices, 14)), need(15)
        except ValueError as exc:
            atr_value, atr_reason = None, str(exc)
    else:
        atr_value, atr_reason = None, "high/low columns not available"
    atr_14 = put("atr_14", atr_value, "Wilder-smoothed mean of the 14 latest true ranges", price_unit, atr_reason)
    ratio("atr_pct", atr_14, Calculated("close", last_close, "latest close", (close_fact,), price_unit),
          "atr_14 / close", "fraction of close", minus_one=False)
    put("realised_vol_63", _last(realised_volatility(close, 63)),
        f"standard deviation of the last 63 daily close-to-close returns x sqrt({TRADING_DAYS_PER_YEAR})",
        "annualised fraction", need(64))
    if "open" in prices.columns:
        put("overnight_gap", _last(overnight_gaps(prices)), "latest open / previous close - 1", "fraction",
            need(2) if n < 2 else "open missing")
    else:
        put("overnight_gap", None, "latest open / previous close - 1", "fraction", "open column not available")

    # Relative strength.
    rs_formula = "asset return - benchmark return over the last {} common sessions"
    bench_basis = dataset_provenance(bench_prices).adjustment if isinstance(bench_prices, MarketDataset) else None
    mismatch = (bench_basis is not None and bench_basis is not provenance.adjustment
                and Adjustment.NOT_APPLICABLE not in (bench_basis, provenance.adjustment))
    if bench_prices is None:
        for lb in (63, 126, 252):
            put(f"rs_{lb}", None, rs_formula.format(lb), "fraction", "no benchmark prices supplied")
        notes.append("Relative strength not computed: no benchmark prices were supplied.")
    elif mismatch:
        reason = (f"the asset ({provenance.adjustment.value}) and the benchmark ({bench_basis.value}) are on different "
                  "price bases, so their returns are not comparable")
        for lb in (63, 126, 252):
            put(f"rs_{lb}", None, rs_formula.format(lb), "fraction", reason)
        notes.append(f"Relative strength not computed: {reason}.")
    else:
        bench_close = _bench_close(bench_prices)
        rs = relative_strength(close, bench_close, (63, 126, 252))
        common, _ = _common_closes(close, bench_close)
        for lb, value in rs.items():
            put(f"rs_{lb}", value, rs_formula.format(lb), "fraction",
                f"needs {lb + 1} sessions shared with the benchmark, only {len(common)} available")
        if len(common) and common.index[-1] < prices.index[-1]:
            notes.append(f"Relative strength is as of {common.index[-1].date()}, the last session shared with the "
                         f"benchmark (the asset's latest session is {as_of}).")

    # Volume and liquidity.
    if "volume" in prices.columns:
        volume = _column(prices, "volume")
        latest_volume = _finite(volume.iloc[-1])
        if n <= 50:
            z_reason = need(51)
        elif latest_volume is None or latest_volume <= 0:
            z_reason = "the latest session's volume is missing or zero"
        else:
            z_reason = (f"fewer than {VOLUME_MIN_VALID_FRACTION:.0%} of the previous 50 sessions report a volume, "
                        "or volume did not vary")
        put("volume_z", _last(volume_zscore(volume, 50)),
            "(ln volume - mean) / std of ln volume over the previous 50 sessions", "standard deviations", z_reason)
        adv_reason = need(20) if n < 20 else "too few of the last 20 sessions report a volume"
        put("avg_dollar_volume_20", _last(average_dollar_volume(prices, 20)),
            "mean of close x volume over the last 20 sessions", f"{price_unit} x shares per session", adv_reason)
    else:
        for name in ("volume_z", "avg_dollar_volume_20"):
            put(name, None, "requires daily volume", None, "volume column not available")

    # 52-week range.
    rng = range_52w(prices)
    basis = rng["basis"]
    high_52w = put("high_52w", rng["high"], f"highest {'high' if basis == 'high/low' else 'close'} of the last 252 "
                   "sessions", price_unit, rng["reason"])
    low_52w = put("low_52w", rng["low"], f"lowest {'low' if basis == 'high/low' else 'close'} of the last 252 "
                  "sessions", price_unit, rng["reason"])
    ratio("pct_from_high_52w", close_fact, high_52w, "close / high_52w - 1")
    ratio("pct_from_low_52w", close_fact, low_52w, "close / low_52w - 1")

    # Support and resistance (heuristic).
    sr = support_resistance(prices, (20, 60))
    for window, level in sr["levels"].items():
        put(f"support_{window}", level["support"], f"heuristic: lowest low of the last {window} sessions",
            price_unit, level["reason"])
        put(f"resistance_{window}", level["resistance"], f"heuristic: highest high of the last {window} sessions",
            price_unit, level["reason"])
    notes.append("Support and resistance are a descriptive heuristic (rolling lowest low / highest high), "
                 "not forecasts.")

    trend = classify_trend(last_close, sma_50.value, sma_200.value)
    if trend == "undetermined":
        notes.append(f"Trend undetermined: {n} sessions, fewer than the 200 needed for the 200-session average.")
    if n < SESSIONS_52W:
        notes.append(f"Only {n} sessions of history: year-long values (52-week range, 252-session relative "
                     "strength) are unavailable.")
    if provenance.adjustment is Adjustment.SPLITS_AND_DIVIDENDS:
        notes.append("Prices are split- and dividend-adjusted (total return): levels before the latest dividend "
                     "are lower than the prices that printed.")
    if provenance.timeliness is Timeliness.SIMULATED:
        notes.append("SIMULATED data: these are not market prices.")
    notes.append("End-of-day data: values describe the close of the latest validated session, not live prices.")
    return TechnicalSnapshot(ticker=ticker, as_of=as_of, close=close_fact, values=values, trend=trend,
                             notes=tuple(notes))


def snapshot_from_dataset(dataset: MarketDataset,
                          bench_prices: pd.DataFrame | pd.Series | MarketDataset | None = None,
                          currency: str | None = None) -> TechnicalSnapshot:
    """:func:`technical_snapshot` of a loaded dataset with its provenance. Refuses unusable datasets."""
    if not dataset.is_usable:
        reasons = "; ".join(i.message for i in dataset.quality.errors) or "no validated observations"
        raise ValueError(f"{dataset.ticker}: the data failed validation ({reasons})")
    return technical_snapshot(dataset.ticker, dataset.prices, dataset_provenance(dataset, currency), bench_prices)

