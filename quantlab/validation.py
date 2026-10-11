"""Input validation and market-data quality checks.

Principles applied here:

* Never invent a price. Rows that cannot be used are *removed and reported*,
  never forward-filled or interpolated.
* Never conceal a problem. Every removal or suspicious observation becomes a
  ``QualityIssue`` that the UI shows before a backtest is run.
* A weekday without data is not automatically "missing": it is compared with an
  approximate NYSE holiday calendar first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

import numpy as np
import pandas as pd

from quantlab.calendar import session_gaps

PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = (*PRICE_COLUMNS, "volume")

# Fewer usable observations than this and a backtest is refused outright.
MIN_OBSERVATIONS = 30

# A close-to-close move larger than this is flagged for a corporate-action check.
EXTREME_RETURN_THRESHOLD = 0.25

# Consecutive observations further apart than this many calendar days are flagged.
MAX_CALENDAR_GAP_DAYS = 7

_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")

_COLUMN_ALIASES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted close": "adj_close",
    "adjusted_close": "adj_close",
    "volume": "volume",
    "dividends": "dividends",
    "stock splits": "stock_splits",
    "stock_splits": "stock_splits",
    "splits": "stock_splits",
    "capital gains": "capital_gains",
}


class InputValidationError(ValueError):
    """Raised when a user-supplied input (ticker, dates, parameters) is invalid."""


class Severity(str, Enum):
    ERROR = "error"  # blocks a backtest
    WARNING = "warning"  # data was altered (rows removed) or looks suspicious
    INFO = "info"  # context the researcher should know about


@dataclass(frozen=True)
class QualityIssue:
    severity: Severity
    code: str
    message: str
    count: int = 0
    dates: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        shown = ", ".join(self.dates[:10]) + (" …" if len(self.dates) > 10 else "")
        return {
            "Severity": self.severity.value.upper(),
            "Check": self.code,
            "Detail": self.message,
            "Rows affected": self.count,
            "Example dates": shown,
        }


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    raw_rows: int = 0
    clean_rows: int = 0
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    removed: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["date", "reason"])
    )

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        dates: pd.Index | list | None = None,
        count: int | None = None,
    ) -> None:
        formatted: tuple[str, ...] = ()
        if dates is not None and len(dates) > 0:
            formatted = tuple(pd.DatetimeIndex(dates).strftime("%Y-%m-%d"))
        self.issues.append(
            QualityIssue(
                severity=severity,
                code=code,
                message=message,
                count=count if count is not None else len(formatted),
                dates=formatted,
            )
        )

    def _of(self, severity: Severity) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is severity]

    @property
    def errors(self) -> list[QualityIssue]:
        return self._of(Severity.ERROR)

    @property
    def warnings(self) -> list[QualityIssue]:
        return self._of(Severity.WARNING)

    @property
    def infos(self) -> list[QualityIssue]:
        return self._of(Severity.INFO)

    @property
    def is_blocking(self) -> bool:
        return bool(self.errors)

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["Severity", "Check", "Detail", "Rows affected", "Example dates"]
            )
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        ranked = sorted(self.issues, key=lambda i: order[i.severity])
        return pd.DataFrame([i.as_dict() for i in ranked])


# --------------------------------------------------------------------------- #
# User inputs
# --------------------------------------------------------------------------- #


def validate_ticker(raw: str) -> str:
    """Normalise and validate a ticker symbol (e.g. ``spy`` -> ``SPY``)."""
    if raw is None:
        raise InputValidationError("Enter a ticker symbol, for example SPY or AAPL.")
    ticker = str(raw).strip().upper()
    if not ticker:
        raise InputValidationError("Enter a ticker symbol, for example SPY or AAPL.")
    if not _TICKER_PATTERN.match(ticker):
        raise InputValidationError(
            f"'{raw}' is not a valid ticker. Use letters, digits, '.', '-', '^' or '=' "
            "(maximum 15 characters), e.g. SPY, BRK-B or ^GSPC."
        )
    return ticker


def validate_date_range(
    start: date | datetime | str,
    end: date | datetime | str,
    today: date | None = None,
    min_calendar_days: int = 45,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Check a requested sample period and return normalised timestamps."""
    try:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
    except (ValueError, TypeError) as exc:
        raise InputValidationError(f"Could not parse the date range: {exc}") from exc
    if start_ts.tzinfo is not None or end_ts.tzinfo is not None:
        start_ts, end_ts = start_ts.tz_localize(None), end_ts.tz_localize(None)
    today_ts = pd.Timestamp(today or date.today()).normalize()
    if start_ts < pd.Timestamp("1970-01-01"):
        raise InputValidationError("Start date must be on or after 1970-01-01.")
    if end_ts > today_ts:
        raise InputValidationError(
            f"End date {end_ts.date()} is in the future; historical data ends today at the latest."
        )
    if end_ts <= start_ts:
        raise InputValidationError("End date must be after the start date.")
    if (end_ts - start_ts).days < min_calendar_days:
        raise InputValidationError(
            f"The sample period must span at least {min_calendar_days} calendar days "
            "so that indicators and metrics have enough observations."
        )
    return start_ts, end_ts


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #


def standardise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common provider column spellings onto lower-case canonical names."""
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance.download returns (field, ticker) columns for a single ticker.
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
    renamed = {}
    for col in frame.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            renamed[col] = _COLUMN_ALIASES[key]
    out = frame.rename(columns=renamed)
    # Drop exact duplicate column names (keep first) after renaming.
    return out.loc[:, ~out.columns.duplicated()]


def to_naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Convert any date-like index to timezone-naive, midnight-normalised dates."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        # Daily bars are stamped at local midnight; keep the local calendar date.
        idx = idx.tz_localize(None)
    return idx.normalize()


def clean_price_frame(
    frame: pd.DataFrame, report: QualityReport | None = None
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate a daily OHLCV frame and return ``(clean_frame, report)``.

    The returned frame has a sorted, unique, timezone-naive ``DatetimeIndex``
    named ``date`` and float columns ``open, high, low, close, volume``. Any
    extra columns (e.g. dividends) are carried through untouched.
    """
    report = report or QualityReport()
    report.raw_rows = 0 if frame is None else len(frame)
    removed: list[tuple[pd.Timestamp, str]] = []

    if frame is None or frame.empty:
        report.add(Severity.ERROR, "empty", "The data source returned no observations.", count=0)
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS)), report

    df = standardise_columns(frame)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add(
            Severity.ERROR,
            "malformed_columns",
            f"Required column(s) missing: {', '.join(missing_cols)}. "
            f"Columns received: {', '.join(map(str, frame.columns))}.",
            count=0,
        )
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS)), report

    # --- index ------------------------------------------------------------
    try:
        idx = to_naive_dates(df.index)
    except (ValueError, TypeError) as exc:
        report.add(Severity.ERROR, "malformed_dates", f"Dates could not be parsed: {exc}.")
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS)), report
    df = df.copy()
    df.index = idx
    df.index.name = "date"
    nat_mask = df.index.isna()
    if nat_mask.any():
        report.add(
            Severity.WARNING,
            "malformed_dates",
            f"{int(nat_mask.sum())} row(s) had unreadable dates and were removed.",
            count=int(nat_mask.sum()),
        )
        df = df.loc[~nat_mask]

    # --- numeric coercion -------------------------------------------------
    numeric_cols = [c for c in df.columns if c in (*REQUIRED_COLUMNS, "adj_close")]
    for col in numeric_cols:
        before_na = df[col].isna()
        coerced = pd.to_numeric(df[col], errors="coerce").astype("float64")
        coerced = coerced.where(np.isfinite(coerced))
        newly_bad = coerced.isna() & ~before_na
        if newly_bad.any():
            report.add(
                Severity.WARNING,
                "malformed_values",
                f"Column '{col}' contained non-numeric or infinite values that were treated as missing.",
                dates=df.index[newly_bad],
            )
        df[col] = coerced

    # --- ordering ---------------------------------------------------------
    if not df.index.is_monotonic_increasing:
        report.add(
            Severity.INFO,
            "unsorted",
            "Observations were not in chronological order and have been sorted.",
            count=len(df),
        )
        df = df.sort_index(kind="mergesort")

    # --- duplicate timestamps ---------------------------------------------
    dup_mask = df.index.duplicated(keep=False)
    if dup_mask.any():
        dup_dates = df.index[dup_mask].unique()
        dups = df.loc[dup_mask]
        conflicting = [
            d for d in dup_dates if len(dups.loc[[d], list(REQUIRED_COLUMNS)].drop_duplicates()) > 1
        ]
        identical = [d for d in dup_dates if d not in set(conflicting)]
        if identical:
            report.add(
                Severity.WARNING,
                "duplicate_timestamps",
                "Identical duplicate rows were found; one copy of each was kept.",
                dates=pd.DatetimeIndex(identical),
            )
        if conflicting:
            report.add(
                Severity.WARNING,
                "conflicting_duplicates",
                "The same date appeared with different prices. Because the correct value "
                "cannot be determined, every copy of these dates was removed.",
                dates=pd.DatetimeIndex(conflicting),
            )
            removed += [(d, "conflicting duplicate timestamp") for d in conflicting]
            df = df.loc[~df.index.isin(conflicting)]
        df = df.loc[~df.index.duplicated(keep="first")]

    # --- missing prices ---------------------------------------------------
    missing_price = df[list(PRICE_COLUMNS)].isna().any(axis=1)
    if missing_price.any():
        report.add(
            Severity.WARNING,
            "missing_prices",
            "Rows with a missing open, high, low or close were removed (prices are never "
            "filled in). No trade can execute on these dates.",
            dates=df.index[missing_price],
        )
        removed += [(d, "missing price field") for d in df.index[missing_price]]
        df = df.loc[~missing_price]

    # --- non-positive prices ----------------------------------------------
    non_positive = (df[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    if non_positive.any():
        report.add(
            Severity.WARNING,
            "non_positive_prices",
            "Rows with a zero or negative price were removed as invalid observations.",
            dates=df.index[non_positive],
        )
        removed += [(d, "non-positive price") for d in df.index[non_positive]]
        df = df.loc[~non_positive]

    # --- volume -----------------------------------------------------------
    negative_volume = df["volume"] < 0
    if negative_volume.any():
        report.add(
            Severity.WARNING,
            "negative_volume",
            "Negative volume values are impossible; they were set to missing (prices kept).",
            dates=df.index[negative_volume],
        )
        df.loc[negative_volume, "volume"] = np.nan
    zero_volume = df["volume"] == 0
    if zero_volume.any():
        report.add(
            Severity.INFO,
            "zero_volume",
            "Sessions with zero reported volume. Prices are kept, but fills on these "
            "dates may not have been achievable in practice.",
            dates=df.index[zero_volume],
        )
    if df["volume"].isna().any():
        report.add(
            Severity.INFO,
            "missing_volume",
            "Volume is missing on some sessions. Volume is informational only and is not "
            "used by the execution model.",
            dates=df.index[df["volume"].isna()],
        )

    # --- OHLC consistency (kept, but flagged) -------------------------------
    if not df.empty:
        tol = 1e-6
        hi_bad = df["high"] < df[["open", "close", "low"]].max(axis=1) * (1 - tol)
        lo_bad = df["low"] > df[["open", "close", "high"]].min(axis=1) * (1 + tol)
        inconsistent = hi_bad | lo_bad
        if inconsistent.any():
            report.add(
                Severity.WARNING,
                "ohlc_inconsistent",
                "High/low do not bracket open/close on these sessions (often a rounding "
                "artefact of price adjustment). Rows kept; executions use open and close only.",
                dates=df.index[inconsistent],
            )

    # --- extreme moves ------------------------------------------------------
    if len(df) > 1:
        rets = df["close"].pct_change()
        extreme = rets.abs() > EXTREME_RETURN_THRESHOLD
        if extreme.any():
            report.add(
                Severity.WARNING,
                "extreme_returns",
                f"Close-to-close moves larger than {EXTREME_RETURN_THRESHOLD:.0%}. Verify these are "
                "genuine and not an unadjusted split or data error.",
                dates=df.index[extreme],
            )

        gaps = df.index.to_series().diff().dt.days
        big_gaps = gaps > MAX_CALENDAR_GAP_DAYS
        if big_gaps.any():
            report.add(
                Severity.WARNING,
                "calendar_gaps",
                f"Consecutive observations more than {MAX_CALENDAR_GAP_DAYS} calendar days apart "
                "(possible trading suspension or missing data). Dates shown are the session after the gap.",
                dates=df.index[big_gaps],
            )

        missing_sessions, unexpected = session_gaps(df.index)
        if len(missing_sessions):
            report.add(
                Severity.INFO,
                "missing_sessions",
                "Weekdays that are not NYSE holidays (approximate calendar) but have no "
                "observation. These may be genuine data gaps or unusual closures; nothing "
                "was filled in.",
                dates=missing_sessions,
            )
        if len(unexpected):
            report.add(
                Severity.WARNING,
                "unexpected_sessions",
                "Observations dated on weekends or NYSE holidays (approximate calendar). "
                "Check the data source's date convention.",
                dates=unexpected,
            )

    check_sampling_frequency(df.index, report)

    if len(df) < MIN_OBSERVATIONS:
        report.add(
            Severity.ERROR,
            "insufficient_observations",
            f"Only {len(df)} usable observations; at least {MIN_OBSERVATIONS} are required.",
            count=len(df),
        )

    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype("float64")

    report.clean_rows = len(df)
    report.first_date = df.index.min() if len(df) else None
    report.last_date = df.index.max() if len(df) else None
    if removed:
        report.removed = pd.DataFrame(
            {"date": [d for d, _ in removed], "reason": [r for _, r in removed]}
        ).sort_values("date", kind="mergesort", ignore_index=True)
    return df, report


def check_sampling_frequency(index: pd.DatetimeIndex, report: QualityReport) -> None:
    """Refuse data that is not daily bars on a five-day (exchange) calendar.

    The engine fills at the next session's open and every annualised metric
    assumes 252 sessions a year. Weekly or monthly bars would turn "next open"
    into next week's open, and a seven-day calendar (crypto, some FX feeds)
    would overstate annualised volatility and Sharpe by about sqrt(365/252).
    """
    if len(index) < 10:
        return
    span_days = (index[-1] - index[0]).days
    median_gap = float(index.to_series().diff().dt.days.median())
    if median_gap >= 5:
        report.add(
            Severity.ERROR,
            "not_daily",
            f"Observations are a median {median_gap:.0f} calendar days apart, which looks like weekly or "
            "monthly data. Quant Lab Pro simulates daily bars (fills at the next session's open, 252 "
            "sessions a year), so lower-frequency data would give wrong timing and annualisation.",
            count=len(index),
        )
        return
    if span_days >= 60:
        per_year = len(index) / span_days * 365.25
        weekend = int((index.dayofweek >= 5).sum())
        if per_year > 300 and weekend / len(index) > 0.2:
            report.add(
                Severity.ERROR,
                "seven_day_calendar",
                f"About {per_year:.0f} observations a year, {weekend / len(index):.0%} of them on weekends: a "
                "seven-day calendar (crypto, some FX feeds). Metrics here annualise with 252 sessions a year "
                "and the NYSE calendar, so they would be misstated. Use daily exchange-traded data.",
                count=weekend,
            )


def check_staleness(
    last_date: pd.Timestamp | None,
    requested_end: pd.Timestamp,
    report: QualityReport,
    today: date | None = None,
    tolerance_sessions: int = 5,
) -> None:
    """Flag data whose last observation is well before the requested end date."""
    if last_date is None:
        return
    horizon = min(pd.Timestamp(requested_end), pd.Timestamp(today or date.today())).normalize()
    lag = len(pd.bdate_range(last_date + pd.Timedelta(days=1), horizon))
    if lag > tolerance_sessions:
        report.add(
            Severity.WARNING,
            "stale_data",
            f"The last observation ({last_date.date()}) is {lag} weekdays before the requested "
            f"end ({horizon.date()}). The security may be delisted, suspended, or the provider "
            "may be lagging.",
            count=lag,
        )
