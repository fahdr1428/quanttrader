"""Historical market-data access.

The provider layer is deliberately small and replaceable: a provider only has to
return a daily OHLCV frame for ``[start, end]``. Everything else — price
adjustment, validation, caching, staleness checks and fingerprinting — is
provider-independent and lives in :func:`load_market_data`.

Price adjustment (how splits and dividends are handled)
-------------------------------------------------------
Yahoo Finance reports two closing prices:

* ``Close`` — adjusted for stock splits only. Dividends are *not* reflected, so
  a strategy using it earns price return only.
* ``Adj Close`` — adjusted for splits *and* dividends (dividends are treated as
  reinvested on the ex-date). It is a total-return series.

In ``TOTAL_RETURN`` mode (the default) we scale open, high, low and close by the
daily factor ``Adj Close / Close``. Every price in the backtest is then on the
same total-return basis, so a buy at the open and a valuation at the close are
consistent. The trade-off: adjusted prices before the most recent dividend are
not the prices that actually printed, and they change every time the company
pays a new dividend. Percentage returns are unaffected because all fees and
slippage in this engine are proportional to notional, but absolute share prices
and share counts in the trade ledger are on the adjusted scale.

In ``PRICE_ONLY`` mode the split-adjusted prices are used unchanged, so
dividends are ignored. This understates the return of income-paying assets and
is offered for comparison and for CSV files that only contain raw prices.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.validation import (
    PRICE_COLUMNS,
    QualityReport,
    Severity,
    check_staleness,
    clean_price_frame,
    standardise_columns,
    to_naive_dates,
    validate_date_range,
    validate_ticker,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_MAX_AGE = timedelta(hours=24)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DataProviderError(Exception):
    """A provider could not deliver data. ``user_message`` is safe to display."""

    kind = "provider_error"

    def __init__(self, user_message: str, detail: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class TickerNotFoundError(DataProviderError):
    kind = "not_found"


class RateLimitError(DataProviderError):
    kind = "rate_limited"


class ProviderConnectionError(DataProviderError):
    kind = "connection"


class MalformedDataError(DataProviderError):
    kind = "malformed"


# --------------------------------------------------------------------------- #
# Adjustment conventions
# --------------------------------------------------------------------------- #


class AdjustmentMode(str, Enum):
    TOTAL_RETURN = "total_return"
    PRICE_ONLY = "price_only"


ADJUSTMENT_LABELS = {
    AdjustmentMode.TOTAL_RETURN: "Total return (split- and dividend-adjusted)",
    AdjustmentMode.PRICE_ONLY: "Price return (split-adjusted, dividends excluded)",
    "as_provided": "As provided in the uploaded file (adjustment unknown)",
}


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class DataProvider(ABC):
    """Interface every market-data source implements."""

    key: str = "base"
    name: str = "Base provider"
    description: str = ""

    @abstractmethod
    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return raw daily bars for ``[start, end]`` (both inclusive).

        The frame must be indexed by date and contain at least open, high, low,
        close and volume (any capitalisation). Raise a :class:`DataProviderError`
        subclass on failure — never return fabricated rows.
        """


class YahooFinanceProvider(DataProvider):
    key = "yahoo"
    name = "Yahoo Finance (via yfinance)"
    description = (
        "Free end-of-day data scraped from Yahoo Finance by the open-source yfinance "
        "library. Not an official API: it can be rate-limited, revised or temporarily "
        "unavailable. Historical data only — nothing here is live or real-time."
    )

    def __init__(
        self,
        timeout: int = 20,
        ticker_factory: Callable[[str], Any] | None = None,
        retries: int = 2,
        backoff_seconds: float = 1.5,
        sleep: Callable[[float], None] | None = None,
    ):
        self.timeout = timeout
        self._ticker_factory = ticker_factory
        # Transient failures (rate limits, dropped connections) are retried with
        # exponential backoff: 1.5 s, then 3 s by default. Unknown tickers and
        # malformed responses are not retried — retrying cannot fix them.
        self.retries = max(0, int(retries))
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep or time.sleep

    def _make_ticker(self, symbol: str) -> Any:
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf  # imported lazily so tests and CSV use need no network stack

        # Make yfinance raise its errors instead of logging them and returning an
        # empty frame; otherwise a network failure is indistinguishable from an
        # unknown ticker. (Replaces the deprecated ``history(raise_errors=True)``.)
        try:
            yf.config.debug.hide_exceptions = False
        except AttributeError:  # pragma: no cover - older yfinance without the config API
            pass
        return yf.Ticker(symbol)

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        for attempt in range(self.retries + 1):
            try:
                return self._fetch_once(ticker, start, end)
            except (RateLimitError, ProviderConnectionError) as exc:
                if attempt == self.retries:
                    raise
                logger.warning("Transient Yahoo Finance error (%s); retrying.", exc.kind)
                self._sleep(self.backoff_seconds * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def _fetch_once(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        try:
            handle = self._make_ticker(ticker)
            # yfinance treats ``end`` as exclusive, so request one extra day.
            frame = handle.history(
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                actions=True,
                timeout=self.timeout,
            )
        except DataProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors below
            raise _map_yahoo_exception(ticker, exc) from exc

        if frame is None or len(frame) == 0:
            raise TickerNotFoundError(
                f"Yahoo Finance returned no data for '{ticker}' between {start.date()} and "
                f"{end.date()}. Check the symbol and the date range (the security may not "
                "have traded in that period, or may be delisted)."
            )
        frame = standardise_columns(frame)
        if "close" not in frame.columns:
            raise MalformedDataError(
                f"Yahoo Finance returned an unexpected format for '{ticker}' "
                f"(columns: {', '.join(map(str, frame.columns))})."
            )
        return frame


def _map_yahoo_exception(ticker: str, exc: Exception) -> DataProviderError:
    name = type(exc).__name__
    text = str(exc)
    lowered = f"{name} {text}".lower()
    if "ratelimit" in lowered.replace(" ", "") or "too many requests" in lowered or "429" in lowered:
        return RateLimitError(
            "Yahoo Finance is rate-limiting requests. Wait a few minutes and try again; "
            "previously cached data (if any) is used in the meantime.",
            detail=f"{name}: {text}",
        )
    if any(
        token in lowered
        for token in ("pricesmissing", "tzmissing", "tickermissing", "delisted", "no timezone", "not found", "404")
    ):
        return TickerNotFoundError(
            f"No price history was found for '{ticker}'. Check the symbol (Yahoo uses e.g. "
            "BRK-B rather than BRK.B) and the date range.",
            detail=f"{name}: {text}",
        )
    if any(
        token in lowered
        for token in ("connection", "timeout", "timed out", "resolve", "network", "ssl", "proxy", "curl", "403")
    ):
        return ProviderConnectionError(
            "Could not reach Yahoo Finance (network error or blocked connection). Check your "
            "internet connection, or upload a CSV file instead.",
            detail=f"{name}: {text}",
        )
    return DataProviderError(
        f"Yahoo Finance could not provide data for '{ticker}'.", detail=f"{name}: {text}"
    )


class CSVProvider(DataProvider):
    """Daily bars from a user-supplied CSV file (offline research or another vendor).

    Required columns (any capitalisation): a date column (``Date``, ``Datetime`` or
    ``Timestamp``, or the first column), ``Open``, ``High``, ``Low``, ``Close``,
    ``Volume``. Optional: ``Adj Close``, ``Dividends``, ``Stock Splits``.
    See :func:`parse_price_csv` for the dialects that are recognised.
    """

    key = "csv"
    name = "Uploaded CSV file"
    description = (
        "Prices supplied by the user. Quant Lab Pro validates the file but cannot verify "
        "where the numbers came from or how they were adjusted."
    )

    def __init__(self, content: bytes, filename: str = "upload.csv"):
        self.content = content
        self.filename = filename
        self.parse_notes: list[str] = []

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        parsed = parse_price_csv_detailed(self.content)
        self.parse_notes = parsed.notes
        frame = parsed.frame
        # Rows whose date could not be read are kept (as NaT) so that validation
        # reports and removes them visibly instead of their vanishing here.
        mask = ((frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))) | frame.index.isna()
        sliced = frame.loc[mask]
        if sliced.index.notna().sum() == 0:
            raise TickerNotFoundError(
                f"The uploaded file has no observations between {pd.Timestamp(start).date()} "
                f"and {pd.Timestamp(end).date()} (file covers {frame.index.min().date()} to "
                f"{frame.index.max().date()})."
            )
        return sliced


# Canonical columns that hold numbers (thousands separators and decimal commas are normalised).
_NUMERIC_CSV_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits",
                        "capital_gains")
_DATE_COLUMN_NAMES = {"date", "datetime", "timestamp", "time", "day"}
# A trailing UTC offset after a clock time (yfinance exports "2024-01-02 00:00:00-05:00").
# Daily bars are labelled with the exchange's local date, so the offset is dropped and the
# local wall-clock date kept; converting to UTC would shift Asian sessions to the previous day.
_TZ_SUFFIX = re.compile(r"(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(?:Z|UTC|GMT|[+-]\d{2}:?\d{2})$", re.IGNORECASE)
_TIME_SUFFIX = re.compile(r"[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$")
# Every format is tried; when several read the file equally well (e.g. day/month vs
# month/day with no day above 12) the choice is resolved by plausibility or refused.
_CSV_DATE_FORMATS = (
    ("YYYY-MM-DD", "%Y-%m-%d"),
    ("MM/DD/YYYY", "%m/%d/%Y"),
    ("DD/MM/YYYY", "%d/%m/%Y"),
    ("DD.MM.YYYY", "%d.%m.%Y"),
    ("DD-MM-YYYY", "%d-%m-%Y"),
    ("MM-DD-YYYY", "%m-%d-%Y"),
    ("YYYY/MM/DD", "%Y/%m/%d"),
    ("YYYYMMDD", "%Y%m%d"),
    ("Mon DD, YYYY", "%b %d, %Y"),
    ("DD-Mon-YYYY", "%d-%b-%Y"),
    ("DD Mon YYYY", "%d %b %Y"),
)


@dataclass
class ParsedCSV:
    frame: pd.DataFrame
    date_format: str
    notes: list[str]


def _decode_csv(content: bytes) -> str:
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def _sniff_dialect(text: str) -> tuple[str, str, str]:
    """Return ``(separator, decimal mark, thousands separator)`` from the file's text."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    sep = max((",", ";", "\t"), key=header.count)
    if not header.count(sep):
        sep = ","
    body = "\n".join(lines[1:50])
    if sep != "," and re.search(r"\d,\d", body):
        return sep, ",", "."
    return sep, ".", ","


def _looks_like_daily_sequence(dates: pd.Series) -> bool:
    d = dates.dropna()
    if len(d) < 3:
        return False
    gaps = d.diff().dropna().dt.days
    monotonic = bool((gaps > 0).all() or (gaps < 0).all())
    return monotonic and float(gaps.abs().median()) <= 5


def _parse_csv_dates(raw: pd.Series, column: str) -> tuple[pd.DatetimeIndex, str]:
    text = raw.astype("string").str.strip().str.replace(_TZ_SUFFIX, r"\1", regex=True)
    times = text.str.extract(_TIME_SUFFIX.pattern.replace("[ T]", "[ T](").replace("$", ")$"), expand=False)
    text = text.str.replace(_TIME_SUFFIX, "", regex=True)
    text = text.str.replace(r"\.0$", "", regex=True)  # 20240102.0 from a spreadsheet
    present = text.notna() & (text != "")
    n_present = int(present.sum())
    if n_present == 0:
        raise MalformedDataError(f"Column '{column}' is empty; it must contain dates.")
    candidates = []
    for label, fmt in _CSV_DATE_FORMATS:
        parsed = pd.to_datetime(text.where(present), format=fmt, errors="coerce")
        candidates.append((int(parsed.notna().sum()), label, parsed))
    best = max(c[0] for c in candidates)
    if best < max(1, 0.5 * n_present):
        examples = ", ".join(map(str, text[present].head(3)))
        raise MalformedDataError(
            f"Column '{column}' could not be parsed as dates (e.g. {examples}). Use ISO dates "
            "(YYYY-MM-DD); MM/DD/YYYY, DD/MM/YYYY, DD.MM.YYYY and YYYYMMDD are also recognised."
        )
    winners = [c for c in candidates if c[0] == best]
    distinct = []
    for c in winners:
        if not any(c[2].equals(d[2]) for d in distinct):
            distinct.append(c)
    if len(distinct) > 1:
        plausible = [c for c in distinct if _looks_like_daily_sequence(c[2])]
        if len(plausible) != 1:
            raise MalformedDataError(
                f"The dates in column '{column}' are ambiguous: they read equally well as "
                f"{' and '.join(c[1] for c in distinct)}. Re-export the file with ISO dates (YYYY-MM-DD)."
            )
        distinct = plausible
    _, label, parsed = distinct[0]
    clock = times.dropna()
    if parsed.dropna().duplicated().any() and (clock.str.strip("0:.") != "").any():
        raise MalformedDataError(
            f"Column '{column}' has several timestamps per day, which looks like intraday bars. "
            "Quant Lab Pro needs one row per trading day."
        )
    return pd.DatetimeIndex(parsed), label


def parse_price_csv_detailed(content: bytes) -> ParsedCSV:
    """Parse CSV bytes into a date-indexed frame with canonical column names.

    Recognised: comma, semicolon or tab separators; UTF-8 (with or without BOM),
    UTF-16 and Windows-1252 text; thousands separators and decimal commas; ISO,
    US, European and YYYYMMDD dates, with or without a time and UTC offset.
    The date format is chosen from the whole column, never row by row, so a
    day/month file is not silently read as month/day. Rows whose date cannot be
    read are kept with a missing date for validation to report and remove.
    """
    if not content:
        raise MalformedDataError("The uploaded file is empty.")
    text = _decode_csv(content)
    sep, decimal, thousands = _sniff_dialect(text)
    try:
        frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skip_blank_lines=True)
    except (pd.errors.ParserError, ValueError) as exc:
        raise MalformedDataError(f"The file could not be read as CSV: {exc}") from exc
    frame = frame.dropna(how="all")
    if frame.empty:
        raise MalformedDataError("The uploaded CSV contains no rows.")
    date_col = next((c for c in frame.columns if str(c).strip().lower() in _DATE_COLUMN_NAMES), frame.columns[0])
    index, date_format = _parse_csv_dates(frame[date_col], str(date_col))
    frame = frame.drop(columns=[date_col])
    frame.index = index
    frame.index.name = "date"
    frame = standardise_columns(frame)
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in frame.columns]
    if missing:
        raise MalformedDataError(
            f"The CSV is missing required column(s): {', '.join(missing)}. "
            "Expected Date, Open, High, Low, Close, Volume (Adj Close optional)."
        )
    for col in _NUMERIC_CSV_COLUMNS:
        if col in frame.columns:
            values = frame[col].astype("string").str.strip().str.replace("\u00a0", "", regex=False)
            values = values.str.replace(thousands, "", regex=False)
            if decimal != ".":
                values = values.str.replace(decimal, ".", regex=False)
            frame[col] = values
    notes = [f"Dates read as {date_format}"]
    if sep != ",":
        notes.append("Separator: " + {";": "semicolon", "\t": "tab"}[sep])
    if decimal != ".":
        notes.append("Decimal comma converted")
    return ParsedCSV(frame=frame, date_format=date_format, notes=notes)


def parse_price_csv(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes into a date-indexed frame (see :func:`parse_price_csv_detailed`)."""
    return parse_price_csv_detailed(content).frame


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #


class DiskCache:
    """Caches *validated* raw provider responses as CSV + JSON metadata.

    Failures to read or write the cache are logged and ignored — the cache is an
    optimisation and must never be the reason a request fails.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @staticmethod
    def make_key(provider_key: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
        return f"{provider_key}_{safe}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.csv", self.root / f"{key}.json"

    def load(self, key: str) -> tuple[pd.DataFrame, dict] | None:
        data_path, meta_path = self._paths(key)
        if not (data_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
            raw_bytes = data_path.read_bytes()
            if meta.get("sha256") != hashlib.sha256(raw_bytes).hexdigest():
                logger.warning("Cache entry %s failed its checksum; ignoring it.", key)
                return None
            frame = pd.read_csv(io.BytesIO(raw_bytes), index_col=0, float_precision="round_trip")
            frame.index = pd.to_datetime(frame.index, format="%Y-%m-%d")
            frame.index.name = "date"
            return frame, meta
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Could not read cache entry %s: %s", key, exc)
            return None

    def save(self, key: str, frame: pd.DataFrame, meta: dict) -> bool:
        data_path, meta_path = self._paths(key)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            payload = frame.to_csv(date_format="%Y-%m-%d").encode("utf-8")
            meta = {**meta, "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(frame)}
            for path, blob in ((data_path, payload), (meta_path, json.dumps(meta, indent=2).encode())):
                fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
                with os.fdopen(fd, "wb") as handle:
                    handle.write(blob)
                os.replace(tmp, path)  # atomic on POSIX and Windows
            return True
        except OSError as exc:
            logger.warning("Could not write cache entry %s: %s", key, exc)
            return False


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


class CacheStatus(str, Enum):
    DOWNLOADED = "downloaded"
    CACHED = "cached"
    STALE_CACHE = "stale_cache"
    UPLOADED = "uploaded"


CACHE_STATUS_LABELS = {
    CacheStatus.DOWNLOADED: "Freshly downloaded",
    CacheStatus.CACHED: "Served from local cache",
    CacheStatus.STALE_CACHE: "STALE cache — provider unavailable",
    CacheStatus.UPLOADED: "User-uploaded file",
}


@dataclass
class MarketDataset:
    """Validated prices plus everything needed to describe where they came from."""

    ticker: str
    prices: pd.DataFrame
    raw: pd.DataFrame
    provider_key: str
    source_name: str
    adjustment: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    fetched_at: datetime
    cache_status: CacheStatus
    quality: QualityReport
    fingerprint: str
    provider_error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_obs(self) -> int:
        return len(self.prices)

    @property
    def first_date(self) -> pd.Timestamp | None:
        return self.prices.index.min() if self.n_obs else None

    @property
    def last_date(self) -> pd.Timestamp | None:
        return self.prices.index.max() if self.n_obs else None

    @property
    def is_usable(self) -> bool:
        return not self.quality.is_blocking and self.n_obs > 0

    def describe(self) -> dict[str, str]:
        return {
            "Ticker": self.ticker,
            "Data source": self.source_name,
            "Price convention": ADJUSTMENT_LABELS.get(self.adjustment, str(self.adjustment)),
            "Downloaded period (incl. warm-up)": f"{self.requested_start.date()} to {self.requested_end.date()}",
            "Observations": f"{self.n_obs:,}",
            "First observation": str(self.first_date.date()) if self.first_date is not None else "—",
            "Last available observation": str(self.last_date.date()) if self.last_date is not None else "—",
            "Retrieved": self.fetched_at.strftime("%Y-%m-%d %H:%M UTC"),
            "Cache status": CACHE_STATUS_LABELS[self.cache_status],
            "Data fingerprint": self.fingerprint[:16],
        }


def normalise_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical column names and timezone-naive session dates for a provider frame.

    Daily bars are labelled by the exchange's local calendar date. Converting to
    naive dates here keeps caching, alignment and fingerprints independent of
    how a provider happens to encode timezones.
    """
    out = standardise_columns(frame).copy()
    try:
        out.index = to_naive_dates(out.index)
    except (ValueError, TypeError) as exc:
        raise MalformedDataError(f"Dates could not be parsed: {exc}") from exc
    out.index.name = "date"
    return out


def apply_adjustment(
    raw: pd.DataFrame, mode: AdjustmentMode, report: QualityReport
) -> tuple[pd.DataFrame, str]:
    """Return ``(frame, adjustment_label)`` with OHLC on the requested price basis."""
    frame = standardise_columns(raw).copy()
    for col in (*PRICE_COLUMNS, "adj_close"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    if mode is AdjustmentMode.PRICE_ONLY:
        return frame, AdjustmentMode.PRICE_ONLY.value

    if "adj_close" not in frame.columns or frame["adj_close"].isna().all():
        report.add(
            Severity.WARNING,
            "no_adjusted_close",
            "Total-return adjustment was requested but the data has no adjusted close. "
            "Prices are used exactly as provided; dividends may be excluded.",
            count=0,
        )
        return frame, "as_provided"

    factor = frame["adj_close"] / frame["close"]
    bad_factor = ~np.isfinite(factor) | (factor <= 0)
    bad_factor &= frame["close"].notna()
    if bad_factor.any():
        report.add(
            Severity.WARNING,
            "invalid_adjustment",
            "The adjustment factor (Adj Close / Close) is missing or invalid on these "
            "sessions, so their prices cannot be adjusted; they will be removed.",
            dates=frame.index[bad_factor],
        )
        factor = factor.where(~bad_factor)
    for col in PRICE_COLUMNS:
        frame[col] = frame[col] * factor
    frame["adjustment_factor"] = factor
    return frame, AdjustmentMode.TOTAL_RETURN.value


def fingerprint_prices(prices: pd.DataFrame) -> str:
    """Stable SHA-256 of the prices actually used, to detect provider revisions."""
    if prices.empty:
        return hashlib.sha256(b"empty").hexdigest()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in prices.columns]
    canonical = prices[cols].to_csv(float_format="%.10g", date_format="%Y-%m-%d")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_market_data(
    ticker: str,
    start: date | str | pd.Timestamp,
    end: date | str | pd.Timestamp,
    provider: DataProvider,
    adjustment: AdjustmentMode = AdjustmentMode.TOTAL_RETURN,
    cache: DiskCache | None = None,
    max_cache_age: timedelta = DEFAULT_CACHE_MAX_AGE,
    force_refresh: bool = False,
    now: datetime | None = None,
    today: date | None = None,
) -> MarketDataset:
    """Fetch, adjust, validate and describe daily prices for one ticker.

    Raises :class:`InputValidationError` for bad inputs and
    :class:`DataProviderError` if no data (fresh or cached) is available.
    """
    ticker = validate_ticker(ticker)
    start_ts, end_ts = validate_date_range(start, end, today=today)
    now = now or datetime.now(timezone.utc)
    uploaded = isinstance(provider, CSVProvider)
    use_cache = cache is not None and not uploaded
    key = DiskCache.make_key(provider.key, ticker, start_ts, end_ts)

    cached = cache.load(key) if use_cache else None
    cached_age: timedelta | None = None
    if cached is not None:
        try:
            cached_at = datetime.fromisoformat(cached[1]["fetched_at"])
            cached_age = now - cached_at
        except (KeyError, ValueError):
            cached = None

    provider_error: str | None = None
    if cached is not None and not force_refresh and cached_age is not None and cached_age <= max_cache_age:
        raw, status, fetched_at = cached[0], CacheStatus.CACHED, now - cached_age
    else:
        try:
            raw = normalise_raw(provider.fetch(ticker, start_ts, end_ts))
            status = CacheStatus.UPLOADED if uploaded else CacheStatus.DOWNLOADED
            fetched_at = now
        except DataProviderError as exc:
            if cached is None or cached_age is None:
                raise
            provider_error = exc.user_message
            raw, status, fetched_at = cached[0], CacheStatus.STALE_CACHE, now - cached_age

    report = QualityReport()
    prepared, adjustment_label = apply_adjustment(raw, adjustment, report)
    clean, report = clean_price_frame(prepared, report)
    check_staleness(report.last_date, end_ts, report, today=today)
    if status is CacheStatus.STALE_CACHE:
        report.add(
            Severity.WARNING,
            "stale_cache",
            f"The provider could not be reached ({provider_error}). Showing data cached on "
            f"{fetched_at:%Y-%m-%d %H:%M} UTC, which may be out of date.",
            count=0,
        )

    if status is CacheStatus.DOWNLOADED and use_cache and not report.is_blocking:
        cache.save(  # type: ignore[union-attr]
            key,
            raw,
            {
                "provider": provider.key,
                "ticker": ticker,
                "start": str(start_ts.date()),
                "end": str(end_ts.date()),
                "fetched_at": now.isoformat(),
            },
        )

    notes = []
    if uploaded:
        notes.append(f"Source file: {getattr(provider, 'filename', 'upload.csv')}")
        notes.extend(getattr(provider, "parse_notes", []))
    return MarketDataset(
        ticker=ticker,
        prices=clean,
        raw=raw,
        provider_key=provider.key,
        source_name=provider.name,
        adjustment=adjustment_label,
        requested_start=start_ts,
        requested_end=end_ts,
        fetched_at=fetched_at,
        cache_status=status,
        quality=report,
        fingerprint=fingerprint_prices(clean),
        provider_error=provider_error,
        notes=notes,
    )


def align_benchmark(
    asset: pd.DataFrame, benchmark: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Restrict two price frames to their common sessions.

    Returns ``(asset_aligned, benchmark_aligned, dropped_dates)``. Prices are
    never forward-filled across calendars: a date missing from either series is
    dropped from both, and the dropped dates are returned for disclosure.
    """
    common = asset.index.intersection(benchmark.index)
    dropped = asset.index.symmetric_difference(benchmark.index)
    return asset.loc[common], benchmark.loc[common], dropped
