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

    def fetch_notes(self) -> list[str]:
        """Disclosures about the last fetch (e.g. rows the provider dropped), shown with the dataset."""
        return list(getattr(self, "_notes", []))

    def invalidate(self) -> None:
        """Forget anything the provider cached in memory, so the next fetch is fresh."""
        return None


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
        clock: Callable[[], datetime] | None = None,
    ):
        self.timeout = timeout
        self._ticker_factory = ticker_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._notes: list[str] = []
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

    def invalidate(self) -> None:
        """Clear yfinance's process-wide response cache.

        yfinance memoises HTTP responses for historical ranges (including error
        responses), so without this a retry or a forced refresh would replay the
        previous answer instead of asking Yahoo again.
        """
        if self._ticker_factory is not None:
            return
        try:
            from yfinance.data import YfData

            YfData().cache_get.cache_clear()
        except Exception:  # noqa: BLE001 - best effort; yfinance internals may change
            logger.debug("Could not clear the yfinance response cache.", exc_info=True)

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self._notes = []
        for attempt in range(self.retries + 1):
            try:
                return self._fetch_once(ticker, start, end)
            except (RateLimitError, ProviderConnectionError) as exc:
                self.invalidate()  # never replay a cached error response
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
        return self._drop_unfinished_session(frame, handle)

    def _drop_unfinished_session(self, frame: pd.DataFrame, handle: Any) -> pd.DataFrame:
        """Remove today's bar while the exchange's regular session is still open.

        During trading hours Yahoo reports the current session as a daily bar
        whose "close" is just the latest price. Treating it as a completed session
        would put a moving number into the backtest (and the cache).
        """
        if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            return frame
        tz = frame.index.tz
        now_local = pd.Timestamp(self._clock()).tz_convert(tz)
        last = frame.index[-1]
        if last.date() != now_local.date():
            return frame
        close_time = None
        try:
            meta = getattr(handle, "history_metadata", None) or {}
            end_epoch = meta.get("currentTradingPeriod", {}).get("regular", {}).get("end")
            if end_epoch:
                close_time = pd.Timestamp(int(end_epoch), unit="s", tz="UTC").tz_convert(tz)
        except Exception:  # noqa: BLE001 - metadata is optional
            close_time = None
        if close_time is None or close_time.date() != now_local.date():
            close_time = now_local.normalize() + pd.Timedelta(hours=16)  # NYSE regular close
        if now_local < close_time + pd.Timedelta(minutes=15):
            self._notes.append(
                f"Dropped the {last:%Y-%m-%d} bar: that session had not closed when the data was requested, so "
                "its 'close' was only the latest price."
            )
            return frame.iloc[:-1]
        return frame


_HTTP_STATUS = re.compile(r"\b(?:status|http|error|code)\D{0,12}(\d{3})\b", re.IGNORECASE)


def _yahoo_exception_kind(exc: Exception) -> str | None:
    """Classify by exception type first; yfinance messages contain tickers and epochs that look like codes."""
    try:
        from yfinance import exceptions as yfe

        if isinstance(exc, yfe.YFRateLimitError):
            return "rate_limit"
        if isinstance(exc, yfe.YFTickerMissingError):
            return "not_found"
    except ImportError:  # pragma: no cover
        pass
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "connection"
    return None


def _map_yahoo_exception(ticker: str, exc: Exception) -> DataProviderError:
    name = type(exc).__name__
    text = str(exc)
    lowered = f"{name} {text}".lower()
    kind = _yahoo_exception_kind(exc)
    status = _HTTP_STATUS.search(text)
    code = status.group(1) if status else None
    if kind == "rate_limit" or (kind is None and (
            "ratelimit" in lowered.replace(" ", "") or "too many requests" in lowered or code == "429")):
        return RateLimitError(
            "Yahoo Finance is rate-limiting requests. Wait a few minutes and try again; "
            "previously cached data (if any) is used in the meantime.",
            detail=f"{name}: {text}",
        )
    if kind == "not_found" or (kind is None and (code == "404" or any(
        token in lowered
        for token in ("pricesmissing", "tzmissing", "tickermissing", "delisted", "no timezone", "not found")
    ))):
        return TickerNotFoundError(
            f"No price history was found for '{ticker}'. Check the symbol (Yahoo uses e.g. "
            "BRK-B rather than BRK.B) and the date range.",
            detail=f"{name}: {text}",
        )
    if kind == "connection" or code == "403" or any(
        token in lowered
        for token in ("connection", "timeout", "timed out", "resolve", "network", "ssl", "proxy", "curl")
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


class SyntheticProvider(DataProvider):
    """Simulated prices from :mod:`quantlab.simulation` — never real market data.

    Used for the offline demo mode and for controlled experiments. The ticker
    encodes the market and seed (e.g. ``SIM-RANDOM-1``); the data goes through
    exactly the same adjustment, validation and fingerprinting as real data.
    """

    key = "synthetic"
    name = "Synthetic demo data (simulated, not market prices)"
    description = (
        "Prices simulated by Quant Lab Pro from a known random process so the app can be explored "
        "without a network connection. They are not, and do not resemble any specific, real security."
    )

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        from quantlab.simulation import SIM_END, SIM_START, parse_demo_ticker, simulate_market

        parsed = parse_demo_ticker(ticker)
        if parsed is None:
            raise TickerNotFoundError(f"'{ticker}' is not one of the synthetic demo markets.")
        frame = simulate_market(*parsed)
        sliced = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        if sliced.empty:
            raise TickerNotFoundError(
                f"The synthetic series covers {SIM_START.date()} to {SIM_END.date()}; choose dates in that range."
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

    # Entries older than this, or beyond this many, are pruned (at most hourly per process).
    MAX_AGE = timedelta(days=14)
    MAX_ENTRIES = 200
    _last_prune: dict[str, float] = {}

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @staticmethod
    def make_key(provider_key: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
        # The readable part replaces symbols such as '=' and '.', so a short hash of the
        # exact ticker keeps e.g. GC=F and GC.F in separate entries.
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
        digest = hashlib.sha1(ticker.encode("utf-8")).hexdigest()[:8]
        return f"{provider_key}_{safe}_{digest}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.csv", self.root / f"{key}.json"

    def load(self, key: str, expect: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict] | None:
        """Return ``(frame, meta)`` for ``key``, or None if absent, corrupt or for a different request.

        ``expect`` (e.g. provider, ticker, start, end) must match the stored
        metadata exactly, so a key collision can never serve another series.
        """
        data_path, meta_path = self._paths(key)
        if not (data_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
            if expect and any(str(meta.get(k)) != str(v) for k, v in expect.items()):
                logger.warning("Cache entry %s belongs to a different request; ignoring it.", key)
                return None
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
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(blob)
                    os.replace(tmp, path)  # atomic on POSIX and Windows
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
        except OSError as exc:
            logger.warning("Could not write cache entry %s: %s", key, exc)
            return False
        self._maybe_prune()
        return True

    def find_latest(self, provider_key: str, ticker: str, start: pd.Timestamp,
                    end: pd.Timestamp) -> tuple[pd.DataFrame, dict] | None:
        """The most recent entry for this provider and ticker that starts on or before ``start``.

        Used only as a stale fallback when the provider is unreachable: an entry
        saved for a different end date is better than nothing, as long as it is
        clearly labelled. The frame is trimmed to ``[start, end]``.
        """
        best: tuple[str, dict] | None = None
        try:
            for meta_path in self.root.glob(f"{provider_key}_*.json"):
                meta = json.loads(meta_path.read_text())
                if meta.get("provider") != provider_key or meta.get("ticker") != ticker:
                    continue
                if pd.Timestamp(meta.get("start", "2100-01-01")) > pd.Timestamp(start):
                    continue
                if best is None or meta.get("fetched_at", "") > best[1].get("fetched_at", ""):
                    best = (meta_path.stem, meta)
        except (OSError, ValueError) as exc:
            logger.warning("Could not scan the data cache: %s", exc)
            return None
        if best is None:
            return None
        loaded = self.load(best[0], {"provider": provider_key, "ticker": ticker})
        if loaded is None:
            return None
        frame, meta = loaded
        frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        return (frame, meta) if len(frame) else None

    def _maybe_prune(self) -> None:
        root = str(self.root.resolve())
        if time.time() - self._last_prune.get(root, 0.0) < 3600:
            return
        self._last_prune[root] = time.time()
        self.prune()

    def prune(self, max_age: timedelta | None = None, max_entries: int | None = None) -> int:
        """Delete entries older than ``max_age`` and the oldest beyond ``max_entries``. Returns the count."""
        max_age = max_age or self.MAX_AGE
        max_entries = max_entries or self.MAX_ENTRIES
        removed = 0
        try:
            now = time.time()
            for tmp in self.root.glob("*.tmp"):
                if now - tmp.stat().st_mtime > 3600:
                    tmp.unlink(missing_ok=True)
            entries = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for i, meta_path in enumerate(entries):
                if i >= max_entries or now - meta_path.stat().st_mtime > max_age.total_seconds():
                    meta_path.unlink(missing_ok=True)
                    meta_path.with_suffix(".csv").unlink(missing_ok=True)
                    removed += 1
        except OSError as exc:
            logger.warning("Could not prune the data cache: %s", exc)
        return removed


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


class CacheStatus(str, Enum):
    DOWNLOADED = "downloaded"
    CACHED = "cached"
    STALE_CACHE = "stale_cache"
    UPLOADED = "uploaded"
    SYNTHETIC = "synthetic"


CACHE_STATUS_LABELS = {
    CacheStatus.DOWNLOADED: "Freshly downloaded",
    CacheStatus.CACHED: "Served from local cache",
    CacheStatus.STALE_CACHE: "STALE cache — provider unavailable",
    CacheStatus.UPLOADED: "User-uploaded file",
    CacheStatus.SYNTHETIC: "SYNTHETIC — simulated, not market data",
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
            coerced = pd.to_numeric(frame[col], errors="coerce").astype("float64")
            unreadable = coerced.isna() & frame[col].notna()
            if unreadable.any():
                report.add(
                    Severity.WARNING,
                    "malformed_values",
                    f"Column '{col}' contained values that are not numbers (e.g. "
                    f"{frame.loc[unreadable, col].astype(str).iloc[0]!r}); they were treated as missing.",
                    dates=frame.index[unreadable],
                )
            frame[col] = coerced

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
    _check_adjustment_against_distributions(frame, factor, report)
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


def _check_adjustment_against_distributions(frame: pd.DataFrame, factor: pd.Series, report: QualityReport) -> None:
    """Warn when Adj Close does not reflect the dividends the provider itself reports.

    On an ex-date the adjustment factor should step by about ``1 - D / previous
    close``. A factor that never moves despite reported dividends means the
    "adjusted" close is really unadjusted; a step far from the dividend size
    points to a provider error. Prices are never changed here — only disclosed.
    """
    dist = pd.Series(0.0, index=frame.index)
    for col in ("dividends", "capital_gains"):
        if col in frame.columns:
            dist = dist + pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    events = dist[dist > 0]
    f = factor.where(np.isfinite(factor) & (factor > 0))
    if events.empty or f.dropna().empty:
        return
    if float(f.max() - f.min()) <= 1e-9 * float(f.abs().max()):
        report.add(
            Severity.WARNING,
            "adjustment_missing",
            "The data reports dividends or distributions, but Adj Close equals Close times a constant, so it does "
            "not include them. 'Total return' results here are price-only returns.",
            dates=events.index,
        )
        return
    prev_close = frame["close"].shift(1)
    implied = prev_close * (1 - f.shift(1) / f)  # distribution implied by the factor step on each ex-date
    rel_gap = ((implied - dist) / dist).abs()[events.index].dropna()
    if len(rel_gap) and float(rel_gap.median()) > 0.25:
        report.add(
            Severity.WARNING,
            "adjustment_inconsistent",
            "The step in the adjustment factor (Adj Close / Close) on ex-dates differs from the reported "
            f"distributions by a median of {float(rel_gap.median()):.0%}. The provider's adjustment may be "
            "wrong; total-return figures could be misstated.",
            dates=rel_gap.index[rel_gap > 0.25],
        )


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
    synthetic = isinstance(provider, SyntheticProvider)
    use_cache = cache is not None and not uploaded and not synthetic
    key = DiskCache.make_key(provider.key, ticker, start_ts, end_ts)
    expect = {"provider": provider.key, "ticker": ticker, "start": str(start_ts.date()), "end": str(end_ts.date())}
    if force_refresh:
        provider.invalidate()

    cached = cache.load(key, expect) if use_cache else None
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
            status = (CacheStatus.UPLOADED if uploaded else
                      CacheStatus.SYNTHETIC if synthetic else CacheStatus.DOWNLOADED)
            fetched_at = now
        except DataProviderError as exc:
            if cached is None and use_cache:
                # No entry for this exact request: fall back to the latest entry for the ticker.
                cached = cache.find_latest(provider.key, ticker, start_ts, end_ts)  # type: ignore[union-attr]
                if cached is not None:
                    try:
                        cached_age = now - datetime.fromisoformat(cached[1]["fetched_at"])
                    except (KeyError, ValueError):
                        cached = None
            if cached is None or cached_age is None:
                raise
            provider_error = exc.user_message
            raw, status, fetched_at = cached[0], CacheStatus.STALE_CACHE, now - cached_age

    report = QualityReport()
    prepared, adjustment_label = apply_adjustment(raw, adjustment, report)
    if uploaded and adjustment_label == AdjustmentMode.PRICE_ONLY.value:
        # The app cannot know whether an uploaded Close was split-adjusted, so it claims nothing.
        adjustment_label = "as_provided"
    clean, report = clean_price_frame(prepared, report)
    if not synthetic:  # a simulated series ends where it was designed to end
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
    if status in (CacheStatus.DOWNLOADED, CacheStatus.UPLOADED, CacheStatus.SYNTHETIC):
        notes.extend(provider.fetch_notes())
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
