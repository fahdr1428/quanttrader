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

    def __init__(self, timeout: int = 20, ticker_factory: Callable[[str], Any] | None = None):
        self.timeout = timeout
        self._ticker_factory = ticker_factory

    def _make_ticker(self, symbol: str) -> Any:
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf  # imported lazily so tests and CSV use need no network stack

        return yf.Ticker(symbol)

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        try:
            handle = self._make_ticker(ticker)
            # yfinance treats ``end`` as exclusive, so request one extra day.
            frame = handle.history(
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                actions=True,
                raise_errors=True,
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

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = parse_price_csv(self.content)
        mask = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        sliced = frame.loc[mask]
        if sliced.empty:
            raise TickerNotFoundError(
                f"The uploaded file has no observations between {pd.Timestamp(start).date()} "
                f"and {pd.Timestamp(end).date()} (file covers {frame.index.min().date()} to "
                f"{frame.index.max().date()})."
            )
        return sliced


def parse_price_csv(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes into a date-indexed frame with canonical column names."""
    if not content:
        raise MalformedDataError("The uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise MalformedDataError(f"The file could not be read as CSV: {exc}") from exc
    if frame.empty:
        raise MalformedDataError("The uploaded CSV contains no rows.")
    date_col = next(
        (c for c in frame.columns if str(c).strip().lower() in {"date", "datetime", "timestamp", "time"}),
        frame.columns[0],
    )
    try:
        dates = pd.to_datetime(frame[date_col], errors="coerce", utc=False)
    except (ValueError, TypeError) as exc:
        raise MalformedDataError(f"Column '{date_col}' could not be parsed as dates: {exc}") from exc
    if dates.isna().all():
        raise MalformedDataError(
            f"Column '{date_col}' could not be parsed as dates. Include a 'Date' column in YYYY-MM-DD format."
        )
    frame = frame.drop(columns=[date_col])
    frame.index = pd.DatetimeIndex(dates)
    frame.index.name = "date"
    frame = standardise_columns(frame)
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in frame.columns]
    if missing:
        raise MalformedDataError(
            f"The CSV is missing required column(s): {', '.join(missing)}. "
            "Expected Date, Open, High, Low, Close, Volume (Adj Close optional)."
        )
    return frame


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
            "Requested period": f"{self.requested_start.date()} to {self.requested_end.date()}",
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
