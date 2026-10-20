"""Synthetic datasets for the technicals, signals, scan and alerts tests. Nothing here is real market data."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from quantlab.calendar import expected_sessions
from quantlab.data import (
    AdjustmentMode,
    CacheStatus,
    CSVProvider,
    MarketDataset,
    fingerprint_prices,
    load_market_data,
)
from quantlab.provenance import Provenance, Timeliness
from quantlab.validation import clean_price_frame

NOW = datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)


def sessions(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    """``n`` NYSE sessions from ``start``, so validation sees no weekend or holiday rows."""
    first = pd.Timestamp(start)
    return pd.DatetimeIndex(expected_sessions(first, first + pd.Timedelta(days=int(n * 1.5) + 30))[:n], name="date")


def ohlcv(closes, opens=None, volumes=None, start: str = "2021-01-04", spread: float = 0.01) -> pd.DataFrame:
    """A clean OHLCV frame on NYSE sessions; highs and lows bracket open and close by ``spread``."""
    closes = np.asarray(closes, dtype="float64")
    opens = closes.copy() if opens is None else np.asarray(opens, dtype="float64")
    volumes = np.full(len(closes), 1_000_000.0) if volumes is None else np.asarray(volumes, dtype="float64")
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes) * (1 + spread),
         "low": np.minimum(opens, closes) * (1 - spread), "close": closes, "volume": volumes},
        index=sessions(len(closes), start),
    )


def walk(n: int = 400, seed: int = 7, drift: float = 0.0003, vol: float = 0.012, start: str = "2021-01-04",
         level: float = 100.0) -> pd.DataFrame:
    """Reproducible geometric random walk with overnight gaps and varying volume."""
    rng = np.random.default_rng(seed)
    closes = level * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    opens = np.r_[level, closes[:-1]] * np.exp(rng.normal(0, 0.004, n))
    volumes = np.round(1_000_000 * np.exp(rng.normal(0, 0.3, n)))
    return ohlcv(closes, opens, volumes, start=start)


def trend(n: int, daily: float, level: float = 50.0, start: str = "2021-01-04", volume: float = 1e6) -> pd.DataFrame:
    """Prices compounding at ``daily`` per session (a clean up- or down-trend)."""
    closes = level * (1 + daily) ** np.arange(n)
    return ohlcv(closes, volumes=np.full(n, volume), start=start)


def prov(**kw) -> Provenance:
    kw.setdefault("timeliness", Timeliness.END_OF_DAY)
    return Provenance(source="Synthetic test frame", retrieved_at=NOW, **kw)


def dataset(ticker: str, frame: pd.DataFrame, cache_status: CacheStatus = CacheStatus.UPLOADED,
            provider_key: str = "csv", adjustment: str = "as_provided", fetched_at: datetime = NOW,
            source_name: str = "Synthetic test frame", provider_error: str | None = None) -> MarketDataset:
    """A MarketDataset validated exactly as load_market_data validates (no provider involved)."""
    clean, report = clean_price_frame(frame)
    return MarketDataset(
        ticker=ticker, prices=clean, raw=frame, provider_key=provider_key, source_name=source_name,
        adjustment=adjustment, requested_start=frame.index.min() if len(frame) else pd.Timestamp("2021-01-04"),
        requested_end=frame.index.max() if len(frame) else pd.Timestamp("2021-12-31"), fetched_at=fetched_at,
        cache_status=cache_status, quality=report, fingerprint=fingerprint_prices(clean),
        provider_error=provider_error,
    )


def csv_dataset(ticker: str, frame: pd.DataFrame) -> MarketDataset:
    """Load ``frame`` through the real pipeline: CSV bytes -> CSVProvider -> load_market_data."""
    out = frame.rename(columns=str.capitalize)
    content = out.to_csv(index_label="Date", date_format="%Y-%m-%d").encode()
    first, last = frame.index[0], frame.index[-1]
    return load_market_data(ticker, first, last, CSVProvider(content, f"{ticker}.csv"),
                            adjustment=AdjustmentMode.PRICE_ONLY, now=NOW, today=date(last.year, last.month, last.day))


def loader(entries: dict[str, MarketDataset | BaseException | object],
           calls: list[str] | None = None) -> Callable[[str], MarketDataset]:
    """A ``load_prices`` callable: returns the dataset for a ticker, or raises the stored exception."""

    def load(ticker: str) -> MarketDataset:
        if calls is not None:
            calls.append(ticker)
        if ticker not in entries:
            from quantlab.data import TickerNotFoundError

            raise TickerNotFoundError(f"No data for '{ticker}' in the test fixture.")
        entry = entries[ticker]
        if isinstance(entry, BaseException):
            raise entry
        return entry  # type: ignore[return-value]

    return load
