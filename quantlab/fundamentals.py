"""Company fundamentals: statements, profile, events, news, filings and insider trades.

Two providers implement one interface:

* :class:`YahooFundamentalsProvider` reads what the yfinance library scrapes from Yahoo
  Finance. It is broad (profile, statements, events, news, insider trades) but it is an
  aggregator: figures can be revised without notice and it gives no filing dates, so a
  value cannot be shown to have been public on a past date.
* :class:`SecEdgarProvider` reads the SEC's own XBRL data (``companyfacts``) and filing
  index (``submissions``). It is the primary source for US filers and keeps each value's
  filing date, so statements can be rebuilt as they were known on a past date (``as_of``).

Both map their source's labels onto one set of canonical line items, so nothing
downstream depends on a vendor's naming. Cash-flow sign convention: outflows are
negative (capital expenditure and dividends paid are <= 0), as printed on a cash-flow
statement; the SEC's "payments" concepts are reported positive and are negated.

:func:`fundamental_metrics` derives valuation, profitability and leverage ratios, each
with an explicit formula, its inputs and its period basis (TTM from the last four
quarters when possible, otherwise the latest fiscal year). A ratio whose denominator is
not positive is not reported: P/E with a loss is undefined, not negative.

Nothing here fabricates a value. Providers raise :class:`~quantlab.data.DataProviderError`
subclasses on failure, missing values stay missing (NaN in a frame, ``None`` in a
result), and every value carries a :class:`~quantlab.provenance.Provenance`. The module
is pure library code: no Streamlit, and the network is reached only through injectable
callables, so tests run offline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from datetime import time as clock_time
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab.config import data_dir
from quantlab.data import (
    DataProviderError,
    MalformedDataError,
    ProviderConnectionError,
    RateLimitError,
    TickerNotFoundError,
    _map_yahoo_exception,
)
from quantlab.provenance import Calculated, Fact, Provenance, Timeliness
from quantlab.serialization import clean_json
from quantlab.validation import validate_ticker

logger = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")


class NotSupportedError(DataProviderError):
    """The provider does not offer this kind of data at all.

    Distinct from an empty result: ``[]`` means "the source was asked and had nothing",
    this means "the source was never able to answer", so a caller must not read it as
    "no news" or "no insider trades".
    """

    kind = "not_supported"


# --------------------------------------------------------------------------- #
# Canonical line items
# --------------------------------------------------------------------------- #


class StatementPeriod(str, Enum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


INCOME_ITEMS = ("revenue", "gross_profit", "operating_income", "net_income", "ebitda",
                "depreciation_amortization", "eps_diluted", "diluted_shares")
BALANCE_ITEMS = ("total_assets", "total_liabilities", "total_equity", "cash", "total_debt",
                 "current_assets", "current_liabilities", "shares_outstanding")
CASHFLOW_ITEMS = ("operating_cash_flow", "capital_expenditure", "free_cash_flow", "dividends_paid")

STATEMENT_ITEMS = {"income": INCOME_ITEMS, "balance": BALANCE_ITEMS, "cashflow": CASHFLOW_ITEMS}
STATEMENT_OF = {item: name for name, items in STATEMENT_ITEMS.items() for item in items}
STATEMENT_LABELS = {"income": "Income statement", "balance": "Balance sheet", "cashflow": "Cash-flow statement"}
PER_SHARE_ITEMS = frozenset({"eps_diluted"})
SHARE_COUNT_ITEMS = frozenset({"diluted_shares", "shares_outstanding"})
# Flows that add up over time, so a quarter can be derived from year-to-date totals.
# Per-share figures and weighted share counts do not add up.
ADDITIVE_FLOW_ITEMS = frozenset(INCOME_ITEMS + CASHFLOW_ITEMS) - PER_SHARE_ITEMS - SHARE_COUNT_ITEMS

# Period lengths in days (52/53-week fiscal years and 13/14-week quarters included).
_QUARTER_DAYS = (80, 100)
_YEAR_DAYS = (340, 390)


def item_unit(item: str, currency: str | None) -> str | None:
    """The unit of a canonical line item's values."""
    if item in PER_SHARE_ITEMS:
        return f"{currency}/share" if currency else "per share (currency unknown)"
    if item in SHARE_COUNT_ITEMS:
        return "shares"
    return currency


def _statement_units(currency: str | None) -> str:
    money = currency or "currency unknown"
    return f"{money}; EPS in {money}/share; share counts in shares"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (date, datetime)) else value


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(index=pd.Index([], dtype=object), columns=pd.DatetimeIndex([]), dtype="float64")


@dataclass(frozen=True)
class FinancialStatements:
    """Income statement, balance sheet and cash-flow statement on canonical line items.

    Each frame has canonical items as rows (an item the source did not report is simply
    absent) and fiscal period ends as columns, newest first. Values are in ``currency``
    (EPS per share, share counts in shares); a NaN cell means the source did not report
    that value — it is never zero-filled.

    ``filed`` maps a period end to the time by which every value in that column was
    public. Only SEC EDGAR provides it; Yahoo's map is empty, and using Yahoo figures
    historically risks look-ahead bias.
    """

    ticker: str
    currency: str | None
    income: pd.DataFrame
    balance: pd.DataFrame
    cashflow: pd.DataFrame
    period: StatementPeriod
    provenance: Provenance
    filed: dict[pd.Timestamp, datetime]
    # Additions beyond the contract: the start of each column's period when known (SEC),
    # the filing time of each individual value, and which source labels/concepts fed each item.
    period_starts: dict[pd.Timestamp, date] = field(default_factory=dict)
    value_filed: dict[tuple[str, pd.Timestamp], datetime] = field(default_factory=dict)
    item_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return {"income": self.income, "balance": self.balance, "cashflow": self.cashflow}

    def frame_for(self, item: str) -> pd.DataFrame:
        if item not in STATEMENT_OF:
            raise KeyError(f"'{item}' is not a canonical line item")
        return self.frames[STATEMENT_OF[item]]

    def row(self, item: str) -> pd.Series:
        """The item's values by period end, newest first (empty if the item is absent)."""
        frame = self.frame_for(item)
        if item not in frame.index:
            return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        return frame.loc[item].astype("float64")

    @property
    def periods(self) -> list[pd.Timestamp]:
        """Every period end that appears in any statement, newest first."""
        ends = {pd.Timestamp(c) for f in self.frames.values() for c in f.columns}
        return sorted(ends, reverse=True)

    @property
    def latest_period(self) -> pd.Timestamp | None:
        periods = self.periods
        return periods[0] if periods else None

    @property
    def is_empty(self) -> bool:
        return all(f.empty for f in self.frames.values())

    def value(self, item: str, period_end: Any) -> float | None:
        row = self.row(item)
        key = pd.Timestamp(period_end)
        if key not in row.index:
            return None
        v = row.loc[key]
        return None if pd.isna(v) else float(v)

    def period_label(self, period_end: pd.Timestamp) -> str:
        kind = "fiscal year" if self.period is StatementPeriod.ANNUAL else "quarter"
        return f"{kind} ended {pd.Timestamp(period_end):%Y-%m-%d}"

    def fact(self, item: str, period_end: Any = None) -> Fact:
        """One value as a :class:`Fact` with its own provenance (period, filing time, unit).

        ``period_end`` defaults to the newest column of the item's statement; a missing
        value comes back as ``Fact(value=None)`` with the reason in ``note``.
        """
        frame = self.frame_for(item)
        unit = item_unit(item, self.currency)
        if period_end is None:
            if not len(frame.columns):
                return Fact(item, None, self.provenance, unit=unit,
                            note=f"{STATEMENT_LABELS[STATEMENT_OF[item]]} has no periods.")
            period_end = frame.columns[0]
        end = pd.Timestamp(period_end)
        start = self.period_starts.get(end)
        published = self.value_filed.get((item, end)) or self.filed.get(end)
        prov = replace(self.provenance, period_start=start if start and start <= end.date() else None,
                       period_end=end.date(), published_at=published, units=unit)
        name = f"{item} ({self.period_label(end)})"
        value = self.value(item, end)
        note = "" if value is not None else f"{self.provenance.source} did not report {item} for this period."
        return Fact(name, value, prov, unit=unit, note=note)

    def to_dict(self) -> dict[str, Any]:
        def frame_dict(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
            return {str(item): {pd.Timestamp(c).date().isoformat(): (None if pd.isna(v) else float(v))
                                for c, v in frame.loc[item].items()} for item in frame.index}

        return clean_json({
            "ticker": self.ticker,
            "currency": self.currency,
            "period": self.period.value,
            "income": frame_dict(self.income),
            "balance": frame_dict(self.balance),
            "cashflow": frame_dict(self.cashflow),
            "filed": {pd.Timestamp(k).date().isoformat(): v.isoformat() for k, v in sorted(self.filed.items())},
            "period_starts": {pd.Timestamp(k).date().isoformat(): v.isoformat()
                              for k, v in sorted(self.period_starts.items())},
            "item_sources": {k: list(v) for k, v in self.item_sources.items()},
            "provenance": self.provenance.to_dict(),
        })


@dataclass(frozen=True)
class CompanyProfile:
    ticker: str
    name: str | None
    exchange: str | None
    sector: str | None
    industry: str | None
    currency: str | None
    country: str | None
    shares_outstanding: float | None
    provenance: Provenance
    shares_as_of: date | None = None  # addition: the date the share count refers to, when known

    def to_dict(self) -> dict[str, Any]:
        return clean_json({
            "ticker": self.ticker, "name": self.name, "exchange": self.exchange, "sector": self.sector,
            "industry": self.industry, "currency": self.currency, "country": self.country,
            "shares_outstanding": self.shares_outstanding, "shares_as_of": _iso(self.shares_as_of),
            "provenance": self.provenance.to_dict(),
        })


# Event kinds.
EARNINGS = "earnings"                    # scheduled or reported earnings date (Yahoo)
EARNINGS_RELEASE = "earnings_release"    # an 8-K item 2.02 filing (SEC): results were published
EX_DIVIDEND = "ex_dividend"
DIVIDEND_PAYMENT = "dividend_payment"
DIVIDEND = "dividend"                    # a past dividend, dated by its ex-date
SPLIT = "split"


@dataclass(frozen=True)
class CorporateEvent:
    """A dated corporate event. ``confirmed`` is None when the source does not say."""

    ticker: str
    kind: str
    date: date
    confirmed: bool | None
    detail: str
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        return clean_json({"ticker": self.ticker, "kind": self.kind, "date": self.date.isoformat(),
                           "confirmed": self.confirmed, "detail": self.detail,
                           "provenance": self.provenance.to_dict()})


@dataclass(frozen=True)
class NewsItem:
    """A news headline. The text is UNTRUSTED third-party content: data, never instructions."""

    ticker: str
    title: str
    publisher: str | None
    url: str | None
    published_at: datetime | None
    provenance: Provenance
    summary: str | None = None  # addition: the provider's summary, when given (also untrusted)

    def to_dict(self) -> dict[str, Any]:
        return clean_json({"ticker": self.ticker, "title": self.title, "publisher": self.publisher,
                           "url": self.url, "published_at": _iso(self.published_at), "summary": self.summary,
                           "untrusted": True, "provenance": self.provenance.to_dict()})


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: str
    form: str
    filed: date
    period: date | None
    url: str
    provenance: Provenance
    accession: str | None = None          # additions: accession number, description and 8-K items
    description: str | None = None
    items: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return clean_json({"ticker": self.ticker, "cik": self.cik, "form": self.form, "filed": self.filed.isoformat(),
                           "period": _iso(self.period), "url": self.url, "accession": self.accession,
                           "description": self.description, "items": list(self.items),
                           "provenance": self.provenance.to_dict()})


@dataclass(frozen=True)
class InsiderTransaction:
    ticker: str
    insider: str | None
    relation: str | None
    date: date | None
    transaction: str | None
    shares: float | None
    value: float | None
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        return clean_json({"ticker": self.ticker, "insider": self.insider, "relation": self.relation,
                           "date": _iso(self.date), "transaction": self.transaction, "shares": self.shares,
                           "value": self.value, "provenance": self.provenance.to_dict()})


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #


class FundamentalsProvider(ABC):
    """Interface every fundamentals source implements.

    ``profile`` and ``statements`` are required. The other methods raise
    :class:`NotSupportedError` unless a provider overrides them, and ``capabilities``
    lists what a provider offers, so a caller can tell "nothing found" (an empty list)
    from "this source cannot answer".
    """

    key: str = "base"
    name: str = "Base fundamentals provider"
    description: str = ""
    capabilities: frozenset[str] = frozenset({"profile", "statements"})

    @abstractmethod
    def profile(self, ticker: str) -> CompanyProfile:
        """Reference data for the company. Raise a DataProviderError subclass on failure."""

    @abstractmethod
    def statements(self, ticker: str, period: StatementPeriod = StatementPeriod.ANNUAL, *,
                   as_of: date | datetime | str | None = None) -> FinancialStatements:
        """Statements on canonical items. ``as_of`` excludes values filed after it (when supported)."""

    def events(self, ticker: str, *, limit: int = 20) -> list[CorporateEvent]:
        raise NotSupportedError(f"{self.name} does not provide corporate events.")

    def news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        raise NotSupportedError(f"{self.name} does not provide news.")

    def filings(self, ticker: str, forms: Iterable[str] | None = ("10-K", "10-Q", "8-K"),
                limit: int = 20) -> list[Filing]:
        raise NotSupportedError(f"{self.name} does not provide a filings index; use SEC EDGAR.")

    def insider_transactions(self, ticker: str, limit: int = 50) -> list[InsiderTransaction]:
        raise NotSupportedError(f"{self.name} does not provide insider transactions.")


def _check_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit!r}")
    return int(limit)


# --------------------------------------------------------------------------- #
# Small parsing helpers (sources disagree on types; never guess a value)
# --------------------------------------------------------------------------- #

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HTML_TAG = re.compile(r"<[^>]{0,500}>")


def _label_key(label: Any) -> str:
    """'Total Revenue', 'TotalRevenue' and 'total_revenue' compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def _finite(value: Any) -> float | None:
    """A finite float, or None for missing, NaN, infinite or non-numeric values (never 0)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _clean_text(value: Any, max_len: int = 500) -> str | None:
    """Plain single-line text: tags and control characters removed, whitespace collapsed, length capped."""
    if not isinstance(value, str):
        return None
    text = _CONTROL_CHARS.sub(" ", _HTML_TAG.sub(" ", value))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _http_url(value: Any) -> str | None:
    if isinstance(value, str) and re.match(r"^https?://[^\s<>\"']+$", value.strip()):
        return value.strip()
    return None


def _epoch_to_datetime(value: Any) -> datetime | None:
    number = _finite(value)
    if number is None:
        return None
    if abs(number) > 1e11:  # milliseconds
        number /= 1000.0
    try:
        moment = datetime.fromtimestamp(number, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return moment if 1980 <= moment.year <= 2200 else None


def _to_datetime_utc(value: Any) -> datetime | None:
    """A timezone-aware UTC datetime from an epoch, ISO string or datetime (naive is read as UTC)."""
    if value is None or value is pd.NaT or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _epoch_to_datetime(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            try:
                value = pd.Timestamp(text)
            except (ValueError, TypeError):
                return None
            if pd.isna(value):
                return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, clock_time(0), tzinfo=timezone.utc)
    return None


def _to_date(value: Any) -> date | None:
    """A calendar date; a timestamp keeps its own (exchange-local) date."""
    if value is None or value is pd.NaT or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        moment = _epoch_to_datetime(value)
        return moment.date() if moment else None
    if isinstance(value, str):
        text = value.strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            try:
                stamp = pd.Timestamp(text)
            except (ValueError, TypeError):
                return None
            return None if pd.isna(stamp) else stamp.date()
    return None


def _filing_time(filed: date) -> datetime:
    """When a filing dated ``filed`` is taken to be public: the end of that day in New York, in UTC.

    companyfacts gives only the filing date. Taking the end of the day is the
    conservative choice for point-in-time use: a value is never treated as known
    before it could have been.
    """
    return datetime.combine(filed, clock_time(23, 59, 59), tzinfo=NEW_YORK).astimezone(timezone.utc)


def _as_of_cutoff(as_of: date | datetime | str | None) -> Callable[[date], bool] | None:
    """A predicate on filing dates: True when a filing was public by ``as_of``."""
    if as_of is None:
        return None
    if isinstance(as_of, str):
        try:
            as_of = datetime.fromisoformat(as_of.strip()) if len(as_of.strip()) > 10 else date.fromisoformat(as_of)
        except ValueError as exc:
            raise ValueError(f"as_of must be an ISO date or datetime, got {as_of!r}") from exc
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware when it is a datetime")
        moment = as_of
        return lambda filed: _filing_time(filed) <= moment
    if isinstance(as_of, date):
        day = as_of
        return lambda filed: filed <= day
    raise ValueError(f"as_of must be a date or datetime, got {type(as_of).__name__}")


# --------------------------------------------------------------------------- #
# Yahoo Finance (yfinance)
# --------------------------------------------------------------------------- #

# Canonical item -> candidate (statement, yfinance row label) pairs, in priority order.
# Labels are compared after _label_key, so the "pretty" and CamelCase forms both match.
YAHOO_LABELS: dict[str, tuple[tuple[str, str], ...]] = {
    "revenue": (("income", "Total Revenue"), ("income", "Operating Revenue")),
    "gross_profit": (("income", "Gross Profit"),),
    "operating_income": (("income", "Operating Income"),),
    "net_income": (("income", "Net Income"), ("income", "Net Income Common Stockholders")),
    "ebitda": (("income", "EBITDA"),),
    "depreciation_amortization": (("income", "Reconciled Depreciation"),
                                  ("cashflow", "Depreciation And Amortization"),
                                  ("cashflow", "Depreciation Amortization Depletion")),
    "eps_diluted": (("income", "Diluted EPS"),),
    "diluted_shares": (("income", "Diluted Average Shares"),),
    "total_assets": (("balance", "Total Assets"),),
    "total_liabilities": (("balance", "Total Liabilities Net Minority Interest"),),
    "total_equity": (("balance", "Stockholders Equity"), ("balance", "Common Stock Equity")),
    "cash": (("balance", "Cash And Cash Equivalents"),),
    "total_debt": (("balance", "Total Debt"),),
    "current_assets": (("balance", "Current Assets"),),
    "current_liabilities": (("balance", "Current Liabilities"),),
    "shares_outstanding": (("balance", "Ordinary Shares Number"),),
    "operating_cash_flow": (("cashflow", "Operating Cash Flow"),
                            ("cashflow", "Cash Flow From Continuing Operating Activities")),
    "capital_expenditure": (("cashflow", "Capital Expenditure"),),
    "free_cash_flow": (("cashflow", "Free Cash Flow"),),
    "dividends_paid": (("cashflow", "Cash Dividends Paid"), ("cashflow", "Common Stock Dividend Paid")),
}

_YAHOO_ATTRIBUTES = {
    StatementPeriod.ANNUAL: {"income": "income_stmt", "balance": "balance_sheet", "cashflow": "cashflow"},
    StatementPeriod.QUARTERLY: {"income": "quarterly_income_stmt", "balance": "quarterly_balance_sheet",
                                "cashflow": "quarterly_cashflow"},
}


def _column_dates(columns: Iterable[Any]) -> pd.DatetimeIndex:
    """Period-end dates from column labels; anything that is not a date (e.g. an integer) becomes NaT."""
    out = []
    for label in columns:
        day = _to_date(label) if isinstance(label, (date, str)) else None  # pd.Timestamp is a date
        out.append(pd.Timestamp(day) if day is not None else pd.NaT)
    return pd.DatetimeIndex(out)


def _clean_yahoo_frame(raw: Any) -> pd.DataFrame:
    """Numeric values, date columns (newest first, duplicates merged), unreadable columns dropped."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return _empty_frame()
    frame = raw.copy()
    frame.columns = _column_dates(frame.columns)
    frame = frame.loc[:, frame.columns.notna()]
    if frame.empty:
        return _empty_frame()
    frame = frame.apply(lambda col: pd.to_numeric(col, errors="coerce")).astype("float64")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if frame.columns.has_duplicates:  # keep the first reported value for each period
        frame = frame.T.groupby(level=0, sort=False).first().T
    return frame.reindex(columns=sorted(frame.columns, reverse=True))


def map_yahoo_statements(raw: Mapping[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[str, ...]]]:
    """Map yfinance statement frames (keys income/balance/cashflow) onto canonical items.

    A canonical row takes its first candidate label and fills that label's gaps from
    later candidates, cell by cell; the labels actually used are returned so the
    definition is visible. Rows and columns with no value at all are dropped.
    """
    cleaned = {name: _clean_yahoo_frame(raw.get(name)) for name in STATEMENT_ITEMS}
    lookup = {name: {_label_key(label): label for label in reversed(list(frame.index))}
              for name, frame in cleaned.items()}
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, tuple[str, ...]] = {}
    for name, items in STATEMENT_ITEMS.items():
        columns = cleaned[name].columns
        rows: dict[str, pd.Series] = {}
        for item in items:
            combined = pd.Series(np.nan, index=columns, dtype="float64")
            used: list[str] = []
            for source, label in YAHOO_LABELS.get(item, ()):
                actual = lookup[source].get(_label_key(label))
                if actual is None or not len(columns):
                    continue
                values = cleaned[source].loc[actual]
                if isinstance(values, pd.DataFrame):  # duplicated label: first occurrence
                    values = values.iloc[0]
                values = values.reindex(columns)
                fills = combined.isna() & values.notna()
                if fills.any():
                    combined = combined.where(~fills, values)
                    used.append(f"{source}: {actual}")
            if combined.notna().any():
                rows[item] = combined
                sources[item] = tuple(used)
        frame = pd.DataFrame(rows).T if rows else _empty_frame()
        if rows:
            frame = frame.reindex(columns=columns).astype("float64")
            frame = frame.loc[:, frame.notna().any(axis=0)]
        frames[name] = frame
    return frames, sources


def _yahoo_error(symbol: str, what: str, exc: Exception) -> DataProviderError:
    """Classify like the price provider, with a message about fundamentals rather than prices."""
    mapped = _map_yahoo_exception(symbol, exc)
    messages = {
        TickerNotFoundError: f"Yahoo Finance has no {what} for '{symbol}'. Check the symbol (Yahoo uses e.g. "
                             "BRK-B rather than BRK.B).",
        RateLimitError: f"Yahoo Finance is rate-limiting requests ({what} for '{symbol}'). Wait a few minutes "
                        "and try again.",
        ProviderConnectionError: f"Could not reach Yahoo Finance for the {what} of '{symbol}' (network error or "
                                 "blocked connection).",
    }
    cls = type(mapped)
    return cls(messages.get(cls, f"Yahoo Finance could not provide the {what} of '{symbol}'."), detail=mapped.detail)


class YahooFundamentalsProvider(FundamentalsProvider):
    key = "yahoo"
    name = "Yahoo Finance (via yfinance)"
    description = (
        "Fundamentals scraped from Yahoo Finance by the open-source yfinance library. Broad but unofficial: "
        "figures are aggregated, can be revised without notice, and carry no filing dates. Prefer SEC EDGAR "
        "for US filers' statements."
    )
    capabilities = frozenset({"profile", "statements", "events", "news", "insider_transactions"})

    def __init__(
        self,
        ticker_factory: Callable[[str], Any] | None = None,
        retries: int = 2,
        backoff_seconds: float = 1.5,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], datetime] | None = None,
        history_limit: int = 8,
    ):
        self._ticker_factory = ticker_factory
        self.retries = max(0, int(retries))
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep or time.sleep
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.history_limit = _check_limit(history_limit)  # past dividends/splits/earnings kept per call

    # -- plumbing ----------------------------------------------------------- #

    def _make_ticker(self, symbol: str) -> Any:
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf  # lazily: tests and offline use need no network stack

        try:  # raise errors instead of logging them and returning empty data
            yf.config.debug.hide_exceptions = False
        except AttributeError:  # pragma: no cover - older yfinance
            pass
        return yf.Ticker(symbol)

    def _get(self, symbol: str, what: str, getter: Callable[[Any], Any]) -> Any:
        """Call ``getter`` on a fresh Ticker; transient failures are retried with backoff."""
        for attempt in range(self.retries + 1):
            try:
                return getter(self._make_ticker(symbol))
            except DataProviderError as exc:
                error = exc
            except Exception as exc:  # noqa: BLE001 - mapped to typed errors
                error = _yahoo_error(symbol, what, exc)
                error.__cause__ = exc
            if not isinstance(error, (RateLimitError, ProviderConnectionError)) or attempt == self.retries:
                raise error
            logger.warning("Transient Yahoo Finance error for %s (%s); retrying.", what, error.kind)
            self._sleep(self.backoff_seconds * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def _provenance(self, what: str, url: str, timeliness: Timeliness, **kw: Any) -> Provenance:
        return Provenance(source=f"{self.name}: {what}", retrieved_at=self._clock(), timeliness=timeliness,
                          source_url=url, cache_status="downloaded", **kw)

    # -- profile and statements -------------------------------------------- #

    def _info(self, symbol: str) -> dict:
        info = self._get(symbol, "company profile", lambda h: h.info)
        if not isinstance(info, Mapping) or not (info.get("longName") or info.get("shortName")):
            raise TickerNotFoundError(f"Yahoo Finance has no company profile for '{symbol}'. Check the symbol.")
        return dict(info)

    def profile(self, ticker: str) -> CompanyProfile:
        symbol = validate_ticker(ticker)
        info = self._info(symbol)
        currency = _clean_text(info.get("currency"), 10)
        warnings = []
        reporting = _clean_text(info.get("financialCurrency"), 10)
        if reporting and currency and reporting != currency:
            warnings.append(f"Statements are reported in {reporting} while the stock trades in {currency}.")
        shares = _positive(info.get("sharesOutstanding"))
        if shares is None:
            warnings.append("Yahoo Finance did not report a share count.")
        else:
            warnings.append("The share count is Yahoo's latest figure; its as-of date is not given.")
        prov = self._provenance("company profile", f"https://finance.yahoo.com/quote/{symbol}/profile",
                                Timeliness.STATIC, currency=currency, warnings=tuple(warnings))
        return CompanyProfile(
            ticker=symbol,
            name=_clean_text(info.get("longName"), 200) or _clean_text(info.get("shortName"), 200),
            exchange=_clean_text(info.get("fullExchangeName"), 100) or _clean_text(info.get("exchange"), 100),
            sector=_clean_text(info.get("sector"), 100),
            industry=_clean_text(info.get("industry"), 100),
            currency=currency,
            country=_clean_text(info.get("country"), 100),
            shares_outstanding=shares,
            provenance=prov,
        )

    def statements(self, ticker: str, period: StatementPeriod = StatementPeriod.ANNUAL, *,
                   as_of: date | datetime | str | None = None) -> FinancialStatements:
        symbol = validate_ticker(ticker)
        period = StatementPeriod(period)
        if as_of is not None:
            raise NotSupportedError(
                "Yahoo Finance gives no filing dates, so statements cannot be rebuilt as of a past date. "
                "Use SEC EDGAR for point-in-time statements."
            )
        raw = {name: self._get(symbol, f"{period.value} {STATEMENT_LABELS[name].lower()}",
                               lambda h, attr=attr: getattr(h, attr))
               for name, attr in _YAHOO_ATTRIBUTES[period].items()}
        frames, sources = map_yahoo_statements(raw)
        if all(f.empty for f in frames.values()):
            raise TickerNotFoundError(
                f"Yahoo Finance has no {period.value} financial statements for '{symbol}'. Funds, ETFs and "
                "indices have none; for a US company, try SEC EDGAR."
            )
        warnings = [
            "Yahoo Finance gives no filing dates: these figures cannot be shown to have been public on a past "
            "date, so historical use risks look-ahead bias.",
            "Aggregated by Yahoo Finance from company reports; prefer SEC filings (primary source) where available.",
        ]
        currency = None
        try:
            info = self._info(symbol)
            currency = _clean_text(info.get("financialCurrency"), 10) or _clean_text(info.get("currency"), 10)
            if currency is None:
                warnings.append("Yahoo Finance did not state the reporting currency.")
        except DataProviderError as exc:
            warnings.append(f"Reporting currency unknown: the company profile could not be read ({exc.user_message})")
        ends = sorted({c for f in frames.values() for c in f.columns}, reverse=True)
        prov = self._provenance(
            f"{period.value} financial statements", f"https://finance.yahoo.com/quote/{symbol}/financials",
            Timeliness.PERIODIC_FILING, period_end=ends[0].date(), currency=currency,
            units=_statement_units(currency), warnings=tuple(warnings),
        )
        return FinancialStatements(
            ticker=symbol, currency=currency, income=frames["income"], balance=frames["balance"],
            cashflow=frames["cashflow"], period=period, provenance=prov, filed={}, item_sources=sources,
        )

    # -- events ------------------------------------------------------------- #

    def events(self, ticker: str, *, limit: int = 20) -> list[CorporateEvent]:
        """Earnings dates (scheduled and reported), dividend dates and splits, newest first.

        Each part comes from a separate Yahoo request. Parts that fail are named in
        every returned event's provenance warnings; if nothing could be read and a part
        failed, the error is raised rather than returning an empty list that would read
        as "no events".
        """
        symbol = validate_ticker(ticker)
        limit = _check_limit(limit)
        parts = (("earnings dates", self._earnings_dates), ("calendar", self._calendar_events),
                 ("dividend history", self._dividend_events), ("split history", self._split_events))
        found: dict[tuple[str, date], CorporateEvent] = {}
        failures: list[tuple[str, DataProviderError]] = []
        for label, reader in parts:
            try:
                for event in reader(symbol):
                    found.setdefault((event.kind, event.date), event)  # earlier parts carry more detail
            except DataProviderError as exc:
                failures.append((label, exc))
        events = sorted(found.values(), key=lambda e: (e.date, e.kind), reverse=True)[:limit]
        if failures:
            if not events:
                raise failures[0][1]
            note = "Not retrieved: " + "; ".join(f"{label} ({exc.user_message})" for label, exc in failures)
            events = [replace(e, provenance=e.provenance.with_warning(note)) for e in events]
        return events

    def _event(self, symbol: str, kind: str, day: date, confirmed: bool | None, detail: str, what: str,
               url: str) -> CorporateEvent:
        prov = self._provenance(what, url, Timeliness.EVENT, period_start=day, period_end=day)
        return CorporateEvent(symbol, kind, day, confirmed, detail, prov)

    def _earnings_dates(self, symbol: str) -> list[CorporateEvent]:
        def read(h: Any) -> Any:
            getter = getattr(h, "get_earnings_dates", None)
            return getter(limit=12) if callable(getter) else getattr(h, "earnings_dates", None)

        frame = self._get(symbol, "earnings dates", read)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []
        columns = {_label_key(c): c for c in frame.columns}
        reported_col, estimate_col = columns.get("reportedeps"), columns.get("epsestimate")
        url = f"https://finance.yahoo.com/calendar/earnings?symbol={symbol}"
        out = []
        for stamp, row in frame.iterrows():
            day = _to_date(stamp)
            if day is None:
                continue
            reported = _finite(row.get(reported_col)) if reported_col is not None else None
            estimate = _finite(row.get(estimate_col)) if estimate_col is not None else None
            if reported is not None:
                detail = f"Earnings reported: EPS {reported:g}" + (f" vs estimate {estimate:g}" if estimate is not None
                                                                   else "")
                confirmed: bool | None = True
            else:
                detail = "Scheduled earnings date" + (f"; EPS estimate {estimate:g}" if estimate is not None else "")
                detail += " (Yahoo does not say whether the date is confirmed)"
                confirmed = None
            out.append(self._event(symbol, EARNINGS, day, confirmed, detail, "earnings dates", url))
        out.sort(key=lambda e: e.date, reverse=True)
        return out[: max(self.history_limit, 12)]

    def _calendar_events(self, symbol: str) -> list[CorporateEvent]:
        calendar = self._get(symbol, "event calendar", lambda h: h.calendar)
        if isinstance(calendar, pd.DataFrame):  # older yfinance: labels in the index, values in columns
            calendar = {str(k): [v for v in row if not (isinstance(v, float) and math.isnan(v))]
                        for k, row in calendar.iterrows()} if not calendar.empty else {}
        if not isinstance(calendar, Mapping):
            return []
        entries = {_label_key(k): v for k, v in calendar.items()}
        for key, value in entries.items():  # one-element lists (older layout) hold a single value
            if key != "earningsdate" and isinstance(value, (list, tuple)):
                entries[key] = value[0] if len(value) == 1 else None
        url = f"https://finance.yahoo.com/quote/{symbol}/"
        out = []
        earnings = entries.get("earningsdate")
        dates = [d for d in (_to_date(v) for v in (earnings if isinstance(earnings, (list, tuple)) else [earnings]))
                 if d is not None]
        if dates:
            dates.sort()
            average = _finite(entries.get("earningsaverage"))
            estimate = f"; consensus EPS estimate {average:g}" if average is not None else ""
            if len(dates) > 1:
                detail = (f"Expected earnings window {dates[0]} to {dates[-1]} (an estimate, not a confirmed date)"
                          f"{estimate}")
                out.append(self._event(symbol, EARNINGS, dates[0], False, detail, "event calendar", url))
            else:
                detail = f"Expected earnings date (Yahoo does not say whether it is confirmed){estimate}"
                out.append(self._event(symbol, EARNINGS, dates[0], None, detail, "event calendar", url))
        for key, kind, text in (("exdividenddate", EX_DIVIDEND, "Ex-dividend date"),
                                ("dividenddate", DIVIDEND_PAYMENT, "Dividend payment date")):
            day = _to_date(entries.get(key))
            if day is not None:
                out.append(self._event(symbol, kind, day, None, f"{text} (from Yahoo's event calendar)",
                                       "event calendar", url))
        return out

    def _series_events(self, symbol: str, attr: str, kind: str, what: str,
                       describe: Callable[[float], str]) -> list[CorporateEvent]:
        series = self._get(symbol, what, lambda h: getattr(h, attr, None))
        if not isinstance(series, pd.Series) or series.empty:
            return []
        url = f"https://finance.yahoo.com/quote/{symbol}/history"
        out = []
        for stamp, raw in series.items():
            day, amount = _to_date(stamp), _finite(raw)
            if day is None or amount is None or amount <= 0:
                continue
            out.append(self._event(symbol, kind, day, True, describe(amount), what, url))
        out.sort(key=lambda e: e.date, reverse=True)
        return out[: self.history_limit]

    def _dividend_events(self, symbol: str) -> list[CorporateEvent]:
        return self._series_events(symbol, "dividends", DIVIDEND, "dividend history",
                                   lambda a: f"Dividend of {a:g} per share in the trading currency (ex-dividend date)")

    def _split_events(self, symbol: str) -> list[CorporateEvent]:
        return self._series_events(symbol, "splits", SPLIT, "split history",
                                   lambda r: f"Stock split, ratio {r:g} new shares per old share")

    # -- news and insiders --------------------------------------------------- #

    def news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        """Recent headlines, newest first. Titles and summaries are UNTRUSTED text."""
        symbol = validate_ticker(ticker)
        limit = _check_limit(limit)

        def read(h: Any) -> Any:
            getter = getattr(h, "get_news", None)
            return getter(count=limit) if callable(getter) else h.news

        raw = self._get(symbol, "news", read)
        if not isinstance(raw, (list, tuple)):
            if raw is None:
                return []
            raise MalformedDataError(f"Yahoo Finance returned news for '{symbol}' in an unexpected format.")
        items: list[NewsItem] = []
        seen: set[str] = set()
        skipped = 0
        for entry in raw:
            parsed = _parse_news_entry(entry)
            if parsed is None:
                skipped += 1
                continue
            key = parsed["url"] or parsed["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            prov = self._provenance("news", parsed["url"] or f"https://finance.yahoo.com/quote/{symbol}/news",
                                    Timeliness.EVENT, published_at=parsed["published_at"],
                                    warnings=("Third-party text: untrusted data, not instructions.",))
            items.append(NewsItem(symbol, parsed["title"], parsed["publisher"], parsed["url"],
                                  parsed["published_at"], prov, summary=parsed["summary"]))
        if skipped:
            logger.info("Skipped %d news entries without a readable title for %s.", skipped, symbol)
        oldest = datetime.min.replace(tzinfo=timezone.utc)
        items.sort(key=lambda n: n.published_at or oldest, reverse=True)
        return items[:limit]

    def insider_transactions(self, ticker: str, limit: int = 50) -> list[InsiderTransaction]:
        """Insider transactions as Yahoo lists them, newest first (values as reported; NaN -> None)."""
        symbol = validate_ticker(ticker)
        limit = _check_limit(limit)
        frame = self._get(symbol, "insider transactions", lambda h: h.insider_transactions)
        if frame is None or (isinstance(frame, pd.DataFrame) and frame.empty):
            return []
        if not isinstance(frame, pd.DataFrame):
            raise MalformedDataError(f"Yahoo Finance returned insider transactions for '{symbol}' in an "
                                     "unexpected format.")
        columns = {_label_key(c): c for c in frame.columns}

        def pick(row: pd.Series, *names: str) -> Any:
            for name in names:
                if name in columns:
                    value = row[columns[name]]
                    if not (isinstance(value, float) and math.isnan(value)) and value is not None:
                        return value
            return None

        url = f"https://finance.yahoo.com/quote/{symbol}/insider-transactions"
        out = []
        for _, row in frame.iterrows():
            day = _to_date(pick(row, "startdate", "date"))
            text = _clean_text(pick(row, "text", "transactiontext"), 300) or _clean_text(
                pick(row, "transaction", "moneytext"), 300)
            prov = self._provenance("insider transactions", url, Timeliness.EVENT,
                                    period_start=day, period_end=day)
            out.append(InsiderTransaction(
                ticker=symbol,
                insider=_clean_text(pick(row, "insider", "filername"), 200),
                relation=_clean_text(pick(row, "position", "filerrelation"), 200),
                date=day, transaction=text,
                shares=_finite(pick(row, "shares")), value=_finite(pick(row, "value")),
                provenance=prov,
            ))
        out.sort(key=lambda t: t.date or date.min, reverse=True)
        return out[:limit]


def _parse_news_entry(entry: Any) -> dict[str, Any] | None:
    """Read one yfinance news entry in either shape: flat (``title``, ``link``,
    ``providerPublishTime``) or nested under ``content`` (``title``, ``pubDate``,
    ``provider.displayName``, ``canonicalUrl.url``). None when there is no title."""
    if not isinstance(entry, Mapping):
        return None
    content = entry.get("content")
    src: Mapping = content if isinstance(content, Mapping) else entry
    title = _clean_text(src.get("title"), 500)
    if title is None:
        return None
    provider = src.get("provider")
    publisher = _clean_text(src.get("publisher"), 200) or (
        _clean_text(provider.get("displayName"), 200) if isinstance(provider, Mapping) else None)

    def nested_url(key: str) -> str | None:
        value = src.get(key)
        return _http_url(value.get("url")) if isinstance(value, Mapping) else _http_url(value)

    url = (_http_url(src.get("link")) or nested_url("canonicalUrl") or nested_url("clickThroughUrl")
           or _http_url(entry.get("link")))
    published = None
    for key in ("providerPublishTime", "pubDate", "displayTime"):
        published = _to_datetime_utc(src.get(key))
        if published is not None:
            break
    summary = _clean_text(src.get("summary"), 1000) or _clean_text(src.get("description"), 1000)
    return {"title": title, "publisher": publisher, "url": url, "published_at": published, "summary": summary}


# --------------------------------------------------------------------------- #
# SEC EDGAR: transport (user agent, throttle, cache)
# --------------------------------------------------------------------------- #

SEC_USER_AGENT_ENV = "QLP_SEC_USER_AGENT"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{document}"


class RequestThrottle:
    """Spaces requests at least ``1 / max_per_second`` seconds apart, across threads.

    The SEC asks automated clients to stay under 10 requests per second and blocks
    those that do not; 5 per second leaves a margin. One instance is shared by
    every :class:`SecEdgarProvider` in the process unless another is injected.
    """

    def __init__(self, max_per_second: float = 5.0, clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] | None = None):
        if not max_per_second or max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self.interval = 1.0 / max_per_second
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._next_allowed: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Block until a request may be sent; returns the seconds waited."""
        with self._lock:
            now = self._clock()
            delay = 0.0 if self._next_allowed is None else max(0.0, self._next_allowed - now)
            if delay > 0:
                self._sleep(delay)
            self._next_allowed = now + delay + self.interval
            return delay


SEC_THROTTLE = RequestThrottle(5.0)


class JsonCache:
    """On-disk cache of JSON responses, one file per URL, written atomically.

    Like :class:`quantlab.data.DiskCache`, failures to read or write are logged and
    ignored: the cache is an optimisation and must never be the reason a request fails.
    Each entry records its URL, so a file can never serve a different request.
    """

    MAX_AGE = timedelta(days=30)   # entries older than this are pruned
    MAX_ENTRIES = 300
    _last_prune: dict[str, float] = {}

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path_for(self, url: str) -> Path:
        tail = re.sub(r"[^A-Za-z0-9_-]", "_", url.rsplit("/", 1)[-1].removesuffix(".json"))[:60]
        return self.root / f"{tail}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}.json"

    def load(self, url: str) -> tuple[Any, datetime] | None:
        path = self.path_for(url)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("url") != url:
                logger.warning("Cache entry %s belongs to a different request; ignoring it.", path.name)
                return None
            fetched_at = datetime.fromisoformat(record["fetched_at"])
            if fetched_at.tzinfo is None:
                raise ValueError("fetched_at is not timezone-aware")
            return record["payload"], fetched_at
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Could not read SEC cache entry %s: %s", path.name, exc)
            return None

    def save(self, url: str, payload: Any, fetched_at: datetime) -> bool:
        path = self.path_for(url)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            blob = json.dumps({"url": url, "fetched_at": fetched_at.isoformat(), "payload": payload}).encode("utf-8")
            fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(blob)
                os.replace(tmp, path)  # atomic on POSIX and Windows
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not write SEC cache entry %s: %s", path.name, exc)
            return False
        self._maybe_prune()
        return True

    def _maybe_prune(self) -> None:
        root = str(self.root.resolve())
        if time.time() - self._last_prune.get(root, 0.0) < 3600:
            return
        self._last_prune[root] = time.time()
        self.prune()

    def prune(self, max_age: timedelta | None = None, max_entries: int | None = None) -> int:
        """Delete entries older than ``max_age`` and the oldest beyond ``max_entries``."""
        max_age = max_age or self.MAX_AGE
        max_entries = max_entries or self.MAX_ENTRIES
        removed = 0
        try:
            now = time.time()
            entries = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for i, path in enumerate(entries):
                if i >= max_entries or now - path.stat().st_mtime > max_age.total_seconds():
                    path.unlink(missing_ok=True)
                    removed += 1
        except OSError as exc:
            logger.warning("Could not prune the SEC cache: %s", exc)
        return removed


def urllib_get(url: str, headers: Mapping[str, str], timeout: float = 20.0) -> bytes:
    """Default HTTP GET (standard library only); returns the decoded response body."""
    request = urllib.request.Request(url, headers={**headers, "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URLs
        body = response.read()
        encoding = (response.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "deflate":
        return zlib.decompress(body)
    return body


def _map_http_error(what: str, exc: Exception) -> tuple[DataProviderError, bool]:
    """A typed error for a failed SEC request, and whether retrying could help."""
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status", None)
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(code, int) and not isinstance(code, bool):
        if code == 404:
            return TickerNotFoundError(f"SEC EDGAR has no {what}.", detail=detail), False
        if code == 429:
            return RateLimitError("SEC EDGAR is rate-limiting requests. Wait a few minutes and try again.",
                                  detail=detail), True
        if code == 403:
            return ProviderConnectionError(
                "SEC EDGAR refused the request (HTTP 403). The SEC blocks requests without an accepted "
                f"User-Agent (set {SEC_USER_AGENT_ENV} to 'Your Name your@email.com') and clients that exceed "
                "its rate limit; wait about ten minutes before retrying.", detail=detail), False
        if code >= 500:
            return ProviderConnectionError(f"SEC EDGAR is temporarily unavailable (HTTP {code}) for {what}.",
                                           detail=detail), True
        return DataProviderError(f"SEC EDGAR could not provide {what} (HTTP {code}).", detail=detail), False
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError)):
        return ProviderConnectionError(
            f"Could not reach SEC EDGAR for {what} (network error or blocked connection).", detail=detail), True
    return DataProviderError(f"SEC EDGAR could not provide {what}.", detail=detail), False


def _parse_json_body(body: Any, what: str) -> Any:
    if isinstance(body, (dict, list)):
        return body
    try:
        text = body.decode("utf-8-sig") if isinstance(body, (bytes, bytearray)) else str(body)
        return json.loads(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise MalformedDataError(
            f"SEC EDGAR returned something other than JSON for {what} (the request may have been blocked).",
            detail=str(exc)[:200],
        ) from exc


@dataclass(frozen=True)
class _Response:
    payload: Any
    url: str
    fetched_at: datetime
    cache_status: str
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# SEC EDGAR: XBRL facts -> canonical statements
# --------------------------------------------------------------------------- #

# Canonical item -> us-gaap concepts in priority order. The first concept with a value for
# a period wins (companies switch concepts over time, e.g. SalesRevenueNet before ASC 606).
SEC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet",
                "SalesRevenueServicesNet"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"),
    "depreciation_amortization": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                                  "DepreciationAmortizationAndAccretionNet"),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "diluted_shares": ("WeightedAverageNumberOfDilutedSharesOutstanding",
                       "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "total_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "Cash"),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "shares_outstanding": ("CommonStockSharesOutstanding",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "dividends_paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
}
# Reported as positive payments; negated to the cash-flow sign convention (outflows < 0).
SEC_NEGATED_ITEMS = frozenset({"capital_expenditure", "dividends_paid"})
# total_debt has no single us-gaap concept: it is the sum of the components a company reports.
SEC_DEBT_LONG_TERM = ("LongTermDebt",)                       # includes current maturities
SEC_DEBT_LONG_TERM_SPLIT = ("LongTermDebtNoncurrent", "LongTermDebtCurrent")
SEC_DEBT_SHORT_TERM = ("ShortTermBorrowings", "CommercialPaper")  # first reported one only (overlap)
SEC_SHARES_COVER = "EntityCommonStockSharesOutstanding"        # dei, cover-page date

SEC_ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
SEC_QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A"})
SEC_STATEMENT_FORMS = SEC_ANNUAL_FORMS | SEC_QUARTERLY_FORMS
_SEC_MONETARY_ITEMS = tuple(i for i in SEC_CONCEPTS if i not in PER_SHARE_ITEMS and i not in SHARE_COUNT_ITEMS)


@dataclass(frozen=True)
class XbrlFact:
    """One companyfacts observation. ``start`` is None for an instant (balance-sheet) value."""

    concept: str
    unit: str
    start: date | None
    end: date
    value: float
    filed: date
    form: str
    fp: str | None
    accn: str | None

    @property
    def days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days


def parse_companyfacts(payload: Any, concepts: Iterable[str] | None = None) -> tuple[dict[str, list[XbrlFact]], int]:
    """Observations by ``taxonomy:concept`` from a companyfacts response, and how many were unreadable.

    Only ``concepts`` (names without taxonomy) are read when given, to keep memory small.
    """
    if not isinstance(payload, Mapping) or not isinstance(payload.get("facts"), Mapping):
        raise MalformedDataError("The SEC companyfacts response has no 'facts' section.")
    wanted = set(concepts) if concepts is not None else None
    out: dict[str, list[XbrlFact]] = {}
    unreadable = 0
    for taxonomy, block in payload["facts"].items():
        if not isinstance(block, Mapping):
            continue
        for concept, body in block.items():
            if wanted is not None and concept not in wanted:
                continue
            units = body.get("units") if isinstance(body, Mapping) else None
            if not isinstance(units, Mapping):
                unreadable += 1
                continue
            for unit, entries in units.items():
                for entry in entries if isinstance(entries, list) else ():
                    try:
                        value = entry["val"]
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise ValueError("not a finite number")
                        start = date.fromisoformat(entry["start"]) if entry.get("start") else None
                        end = date.fromisoformat(entry["end"])
                        if start is not None and start > end:
                            raise ValueError("start after end")
                        fact = XbrlFact(concept=concept, unit=str(unit), start=start, end=end, value=float(value),
                                        filed=date.fromisoformat(entry["filed"]), form=str(entry.get("form") or ""),
                                        fp=entry.get("fp"), accn=entry.get("accn"))
                    except (KeyError, TypeError, ValueError, AttributeError):
                        unreadable += 1
                        continue
                    out.setdefault(f"{taxonomy}:{concept}", []).append(fact)
    return out, unreadable


@dataclass(frozen=True)
class _Chosen:
    """The value used for one (concept, period), after de-duplication across filings."""

    value: float
    start: date | None
    end: date
    filed: date
    concept: str
    derived: bool = False
    revised_from: tuple[float, date] | None = None   # another filing's different value for the period


def _choose(observations: list[XbrlFact], latest: bool) -> _Chosen:
    """The first-filed value for a period (or the latest filed when ``latest``), noting revisions."""
    ordered = sorted(observations, key=lambda o: (o.filed, o.accn or ""))
    pick = ordered[-1] if latest else ordered[0]
    other = next((o for o in (ordered if latest else reversed(ordered))
                  if abs(o.value - pick.value) > 1e-9 * max(1.0, abs(pick.value))), None)
    return _Chosen(pick.value, pick.start, pick.end, pick.filed, pick.concept,
                   revised_from=(other.value, other.filed) if other else None)


def _in(days: int | None, bounds: tuple[int, int]) -> bool:
    return days is not None and bounds[0] <= days <= bounds[1]


def _dedupe(observations: list[XbrlFact], latest: bool) -> dict[tuple[date | None, date], _Chosen]:
    groups: dict[tuple[date | None, date], list[XbrlFact]] = {}
    for o in observations:
        groups.setdefault((o.start, o.end), []).append(o)
    return {key: _choose(group, latest) for key, group in groups.items()}


def _closest(candidates: list[_Chosen], target_days: int) -> _Chosen:
    return min(candidates, key=lambda c: (abs((c.end - c.start).days - target_days), c.start))


def _annual_flows(observations: list[XbrlFact], latest: bool) -> dict[date, _Chosen]:
    """Fiscal-year values: 10-K facts with fp FY and a duration of about one year."""
    usable = [o for o in observations if o.form in SEC_ANNUAL_FORMS and o.fp in (None, "FY") and _in(o.days, _YEAR_DAYS)]
    by_end: dict[date, list[_Chosen]] = {}
    for chosen in _dedupe(usable, latest).values():
        by_end.setdefault(chosen.end, []).append(chosen)
    return {end: _closest(group, 365) for end, group in by_end.items()}


def _quarterly_flows(observations: list[XbrlFact], latest: bool, additive: bool) -> dict[date, _Chosen]:
    """Quarter values: reported three-month figures, plus — for additive items — quarters derived
    from consecutive year-to-date totals with the same start (10-Q cash flows are year-to-date only,
    and no filing reports the fourth quarter on its own: Q4 = fiscal year − nine months)."""
    usable = [o for o in observations if o.form in SEC_STATEMENT_FORMS and _in(o.days, (_QUARTER_DAYS[0], _YEAR_DAYS[1]))]
    chosen = _dedupe(usable, latest)
    direct_by_end: dict[date, list[_Chosen]] = {}
    for c in chosen.values():
        if _in((c.end - c.start).days, _QUARTER_DAYS):
            direct_by_end.setdefault(c.end, []).append(c)
    out = {end: _closest(group, 91) for end, group in direct_by_end.items()}
    if not additive:
        return out
    by_start: dict[date, list[_Chosen]] = {}
    for c in chosen.values():
        by_start.setdefault(c.start, []).append(c)
    for _start, group in sorted(by_start.items()):
        group.sort(key=lambda c: c.end)
        for shorter, longer in zip(group, group[1:]):
            if longer.end in out or not _in((longer.end - shorter.end).days, _QUARTER_DAYS):
                continue
            out[longer.end] = _Chosen(longer.value - shorter.value, shorter.end + timedelta(days=1), longer.end,
                                      max(shorter.filed, longer.filed), longer.concept, derived=True)
    return out


def _instants(observations: list[XbrlFact], latest: bool) -> dict[date, _Chosen]:
    usable = [o for o in observations if o.start is None and o.form in SEC_STATEMENT_FORMS]
    return {end: c for (_, end), c in _dedupe(usable, latest).items()}


def _statement_currency(facts: Mapping[str, list[XbrlFact]]) -> tuple[str | None, list[str]]:
    """The currency of the monetary items: the ISO code with the most observations."""
    counts: Counter[str] = Counter()
    for item in _SEC_MONETARY_ITEMS:
        for concept in SEC_CONCEPTS[item]:
            for o in facts.get(f"us-gaap:{concept}", ()):
                if re.fullmatch(r"[A-Z]{3}", o.unit):
                    counts[o.unit] += 1
    if not counts:
        return None, []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] != "USD", kv[0]))
    currency = ranked[0][0]
    notes = []
    if len(ranked) > 1:
        others = ", ".join(code for code, _ in ranked[1:])
        notes.append(f"Some values are reported in other currencies ({others}); only {currency} values are used.")
    return currency, notes


@dataclass
class _SecBuild:
    columns: list[date]
    values: dict[str, dict[date, _Chosen]]
    sources: dict[str, tuple[str, ...]]
    warnings: list[str]
    currency: str | None


def build_sec_statements(facts: Mapping[str, list[XbrlFact]], period: StatementPeriod, *,
                         latest: bool = False, max_periods: int | None = 12) -> _SecBuild:
    """Assemble canonical statement values from (already as_of-filtered) XBRL observations."""
    currency, warnings = _statement_currency(facts)
    if currency is None:
        warnings.append("No monetary values in a currency were found; only share data could be read.")

    def unit_for(item: str) -> str | None:
        if item in PER_SHARE_ITEMS:
            return f"{currency}/shares" if currency else None
        if item in SHARE_COUNT_ITEMS:
            return "shares"
        return currency

    def observations(concept: str, unit: str | None) -> list[XbrlFact]:
        return [o for o in facts.get(f"us-gaap:{concept}", ()) if o.unit == unit] if unit else []

    values: dict[str, dict[date, _Chosen]] = {}
    for item, concepts in SEC_CONCEPTS.items():
        per_end: dict[date, _Chosen] = {}
        for concept in concepts:  # priority order: an earlier concept wins a period
            obs = observations(concept, unit_for(item))
            if not obs:
                continue
            if STATEMENT_OF[item] == "balance":
                found = _instants(obs, latest)
            elif period is StatementPeriod.ANNUAL:
                found = _annual_flows(obs, latest)
            else:
                found = _quarterly_flows(obs, latest, item in ADDITIVE_FLOW_ITEMS)
            for end, chosen in found.items():
                per_end.setdefault(end, chosen)
        if per_end:
            if item in SEC_NEGATED_ITEMS:
                per_end = {e: replace(c, value=-c.value, revised_from=None if c.revised_from is None else
                                      (-c.revised_from[0], c.revised_from[1])) for e, c in per_end.items()}
            values[item] = per_end

    debt = _sec_total_debt({c: _instants(observations(c, currency), latest)
                            for c in SEC_DEBT_LONG_TERM + SEC_DEBT_LONG_TERM_SPLIT + SEC_DEBT_SHORT_TERM})
    if debt:
        values["total_debt"] = debt

    flow_ends = {e for item, v in values.items() if STATEMENT_OF[item] != "balance" for e in v}
    if flow_ends:
        columns = sorted(flow_ends, reverse=True)
    else:  # balance-sheet data only: use the period ends of the matching filings
        forms = SEC_ANNUAL_FORMS if period is StatementPeriod.ANNUAL else SEC_STATEMENT_FORMS
        columns = sorted({o.end for lst in facts.values() for o in lst
                          if o.start is None and o.form in forms and o.concept in _balance_concepts()}, reverse=True)
    if max_periods is not None:
        columns = columns[:max_periods]
    keep = set(columns)
    values = {item: {e: c for e, c in v.items() if e in keep} for item, v in values.items()}
    values = {item: v for item, v in values.items() if v}

    sources = {item: tuple(sorted({f"us-gaap:{c.concept}" for c in v.values()}, key=lambda s: s))
               for item, v in values.items()}
    derived = sorted({(item, e) for item, v in values.items() for e, c in v.items() if c.derived},
                     key=lambda x: (x[1], x[0]))
    if derived:
        items = sorted({item for item, _ in derived})
        warnings.append(
            f"{len(derived)} quarterly value(s) were derived as the difference of year-to-date totals (10-Q cash "
            "flows are year-to-date, and Q4 = fiscal year − nine months), not reported directly: "
            + ", ".join(items) + "."
        )
    revised = [(item, e, c) for item, v in values.items() for e, c in sorted(v.items()) if c.revised_from]
    if revised:
        which = "first filed (as originally reported)" if not latest else "the latest filed"
        examples = "; ".join(f"{item} for the period ended {e}: {c.value:g} (filed {c.filed}) vs {c.revised_from[0]:g} "
                             f"(filed {c.revised_from[1]})" for item, e, c in revised[:3])
        warnings.append(f"{len(revised)} value(s) differ between filings (restated or reclassified); shown is the value "
                        f"{which}. E.g. {examples}.")
    if "total_debt" in values:
        warnings.append("total_debt is the sum of the debt concepts the company reported at each date ("
                        + ", ".join(sources["total_debt"]) + "); components it did not tag, and lease liabilities, "
                        "are not included.")
    return _SecBuild(columns=columns, values=values, sources=sources, warnings=warnings, currency=currency)


def _balance_concepts() -> set[str]:
    return {c for item, cs in SEC_CONCEPTS.items() if STATEMENT_OF[item] == "balance" for c in cs}


def _sec_total_debt(instants: Mapping[str, dict[date, _Chosen]]) -> dict[date, _Chosen]:
    """Long-term debt (LongTermDebt, else non-current + current portion) plus short-term borrowings."""
    out: dict[date, _Chosen] = {}
    ends = {e for v in instants.values() for e in v}
    for end in ends:
        parts: list[_Chosen] = []
        if end in instants.get("LongTermDebt", {}):
            parts.append(instants["LongTermDebt"][end])
        elif end in instants.get("LongTermDebtNoncurrent", {}):
            parts.append(instants["LongTermDebtNoncurrent"][end])
            if end in instants.get("LongTermDebtCurrent", {}):
                parts.append(instants["LongTermDebtCurrent"][end])
        if not parts:
            continue
        for concept in SEC_DEBT_SHORT_TERM:
            if end in instants.get(concept, {}):
                parts.append(instants[concept][end])
                break
        out[end] = _Chosen(sum(p.value for p in parts), None, end, max(p.filed for p in parts),
                           "+".join(p.concept for p in parts))
    return out


def _frame_from(items: tuple[str, ...], columns: list[date], values: Mapping[str, dict[date, _Chosen]]) -> pd.DataFrame:
    rows = {item: [values[item][c].value if c in values[item] else np.nan for c in columns]
            for item in items if item in values}
    if not rows:
        return _empty_frame()
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=pd.DatetimeIndex([pd.Timestamp(c) for c in columns]),
                                   dtype="float64")
    return frame.loc[:, frame.notna().any(axis=0)]


def _filing_rows(block: Any) -> list[dict[str, Any]]:
    """Rows from the column-oriented filing arrays of a submissions response."""
    if not isinstance(block, Mapping):
        return []
    accessions = block.get("accessionNumber")
    if not isinstance(accessions, list):
        return []
    keys = ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "primaryDocument",
            "primaryDocDescription", "items")
    columns = {k: block.get(k) if isinstance(block.get(k), list) else [] for k in keys}
    return [{k: (columns[k][i] if i < len(columns[k]) else None) for k in keys} for i in range(len(accessions))]


# --------------------------------------------------------------------------- #
# SEC EDGAR provider
# --------------------------------------------------------------------------- #


class SecEdgarProvider(FundamentalsProvider):
    """Statements, profile, filings and earnings releases straight from SEC EDGAR.

    * ticker -> CIK through ``company_tickers.json`` (a numeric ticker is taken as a CIK);
    * statements from ``/api/xbrl/companyfacts`` (us-gaap concepts with alternates);
      annual values from 10-K facts with ``fp`` FY and a one-year duration, quarterly
      values from 10-Q facts (and quarters derived from year-to-date totals);
    * restated values are de-duplicated: by default each period keeps the value first
      filed, so ``filed`` is the date that exact figure became public; with ``as_of``
      the latest value filed by then is used (what an investor could know on that
      date). ``restated=True/False`` overrides either default;
    * a User-Agent "Name email" from ``QLP_SEC_USER_AGENT`` is required (never
      hard-coded); requests are throttled and cached as JSON under ``QLP_DATA_DIR/sec``.
    """

    key = "sec_edgar"
    name = "SEC EDGAR"
    description = (
        "Primary-source data from the U.S. Securities and Exchange Commission: XBRL financial statement facts "
        "(companyfacts) and the filings index (submissions). Covers companies that file 10-K/10-Q reports; each "
        "value keeps its filing date for point-in-time use."
    )
    capabilities = frozenset({"profile", "statements", "events", "filings"})

    def __init__(
        self,
        user_agent: str | None = None,
        http_get: Callable[[str, Mapping[str, str]], Any] | None = None,
        cache_dir: Path | str | None = None,
        use_cache: bool = True,
        max_cache_age: timedelta = timedelta(hours=24),
        ticker_map_max_age: timedelta = timedelta(days=7),
        throttle: RequestThrottle | None = None,
        clock: Callable[[], datetime] | None = None,
        timeout: float = 20.0,
        retries: int = 1,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] | None = None,
        max_periods: int | None = 12,
        submission_pages: int = 3,
    ):
        self._user_agent = user_agent
        self._http_get = http_get or partial(urllib_get, timeout=timeout)
        self._cache = JsonCache(Path(cache_dir) if cache_dir is not None else data_dir() / "sec") if use_cache else None
        self.max_cache_age = max_cache_age
        self.ticker_map_max_age = ticker_map_max_age
        self._throttle = throttle or SEC_THROTTLE
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.retries = max(0, int(retries))
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep or time.sleep
        self.max_periods = max_periods
        self.submission_pages = max(0, int(submission_pages))

    # -- transport ----------------------------------------------------------- #

    @property
    def user_agent(self) -> str:
        value = self._user_agent if self._user_agent is not None else os.environ.get(SEC_USER_AGENT_ENV, "")
        return value.strip()

    def _require_user_agent(self) -> str:
        agent = self.user_agent
        if not agent:
            raise ProviderConnectionError(
                f"SEC EDGAR requires a User-Agent that identifies you. Set the environment variable "
                f"{SEC_USER_AGENT_ENV} to your name and e-mail address, e.g. 'Jane Doe jane@example.com'."
            )
        if any(ch in agent for ch in "\r\n") or len(agent) > 200:
            raise ProviderConnectionError(f"{SEC_USER_AGENT_ENV} must be a single line of at most 200 characters.")
        return agent

    def _download(self, url: str, what: str) -> Any:
        headers = {"User-Agent": self._require_user_agent(), "Accept": "application/json"}
        for attempt in range(self.retries + 1):
            self._throttle.wait()
            try:
                body = self._http_get(url, headers)
            except DataProviderError as exc:
                error, transient = exc, isinstance(exc, (RateLimitError, ProviderConnectionError))
            except Exception as exc:  # noqa: BLE001 - mapped to typed errors
                error, transient = _map_http_error(what, exc)
                error.__cause__ = exc
            else:
                return _parse_json_body(body, what)
            if not transient or attempt == self.retries:
                raise error
            logger.warning("Transient SEC EDGAR error for %s (%s); retrying.", what, error.kind)
            self._sleep(self.backoff_seconds * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def _get_json(self, url: str, what: str, max_age: timedelta) -> _Response:
        """A fresh cache entry, else a download; a stale entry only when the SEC cannot be reached."""
        now = self._clock()
        cached = self._cache.load(url) if self._cache is not None else None
        if cached is not None and timedelta(0) <= now - cached[1] <= max_age:
            return _Response(cached[0], url, cached[1], "cached")
        try:
            payload = self._download(url, what)
        except (ProviderConnectionError, RateLimitError) as exc:
            if cached is None:
                raise
            note = (f"SEC EDGAR could not be reached ({exc.user_message}); using a copy cached on "
                    f"{cached[1]:%Y-%m-%d %H:%M} UTC, which may be out of date.")
            return _Response(cached[0], url, cached[1], "stale_cache", (note,))
        if self._cache is not None:
            self._cache.save(url, payload, now)
        return _Response(payload, url, now, "downloaded")

    # -- identifiers ----------------------------------------------------------- #

    def resolve_cik(self, ticker: str) -> tuple[str, str | None]:
        """``(10-digit CIK, company name)`` for a ticker; a purely numeric ticker is taken as the CIK."""
        symbol = validate_ticker(ticker)
        if symbol.isdigit():
            if len(symbol) > 10:
                raise TickerNotFoundError(f"'{symbol}' is not a valid SEC CIK (at most 10 digits).")
            return symbol.zfill(10), None
        response = self._get_json(SEC_TICKERS_URL, "the list of company tickers", self.ticker_map_max_age)
        table = _ticker_table(response.payload)
        for candidate in dict.fromkeys((symbol, symbol.replace(".", "-"), symbol.replace("-", "."))):
            if candidate in table:
                return table[candidate]
        raise TickerNotFoundError(
            f"'{symbol}' is not in the SEC's list of company tickers. SEC EDGAR covers companies that file with "
            "the SEC; funds, indices and most foreign listings are not included."
        )

    def _provenance(self, response: _Response, what: str, timeliness: Timeliness, **kw: Any) -> Provenance:
        warnings = tuple(response.warnings) + tuple(kw.pop("warnings", ()))
        return Provenance(source=f"{self.name}: {what}", retrieved_at=response.fetched_at, timeliness=timeliness,
                          source_url=response.url, cache_status=response.cache_status, warnings=warnings, **kw)

    def _facts(self, cik: str) -> tuple[_Response, dict[str, list[XbrlFact]], int]:
        url = SEC_FACTS_URL.format(cik=cik)
        response = self._get_json(url, f"XBRL financial data for CIK {cik}", self.max_cache_age)
        wanted = {c for cs in SEC_CONCEPTS.values() for c in cs}
        wanted |= set(SEC_DEBT_LONG_TERM + SEC_DEBT_LONG_TERM_SPLIT + SEC_DEBT_SHORT_TERM) | {SEC_SHARES_COVER}
        facts, unreadable = parse_companyfacts(response.payload, wanted)
        return response, facts, unreadable

    # -- statements -------------------------------------------------------------- #

    def statements(self, ticker: str, period: StatementPeriod = StatementPeriod.ANNUAL, *,
                   as_of: date | datetime | str | None = None, restated: bool | None = None) -> FinancialStatements:
        symbol = validate_ticker(ticker)
        period = StatementPeriod(period)
        cutoff = _as_of_cutoff(as_of)
        latest = (as_of is not None) if restated is None else bool(restated)
        cik, _ = self.resolve_cik(symbol)
        response, facts, unreadable = self._facts(cik)
        if cutoff is not None:
            facts = {k: [o for o in v if cutoff(o.filed)] for k, v in facts.items()}
        build = build_sec_statements(facts, period, latest=latest, max_periods=self.max_periods)
        if not build.columns or not build.values:
            by = f" filed by {as_of}" if as_of is not None else ""
            raise TickerNotFoundError(
                f"SEC EDGAR has no {period.value} 10-K/10-Q statement figures for '{symbol}'{by} "
                "(the company may not file XBRL statements, or files under IFRS as a foreign issuer)."
            )
        columns = [pd.Timestamp(c) for c in build.columns]
        value_filed = {(item, pd.Timestamp(e)): _filing_time(c.filed)
                       for item, v in build.values.items() for e, c in v.items()}
        filed: dict[pd.Timestamp, datetime] = {}
        for (_item, end), moment in value_filed.items():
            filed[end] = max(filed.get(end, moment), moment)
        starts: dict[pd.Timestamp, date] = {}
        for end in build.columns:
            found = Counter(v[end].start for item, v in build.values.items()
                            if STATEMENT_OF[item] != "balance" and end in v and v[end].start is not None)
            if found:
                starts[pd.Timestamp(end)] = found.most_common(1)[0][0]
        frames = {name: _frame_from(items, build.columns, build.values) for name, items in STATEMENT_ITEMS.items()}
        warnings = list(build.warnings)
        if unreadable:
            warnings.append(f"{unreadable} XBRL observation(s) could not be read and were skipped.")
        if as_of is not None:
            warnings.append(f"Point-in-time view: only values filed by {_iso(as_of)} are included.")
        prov = self._provenance(
            response, f"XBRL financial statements ({period.value})", Timeliness.PERIODIC_FILING,
            period_start=min(starts.values()) if starts else None, period_end=columns[0].date(),
            published_at=max(filed.values()) if filed else None, currency=build.currency,
            units=_statement_units(build.currency), warnings=tuple(warnings),
        )
        return FinancialStatements(
            ticker=symbol, currency=build.currency, income=frames["income"], balance=frames["balance"],
            cashflow=frames["cashflow"], period=period, provenance=prov, filed=filed, period_starts=starts,
            value_filed=value_filed, item_sources=build.sources,
        )

    # -- profile ------------------------------------------------------------------- #

    def profile(self, ticker: str) -> CompanyProfile:
        symbol = validate_ticker(ticker)
        cik, title = self.resolve_cik(symbol)
        response = self._get_json(SEC_SUBMISSIONS_URL.format(cik=cik), f"the filings index for CIK {cik}",
                                  self.max_cache_age)
        payload = response.payload
        if not isinstance(payload, Mapping):
            raise MalformedDataError(f"The SEC submissions response for CIK {cik} is not a JSON object.")
        tickers = [str(t).upper() for t in payload.get("tickers") or [] if t]
        exchanges = list(payload.get("exchanges") or [])
        exchange = None
        if symbol in tickers and tickers.index(symbol) < len(exchanges):
            exchange = exchanges[tickers.index(symbol)]
        elif exchanges:
            exchange = exchanges[0]
        warnings = ["SEC EDGAR has no sector classification; the industry is the SIC description."]
        shares = shares_as_of = currency = None
        try:
            facts_response, facts, _ = self._facts(cik)
            currency, _ = _statement_currency(facts)
            cover = [o for o in facts.get(f"dei:{SEC_SHARES_COVER}", ()) if o.unit == "shares" and o.value > 0]
            if cover:
                newest = max(cover, key=lambda o: (o.end, o.filed))
                shares, shares_as_of = newest.value, newest.end
                warnings.append(f"Shares outstanding as stated on the cover of the {newest.form} filed {newest.filed}.")
            else:
                warnings.append("No share count on the filings' cover pages.")
        except DataProviderError as exc:
            warnings.append(f"Shares outstanding and currency unknown: {exc.user_message}")
        prov = self._provenance(response, "company filings index (submissions)", Timeliness.STATIC,
                                currency=currency, warnings=tuple(warnings))
        return CompanyProfile(
            ticker=symbol,
            name=_clean_text(payload.get("name"), 200) or title,
            exchange=_clean_text(exchange, 50),
            sector=None,
            industry=_clean_text(payload.get("sicDescription"), 200),
            currency=currency,
            country=_sec_country(payload.get("addresses")),
            shares_outstanding=shares,
            provenance=prov,
            shares_as_of=shares_as_of,
        )

    # -- filings and events ----------------------------------------------------------- #

    def _submission_rows(self, cik: str, keep: Callable[[dict[str, Any]], bool], limit: int,
                         ) -> tuple[list[dict[str, Any]], _Response]:
        """Matching filing rows, newest first: the 'recent' block, then older pages while more are needed."""
        response = self._get_json(SEC_SUBMISSIONS_URL.format(cik=cik), f"the filings index for CIK {cik}",
                                  self.max_cache_age)
        payload = response.payload
        if not isinstance(payload, Mapping) or not isinstance(payload.get("filings"), Mapping):
            raise MalformedDataError(f"The SEC submissions response for CIK {cik} has no 'filings' section.")
        rows = [r for r in _filing_rows(payload["filings"].get("recent")) if keep(r)]
        pages = payload["filings"].get("files") or []
        for page in pages[: self.submission_pages]:
            if len(rows) >= limit:
                break
            name = page.get("name") if isinstance(page, Mapping) else None
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name):
                continue
            older = self._get_json(SEC_SUBMISSIONS_PAGE_URL.format(name=name), f"older filings for CIK {cik}",
                                   self.max_cache_age)
            rows.extend(r for r in _filing_rows(older.payload) if keep(r))
        rows = [r for r in rows if _to_date(r.get("filingDate")) is not None]
        rows.sort(key=lambda r: (_to_date(r["filingDate"]), str(r.get("accessionNumber") or "")), reverse=True)
        return rows[:limit], response

    def _filing(self, symbol: str, cik: str, row: dict[str, Any], response: _Response) -> Filing:
        accession = str(row.get("accessionNumber") or "")
        folder = accession.replace("-", "")
        document = row.get("primaryDocument")
        if isinstance(document, str) and re.fullmatch(r"[A-Za-z0-9_.\-/]+", document) and ".." not in document:
            url = SEC_ARCHIVE_URL.format(cik=int(cik), folder=folder, document=document)
        else:
            url = SEC_ARCHIVE_URL.format(cik=int(cik), folder=folder, document=f"{accession}-index.htm")
        filed_day = _to_date(row.get("filingDate"))
        accepted = _to_datetime_utc(row.get("acceptanceDateTime")) or _filing_time(filed_day)
        items = tuple(i.strip() for i in str(row.get("items") or "").split(",") if i.strip())
        prov = self._provenance(response, "filings index (submissions)", Timeliness.EVENT, published_at=accepted)
        return Filing(ticker=symbol, cik=cik, form=str(row.get("form") or ""), filed=filed_day,
                      period=_to_date(row.get("reportDate")) if row.get("reportDate") else None, url=url,
                      provenance=prov, accession=accession or None,
                      description=_clean_text(row.get("primaryDocDescription"), 200), items=items)

    def filings(self, ticker: str, forms: Iterable[str] | None = ("10-K", "10-Q", "8-K"),
                limit: int = 20) -> list[Filing]:
        """Filings newest first; ``forms`` are matched exactly (amendments such as 10-K/A must be named)."""
        symbol = validate_ticker(ticker)
        limit = _check_limit(limit)
        wanted = None if forms is None else {str(f).strip().upper() for f in forms}
        cik, _ = self.resolve_cik(symbol)
        rows, response = self._submission_rows(
            cik, lambda r: wanted is None or str(r.get("form") or "").upper() in wanted, limit)
        return [self._filing(symbol, cik, r, response) for r in rows]

    def events(self, ticker: str, *, limit: int = 20) -> list[CorporateEvent]:
        """Earnings releases: Form 8-K filings with item 2.02 (Results of Operations), newest first."""
        symbol = validate_ticker(ticker)
        limit = _check_limit(limit)
        cik, _ = self.resolve_cik(symbol)
        rows, response = self._submission_rows(
            cik, lambda r: str(r.get("form") or "").upper() in {"8-K", "8-K/A"}
            and "2.02" in str(r.get("items") or ""), limit)
        out = []
        for row in rows:
            filing = self._filing(symbol, cik, row, response)
            detail = (f"Form {filing.form} item 2.02 (Results of Operations and Financial Condition) filed "
                      f"{filing.filed}: {filing.url}")
            prov = replace(filing.provenance, period_start=filing.filed, period_end=filing.filed)
            out.append(CorporateEvent(symbol, EARNINGS_RELEASE, filing.filed, True, detail, prov))
        return out

    def news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        raise NotSupportedError("SEC EDGAR does not publish news; use the Yahoo Finance provider for headlines.")

    def insider_transactions(self, ticker: str, limit: int = 50) -> list[InsiderTransaction]:
        raise NotSupportedError(
            "Insider transactions are not parsed from SEC EDGAR; list Form 4 filings with filings(forms=('4',)), "
            "or use the Yahoo Finance provider."
        )


def _ticker_table(payload: Any) -> dict[str, tuple[str, str | None]]:
    """Ticker -> (10-digit CIK, title) from company_tickers.json (or the 'fields'/'data' variant)."""
    rows: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("fields"), list) and isinstance(payload.get("data"), list):
        fields = payload["fields"]
        rows = [dict(zip(fields, r)) for r in payload["data"] if isinstance(r, list)]
    elif isinstance(payload, Mapping):
        rows = [r for r in payload.values() if isinstance(r, Mapping)]
    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, Mapping)]
    table: dict[str, tuple[str, str | None]] = {}
    for row in rows:
        ticker, cik = row.get("ticker"), row.get("cik_str", row.get("cik"))
        if not ticker or cik is None or isinstance(cik, bool):
            continue
        try:
            number = int(cik)
        except (TypeError, ValueError):
            continue
        table.setdefault(str(ticker).strip().upper(), (f"{number:010d}", _clean_text(row.get("title") or row.get("name"))))
    if not table:
        raise MalformedDataError("The SEC company ticker list is empty or in an unexpected format.")
    return table


def _sec_country(addresses: Any) -> str | None:
    business = addresses.get("business") if isinstance(addresses, Mapping) else None
    if not isinstance(business, Mapping):
        return None
    country = _clean_text(business.get("country"), 100)
    if country:
        return country
    foreign = business.get("isForeignLocation")
    if foreign in (0, False):
        return "United States"
    if foreign in (1, True):
        return _clean_text(business.get("stateOrCountryDescription"), 100)
    return None


FUNDAMENTALS_PROVIDERS: dict[str, type[FundamentalsProvider]] = {
    YahooFundamentalsProvider.key: YahooFundamentalsProvider,
    SecEdgarProvider.key: SecEdgarProvider,
}


# --------------------------------------------------------------------------- #
# Fundamental metrics
# --------------------------------------------------------------------------- #

METRIC_KEYS = ("revenue_growth_yoy", "earnings_growth_yoy", "gross_margin", "operating_margin", "net_margin",
               "free_cash_flow", "fcf_margin", "market_cap", "enterprise_value", "pe_trailing", "price_to_sales",
               "ev_to_ebitda", "fcf_yield", "net_debt_to_ebitda", "debt_to_equity", "current_ratio",
               "cash_conversion")
FRACTION = "fraction"   # 0.25 means 25 %
MULTIPLE = "multiple"   # times, e.g. a P/E of 18.0
_FLOW_STALE_DAYS = {"ttm": 200, "annual": 460}

Value = Fact | Calculated


@dataclass
class _Basis:
    """Flow values on one period basis: TTM (sum of four quarters) or one fiscal year."""

    kind: str                       # "ttm" | "annual"
    label: str                      # e.g. "TTM: four quarters ended 2024-03-31"
    end: pd.Timestamp | None
    current: dict[str, Value] = field(default_factory=dict)
    prior: dict[str, Value] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)


def _consecutive(ends: list[pd.Timestamp], bounds: tuple[int, int]) -> bool:
    return all(_in((a - b).days, bounds) for a, b in zip(ends, ends[1:]))


def _flow_end(s: FinancialStatements) -> pd.Timestamp | None:
    """The newest period of the income and cash-flow statements (the end of any flow basis)."""
    ends = [pd.Timestamp(f.columns[0]) for f in (s.income, s.cashflow) if len(f.columns)]
    return max(ends) if ends else None


def _sum_quarters(q: FinancialStatements, item: str, window: list[pd.Timestamp]) -> tuple[Calculated | None, str]:
    if len(window) < 4 or not _consecutive(window, _QUARTER_DAYS):
        return None, "fewer than four consecutive quarters"
    facts = [q.fact(item, c) for c in window]
    gaps = [f"{c:%Y-%m-%d}" for c, f in zip(window, facts) if f.value is None]
    if gaps:
        return None, "no value for the quarter(s) ended " + ", ".join(gaps)
    return Calculated(
        f"{item} (TTM to {window[0]:%Y-%m-%d})", float(sum(f.value for f in facts)),
        f"{item} TTM = sum of {item} over the four quarters ended " + ", ".join(f"{c:%Y-%m-%d}" for c in window),
        tuple(facts), unit=item_unit(item, q.currency)), ""


def _ttm_basis(q: FinancialStatements, items: Iterable[str]) -> _Basis:
    end = _flow_end(q)
    basis = _Basis("ttm", f"TTM: four quarters ended {end:%Y-%m-%d}" if end is not None else "TTM", end)
    for item in items:
        frame = q.frame_for(item)
        cols = [pd.Timestamp(c) for c in frame.columns]
        if item not in frame.index or end is None:
            basis.missing[item] = "not reported quarterly"
            continue
        if not cols or cols[0] != end:
            basis.missing[item] = f"its statement's latest quarter is not {end:%Y-%m-%d}"
            continue
        current, reason = _sum_quarters(q, item, cols[:4])
        if current is None:
            basis.missing[item] = reason
            continue
        basis.current[item] = current
        if len(cols) >= 8 and _in((cols[3] - cols[4]).days, _QUARTER_DAYS):
            prior, _ = _sum_quarters(q, item, cols[4:8])
            if prior is not None:
                basis.prior[item] = prior
    return basis


def _annual_basis(a: FinancialStatements, items: Iterable[str], note: str) -> _Basis:
    end = _flow_end(a)
    label = f"latest fiscal year, ended {end:%Y-%m-%d}" if end is not None else "latest fiscal year"
    basis = _Basis("annual", label + (f" ({note})" if note else ""), end)
    for item in items:
        frame = a.frame_for(item)
        cols = [pd.Timestamp(c) for c in frame.columns]
        if item not in frame.index or end is None or not cols or cols[0] != end:
            basis.missing[item] = "not reported for the latest fiscal year"
            continue
        fact = a.fact(item, cols[0])
        if fact.value is None:
            basis.missing[item] = f"not reported for the fiscal year ended {end:%Y-%m-%d}"
            continue
        basis.current[item] = fact
        if len(cols) > 1 and _in((cols[0] - cols[1]).days, _YEAR_DAYS):
            prior = a.fact(item, cols[1])
            if prior.value is not None:
                basis.prior[item] = prior
    return basis


def _unavailable(name: str, formula: str, reason: str, unit: str | None,
                 inputs: tuple[Value, ...] = (), warnings: tuple[str, ...] = ()) -> Calculated:
    return Calculated(name, None, formula, inputs, unit=unit, reason=reason, warnings=warnings)


def _divide(name: str, formula: str, numerator: Value, denominator: Value, unit: str | None,
            warnings: tuple[str, ...] = (), minus_one: bool = False) -> Calculated:
    """numerator / denominator (− 1 for growth); None with a reason unless the denominator is positive."""
    inputs = (numerator, denominator)
    if numerator.value is None or denominator.value is None:
        missing = numerator if numerator.value is None else denominator
        return _unavailable(name, formula, f"{missing.name} is not available", unit, inputs, warnings)
    if denominator.value <= 0:
        return _unavailable(name, formula, f"{denominator.name} is {denominator.value:g}, not positive; the ratio is "
                            "not meaningful", unit, inputs, warnings)
    value = float(numerator.value) / float(denominator.value) - (1.0 if minus_one else 0.0)
    return Calculated(name, value, formula, inputs, unit=unit, warnings=warnings)


def fundamental_metrics(statements: FinancialStatements, price: Fact, shares: Fact | None = None,
                        ttm: FinancialStatements | None = None) -> dict[str, Calculated]:
    """Valuation, profitability and leverage metrics, each with a formula, inputs and basis.

    Flow items (revenue, earnings, cash flow) use TTM — the sum of the last four
    consecutive quarters, from ``ttm`` (or ``statements`` when it is quarterly) — when
    every item a metric needs is available that way, otherwise the latest fiscal year;
    the formula says which, and both parts of a ratio always share one basis. A basis is
    abandoned only for missing data, never to avoid an unwelcome value. Balance-sheet
    items use the most recent balance sheet. Ratios whose denominator is not positive,
    and price-based metrics when the price and statements are in different currencies,
    are None with a reason.
    """
    if ttm is not None:
        if ttm.period is not StatementPeriod.QUARTERLY:
            raise ValueError("ttm must be quarterly statements")
        if ttm.ticker != statements.ticker:
            raise ValueError(f"ttm is for {ttm.ticker}, statements for {statements.ticker}")
        if ttm.currency and statements.currency and ttm.currency != statements.currency:
            raise ValueError(f"ttm is in {ttm.currency}, statements in {statements.currency}")
    annual = statements if statements.period is StatementPeriod.ANNUAL else None
    quarterly = ttm if ttm is not None else (statements if statements.period is StatementPeriod.QUARTERLY else None)
    currency = statements.currency or (quarterly.currency if quarterly else None)
    flow_items = INCOME_ITEMS + CASHFLOW_ITEMS
    bases: list[_Basis] = []
    ttm_note = ""
    if quarterly is not None and not quarterly.is_empty:
        tb = _ttm_basis(quarterly, flow_items)
        bases.append(tb)
        ttm_note = "TTM not available for this metric"
    else:
        ttm_note = "TTM not used: no quarterly statements supplied"
    if annual is not None and not annual.is_empty:
        bases.append(_annual_basis(annual, flow_items, ttm_note))

    price_value = _positive(price.value)
    price_date = price.provenance.period_end or price.provenance.retrieved_at.date()
    price_currency = price.provenance.currency or (price.unit if price.unit and re.fullmatch(r"[A-Z]{3}", price.unit)
                                                   else None)
    currency_problem = None
    currency_warnings: tuple[str, ...] = ()
    if price_currency and currency and price_currency != currency:
        currency_problem = (f"the price is in {price_currency} but the statements are in {currency}; no currency "
                            "conversion is applied")
    elif not price_currency or not currency:
        currency_warnings = ("The currency of the price or the statements is unknown; they are assumed to match.",)

    def pick(*items: str, prior: bool = False) -> tuple[_Basis | None, str]:
        """The first basis on which every item is available (and its prior-year value, for growth)."""
        reasons = []
        for basis in bases:
            pool = basis.current
            absent = [i for i in items if i not in pool or (prior and i not in basis.prior)]
            if not absent:
                return basis, ""
            details = [basis.missing.get(i) or ("no value one year earlier" if i in pool else "not available")
                       for i in absent]
            reasons.append(f"{basis.label}: " + "; ".join(f"{i} {d}" for i, d in zip(absent, details)))
        return None, " | ".join(reasons) or "no statements"

    def stale(basis: _Basis) -> tuple[str, ...]:
        if basis.end is None:
            return ()
        lag = (pd.Timestamp(price_date) - basis.end).days
        if lag > _FLOW_STALE_DAYS[basis.kind]:
            return (f"The fundamentals ({basis.label}) end {lag} days before the price date {price_date}; they may "
                    "be out of date.",)
        return ()

    def latest_balance(item: str) -> Fact | None:
        candidates = []
        for source in (quarterly, annual):
            if source is None:
                continue
            frame = source.balance
            if item in frame.index and len(frame.columns):
                fact = source.fact(item, frame.columns[0])
                if fact.value is not None:
                    candidates.append((pd.Timestamp(frame.columns[0]), fact))
        return max(candidates, key=lambda c: c[0])[1] if candidates else None

    out: dict[str, Calculated] = {}

    # Growth ---------------------------------------------------------------------------------
    for key, item in (("revenue_growth_yoy", "revenue"), ("earnings_growth_yoy", "net_income")):
        basis, why = pick(item, prior=True)
        formula = f"{key} = {item} / {item} one year earlier − 1"
        if basis is None:
            out[key] = _unavailable(key, formula, why, FRACTION)
            continue
        cur, prev = basis.current[item], basis.prior[item]
        result = _divide(key, f"{formula}; basis: {basis.label}", cur, prev, FRACTION, stale(basis), minus_one=True)
        if result.value is None and prev.value is not None and prev.value <= 0:
            result = replace(result, reason=f"growth from a non-positive base ({prev.value:g}) is not meaningful")
        out[key] = result

    # Margins ----------------------------------------------------------------------------------
    for key, item in (("gross_margin", "gross_profit"), ("operating_margin", "operating_income"),
                      ("net_margin", "net_income")):
        formula = f"{key} = {item} / revenue"
        basis, why = pick(item, "revenue")
        out[key] = (_unavailable(key, formula, why, FRACTION) if basis is None else
                    _divide(key, f"{formula}; basis: {basis.label}", basis.current[item], basis.current["revenue"],
                            FRACTION, stale(basis)))

    # Free cash flow -----------------------------------------------------------------------------
    fcf: Calculated | None = None
    fcf_basis: _Basis | None = None
    fcf_reasons = []
    for basis in bases:
        pool = basis.current
        if "operating_cash_flow" in pool and "capital_expenditure" in pool:
            ocf, capex = pool["operating_cash_flow"], pool["capital_expenditure"]
            warns = stale(basis)
            if capex.value > 0:
                warns += ("capital_expenditure is positive (an inflow), contrary to the cash-flow sign convention; "
                          "its absolute value is subtracted.",)
            fcf = Calculated("free_cash_flow", float(ocf.value - abs(capex.value)),
                             f"free_cash_flow = operating_cash_flow − |capital_expenditure|; basis: {basis.label}",
                             (ocf, capex), unit=currency, warnings=warns)
            fcf_basis = basis
            break
        if "free_cash_flow" in pool:
            reported = pool["free_cash_flow"]
            fcf = Calculated("free_cash_flow", float(reported.value),
                             f"free_cash_flow as reported by the source (operating cash flow or capital expenditure "
                             f"not available); basis: {basis.label}", (reported,), unit=currency,
                             warnings=stale(basis))
            fcf_basis = basis
            break
        fcf_reasons.append(f"{basis.label}: operating_cash_flow or capital_expenditure not available")
    out["free_cash_flow"] = fcf or _unavailable(
        "free_cash_flow", "free_cash_flow = operating_cash_flow − |capital_expenditure|",
        " | ".join(fcf_reasons) or "no statements", currency)
    if fcf is not None and "revenue" in fcf_basis.current:
        out["fcf_margin"] = _divide("fcf_margin", f"fcf_margin = free_cash_flow / revenue; basis: {fcf_basis.label}",
                                    fcf, fcf_basis.current["revenue"], FRACTION, stale(fcf_basis))
    else:
        out["fcf_margin"] = _unavailable("fcf_margin", "fcf_margin = free_cash_flow / revenue",
                                         "free cash flow or revenue on the same basis is not available", FRACTION)

    # Market value ---------------------------------------------------------------------------------
    share_warnings: tuple[str, ...] = ()
    share_fact: Fact | None = shares if shares is not None and _positive(shares.value) is not None else None
    if share_fact is None:
        share_fact = latest_balance("shares_outstanding")
        if share_fact is not None:
            share_warnings = (f"Shares outstanding are from the balance sheet ({share_fact.name}); issuance or "
                              "buybacks since then are not reflected.",)
    mc_formula = "market_cap = price × shares_outstanding"
    mc_unit = price_currency or currency
    if price_value is None:
        market_cap = _unavailable("market_cap", mc_formula, f"the price ({price.value!r}) is not a positive number",
                                  mc_unit, (price,))
    elif share_fact is None or _positive(share_fact.value) is None:
        market_cap = _unavailable("market_cap", mc_formula, "no positive share count is available", mc_unit, (price,))
    else:
        market_cap = Calculated("market_cap", price_value * float(share_fact.value), mc_formula, (price, share_fact),
                                unit=mc_unit, warnings=share_warnings)
    out["market_cap"] = market_cap

    debt, cash = latest_balance("total_debt"), latest_balance("cash")
    ev_formula = "enterprise_value = market_cap + total_debt − cash"
    if currency_problem:
        out["enterprise_value"] = _unavailable("enterprise_value", ev_formula, currency_problem, currency)
    elif market_cap.value is None:
        out["enterprise_value"] = _unavailable("enterprise_value", ev_formula, "market_cap is not available",
                                               currency, (market_cap,))
    elif debt is None or cash is None:
        absent = " and ".join(n for n, f in (("total_debt", debt), ("cash", cash)) if f is None)
        out["enterprise_value"] = _unavailable("enterprise_value", ev_formula,
                                               f"{absent} not reported (a missing value is not treated as zero)",
                                               currency, (market_cap,))
    else:
        out["enterprise_value"] = Calculated(
            "enterprise_value", float(market_cap.value + debt.value - cash.value), ev_formula,
            (market_cap, debt, cash), unit=currency, warnings=currency_warnings)
    ev = out["enterprise_value"]

    # Valuation multiples -------------------------------------------------------------------------
    pe = None
    pe_reasons = []
    for basis in bases:
        if "eps_diluted" in basis.current:
            pe = _divide("pe_trailing", f"pe_trailing = price / eps_diluted; basis: {basis.label}",
                         price, basis.current["eps_diluted"], MULTIPLE, stale(basis) + currency_warnings)
            break
        if "net_income" in basis.current:
            pe = _divide("pe_trailing", f"pe_trailing = market_cap / net_income (diluted EPS not available); basis: "
                         f"{basis.label}", market_cap, basis.current["net_income"], MULTIPLE,
                         stale(basis) + currency_warnings)
            break
        pe_reasons.append(f"{basis.label}: eps_diluted and net_income not available")
    if currency_problem:
        pe = _unavailable("pe_trailing", "pe_trailing = price / eps_diluted", currency_problem, MULTIPLE)
    out["pe_trailing"] = pe or _unavailable("pe_trailing", "pe_trailing = price / eps_diluted",
                                            " | ".join(pe_reasons) or "no statements", MULTIPLE)

    ps_formula = "price_to_sales = market_cap / revenue"
    basis, why = pick("revenue")
    if currency_problem:
        out["price_to_sales"] = _unavailable("price_to_sales", ps_formula, currency_problem, MULTIPLE)
    elif basis is None:
        out["price_to_sales"] = _unavailable("price_to_sales", ps_formula, why, MULTIPLE)
    else:
        out["price_to_sales"] = _divide("price_to_sales", f"{ps_formula}; basis: {basis.label}", market_cap,
                                        basis.current["revenue"], MULTIPLE, stale(basis) + currency_warnings)

    ebitda: Value | None = None
    ebitda_reasons = []
    for basis in bases:
        pool = basis.current
        if "ebitda" in pool:
            ebitda = Calculated("ebitda", float(pool["ebitda"].value), f"EBITDA as reported; basis: {basis.label}",
                                (pool["ebitda"],), unit=currency, warnings=stale(basis))
            break
        if "operating_income" in pool and "depreciation_amortization" in pool:
            oi, da = pool["operating_income"], pool["depreciation_amortization"]
            ebitda = Calculated("ebitda", float(oi.value + abs(da.value)),
                                f"EBITDA ≈ operating_income + |depreciation_amortization| (not reported; "
                                f"approximation); basis: {basis.label}", (oi, da), unit=currency,
                                warnings=stale(basis))
            break
        ebitda_reasons.append(f"{basis.label}: ebitda (or operating_income and depreciation_amortization) not "
                              "available")
    ebitda_missing = " | ".join(ebitda_reasons) or "no statements"

    if currency_problem:
        out["ev_to_ebitda"] = _unavailable("ev_to_ebitda", "ev_to_ebitda = enterprise_value / EBITDA", currency_problem,
                                           MULTIPLE)
    elif ebitda is None:
        out["ev_to_ebitda"] = _unavailable("ev_to_ebitda", "ev_to_ebitda = enterprise_value / EBITDA", ebitda_missing,
                                           MULTIPLE, (ev,))
    else:
        out["ev_to_ebitda"] = _divide("ev_to_ebitda", "ev_to_ebitda = enterprise_value / EBITDA", ev, ebitda, MULTIPLE,
                                      currency_warnings)

    fy_formula = "fcf_yield = free_cash_flow / market_cap"
    if currency_problem:
        out["fcf_yield"] = _unavailable("fcf_yield", fy_formula, currency_problem, FRACTION)
    else:
        out["fcf_yield"] = _divide("fcf_yield", fy_formula, out["free_cash_flow"], market_cap, FRACTION,
                                   currency_warnings)

    # Leverage and liquidity ---------------------------------------------------------------------------
    nd_formula = "net_debt_to_ebitda = (total_debt − cash) / EBITDA"
    if debt is None or cash is None:
        absent = " and ".join(n for n, f in (("total_debt", debt), ("cash", cash)) if f is None)
        out["net_debt_to_ebitda"] = _unavailable("net_debt_to_ebitda", nd_formula, f"{absent} not reported",
                                                 MULTIPLE)
    elif ebitda is None:
        out["net_debt_to_ebitda"] = _unavailable("net_debt_to_ebitda", nd_formula, ebitda_missing, MULTIPLE,
                                                 (debt, cash))
    else:
        net_debt = Calculated("net_debt", float(debt.value - cash.value), "net_debt = total_debt − cash", (debt, cash),
                              unit=currency)
        result = _divide("net_debt_to_ebitda", nd_formula, net_debt, ebitda, MULTIPLE)
        if result.value is not None and result.value < 0:
            result = replace(result, warnings=result.warnings + ("Negative: cash exceeds debt (net cash).",))
        out["net_debt_to_ebitda"] = result

    for key, num_item, den_item in (("debt_to_equity", "total_debt", "total_equity"),
                                    ("current_ratio", "current_assets", "current_liabilities")):
        formula = f"{key} = {num_item} / {den_item}"
        num, den = latest_balance(num_item), latest_balance(den_item)
        if num is None or den is None:
            absent = " and ".join(n for n, f in ((num_item, num), (den_item, den)) if f is None)
            out[key] = _unavailable(key, formula, f"{absent} not reported", MULTIPLE)
            continue
        warns = ()
        if num.provenance.period_end != den.provenance.period_end:
            warns = (f"{num_item} and {den_item} come from balance sheets of different dates.",)
        out[key] = _divide(key, f"{formula}; balance sheet of {den.provenance.period_end}", num, den, MULTIPLE, warns)

    cc_formula = "cash_conversion = operating_cash_flow / net_income"
    basis, why = pick("operating_cash_flow", "net_income")
    out["cash_conversion"] = (_unavailable("cash_conversion", cc_formula, why, MULTIPLE) if basis is None else
                              _divide("cash_conversion", f"{cc_formula}; basis: {basis.label}",
                                      basis.current["operating_cash_flow"], basis.current["net_income"], MULTIPLE,
                                      stale(basis)))
    return {key: out[key] for key in METRIC_KEYS}


# --------------------------------------------------------------------------- #
# Statement quality
# --------------------------------------------------------------------------- #

_MUST_BE_POSITIVE = ("total_assets", "shares_outstanding", "diluted_shares")
_MUST_BE_NON_NEGATIVE = ("revenue", "cash", "total_debt", "current_assets", "current_liabilities", "total_liabilities")
_MUST_BE_NON_POSITIVE = ("capital_expenditure", "dividends_paid")   # cash-flow sign convention: outflows < 0


def statements_quality(statements: FinancialStatements, today: date | None = None) -> list[str]:
    """Plain-language warnings about the statements: missing items, stale or irregular periods,
    gaps in the latest period, sign anomalies and accounting-identity breaks.

    ``today`` defaults to the retrieval date. An empty list means none of these checks fired,
    not that the figures are correct.
    """
    if statements.is_empty:
        return ["No statement data: every statement is empty."]
    today = today or statements.provenance.retrieved_at.date()
    annual = statements.period is StatementPeriod.ANNUAL
    spacing = _YEAR_DAYS if annual else _QUARTER_DAYS
    warnings: list[str] = []
    if statements.currency is None:
        warnings.append("The reporting currency is unknown.")

    for name, items in STATEMENT_ITEMS.items():
        frame = statements.frames[name]
        label = STATEMENT_LABELS[name]
        if frame.empty:
            warnings.append(f"{label}: no data.")
            continue
        absent = [i for i in items if i not in frame.index or frame.loc[i].isna().all()]
        if absent:
            warnings.append(f"{label}: not reported — {', '.join(absent)}.")
        cols = [pd.Timestamp(c) for c in frame.columns]
        if cols != sorted(cols, reverse=True) or len(set(cols)) != len(cols):
            warnings.append(f"{label}: periods are duplicated or out of order.")
        latest = cols[0]
        gaps = [i for i in items if i in frame.index and frame.loc[i].notna().any() and pd.isna(frame.loc[i, latest])]
        if gaps:
            warnings.append(f"{label}: no value for the latest period ({latest:%Y-%m-%d}) — {', '.join(gaps)}.")
        for newer, older in zip(cols, cols[1:]):
            days = (newer - older).days
            if not _in(days, spacing):
                warnings.append(f"{label}: {days} days between the periods ended {older:%Y-%m-%d} and "
                                f"{newer:%Y-%m-%d} — a period is missing or the fiscal year changed.")

    latest = statements.latest_period
    limit = 460 if annual else 140
    age = (pd.Timestamp(today) - latest).days
    if age > limit:
        kind = "fiscal year" if annual else "quarter"
        warnings.append(f"Stale: the latest {kind} ended {latest:%Y-%m-%d}, {age} days before {today}; newer "
                        "figures are probably available.")
    elif age < 0:
        warnings.append(f"The latest period ends {latest:%Y-%m-%d}, after {today}: check the dates.")

    def bad(item: str, test: Callable[[float], bool]) -> list[str]:
        row = statements.row(item)
        return [f"{c:%Y-%m-%d}" for c, v in row.items() if pd.notna(v) and test(float(v))]

    for item in _MUST_BE_POSITIVE:
        if dates := bad(item, lambda v: v <= 0):
            warnings.append(f"Sign anomaly: {item} is zero or negative ({', '.join(dates[:3])}).")
    for item in _MUST_BE_NON_NEGATIVE:
        if dates := bad(item, lambda v: v < 0):
            warnings.append(f"Sign anomaly: {item} is negative ({', '.join(dates[:3])}).")
    for item in _MUST_BE_NON_POSITIVE:
        if dates := bad(item, lambda v: v > 0):
            warnings.append(f"Sign anomaly: {item} is positive ({', '.join(dates[:3])}); outflows should be negative.")

    def pairs(a: str, b: str) -> list[tuple[pd.Timestamp, float, float]]:
        ra, rb = statements.row(a), statements.row(b)
        common = ra.index.intersection(rb.index)
        return [(c, float(ra[c]), float(rb[c])) for c in common if pd.notna(ra[c]) and pd.notna(rb[c])]

    if over := [f"{c:%Y-%m-%d}" for c, gp, rev in pairs("gross_profit", "revenue") if gp > rev > 0]:
        warnings.append(f"Inconsistent: gross_profit exceeds revenue ({', '.join(over[:3])}).")
    if over := [f"{c:%Y-%m-%d}" for c, ca, ta in pairs("current_assets", "total_assets") if ca > ta > 0]:
        warnings.append(f"Inconsistent: current_assets exceed total_assets ({', '.join(over[:3])}).")
    sign_flip = [f"{c:%Y-%m-%d}" for c, eps, ni in pairs("eps_diluted", "net_income") if eps * ni < 0]
    if sign_flip:
        warnings.append(f"Inconsistent: eps_diluted and net_income have opposite signs ({', '.join(sign_flip[:3])}).")
    assets, liabilities, equity = (statements.row(i) for i in ("total_assets", "total_liabilities", "total_equity"))
    broken = []
    for c in assets.index.intersection(liabilities.index).intersection(equity.index):
        a, li, e = assets[c], liabilities[c], equity[c]
        if pd.notna(a) and pd.notna(li) and pd.notna(e) and a > 0 and abs(a - (li + e)) / a > 0.10:
            broken.append(f"{c:%Y-%m-%d}")
    if broken:
        warnings.append("total_assets differ from total_liabilities + total_equity by more than 10% "
                        f"({', '.join(broken[:3])}); minority interests can explain part of a gap, a mis-mapped "
                        "line item the rest.")
    return warnings
