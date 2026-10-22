"""Where a number came from, and how far it can be trusted.

Every value the research agent reports is one of three kinds, and the kinds are
never mixed:

* a **fact** — a value read from a named source (a price, a reported revenue
  figure, a filing date), carried with its :class:`Provenance`;
* a **calculated metric** — derived from facts by a stated formula, carrying
  the facts it used;
* an **interpretation** — prose judgement, labelled as such.

``Provenance`` records what the agent's mission requires for every dataset:
source and retrieval time, period covered, publication time, currency and
units, whether the data is real-time, delayed or end-of-day, whether prices
are adjusted for corporate actions, and any data-quality warnings. The module
is pure (no Streamlit, no network).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any


class Timeliness(str, Enum):
    """How current a dataset can be. Nothing in this project is real-time."""

    END_OF_DAY = "end_of_day"            # daily bars after the session closed
    DELAYED = "delayed"                  # a quote or value delayed by the provider
    PERIODIC_FILING = "periodic_filing"  # quarterly or annual reports
    EVENT = "event"                      # dated events (earnings dates, filings, news)
    STATIC = "static"                    # reference data (names, sectors, identifiers)
    SIMULATED = "simulated"              # synthetic data with a known generating process
    USER_SUPPLIED = "user_supplied"      # entered or uploaded by the user

    @property
    def label(self) -> str:
        return {
            "end_of_day": "End-of-day (not live)",
            "delayed": "Delayed (not live)",
            "periodic_filing": "Periodic filing",
            "event": "Dated event",
            "static": "Reference data",
            "simulated": "Simulated (not market data)",
            "user_supplied": "Supplied by the user",
        }[self.value]


class Adjustment(str, Enum):
    """Whether prices reflect corporate actions."""

    SPLITS_AND_DIVIDENDS = "splits_and_dividends"
    SPLITS_ONLY = "splits_only"
    UNADJUSTED = "unadjusted"
    AS_PROVIDED = "as_provided"          # unknown: used exactly as supplied
    NOT_APPLICABLE = "not_applicable"    # not a price series


@dataclass(frozen=True)
class Provenance:
    source: str                                   # e.g. "SEC EDGAR companyfacts (XBRL)"
    retrieved_at: datetime                        # UTC, when this process obtained it
    timeliness: Timeliness
    source_url: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    published_at: datetime | None = None          # e.g. the filing date; None if unknown
    currency: str | None = None
    units: str | None = None                      # e.g. "USD", "USD/share", "shares", "ratio"
    adjustment: Adjustment = Adjustment.NOT_APPLICABLE
    cache_status: str | None = None               # e.g. "downloaded", "cached", "stale_cache"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware (UTC)")
        if self.published_at is not None and self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        if self.period_start and self.period_end and self.period_start > self.period_end:
            raise ValueError("period_start is after period_end")

    def with_warning(self, message: str) -> Provenance:
        return replace(self, warnings=(*self.warnings, message))

    def is_point_in_time_at(self, as_of: datetime) -> bool | None:
        """Whether this value was public by ``as_of``. None when the publication time is unknown —
        then historical use risks look-ahead bias and must be disclosed."""
        if self.published_at is None:
            return None
        return self.published_at <= as_of

    def to_dict(self) -> dict[str, Any]:
        def iso(v):
            return v.isoformat() if isinstance(v, date | datetime) else v

        return {
            "source": self.source,
            "source_url": self.source_url,
            "retrieved_at": iso(self.retrieved_at),
            "timeliness": self.timeliness.value,
            "period_start": iso(self.period_start),
            "period_end": iso(self.period_end),
            "published_at": iso(self.published_at),
            "currency": self.currency,
            "units": self.units,
            "adjustment": self.adjustment.value,
            "cache_status": self.cache_status,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class Fact:
    """A value read from a source. ``value`` is None when the source did not provide it;
    a missing value is never replaced by zero."""

    name: str
    value: float | str | None
    provenance: Provenance
    unit: str | None = None
    note: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fact", "name": self.name, "value": self.value, "unit": self.unit, "note": self.note,
                "provenance": self.provenance.to_dict()}


@dataclass(frozen=True)
class Calculated:
    """A value derived from facts by a stated formula. ``value`` is None when an input is
    missing or the formula is undefined (e.g. a ratio with a non-positive denominator), and
    ``reason`` says why."""

    name: str
    value: float | None
    formula: str
    inputs: tuple[Fact | Calculated, ...] = ()
    unit: str | None = None
    reason: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "calculated", "name": self.name, "value": self.value, "unit": self.unit,
                "formula": self.formula, "reason": self.reason, "warnings": list(self.warnings),
                "inputs": [i.name for i in self.inputs]}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
