"""Provenance travels with every fact; missing values stay missing."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from quantlab.provenance import Adjustment, Calculated, Fact, Provenance, Timeliness

NOW = datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)


def _prov(**kw):
    return Provenance(source="Test source", retrieved_at=NOW, timeliness=Timeliness.END_OF_DAY, **kw)


def test_provenance_requires_timezone_aware_times_and_ordered_periods():
    with pytest.raises(ValueError):
        Provenance(source="x", retrieved_at=datetime(2024, 6, 3), timeliness=Timeliness.STATIC)
    with pytest.raises(ValueError):
        _prov(published_at=datetime(2024, 6, 3))
    with pytest.raises(ValueError):
        _prov(period_start=date(2024, 6, 2), period_end=date(2024, 6, 1))


def test_point_in_time_is_unknown_without_a_publication_time():
    """A value without a publication time cannot be shown to have been public: historical use risks look-ahead."""
    assert _prov().is_point_in_time_at(NOW) is None
    filed = _prov(published_at=datetime(2024, 5, 1, tzinfo=timezone.utc))
    assert filed.is_point_in_time_at(NOW) is True
    assert filed.is_point_in_time_at(datetime(2024, 4, 1, tzinfo=timezone.utc)) is False


def test_facts_and_calculations_serialise_with_their_provenance_and_formula():
    p = _prov(currency="USD", units="USD/share", adjustment=Adjustment.SPLITS_AND_DIVIDENDS).with_warning("stale")
    price = Fact("close", 101.5, p, unit="USD")
    missing = Fact("eps_ttm", None, p, unit="USD/share", note="not reported")
    pe = Calculated("pe_trailing", None, "price / EPS (TTM)", (price, missing), reason="EPS not available")
    assert price.available and not missing.available and not pe.available
    d = price.to_dict()
    assert d["provenance"]["currency"] == "USD" and d["provenance"]["warnings"] == ["stale"]
    assert d["provenance"]["timeliness"] == "end_of_day" and d["provenance"]["adjustment"] == "splits_and_dividends"
    assert pe.to_dict()["inputs"] == ["close", "eps_ttm"] and pe.to_dict()["value"] is None
