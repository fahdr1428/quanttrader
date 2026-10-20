"""JSON helpers shared by exports and the experiment store."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pandas as pd


def clean_json(value: Any) -> Any:
    """Recursively make ``value`` strictly JSON-serialisable.

    NaN and infinities become ``None`` (they are not valid JSON), timestamps
    become ISO strings, and numpy/pandas scalars become Python numbers.
    """
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
