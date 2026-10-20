"""Synthetic price builders for tests. Nothing here is real market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_prices(closes, opens=None, start="2024-01-02", index=None) -> pd.DataFrame:
    """Build a clean OHLCV frame on consecutive weekdays from close (and open) paths."""
    closes = np.asarray(closes, dtype="float64")
    opens = closes.copy() if opens is None else np.asarray(opens, dtype="float64")
    idx = index if index is not None else pd.bdate_range(start, periods=len(closes))
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * 1.01,
            "low": np.minimum(opens, closes) * 0.99,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )
    return frame


def random_walk(n: int = 800, seed: int = 7, drift: float = 0.0003, vol: float = 0.012,
                start: str = "2018-01-02") -> pd.DataFrame:
    """Reproducible geometric random walk with an overnight gap component."""
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    gaps = np.exp(rng.normal(0, 0.004, n))
    opens = np.r_[100.0, closes[:-1]] * gaps
    return make_prices(closes, opens, start=start)
