"""Technical indicators.

Every function here is *causal*: the value at session ``t`` uses only
observations at or before ``t``. Warm-up periods return ``NaN`` rather than a
value computed from fewer observations than the indicator needs, and nothing
is back-filled from the future. ``tests/test_strategies.py`` checks this by
changing future prices and confirming earlier values are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_window(window: int, minimum: int = 1) -> int:
    if int(window) != window or window < minimum:
        raise ValueError(f"window must be an integer >= {minimum}, got {window!r}")
    return int(window)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average of the last ``window`` observations (inclusive of t)."""
    window = _require_window(window)
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average with span ``window`` (alpha = 2 / (window + 1)).

    The recursion starts at the first observation of the series, so early values
    depend on where the data begins. The first ``window - 1`` values are masked
    as warm-up so that only reasonably initialised values are used.
    """
    window = _require_window(window)
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    1. ``delta_t = close_t - close_{t-1}``; gains are positive deltas, losses the
       absolute value of negative deltas.
    2. The first average gain/loss is the simple mean of the first ``window``
       deltas; afterwards ``avg_t = (avg_{t-1} * (window - 1) + x_t) / window``.
    3. ``RSI = 100 - 100 / (1 + avg_gain / avg_loss)``.

    Conventions: if the average loss is zero the RSI is 100; if both averages are
    zero (no price movement) the RSI is 50 (neutral). The first ``window`` values
    are ``NaN`` (a delta needs two prices, so ``window`` deltas need
    ``window + 1`` closes).
    """
    window = _require_window(window, minimum=2)
    values = close.to_numpy(dtype="float64")
    n = len(values)
    out = np.full(n, np.nan)
    if n <= window:
        return pd.Series(out, index=close.index, name=f"rsi_{window}")
    if np.isnan(values).any():
        raise ValueError("rsi_wilder requires a series without missing values")

    delta = np.diff(values)
    gains = np.clip(delta, 0.0, None)
    losses = np.clip(-delta, 0.0, None)
    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    def _rsi(gain: float, loss: float) -> float:
        if loss == 0.0:
            return 50.0 if gain == 0.0 else 100.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    out[window] = _rsi(avg_gain, avg_loss)
    for i in range(window, n - 1):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        out[i + 1] = _rsi(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name=f"rsi_{window}")


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Middle band = SMA(window); upper/lower = middle +/- num_std * rolling std.

    The rolling standard deviation uses the population formula (``ddof=0``),
    following John Bollinger's original definition.
    """
    window = _require_window(window, minimum=2)
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": middle + num_std * std,
            "bb_lower": middle - num_std * std,
            "bb_std": std,
        },
        index=close.index,
    )


def trailing_return(close: pd.Series, lookback: int, skip: int = 0) -> pd.Series:
    """Return from ``t - skip - lookback`` to ``t - skip``: ``close_{t-skip} / close_{t-skip-lookback} - 1``.

    ``skip`` excludes the most recent sessions (e.g. the classic "12-1" momentum
    skips the last month to avoid short-term reversal).
    """
    lookback = _require_window(lookback)
    if int(skip) != skip or skip < 0:
        raise ValueError(f"skip must be a non-negative integer, got {skip!r}")
    skip = int(skip)
    return close.shift(skip) / close.shift(skip + lookback) - 1.0
