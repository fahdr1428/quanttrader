"""Synthetic markets with a *known* data-generating process.

These series are simulated, not market data. They exist for two honest uses:

* a demo mode that works with no network access, clearly labelled as synthetic
  wherever it appears; and
* controlled experiments: on a market with no predictable component every
  timing strategy should fail to beat buy-and-hold after costs except by luck,
  and on a market with a planted effect the right tools should find it.

Each market is generated once over a fixed calendar (NYSE-style weekdays from
2000 to 2024) from a fixed seed, so any date range is a slice of the same path
and its fingerprint does not depend on the day the app is run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from quantlab.calendar import expected_sessions
from quantlab.config import TRADING_DAYS_PER_YEAR

SIM_START = pd.Timestamp("2000-01-03")
SIM_END = pd.Timestamp("2024-12-31")
DEMO_SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class SyntheticMarket:
    key: str
    name: str
    symbol: str
    summary: str
    annual_drift: float  # arithmetic expected return per year (before any planted effect)
    annual_vol: float  # long-run volatility of the GARCH(1,1) process
    momentum_drift: float = 0.0  # daily expected-return tilt after a rising / falling lookback window
    momentum_lookback: int = 63

    def describe(self, seed: int) -> dict[str, str]:
        out = {
            "Market": self.name,
            "What it is": self.summary,
            "Arithmetic drift": f"{self.annual_drift:.1%} per year",
            "Volatility": f"GARCH(1,1), long-run {self.annual_vol:.0%} per year, Student-t (5 d.f.) shocks",
            "Planted effect": (
                f"Expected daily return {self.momentum_drift:+.2%} after a rising {self.momentum_lookback}-session "
                f"window and {-self.momentum_drift:+.2%} after a falling one"
                if self.momentum_drift else "None: returns are unpredictable from past prices"
            ),
            "Calendar": f"NYSE-style weekdays {SIM_START:%Y-%m-%d} to {SIM_END:%Y-%m-%d}",
            "Seed": str(seed),
        }
        return out


SYNTHETIC_MARKETS: dict[str, SyntheticMarket] = {
    "random": SyntheticMarket(
        key="random",
        name="Random walk with drift (no timing signal)",
        symbol="SIM-RANDOM",
        summary="Equity-like drift and volatility clustering, but nothing in past prices predicts future "
                "returns. Any timing edge found here is luck or overfitting.",
        annual_drift=0.07,
        annual_vol=0.18,
    ),
    "trend": SyntheticMarket(
        key="trend",
        name="Planted time-series momentum (control)",
        symbol="SIM-TREND",
        summary="The same process plus a planted momentum effect: expected returns are higher after a "
                "rising quarter and lower after a falling one. Trend tools should detect it.",
        annual_drift=0.07,
        annual_vol=0.18,
        momentum_drift=0.0008,
    ),
}


def demo_ticker(market_key: str, seed: int) -> str:
    return f"{SYNTHETIC_MARKETS[market_key].symbol}-{int(seed)}"


def parse_demo_ticker(ticker: str) -> tuple[str, int] | None:
    for key, market in SYNTHETIC_MARKETS.items():
        prefix = market.symbol + "-"
        if ticker.upper().startswith(prefix) and ticker[len(prefix):].isdigit():
            return key, int(ticker[len(prefix):])
    return None


def _seed_for(market_key: str, seed: int) -> int:
    # Distinct, stable streams per market and seed.
    return 20_240_000 + 1_000 * list(SYNTHETIC_MARKETS).index(market_key) + int(seed)


@lru_cache(maxsize=16)
def _simulate_cached(market_key: str, seed: int) -> pd.DataFrame:
    market = SYNTHETIC_MARKETS[market_key]
    index = expected_sessions(SIM_START, SIM_END)
    n = len(index)
    rng = np.random.default_rng(_seed_for(market_key, seed))

    ppy = TRADING_DAYS_PER_YEAR
    mu = market.annual_drift / ppy
    long_run_var = (market.annual_vol / math.sqrt(ppy)) ** 2
    alpha, beta = 0.08, 0.90
    omega = long_run_var * (1 - alpha - beta)
    dof = 5.0
    shocks = rng.standard_t(dof, n) / math.sqrt(dof / (dof - 2))  # unit variance

    returns = np.empty(n)
    log_close = np.empty(n)
    var, prev_eps, level = long_run_var, 0.0, 0.0
    lb = market.momentum_lookback
    for t in range(n):
        var = omega + alpha * prev_eps**2 + beta * var
        tilt = 0.0
        if market.momentum_drift and t > lb:
            tilt = market.momentum_drift if log_close[t - 1] > log_close[t - 1 - lb] else -market.momentum_drift
        eps = math.sqrt(var) * shocks[t]
        r = max(mu + tilt + eps, -0.5)  # simple return; floor keeps prices positive
        returns[t] = r
        level += math.log1p(r)
        log_close[t] = level
        prev_eps = eps

    close = 100.0 * np.exp(log_close)
    prev_close = np.r_[100.0, close[:-1]]
    daily_vol = math.sqrt(long_run_var)
    # Part of each day's move happens overnight; the close path is unaffected.
    overnight = 0.3 * np.log(close / prev_close) + rng.normal(0.0, 0.25 * daily_vol, n)
    opens = prev_close * np.exp(overnight)
    span = np.abs(rng.normal(0.0, 0.5 * daily_vol, (2, n)))
    high = np.maximum(opens, close) * np.exp(span[0])
    low = np.minimum(opens, close) * np.exp(-span[1])
    volume = np.round(5e6 * np.exp(rng.normal(0.0, 0.3, n) + 8 * np.abs(returns)))
    frame = pd.DataFrame(
        {"Open": opens, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.DatetimeIndex(index, name="date"),
    )
    return frame


def simulate_market(market_key: str, seed: int = 1) -> pd.DataFrame:
    """Full simulated OHLCV history for one market and seed (a copy; safe to modify)."""
    if market_key not in SYNTHETIC_MARKETS:
        raise KeyError(f"Unknown synthetic market '{market_key}'.")
    return _simulate_cached(market_key, int(seed)).copy()
