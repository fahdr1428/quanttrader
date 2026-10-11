"""Robustness analytics: how much of a backtest result could be luck?

Stationary block bootstrap
--------------------------
A backtest produces one path of daily returns. Resampling that path many times
— in random-length blocks, so that volatility clustering and short-term
autocorrelation survive (Politis & Romano, 1994) — gives a distribution of the
metrics the same return process could plausibly have produced. Strategy and
benchmark returns are resampled with the *same* indices so their correlation is
preserved and "probability of beating the benchmark" is meaningful.

The bootstrap assumes the return process is stationary: it cannot anticipate
regime changes that never appeared in the sample, so its intervals are a lower
bound on true uncertainty. A fixed seed makes every run reproducible.

Regime analysis
---------------
Each session is labelled by the benchmark's state *as known at that session's
close*: trend (above or below its 200-session average) and volatility (63-session
realised volatility above or below its median over the sample — the median is
an after-the-fact threshold, so this split is descriptive, not tradeable).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantlab.config import TRADING_DAYS_PER_YEAR

DEFAULT_SEED = 20240101


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """``(n_boot, n)`` index matrix for the Politis-Romano stationary bootstrap.

    Each resampled path starts at a random session; at every step it continues the
    current block with probability ``1 - 1/mean_block`` (wrapping around the end)
    or jumps to a new random session.
    """
    if n < 2:
        raise ValueError("need at least two observations to bootstrap")
    p_new = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    for t in range(1, n):
        jump = rng.random(n_boot) < p_new
        idx[:, t] = np.where(jump, rng.integers(0, n, n_boot), (idx[:, t - 1] + 1) % n)
    return idx


def _path_metrics(returns: np.ndarray, periods_per_year: int, rf_daily: float) -> dict[str, np.ndarray]:
    """Vectorised metrics for a matrix of return paths (one path per row)."""
    n = returns.shape[1]
    growth = np.cumprod(1 + returns, axis=1)
    final = growth[:, -1]
    cagr = np.where(final > 0, final ** (periods_per_year / n) - 1, -1.0)
    excess = returns - rf_daily
    std = excess.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 1e-12, excess.mean(axis=1) / std * math.sqrt(periods_per_year), np.nan)
    peaks = np.maximum.accumulate(np.concatenate([np.ones((returns.shape[0], 1)), growth], axis=1), axis=1)[:, 1:]
    max_dd = (growth / peaks - 1).min(axis=1)
    return {
        "total_return": final - 1,
        "cagr": cagr,
        "volatility": returns.std(axis=1, ddof=1) * math.sqrt(periods_per_year),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def bootstrap_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    n_boot: int = 1000,
    mean_block: float = 21.0,
    seed: int = DEFAULT_SEED,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """One row per bootstrap path with strategy (and benchmark) metrics."""
    r = strategy_returns.to_numpy(dtype="float64")
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(strategy_returns.index).to_numpy(dtype="float64")
        if np.isnan(b).any():
            raise ValueError("benchmark returns must cover the same sessions as the strategy")
    if len(r) < 20:
        raise ValueError("at least 20 daily returns are needed for a meaningful bootstrap")
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(r), n_boot, mean_block, rng)
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    out = pd.DataFrame(_path_metrics(r[idx], periods_per_year, rf_daily))
    if benchmark_returns is not None:
        bench = _path_metrics(b[idx], periods_per_year, rf_daily)
        for key, values in bench.items():
            out[f"benchmark_{key}"] = values
        out["excess_cagr"] = out["cagr"] - out["benchmark_cagr"]
        out["sharpe_difference"] = out["sharpe"] - out["benchmark_sharpe"]
    return out


BOOTSTRAP_LABELS = {
    "sharpe": "Sharpe ratio",
    "cagr": "Annualised return",
    "volatility": "Annualised volatility",
    "max_drawdown": "Maximum drawdown",
    "excess_cagr": "Excess annualised return vs benchmark",
    "sharpe_difference": "Sharpe minus benchmark Sharpe",
}


def bootstrap_summary(samples: pd.DataFrame, observed: dict[str, float | None]) -> pd.DataFrame:
    """Percentile intervals and probabilities for each bootstrapped metric."""
    rows = []
    for key, label in BOOTSTRAP_LABELS.items():
        if key not in samples:
            continue
        values = samples[key].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "key": key,
                "Metric": label,
                "Observed": observed.get(key),
                "5th percentile": float(values.quantile(0.05)),
                "Median": float(values.median()),
                "95th percentile": float(values.quantile(0.95)),
                # Volatility is always positive and drawdowns never are: a probability is meaningless.
                "P(> 0)": None if key in ("volatility", "max_drawdown") else float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def regime_labels(benchmark_close: pd.Series) -> pd.DataFrame:
    """Trend and volatility regime of the benchmark at each session's close."""
    sma200 = benchmark_close.rolling(200, min_periods=200).mean()
    vol63 = benchmark_close.pct_change().rolling(63, min_periods=63).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    trend = pd.Series(np.where(benchmark_close > sma200, "Up-trend", "Down-trend"), index=benchmark_close.index)
    trend = trend.where(sma200.notna())
    median_vol = vol63.median()
    vol = pd.Series(np.where(vol63 > median_vol, "High volatility", "Low volatility"), index=benchmark_close.index)
    vol = vol.where(vol63.notna())
    return pd.DataFrame({"trend": trend, "volatility": vol, "benchmark_vol_63d": vol63})


def regime_performance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    regimes: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Strategy vs benchmark in each regime.

    The regime of the *previous* close is used for each day's return, so the
    label was known before the return happened. Annualised returns here are
    arithmetic (mean daily return x 252) because regime days are not contiguous.
    Sharpe ratios use returns in excess of ``risk_free_rate``, like the headline Sharpe.
    """
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    prior = regimes.shift(1).reindex(strategy_returns.index)
    rows = []
    for dimension in ("trend", "volatility"):
        for regime in sorted(prior[dimension].dropna().unique()):
            mask = prior[dimension] == regime
            s, b = strategy_returns[mask], benchmark_returns.reindex(strategy_returns.index)[mask]
            if len(s) < 2:
                continue
            s_std, b_std = s.std(ddof=1), b.std(ddof=1)
            rows.append(
                {
                    "Dimension": "Trend" if dimension == "trend" else "Volatility",
                    "Regime": regime,
                    "Sessions": int(mask.sum()),
                    "Share of sample": float(mask.mean()),
                    "Strategy return (ann.)": float(s.mean() * periods_per_year),
                    "Benchmark return (ann.)": float(b.mean() * periods_per_year),
                    "Strategy volatility": float(s_std * math.sqrt(periods_per_year)),
                    "Benchmark volatility": float(b_std * math.sqrt(periods_per_year)),
                    "Strategy Sharpe": (float((s.mean() - rf_daily) / s_std * math.sqrt(periods_per_year))
                                        if s_std > 1e-12 else None),
                    "Benchmark Sharpe": (float((b.mean() - rf_daily) / b_std * math.sqrt(periods_per_year))
                                         if b_std > 1e-12 else None),
                    "Strategy up-days": float((s > 0).mean()),
                }
            )
    return pd.DataFrame(rows)
