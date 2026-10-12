"""Statistical inference and tail-risk measures for strategy returns.

Answers "how sure can we be?" questions about a return series: the sampling
error of the Sharpe ratio, autocorrelation-robust (HAC) t-statistics, CAPM
alpha/beta with HAC standard errors, the minimum track record needed before a
Sharpe ratio is statistically convincing, and distribution-tail statistics.

Conventions
-----------
* Inputs are simple per-period (daily) returns ``r_t``. Non-finite values (NaN,
  +/-inf) are dropped first; ``T`` is the number of remaining observations and
  ``P`` the periods per year.
* Every Sharpe-related quantity uses excess returns ``x_t = r_t - rf_d`` with
  ``rf_d = (1 + rf) ** (1/P) - 1`` (:func:`quantlab.metrics.daily_risk_free`), so
  ``SharpeInference.sharpe_annual`` equals :func:`quantlab.metrics.sharpe_ratio`.
* Per-period Sharpe ``SR = mean(x) / std(x, ddof=1)``. Skewness ``g3`` and kurtosis
  ``g4`` are population standardised moments (``ddof=0``). ``g4`` is *raw* (3 for
  a normal) inside Sharpe formulas, as in
  :func:`quantlab.research.probabilistic_sharpe_ratio`; :func:`excess_kurtosis`
  and Cornish-Fisher use ``g4 - 3``.
* VaR and CVaR are reported as *returns*, so losses are negative: ``VaR95 = -0.021``
  means one session in twenty is expected to lose 2.1% or more.
* A statistic is ``None`` when it is undefined (fewer than three observations,
  zero variance, ...); degenerate data never raises. Invalid *arguments* (a
  probability outside (0, 1), an unknown VaR method, negative lags) raise
  ``ValueError``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from quantlab.config import TRADING_DAYS_PER_YEAR
from quantlab.metrics import daily_risk_free

__all__ = [
    "AlphaBeta",
    "MinTRL",
    "SharpeInference",
    "alpha_beta",
    "conditional_value_at_risk",
    "cornish_fisher_z",
    "excess_kurtosis",
    "hac_mean_tstat",
    "min_track_record_length",
    "newey_west_lags",
    "newey_west_variance_of_mean",
    "omega_ratio",
    "sharpe_inference",
    "skewness",
    "tail_ratio",
    "value_at_risk",
]

ReturnsLike = pd.Series | np.ndarray | Sequence[float]
VarMethod = Literal["historical", "cornish_fisher", "normal"]

_EPS = 1e-12
_MIN_OBS = 3
_MIN_OBS_REGRESSION = 30
_QUANTILE_METHOD = "linear"  # Hyndman & Fan type 7, the numpy/pandas default
_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SharpeInference:
    """Sharpe ratio of excess returns with its standard error and confidence interval.

    ``*_per_period`` fields are per session; ``*_annual`` fields are scaled by
    ``sqrt(P)``. The interval ``[ci_lower, ci_upper]`` is annualised and two-sided.
    """

    n: int
    sharpe_per_period: float
    se_per_period: float
    sharpe_annual: float
    se_annual: float
    ci_lower: float
    ci_upper: float
    confidence: float
    skewness: float  # g3 of excess returns
    kurtosis: float  # raw g4 of excess returns (normal = 3)


@dataclass(frozen=True)
class AlphaBeta:
    """OLS of strategy excess returns on benchmark excess returns, with Newey-West errors.

    ``alpha_annual = alpha_per_period * P`` (and likewise its standard error);
    t-statistics test a zero coefficient and are ``None`` when the standard error
    is zero (a perfect fit). ``r_squared`` is ``None`` when the strategy's excess
    return is constant.
    """

    n: int
    lags: int
    alpha_per_period: float
    alpha_annual: float
    alpha_se_annual: float
    alpha_tstat: float | None
    beta: float
    beta_se: float
    beta_tstat: float | None
    r_squared: float | None


@dataclass(frozen=True)
class MinTRL:
    """Minimum track record length (Bailey & López de Prado, 2012).

    ``sessions`` is fractional (round up for a whole number of sessions);
    ``years = sessions / P``. ``sufficient`` is True when the observed sample
    (``n`` returns) is at least that long.
    """

    sessions: float
    years: float
    n: int
    sufficient: bool
    sharpe_annual: float
    target_sharpe_annual: float
    confidence: float


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clean(values: ReturnsLike) -> np.ndarray:
    """1-D float64 array with non-finite values removed."""
    if isinstance(values, (pd.Series, pd.Index)):
        x = values.to_numpy(dtype="float64", na_value=np.nan)
    else:
        x = np.asarray(values, dtype="float64").ravel()
    return x[np.isfinite(x)]


def _as_series(values: ReturnsLike) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.Series(values.to_numpy(dtype="float64", na_value=np.nan), index=values.index)
    return pd.Series(np.asarray(values, dtype="float64").ravel())


def _check_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value!r}")


def _moments(x: np.ndarray) -> tuple[float, float] | None:
    """Population skewness ``g3`` and raw kurtosis ``g4`` (ddof=0 moments)."""
    if len(x) < _MIN_OBS:
        return None
    sd = float(x.std(ddof=0))
    if sd < _EPS:
        return None
    z = (x - x.mean()) / sd
    return float(np.mean(z**3)), float(np.mean(z**4))


def _sharpe_moments(
    returns: ReturnsLike, risk_free_rate: float, periods_per_year: int
) -> tuple[float, float, float, int] | None:
    """Per-period SR, ``g3`` and raw ``g4`` of excess returns, and ``T``."""
    x = _clean(returns) - daily_risk_free(risk_free_rate, periods_per_year)
    m = _moments(x)
    if m is None:
        return None
    return float(x.mean() / x.std(ddof=1)), m[0], m[1], len(x)


def _sharpe_variance_term(sr: float, g3: float, g4: float) -> float:
    """``1 - g3 SR + (g4 - 1)/4 SR^2``: T-1 times the asymptotic variance of SR (Mertens, 2002)."""
    return 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2


def _resolve_lags(lags: int | None, n: int) -> int:
    if lags is None:
        lags = newey_west_lags(n)
    elif lags < 0:
        raise ValueError(f"lags must be non-negative, got {lags!r}")
    return int(min(lags, n - 1))


def _long_run_covariance(h: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-weighted long-run covariance of the mean-zero rows of ``h`` (T x k).

    ``Omega = (1/T) [G_0 + sum_{l=1..L} w_l (G_l + G_l')]`` with
    ``G_l = sum_{t>l} h_t h_{t-l}'`` and ``w_l = 1 - l / (L + 1)`` (Newey & West, 1987).
    """
    s = h.T @ h
    for lag in range(1, lags + 1):
        gamma = h[lag:].T @ h[:-lag]
        s = s + (1.0 - lag / (lags + 1)) * (gamma + gamma.T)
    return s / h.shape[0]


# --------------------------------------------------------------------------- #
# Sharpe-ratio inference
# --------------------------------------------------------------------------- #


def sharpe_inference(
    returns: ReturnsLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    confidence: float = 0.95,
) -> SharpeInference | None:
    """Sharpe ratio, its standard error and a two-sided confidence interval.

    ``SE(SR) = sqrt((1 - g3 SR + (g4 - 1)/4 SR^2) / (T - 1))`` for iid, possibly
    non-normal returns (Lo, 2002; Mertens, 2002) — the same denominator as the PSR
    in :mod:`quantlab.research`. Annualised: ``SR sqrt(P)`` and ``SE sqrt(P)``;
    interval ``SR_a +/- z SE_a`` with ``z = Phi^-1((1 + confidence) / 2)``. The
    formula assumes serially independent returns; with positive autocorrelation
    it understates the error (compare :func:`hac_mean_tstat`).
    ``None`` when ``T < 3``, the variance is zero or the variance term is not positive.
    """
    _check_probability(confidence, "confidence")
    m = _sharpe_moments(returns, risk_free_rate, periods_per_year)
    if m is None:
        return None
    sr, g3, g4, n = m
    term = _sharpe_variance_term(sr, g3, g4)
    if term <= 0:
        return None
    se = math.sqrt(term / (n - 1))
    root = math.sqrt(periods_per_year)
    sr_annual, se_annual = sr * root, se * root
    z = _NORMAL.inv_cdf(0.5 + confidence / 2)
    return SharpeInference(
        n=n,
        sharpe_per_period=sr,
        se_per_period=se,
        sharpe_annual=sr_annual,
        se_annual=se_annual,
        ci_lower=sr_annual - z * se_annual,
        ci_upper=sr_annual + z * se_annual,
        confidence=confidence,
        skewness=g3,
        kurtosis=g4,
    )


def min_track_record_length(
    returns: ReturnsLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    target_sharpe_annual: float = 0.0,
    confidence: float = 0.95,
) -> MinTRL | None:
    """Sessions needed for the observed Sharpe to exceed a target with the given confidence.

    Bailey & López de Prado (2012): ``MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2)
    (z / (SR - SR*))^2`` with per-period ``SR`` (of excess returns),
    ``SR* = target_sharpe_annual / sqrt(P)`` and one-sided ``z = Phi^-1(confidence)``.
    Equivalently, the sample length at which the PSR against ``SR*`` reaches
    ``confidence``. ``None`` when the observed Sharpe is at or below the target, or
    when the Sharpe ratio is undefined.
    """
    _check_probability(confidence, "confidence")
    m = _sharpe_moments(returns, risk_free_rate, periods_per_year)
    if m is None:
        return None
    sr, g3, g4, n = m
    root = math.sqrt(periods_per_year)
    target = target_sharpe_annual / root
    if sr <= target:
        return None
    term = _sharpe_variance_term(sr, g3, g4)
    if term <= 0:
        return None
    sessions = 1.0 + term * (_NORMAL.inv_cdf(confidence) / (sr - target)) ** 2
    return MinTRL(
        sessions=sessions,
        years=sessions / periods_per_year,
        n=n,
        sufficient=n >= sessions,
        sharpe_annual=sr * root,
        target_sharpe_annual=target_sharpe_annual,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# HAC (Newey-West) inference
# --------------------------------------------------------------------------- #


def newey_west_lags(n: int) -> int:
    """Default Newey-West bandwidth ``L = floor(4 (T / 100) ^ (2/9))`` (Newey & West, 1994)."""
    return int(math.floor(4 * (max(n, 0) / 100) ** (2 / 9)))


def newey_west_variance_of_mean(x: ReturnsLike, lags: int | None = None) -> float | None:
    """Newey-West (Bartlett kernel) variance of the sample mean of ``x``.

    ``Var(mean) = (gamma_0 + 2 sum_{l=1..L} (1 - l/(L+1)) gamma_l) / T`` with
    autocovariances ``gamma_l = (1/T) sum_{t>l} (x_t - m)(x_{t-l} - m)`` (divisor
    ``T``, i.e. the ddof=0 convention), so ``lags=0`` gives the iid
    ``var(x, ddof=0) / T``. Default ``L`` is :func:`newey_west_lags`; ``L`` is capped
    at ``T - 1``. ``None`` when ``T < 3`` or ``x`` has zero variance.
    """
    v = _clean(x)
    n = len(v)
    if n < _MIN_OBS or float(v.std(ddof=0)) < _EPS:
        return None
    omega = _long_run_covariance((v - v.mean())[:, None], _resolve_lags(lags, n))
    var = float(omega[0, 0]) / n
    return var if var > 0 else None


def hac_mean_tstat(x: ReturnsLike, lags: int | None = None) -> float | None:
    """t-statistic of ``mean(x) = 0`` with a Newey-West standard error.

    ``t = mean(x) / sqrt(newey_west_variance_of_mean(x, lags))``. ``None`` when that
    variance is undefined.
    """
    var = newey_west_variance_of_mean(x, lags)
    if var is None:
        return None
    return float(_clean(x).mean() / math.sqrt(var))


def alpha_beta(
    strategy_returns: ReturnsLike,
    benchmark_returns: ReturnsLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    lags: int | None = None,
) -> AlphaBeta | None:
    """Jensen's alpha and beta with Newey-West (HAC) standard errors.

    OLS of ``y_t = r_t - rf_d`` on ``[1, b_t - rf_d]`` after aligning both series on
    their common index (positional for arrays) and dropping rows with non-finite
    values. Covariance ``(X'X)^-1 S (X'X)^-1`` with ``S = T * Omega`` the Bartlett
    long-run covariance of the scores ``X_t u_t`` (Newey & West, 1987; no
    small-sample correction), default lags :func:`newey_west_lags`. Alpha is
    annualised arithmetically (``x P``). ``None`` when fewer than 30 aligned
    observations remain or the benchmark has zero variance.
    """
    frame = pd.concat(
        [_as_series(strategy_returns), _as_series(benchmark_returns)], axis=1, join="inner", keys=["s", "b"]
    )
    values = frame.to_numpy(dtype="float64")
    values = values[np.isfinite(values).all(axis=1)]
    n = len(values)
    if n < _MIN_OBS_REGRESSION:
        return None
    rf_d = daily_risk_free(risk_free_rate, periods_per_year)
    y = values[:, 0] - rf_d
    x = values[:, 1] - rf_d
    if float(x.std(ddof=0)) < _EPS:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    beta = float(xc @ yc / (xc @ xc))
    alpha = float(y.mean() - beta * x.mean())
    resid = y - alpha - beta * x
    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(design.T @ design)
    used_lags = _resolve_lags(lags, n)
    cov = xtx_inv @ (n * _long_run_covariance(design * resid[:, None], used_lags)) @ xtx_inv
    alpha_se = math.sqrt(max(float(cov[0, 0]), 0.0))
    beta_se = math.sqrt(max(float(cov[1, 1]), 0.0))
    r_squared = None if float(y.std(ddof=0)) < _EPS else float(1.0 - (resid @ resid) / (yc @ yc))
    return AlphaBeta(
        n=n,
        lags=used_lags,
        alpha_per_period=alpha,
        alpha_annual=alpha * periods_per_year,
        alpha_se_annual=alpha_se * periods_per_year,
        alpha_tstat=alpha / alpha_se if alpha_se > _EPS else None,
        beta=beta,
        beta_se=beta_se,
        beta_tstat=beta / beta_se if beta_se > _EPS else None,
        r_squared=r_squared,
    )


# --------------------------------------------------------------------------- #
# Distribution and tail statistics
# --------------------------------------------------------------------------- #


def skewness(returns: ReturnsLike) -> float | None:
    """Population skewness ``g3 = m3 / m2^(3/2)`` (ddof=0 central moments)."""
    m = _moments(_clean(returns))
    return None if m is None else m[0]


def excess_kurtosis(returns: ReturnsLike) -> float | None:
    """Population excess kurtosis ``g4 - 3 = m4 / m2^2 - 3`` (ddof=0; 0 for a normal)."""
    m = _moments(_clean(returns))
    return None if m is None else m[1] - 3.0


def cornish_fisher_z(z: float, skew: float, excess_kurt: float) -> float:
    """Cornish-Fisher adjusted standard-normal quantile (Cornish & Fisher, 1937; Zangari, 1996).

    ``z_cf = z + (z^2 - 1) S/6 + (z^3 - 3z) K/24 - (2z^3 - 5z) S^2/36`` with skewness
    ``S`` and excess kurtosis ``K``; equals ``z`` when ``S = K = 0``. Unreliable (it
    can become non-monotonic) for large ``|S|`` or ``K``.
    """
    return (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * excess_kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )


def value_at_risk(returns: ReturnsLike, level: float = 0.95, method: VarMethod = "historical") -> float | None:
    """One-period Value at Risk at ``level``, as a return (losses are negative).

    ``VaR95 = -0.021`` means returns fall to -2.1% or below on 5% of sessions.
    The ``1 - level`` quantile of returns, by ``method``:

    * ``"historical"``: empirical quantile with linear interpolation between order
      statistics (numpy ``method="linear"``, Hyndman & Fan type 7). A constant
      series returns that constant.
    * ``"normal"``: ``mean + z sd`` with ``z = Phi^-1(1 - level)`` and ``sd`` ddof=1.
    * ``"cornish_fisher"``: ``mean + z_cf sd`` with :func:`cornish_fisher_z` using
      population skewness and excess kurtosis; equals ``"normal"`` when both are 0.

    ``None`` when ``T < 3``, or for the parametric methods when the variance is zero.
    """
    _check_probability(level, "level")
    if method not in ("historical", "cornish_fisher", "normal"):
        raise ValueError(f"unknown VaR method {method!r}")
    x = _clean(returns)
    if len(x) < _MIN_OBS:
        return None
    if method == "historical":
        return float(np.quantile(x, 1.0 - level, method=_QUANTILE_METHOD))
    m = _moments(x)
    if m is None:
        return None
    z = _NORMAL.inv_cdf(1.0 - level)
    if method == "cornish_fisher":
        z = cornish_fisher_z(z, m[0], m[1] - 3.0)
    return float(x.mean() + z * x.std(ddof=1))


def conditional_value_at_risk(returns: ReturnsLike, level: float = 0.95) -> float | None:
    """Historical expected shortfall (CVaR) at ``level``, as a return (losses are negative).

    Mean of the returns at or below the historical :func:`value_at_risk` (same
    linear-interpolation quantile), so ``CVaR <= VaR``. ``None`` when ``T < 3``.
    """
    x = _clean(returns)
    var = value_at_risk(x, level, "historical")
    if var is None:
        return None
    tail = x[x <= var]
    return float(tail.mean()) if tail.size else var


def omega_ratio(returns: ReturnsLike, threshold: float = 0.0) -> float | None:
    """Omega ratio ``sum(max(r - tau, 0)) / sum(max(tau - r, 0))`` (Keating & Shadwick, 2002).

    ``threshold`` (``tau``) is a per-period return. ``None`` when ``T < 3`` or no
    return falls below the threshold.
    """
    x = _clean(returns)
    if len(x) < _MIN_OBS:
        return None
    gains = float(np.clip(x - threshold, 0.0, None).sum())
    losses = float(np.clip(threshold - x, 0.0, None).sum())
    if losses < _EPS:
        return None
    return gains / losses


def tail_ratio(returns: ReturnsLike, q: float = 0.95) -> float | None:
    """``|Q(q)| / |Q(1 - q)|``: size of the right tail relative to the left tail.

    Quantiles use linear interpolation (as :func:`value_at_risk`). Above 1 means
    large gains are bigger than large losses. ``None`` when ``T < 3``, the
    variance is zero or ``Q(1 - q)`` is zero.
    """
    _check_probability(q, "q")
    x = _clean(returns)
    if len(x) < _MIN_OBS or float(x.std(ddof=0)) < _EPS:
        return None
    upper = float(np.quantile(x, q, method=_QUANTILE_METHOD))
    lower = float(np.quantile(x, 1.0 - q, method=_QUANTILE_METHOD))
    if abs(lower) < _EPS:
        return None
    return abs(upper) / abs(lower)
