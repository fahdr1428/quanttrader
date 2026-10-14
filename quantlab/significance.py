"""Data-snooping tests: does the best configuration of a grid beat its benchmark,
once the search over the whole grid is accounted for?

Setting
-------
K configurations are run over the same T sessions. Their *performance
differentials* against a benchmark are ``d[k, t] = r[k, t] - b[t]`` (larger is
better), where ``b`` is a benchmark's daily return or, without a benchmark, the
daily risk-free return (the comparison is then "versus cash"). The null
hypothesis of every test here is that **no configuration has a positive
expected differential**: ``H0: max_k E[d_k] <= 0``. Testing only the best
in-sample configuration as if it were the only one tried overstates its
significance; these tests compare it with the best of K resampled grids
instead.

Bootstrap
---------
All tests use the Politis & Romano (1994) stationary bootstrap
(``quantlab.robustness.stationary_bootstrap_indices``, mean block length
``mean_block``). One index matrix is drawn per test and applied to *every*
column, so the cross-sectional correlation between configurations — which is
what makes a max-statistic test less conservative than Bonferroni — is
preserved. Only the K bootstrap means of each path are needed, and
``mean*[b, k] = (1/T) * sum_t c[b, t] * d[t, k]`` where ``c[b, t]`` counts how
often session ``t`` appears in path ``b``; a (paths x T) @ (T x K) product
therefore replaces a (paths x T x K) array. Paths are generated in chunks
whose size depends on T only (each index/count array is capped at about
32 MB), so the draws — and the results for a given ``seed`` — never depend on
K, and adding a duplicate column cannot change them.

White's Reality Check (White, 2000)
    ``V = max_k sqrt(T) * mean(d_k)``;
    ``V*_b = max_k sqrt(T) * (mean*_bk - mean(d_k))``; ``p = #{V*_b >= V} / B``.
    Recentring every column at its own mean imposes the least favourable null
    ``E[d_k] = 0`` for all k, so very poor configurations dilute the test
    (a conservative p-value). It is not studentised, so volatile
    configurations dominate the maximum. In short samples it is slightly
    liberal instead: the stationary bootstrap variance of a mean is biased
    down by about ``2 * mean_block / T`` (8% rejections at 5% for 20
    independent zero-mean columns, T = 500, ``mean_block`` = 21; 6% at
    T = 1000 and 2000).

Hansen's test for Superior Predictive Ability (Hansen, 2005)
    Studentised: ``t_k = sqrt(T) * mean(d_k) / w_k`` and
    ``T_SPA = max(0, max_k t_k)``. Bootstrap paths are recentred at an estimate
    ``mu_k`` of the null mean,
    ``T*_b = max(0, max_k sqrt(T) * (mean*_bk - mean(d_k) + mu_k) / s*_bk)``, and
    ``p = #{T*_b >= T_SPA} / B`` with

    * lower:      ``mu_k = min(mean(d_k), 0)``
    * consistent: ``mu_k = mean(d_k)`` if ``sqrt(T) * mean(d_k) / w_k <= -sqrt(2 log log T)``, else 0
    * upper:      ``mu_k = 0`` (the studentised Reality Check)

    Since ``mu_lower <= mu_consistent <= mu_upper`` column by column and all
    three use the same draws and studentisers, ``T*_b`` is ordered path by
    path and the p-values are ordered ``lower <= consistent <= upper``. The consistent p-value is
    the one to report: configurations that are clearly worse than the
    benchmark no longer inflate it.

    Variance estimator. ``w_k**2`` is the stationary-bootstrap kernel
    estimator of the long-run variance of ``sqrt(T) * mean(d_k)`` given by
    Hansen (2005), following Politis & Romano (1994)::

        w_k**2 = g_0 + 2 * sum_{i=1}^{T-1} kappa(T, i) * g_i,
        kappa(T, i) = (1 - i/T) * (1 - q)**i + (i/T) * (1 - q)**(T - i),  q = 1 / mean_block,

    with ``g_i = (1/T) * sum_{t=1}^{T-i} (d_t - mean(d)) (d_{t+i} - mean(d))``.
    It equals the exact conditional variance of ``sqrt(T) * mean*`` under the
    (circular) stationary bootstrap used for the p-values — the bootstrap
    variance with B = infinity — so it is free of resampling noise and needs
    no second bootstrap. All T-1 autocovariances are computed by FFT
    (O(T log T) per column). A column whose ``w_k`` is below 1e-12 (a constant
    differential, e.g. a configuration identical to the benchmark) carries no
    sampling uncertainty; it is excluded from the studentised statistics and
    counted in ``n_excluded``.

    Studentiser of the bootstrap paths, ``s*_bk``. Hansen divides every path
    by the same ``w_k`` (``restudentise=False``). That ignores the sampling
    error of ``w_k`` itself — a kernel estimate whose relative standard error
    is roughly ``sqrt(2 * mean_block / T)`` — so the observed ``t_k`` have
    heavier tails than their bootstrap counterparts and the maximum over K
    columns over-rejects when ``T / mean_block`` is small. By default
    (``restudentise=True``) each path is instead studentised by its own
    block-sum variance (the bootstrap-t principle for block bootstraps of
    Götze & Künsch, 1996)::

        s*_bk**2 = (1/T) * sum_j (S*_jk - L_j * mean*_bk)**2,

    where the path consists of blocks j of lengths ``L_j`` with sums ``S*_jk``.
    Centred at the sample mean instead of ``mean*_bk`` its bootstrap
    expectation would be exactly ``w_k**2`` (two sessions of a path share a
    block with probability ``(1 - q)**lag``, the kernel's weight); centring at
    the path mean lowers it by a factor of about ``1 - 2 * mean_block / T`` —
    the same downward bias ``w_k**2`` has as an estimate of the true long-run
    variance, which is what the bootstrap-t should reproduce. Nothing changes
    asymptotically. A path whose ``s*_bk`` is below 1e-12 (possible for very
    sparse return columns) falls back to ``w_k``. On K = 20 independent
    zero-mean columns, T = 500, ``mean_block`` = 21 and 150 replications, the
    rejection rate at 5% was 4.7% re-studentised against 14.0% with the fixed
    ``w_k`` (4.7% vs 11.3% at T = 1000; 3.3% vs 6.0% at T = 2000).
    Re-studentising roughly doubles the run time.

Romano-Wolf stepdown (Romano & Wolf, 2005; adjusted p-values as in Romano & Wolf, 2016)
    Tests each ``H0_k: E[d_k] <= 0`` with family-wise error control. With the
    studentised ``t_k`` sorted in decreasing order ``t_(1) >= ... >= t_(K)``
    and the recentred bootstrap ``t*_bk = sqrt(T) * (mean*_bk - mean(d_k)) / s*_bk``,
    ``p_(j) = #{max_{l >= j} t*_(l) >= t_(j)} / B`` and the adjusted p-value
    is ``max(p_(1), ..., p_(j))``. Rejecting every ``H0_k`` whose adjusted
    p-value is at most ``alpha`` controls the family-wise error rate at
    ``alpha`` asymptotically. When its ``t_k`` is positive, the adjusted
    p-value of the top configuration equals the upper SPA p-value (same seed
    and options).

Caveats: every test assumes the differentials are stationary and weakly
dependent; the block bootstrap cannot anticipate regimes absent from the
sample. Only the configurations passed in are accounted for — if the grid
was itself chosen after looking at other grids, those searches are not.

References
----------
* H. White (2000), "A Reality Check for Data Snooping", *Econometrica* 68(5), 1097-1126.
* P. R. Hansen (2005), "A Test for Superior Predictive Ability", *Journal of Business &
  Economic Statistics* 23(4), 365-380.
* D. N. Politis and J. P. Romano (1994), "The Stationary Bootstrap", *Journal of the
  American Statistical Association* 89(428), 1303-1313.
* F. Götze and H. R. Künsch (1996), "Second-order correctness of the blockwise bootstrap
  for stationary observations", *Annals of Statistics* 24(5), 1914-1933.
* J. P. Romano and M. Wolf (2005), "Stepwise Multiple Testing as Formalized Data Snooping",
  *Econometrica* 73(4), 1237-1282.
* J. P. Romano and M. Wolf (2016), "Efficient computation of adjusted p-values for
  resampling-based stepdown multiple testing", *Statistics & Probability Letters* 113, 38-40.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.config import TRADING_DAYS_PER_YEAR
from quantlab.metrics import daily_risk_free
from quantlab.robustness import DEFAULT_SEED, stationary_bootstrap_indices

MIN_OBS = 20  # as quantlab.robustness.bootstrap_metrics

_EPS = 1e-12  # long-run standard deviation below this is zero (as in quantlab.metrics)
_CHUNK_BYTES = 32 * 2**20  # cap on each (paths x T) index / count array


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RealityCheckResult:
    """Outcome of :func:`white_reality_check`. Arrays are read-only."""

    statistic: float  # V = max_k sqrt(T) * mean(d_k)
    p_value: float  # #{V*_b >= V} / B
    best_index: int  # column position attaining V (lowest on ties)
    best_label: Hashable
    mean_differentials: pd.Series  # mean(d_k), per session, indexed by column label
    bootstrap_statistics: np.ndarray  # V*_b, shape (n_boot,)
    n_obs: int
    n_configs: int
    n_boot: int
    mean_block: float
    seed: int


@dataclass(frozen=True)
class SPAResult:
    """Outcome of :func:`hansen_spa`. Arrays are read-only.

    ``t_statistics`` is NaN and ``best_index``/``best_label`` are None where
    columns were excluded for having a constant differential.
    """

    statistic: float  # T_SPA = max(0, max_k t_k)
    p_value: float  # consistent p-value (the one to report)
    p_value_lower: float
    p_value_upper: float
    best_index: int | None  # column position with the largest t_k (lowest on ties)
    best_label: Hashable | None
    t_statistics: pd.Series  # t_k = sqrt(T) * mean(d_k) / w_k
    omega: pd.Series  # w_k, long-run standard deviation of sqrt(T) * mean(d_k)
    mean_differentials: pd.Series
    threshold: float  # sqrt(2 log log T): recentring threshold on t_k
    n_excluded: int  # columns with w_k < 1e-12, left out of the studentised max
    bootstrap_statistics: np.ndarray  # consistent T*_b, shape (n_boot,)
    n_obs: int
    n_configs: int
    n_boot: int
    mean_block: float
    seed: int
    restudentised: bool  # bootstrap paths studentised by their own block-sum variance


@dataclass(frozen=True)
class RomanoWolfResult:
    """Outcome of :func:`romano_wolf_stepdown`.

    ``adjusted_p_values`` are family-wise-error adjusted, one-sided
    (``H0_k: E[d_k] <= 0``); excluded (constant) columns get 1.0.
    ``single_step_p_values`` compare each ``t_k`` with the maximum over *all*
    configurations (never smaller than the stepdown values).
    """

    adjusted_p_values: pd.Series
    single_step_p_values: pd.Series
    t_statistics: pd.Series
    omega: pd.Series
    n_excluded: int
    n_obs: int
    n_configs: int
    n_boot: int
    mean_block: float
    seed: int
    restudentised: bool

    def rejected(self, alpha: float = 0.05) -> list[Hashable]:
        """Labels of the configurations whose null ``E[d_k] <= 0`` is rejected at FWER ``alpha``."""
        return list(self.adjusted_p_values.index[self.adjusted_p_values.to_numpy() <= alpha])


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def _numeric_frame(returns_matrix: pd.DataFrame | np.ndarray, name: str) -> pd.DataFrame:
    if isinstance(returns_matrix, pd.Series):
        returns_matrix = returns_matrix.to_frame()
    if not isinstance(returns_matrix, pd.DataFrame):
        arr = np.asarray(returns_matrix)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional (sessions x configurations)")
        returns_matrix = pd.DataFrame(arr)
    try:
        values = returns_matrix.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc
    if returns_matrix.shape[1] < 1:
        raise ValueError(f"{name} has no columns: at least one configuration is required")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values; align and clean the columns first")
    return pd.DataFrame(values, index=returns_matrix.index, columns=returns_matrix.columns)


def loss_differentials(
    returns_matrix: pd.DataFrame | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Performance differentials ``d[k, t] = r[k, t] - b[t]`` (positive = configuration did better).

    ``b`` is ``benchmark_returns`` when given — a Series is aligned on the
    matrix's index and must cover every session; an array must have one value
    per row — and otherwise the daily risk-free return
    ``(1 + risk_free_rate) ** (1 / periods_per_year) - 1`` ("versus cash").

    Raises ValueError for NaN/infinite or non-numeric inputs, an empty
    matrix, or a benchmark that does not cover every session.
    """
    frame = _numeric_frame(returns_matrix, "returns_matrix")
    if benchmark_returns is None:
        if not periods_per_year > 0:
            raise ValueError("periods_per_year must be positive")
        return frame - daily_risk_free(risk_free_rate, periods_per_year)
    if isinstance(benchmark_returns, pd.Series):
        bench = benchmark_returns.reindex(frame.index)
    else:
        bench = np.asarray(benchmark_returns, dtype=np.float64)
        if bench.shape != (len(frame),):
            raise ValueError("an unlabelled benchmark must have exactly one return per session")
        bench = pd.Series(bench, index=frame.index)
    try:
        bench = bench.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark_returns must be numeric") from exc
    if not np.isfinite(bench.to_numpy()).all():
        raise ValueError("benchmark returns must cover every session of returns_matrix with finite values")
    return frame.sub(bench, axis=0)


def _prepare(d: pd.DataFrame | np.ndarray, n_boot: int, mean_block: float) -> tuple[np.ndarray, pd.Index]:
    frame = _numeric_frame(d, "d")
    n_obs = frame.shape[0]
    if n_obs < MIN_OBS:
        raise ValueError(f"at least {MIN_OBS} sessions are needed for a block bootstrap test, got {n_obs}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not (math.isfinite(mean_block) and 1.0 <= mean_block <= n_obs):
        raise ValueError(f"mean_block must be between 1 and the number of sessions ({n_obs})")
    return frame.to_numpy(), frame.columns


# --------------------------------------------------------------------------- #
# Bootstrap machinery
# --------------------------------------------------------------------------- #


def _path_block_variances(idx: np.ndarray, z: np.ndarray, prefix: np.ndarray) -> np.ndarray:
    """``(paths, K)`` block-sum variance of each bootstrap path: ``(1/T) sum_j (S_j - L_j * mean*)**2``.

    A path is a sequence of blocks — runs of consecutive (circular) sessions of
    ``z`` — with sums ``S_j`` and lengths ``L_j``. Block sums come from the
    prefix sums of ``[z; z]`` (a block that wraps past the end is one block), so
    no (paths x T x K) array is formed; paths are processed in groups whose
    (blocks x K) arrays stay under ~8 MB.
    """
    n_paths, n_obs = idx.shape
    n_cols = z.shape[1]
    starts = np.ones((n_paths, n_obs), dtype=bool)
    starts[:, 1:] = idx[:, 1:] != (idx[:, :-1] + 1) % n_obs
    path_of, pos = np.nonzero(starts)  # row-major: grouped by path, increasing position
    per_path = starts.sum(axis=1)
    first = np.concatenate([[0], np.cumsum(per_path)[:-1]])
    end = np.append(pos[1:], n_obs)
    end[first[1:] - 1] = n_obs  # the last block of every path runs to the path's end
    length = end - pos
    origin = idx[path_of, pos]
    full_cycles, rest = np.divmod(length, n_obs)
    cycle_sum = z.sum(axis=0)

    out = np.empty((n_paths, n_cols))
    budget = max(n_obs, (_CHUNK_BYTES // 4) // (8 * n_cols))  # rows per (blocks x K) array; >= one path
    cum = np.cumsum(per_path)
    p0 = 0
    while p0 < n_paths:
        p1 = max(p0 + 1, int(np.searchsorted(cum, first[p0] + budget, side="right")))
        b0, b1 = first[p0], (first[p1] if p1 < n_paths else len(pos))
        o, r = origin[b0:b1], rest[b0:b1]
        sums = prefix[o + r] - prefix[o] + full_cycles[b0:b1, None] * cycle_sum
        offsets = first[p0:p1] - b0
        path_mean = np.add.reduceat(sums, offsets, axis=0) / n_obs
        dev = sums - length[b0:b1, None] * np.repeat(path_mean, per_path[p0:p1], axis=0)
        out[p0:p1] = np.add.reduceat(dev * dev, offsets, axis=0) / n_obs
        p0 = p1
    return out


def _bootstrap_draws(
    x: np.ndarray, n_boot: int, mean_block: float, seed: int, path_variance: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Centred bootstrap means ``mean*_bk - mean_k`` (n_boot x K) and, optionally, per-path variances.

    Every path's index vector is shared by all columns. Means use the count
    matrix product described in the module docstring. Paths come in chunks of
    a size that depends on T only, so the draws for a given seed are the same
    whatever the number of columns.
    """
    n_obs, n_cols = x.shape
    z = x - x.mean(axis=0)
    rng = np.random.default_rng(seed)
    chunk = int(max(1, min(n_boot, _CHUNK_BYTES // (8 * n_obs))))
    dev = np.empty((n_boot, n_cols))
    var = np.empty((n_boot, n_cols)) if path_variance else None
    prefix = None
    if path_variance:
        prefix = np.concatenate([np.zeros((1, n_cols)), np.cumsum(np.concatenate([z, z]), axis=0)])
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = stationary_bootstrap_indices(n_obs, m, mean_block, rng)
        if var is not None:
            var[start : start + m] = _path_block_variances(idx, z, prefix)
        idx += (np.arange(m, dtype=np.int64) * n_obs)[:, None]
        counts = np.bincount(idx.ravel(), minlength=m * n_obs).reshape(m, n_obs).astype(np.float64)
        del idx
        dev[start : start + m] = counts @ z / n_obs
    return dev, var


def stationary_bootstrap_variance(x: np.ndarray, mean_block: float) -> np.ndarray:
    """Hansen's (2005) long-run variance ``w**2`` of ``sqrt(T) * mean`` for each column of ``x``.

    ``w**2 = g_0 + 2 * sum_{i=1}^{T-1} kappa(T, i) g_i`` with the stationary-bootstrap
    kernel ``kappa(T, i) = (1 - i/T)(1 - q)**i + (i/T)(1 - q)**(T - i)``, ``q = 1/mean_block``,
    and biased autocovariances ``g_i`` (divisor T) computed by FFT. This is the exact
    variance of ``sqrt(T) * mean*`` under the circular stationary bootstrap.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_obs = x.shape[0]
    z = x - x.mean(axis=0)
    nfft = 1 << (2 * n_obs - 1).bit_length()  # zero padding >= 2T - 1: no circular wrap-around
    spec = np.fft.rfft(z, n=nfft, axis=0)
    acov = np.fft.irfft(spec.real**2 + spec.imag**2, n=nfft, axis=0)[:n_obs] / n_obs
    lags = np.arange(1, n_obs)
    q = 1.0 / mean_block
    kappa = (1 - lags / n_obs) * (1 - q) ** lags + (lags / n_obs) * (1 - q) ** (n_obs - lags)
    return np.maximum(acov[0] + 2.0 * (kappa @ acov[1:]), 0.0)


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def white_reality_check(
    d: pd.DataFrame | np.ndarray,
    n_boot: int = 1000,
    mean_block: float = 21.0,
    seed: int = DEFAULT_SEED,
) -> RealityCheckResult:
    """White's (2000) Reality Check for data snooping on the T x K differentials ``d``.

    ``V = max_k sqrt(T) * mean(d_k)``, ``V*_b = max_k sqrt(T) * (mean*_bk - mean(d_k))``
    with stationary-bootstrap means sharing one index path across columns, and
    ``p = #{V*_b >= V} / B``. A small p-value says the best configuration's
    advantage over the benchmark is unlikely to be the luck of searching K
    configurations. Not studentised: high-volatility configurations weigh more.

    Raises ValueError for NaN/infinite values, no columns, fewer than
    ``MIN_OBS`` sessions, ``n_boot < 1`` or ``mean_block`` outside [1, T].
    """
    x, labels = _prepare(d, n_boot, mean_block)
    n_obs, n_cols = x.shape
    means = x.mean(axis=0)
    root_n = math.sqrt(n_obs)
    best = int(np.argmax(means))
    statistic = float(root_n * means[best])
    boot = root_n * _bootstrap_draws(x, n_boot, mean_block, seed)[0].max(axis=1)
    return RealityCheckResult(
        statistic=statistic,
        p_value=float(np.mean(boot >= statistic)),
        best_index=best,
        best_label=labels[best],
        mean_differentials=pd.Series(means, index=labels, name="mean_differential"),
        bootstrap_statistics=_frozen(boot),
        n_obs=n_obs,
        n_configs=n_cols,
        n_boot=int(n_boot),
        mean_block=float(mean_block),
        seed=seed,
    )


def _studentised(x: np.ndarray, mean_block: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column means, w_k, validity mask (w_k > eps) and t_k (NaN where invalid)."""
    means = x.mean(axis=0)
    omega = np.sqrt(stationary_bootstrap_variance(x, mean_block))
    valid = omega > _EPS
    t = np.full(x.shape[1], np.nan)
    t[valid] = math.sqrt(x.shape[0]) * means[valid] / omega[valid]
    return means, omega, valid, t


def _studentised_draws(
    x: np.ndarray, omega: np.ndarray, n_boot: int, mean_block: float, seed: int, restudentise: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Centred bootstrap means (n_boot x K) and the standard errors that studentise them.

    With ``restudentise`` each path is scaled by its own block-sum standard
    deviation (falling back to ``w_k`` where a path's is below 1e-12); otherwise
    every path uses ``w_k`` (Hansen's original construction).
    """
    dev, var = _bootstrap_draws(x, n_boot, mean_block, seed, path_variance=restudentise)
    if var is None:
        return dev, np.broadcast_to(omega, dev.shape)
    scale = np.sqrt(var)
    return dev, np.where(scale > _EPS, scale, omega)


def hansen_spa(
    d: pd.DataFrame | np.ndarray,
    n_boot: int = 1000,
    mean_block: float = 21.0,
    seed: int = DEFAULT_SEED,
    restudentise: bool = True,
) -> SPAResult:
    """Hansen's (2005) test for Superior Predictive Ability on the T x K differentials ``d``.

    Returns the consistent p-value with the lower and upper bounds; see the
    module docstring for the statistic, the three recentrings (threshold
    ``-sqrt(2 log log T) * w_k / sqrt(T)`` on ``mean(d_k)``), the variance
    estimator ``w_k`` and ``restudentise`` (default True: each bootstrap path is
    studentised by its own block-sum variance; False: Hansen's fixed ``w_k``).
    ``p = #{T*_b >= T_SPA} / B``, so ``T_SPA = 0`` (no configuration ahead of
    the benchmark) gives p = 1.

    Raises ValueError for NaN/infinite values, no columns, fewer than
    ``MIN_OBS`` sessions, ``n_boot < 1`` or ``mean_block`` outside [1, T].
    """
    x, labels = _prepare(d, n_boot, mean_block)
    n_obs, n_cols = x.shape
    means, omega, valid, t = _studentised(x, mean_block)
    threshold = math.sqrt(2.0 * math.log(math.log(n_obs)))
    root_n = math.sqrt(n_obs)

    if valid.any():
        best = int(np.nanargmax(t))
        statistic = max(0.0, float(t[best]))
        m_v, t_v = means[valid], t[valid]
        dev, scale = _studentised_draws(x[:, valid], omega[valid], n_boot, mean_block, seed, restudentise)
        p_values = {}
        boot_c = None
        for name, mu in (
            ("lower", np.minimum(m_v, 0.0)),
            ("consistent", np.where(t_v <= -threshold, m_v, 0.0)),
            ("upper", np.zeros_like(m_v)),
        ):
            boot = np.maximum((root_n * (dev + mu) / scale).max(axis=1), 0.0)
            p_values[name] = float(np.mean(boot >= statistic))
            if name == "consistent":
                boot_c = boot
    else:  # every differential is constant: nothing to studentise, no evidence of superiority
        best, statistic = None, 0.0
        p_values = {"lower": 1.0, "consistent": 1.0, "upper": 1.0}
        boot_c = np.zeros(n_boot)

    return SPAResult(
        statistic=statistic,
        p_value=p_values["consistent"],
        p_value_lower=p_values["lower"],
        p_value_upper=p_values["upper"],
        best_index=best,
        best_label=None if best is None else labels[best],
        t_statistics=pd.Series(t, index=labels, name="t_statistic"),
        omega=pd.Series(omega, index=labels, name="omega"),
        mean_differentials=pd.Series(means, index=labels, name="mean_differential"),
        threshold=threshold,
        n_excluded=int((~valid).sum()),
        bootstrap_statistics=_frozen(boot_c),
        n_obs=n_obs,
        n_configs=n_cols,
        n_boot=int(n_boot),
        mean_block=float(mean_block),
        seed=seed,
        restudentised=restudentise,
    )


def romano_wolf_stepdown(
    d: pd.DataFrame | np.ndarray,
    n_boot: int = 1000,
    mean_block: float = 21.0,
    seed: int = DEFAULT_SEED,
    restudentise: bool = True,
) -> RomanoWolfResult:
    """Romano-Wolf stepdown adjusted p-values for ``H0_k: E[d_k] <= 0``, one per configuration.

    Studentised exactly as :func:`hansen_spa` (same ``w_k``, same bootstrap
    draws and ``restudentise`` option for a given seed); see the module
    docstring for the algorithm. Ties in ``t_k`` are ordered by column position.

    Raises ValueError for NaN/infinite values, no columns, fewer than
    ``MIN_OBS`` sessions, ``n_boot < 1`` or ``mean_block`` outside [1, T].
    """
    x, labels = _prepare(d, n_boot, mean_block)
    n_obs, n_cols = x.shape
    _, omega, valid, t = _studentised(x, mean_block)
    adjusted = np.ones(n_cols)
    single = np.ones(n_cols)
    if valid.any():
        cols = np.flatnonzero(valid)
        dev, scale = _studentised_draws(x[:, cols], omega[cols], n_boot, mean_block, seed, restudentise)
        sorter = np.argsort(-t[cols], kind="stable")  # decreasing t, ties by column position
        order = cols[sorter]
        t_star = math.sqrt(n_obs) * dev[:, sorter] / scale[:, sorter]
        # max over the hypotheses still in play at step j: columns j..end of the sorted matrix
        tail_max = np.maximum.accumulate(t_star[:, ::-1], axis=1)[:, ::-1]
        p_step = np.mean(tail_max >= t[order], axis=0)
        adjusted[order] = np.maximum.accumulate(p_step)
        single[order] = np.mean(tail_max[:, :1] >= t[order], axis=0)
    return RomanoWolfResult(
        adjusted_p_values=pd.Series(adjusted, index=labels, name="adjusted_p_value"),
        single_step_p_values=pd.Series(single, index=labels, name="single_step_p_value"),
        t_statistics=pd.Series(t, index=labels, name="t_statistic"),
        omega=pd.Series(omega, index=labels, name="omega"),
        n_excluded=int((~valid).sum()),
        n_obs=n_obs,
        n_configs=n_cols,
        n_boot=int(n_boot),
        mean_block=float(mean_block),
        seed=seed,
        restudentised=restudentise,
    )
