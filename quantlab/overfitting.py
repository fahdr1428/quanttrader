"""Probability of backtest overfitting (PBO) by combinatorially symmetric cross-validation.

The question
------------
A parameter grid produces K configurations, each with a daily return path over
the same T sessions. "Pick the configuration with the best in-sample Sharpe
ratio" is itself a procedure, and it can be evaluated. PBO is the probability
that the configuration this procedure selects ranks *below the median of the
grid* out of sample (Bailey, Borwein, López de Prado & Zhu, 2017).

Combinatorially symmetric cross-validation (CSCV)
-------------------------------------------------
1. Partition the T sessions into S (even) contiguous blocks. When T is not a
   multiple of S the first ``T mod S`` blocks hold one extra session, so the
   blocks always partition the sample exactly and deterministically.
2. For each of the ``C(S, S/2)`` ways to choose S/2 blocks as the in-sample (IS)
   set — the other S/2 blocks are the out-of-sample (OOS) set, so every split
   also appears with the roles swapped:

   a. compute every configuration's Sharpe ratio on the IS and on the OOS set,
      ``SR = mean(r - rf_d) / std(r - rf_d, ddof=1) * sqrt(P)`` with
      ``rf_d = (1 + rf) ** (1/P) - 1`` (the convention of ``quantlab.metrics``).
      A sub-sample with (numerically) zero variance has ``SR = 0``, never NaN;
   b. select ``n* = argmax_k SR_IS[k]`` (ties: lowest column position);
   c. its OOS relative rank ``w = rank(SR_OOS[n*]) / (K + 1)``, where ``rank`` is
      the ascending rank in ``1..K`` among the K OOS Sharpe ratios (ties get the
      mid-rank), so ``0 < w < 1``;
   d. the logit ``lambda = ln(w / (1 - w))``. ``lambda > 0`` means the selected
      configuration beat the OOS median.

3. ``PBO = #{lambda <= 0} / C(S, S/2)``.

Because Sharpe ratios only need the count, sum and sum of squares of each
sub-sample, per-block sums and sums of squares (an S x K table) are computed
once; each split then costs O(S * K), vectorised over chunks of splits. Excess
returns are centred on each column's full-sample mean before the sums are
formed, which keeps the ``sum of squares - n * mean^2`` variance formula
accurate; a sub-sample variance below ``1e-10`` times its raw second moment,
or a standard deviation below ``1e-12``, is treated as zero.

Companion statistics (same paper): the *performance degradation* slope — the
OLS slope of the selected configuration's OOS Sharpe on its IS Sharpe across
splits (a slope well below 1, often negative, is the signature of selection
bias) — and the *probability of OOS loss*, the share of splits in which the
selected configuration has a negative OOS Sharpe ratio.

What PBO does **not** measure
-----------------------------
PBO asks whether *choosing among the grid points* is informative out of sample;
it says nothing about whether any edge exists. If every configuration shares
the same edge (or the same lack of one), the IS ranking is noise, the OOS rank
of the winner is uniform and PBO stays near 0.5 — even when every
configuration is profitable. A low PBO says the in-sample ranking carries over
out of sample; it does not say the winner beats cash or a benchmark (use
``quantlab.significance`` for that). CSCV also assumes the blocks are
exchangeable (roughly stationary returns) and, by construction, uses OOS
blocks that precede IS blocks, so it is a statistical diagnostic, not a
walk-forward simulation.

Reference: D. H. Bailey, J. M. Borwein, M. López de Prado and Q. J. Zhu (2017),
"The Probability of Backtest Overfitting", *Journal of Computational Finance*
20(4), 39-69.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from itertools import combinations, islice

import numpy as np
import pandas as pd

from quantlab.config import TRADING_DAYS_PER_YEAR
from quantlab.metrics import daily_risk_free

MIN_OBS_PER_BLOCK = 10
MAX_BLOCKS = 20  # C(20, 10) = 184,756 splits; beyond that the enumeration is impractical

_EPS = 1e-12  # standard deviation below this is zero (as in quantlab.metrics)
# A sub-sample variance below this share of its raw second moment is floating-point
# residue of the ``sum of squares - n * mean^2`` formula, i.e. a constant sub-sample.
_REL_VAR_TOL = 1e-10
_SPLIT_CHUNK = 1024  # splits evaluated together (bounds memory: ~16 MB per gathered array at S=20, K=200)


@dataclass(frozen=True)
class PBOResult:
    """Outcome of :func:`probability_of_backtest_overfitting`.

    All per-split arrays have length ``n_splits`` and follow the lexicographic
    order of ``itertools.combinations(range(n_blocks), n_blocks // 2)`` (the IS
    block positions). Sharpe ratios are annualised. Arrays are read-only.
    """

    pbo: float  # share of splits with logit <= 0
    logits: np.ndarray  # lambda = ln(w / (1 - w)) per split
    oos_relative_ranks: np.ndarray  # w = rank / (K + 1) per split
    is_sharpes: np.ndarray  # selected configuration's in-sample Sharpe per split
    oos_sharpes: np.ndarray  # the same configuration's out-of-sample Sharpe per split
    selected: np.ndarray  # column position selected in sample, per split
    degradation_slope: float | None  # OLS slope of oos_sharpes on is_sharpes (None if IS Sharpes are constant)
    degradation_intercept: float | None
    prob_oos_loss: float  # share of splits with oos_sharpes < 0
    n_splits: int
    n_configs: int
    n_blocks: int
    n_obs: int
    block_edges: np.ndarray  # length n_blocks + 1; block b covers rows [edges[b], edges[b + 1])
    config_labels: tuple[Hashable, ...]

    def selection_counts(self) -> pd.Series:
        """How many splits selected each configuration in sample (indexed by column label)."""
        counts = np.bincount(self.selected, minlength=self.n_configs)
        return pd.Series(counts, index=pd.Index(self.config_labels, tupleize_cols=False), name="times_selected")


def block_boundaries(n_obs: int, n_blocks: int) -> np.ndarray:
    """Edges of ``n_blocks`` contiguous blocks that partition ``range(n_obs)`` exactly.

    Block sizes differ by at most one: the first ``n_obs mod n_blocks`` blocks
    get ``n_obs // n_blocks + 1`` sessions, the rest ``n_obs // n_blocks``.
    Returns an integer array ``e`` of length ``n_blocks + 1`` with ``e[0] = 0`` and
    ``e[-1] = n_obs``; block ``b`` is ``[e[b], e[b + 1])``.
    """
    if n_blocks < 1 or n_obs < n_blocks:
        raise ValueError(f"cannot split {n_obs} observations into {n_blocks} non-empty blocks")
    base, extra = divmod(n_obs, n_blocks)
    sizes = np.full(n_blocks, base, dtype=np.int64)
    sizes[:extra] += 1
    return np.concatenate([[0], np.cumsum(sizes)])


def _subsample_sharpes(n: np.ndarray, sums: np.ndarray, sqs: np.ndarray, shift: np.ndarray, ann: float) -> np.ndarray:
    """Annualised Sharpe ratios from counts and sums of centred excess returns.

    ``n`` is (m,), ``sums``/``sqs`` are (m, K) sums of ``z`` and ``z**2`` where
    ``z = excess - shift``. Zero-variance sub-samples get Sharpe 0.
    """
    n = n[:, None]
    dev_mean = sums / n
    var = (sqs - sums * dev_mean) / (n - 1)
    var = np.where(var > _REL_VAR_TOL * (sqs / n), var, 0.0)  # also clears round-off negatives
    std = np.sqrt(var)
    out = np.zeros_like(var)
    np.divide(dev_mean + shift, std, out=out, where=std > _EPS)
    return out * ann


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame | np.ndarray,
    n_blocks: int = 12,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PBOResult | None:
    """Probability of backtest overfitting of "pick the best in-sample Sharpe" over a grid.

    Parameters
    ----------
    returns_matrix
        T x K daily net returns, one column per configuration, all on the same
        sessions (a 2-D array is accepted; columns are then labelled 0..K-1).
    n_blocks
        Number S of contiguous blocks; must be even and between 2 and
        ``MAX_BLOCKS``. The default 12 gives 924 splits.
    risk_free_rate
        Annual risk-free rate; Sharpe ratios are computed on returns in excess
        of its daily equivalent.
    periods_per_year
        Annualisation factor for the reported Sharpe ratios (ranks do not depend on it).

    Returns
    -------
    PBOResult, or ``None`` when the question cannot be asked of the data: fewer
    than two configurations, or fewer than ``MIN_OBS_PER_BLOCK`` sessions in the
    smallest block (T < S * ``MIN_OBS_PER_BLOCK``).

    Raises
    ------
    ValueError
        For an odd or out-of-range ``n_blocks``, a non-positive
        ``periods_per_year``, or a matrix that is not two-dimensional, not
        numeric, or contains NaN or infinite values.

    See the module docstring for the method, the formulas and what PBO does
    not measure.
    """
    if isinstance(n_blocks, bool) or int(n_blocks) != n_blocks:
        raise ValueError("n_blocks must be an integer")
    n_blocks = int(n_blocks)
    if n_blocks < 2 or n_blocks % 2 or n_blocks > MAX_BLOCKS:
        raise ValueError(f"n_blocks must be an even number between 2 and {MAX_BLOCKS}, got {n_blocks}")
    if not periods_per_year > 0:
        raise ValueError("periods_per_year must be positive")
    if isinstance(returns_matrix, pd.DataFrame):
        labels = tuple(returns_matrix.columns)
        raw = returns_matrix
    else:
        raw = np.asarray(returns_matrix)
        if raw.ndim != 2:
            raise ValueError("returns_matrix must be two-dimensional (sessions x configurations)")
        labels = tuple(range(raw.shape[1]))
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns_matrix must contain only numeric returns") from exc
    if not np.isfinite(values).all():
        raise ValueError("returns_matrix contains NaN or infinite values; align and clean the columns first")

    n_obs, n_configs = values.shape
    if n_configs < 2 or n_obs < n_blocks * MIN_OBS_PER_BLOCK:
        return None

    excess = values - daily_risk_free(risk_free_rate, periods_per_year)
    shift = excess.mean(axis=0)
    centred = excess - shift
    edges = block_boundaries(n_obs, n_blocks)
    sizes = np.diff(edges).astype(np.float64)
    block_sums = np.add.reduceat(centred, edges[:-1], axis=0)  # S x K
    block_sqs = np.add.reduceat(centred * centred, edges[:-1], axis=0)
    ann = math.sqrt(periods_per_year)

    logits, ranks, is_sr, oos_sr, selected = [], [], [], [], []
    splits = combinations(range(n_blocks), n_blocks // 2)
    while chunk := list(islice(splits, _SPLIT_CHUNK)):
        m = len(chunk)
        rows = np.arange(m)
        is_blocks = np.asarray(chunk)  # (m, S/2), increasing
        member = np.zeros((m, n_blocks), dtype=bool)
        member[rows[:, None], is_blocks] = True
        oos_blocks = np.nonzero(~member)[1].reshape(m, -1)  # complements, increasing
        # Gather-and-sum (not a BLAS product) adds blocks in the same order for every column, so
        # identical columns get bit-identical Sharpe ratios and the lowest-position tie-break holds.
        sr_is, sr_oos = (
            _subsample_sharpes(sizes[b].sum(axis=1), block_sums[b].sum(axis=1), block_sqs[b].sum(axis=1), shift, ann)
            for b in (is_blocks, oos_blocks)
        )
        best = np.argmax(sr_is, axis=1)  # first maximum: ties go to the lowest column position
        chosen = sr_oos[rows, best]
        below = (sr_oos < chosen[:, None]).sum(axis=1)
        tied = (sr_oos == chosen[:, None]).sum(axis=1)  # includes the selected configuration itself
        w = (below + (tied + 1) / 2.0) / (n_configs + 1)
        logits.append(np.log(w / (1.0 - w)))
        ranks.append(w)
        is_sr.append(sr_is[rows, best])
        oos_sr.append(chosen)
        selected.append(best)

    lam = np.concatenate(logits)
    x = np.concatenate(is_sr)
    y = np.concatenate(oos_sr)
    slope = intercept = None
    if np.ptp(x) > _EPS * max(1.0, float(np.abs(x).max())):
        xc = x - x.mean()
        slope = float(xc @ (y - y.mean()) / (xc @ xc))
        intercept = float(y.mean() - slope * x.mean())

    return PBOResult(
        pbo=float(np.mean(lam <= 0)),
        logits=_frozen(lam),
        oos_relative_ranks=_frozen(np.concatenate(ranks)),
        is_sharpes=_frozen(x),
        oos_sharpes=_frozen(y),
        selected=_frozen(np.concatenate(selected).astype(np.int64)),
        degradation_slope=slope,
        degradation_intercept=intercept,
        prob_oos_loss=float(np.mean(y < 0)),
        n_splits=int(lam.size),
        n_configs=int(n_configs),
        n_blocks=n_blocks,
        n_obs=int(n_obs),
        block_edges=_frozen(edges),
        config_labels=labels,
    )
