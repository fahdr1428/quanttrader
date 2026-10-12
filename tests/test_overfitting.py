"""Probability of backtest overfitting (CSCV). All data are seeded synthetic returns."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from quantlab.metrics import sharpe_ratio
from quantlab.overfitting import block_boundaries, probability_of_backtest_overfitting


def _matrix(n=240, k=5, seed=0, sd=0.01) -> pd.DataFrame:
    """Independent zero-mean (in population) daily returns; sample means are *not* forced to zero."""
    idx = pd.bdate_range("2016-01-04", periods=n)
    values = np.random.default_rng(seed).normal(0.0, sd, (n, k))
    return pd.DataFrame(values, index=idx, columns=[f"cfg{j}" for j in range(k)])


def _naive_pbo(frame: pd.DataFrame, n_blocks: int, risk_free_rate: float = 0.0) -> dict[str, np.ndarray]:
    """Brute-force CSCV: explicit row selection and ``quantlab.metrics.sharpe_ratio`` per column."""
    n, k = frame.shape
    blocks = np.array_split(np.arange(n), n_blocks)  # first n % S blocks one longer, as specified
    out = {"logits": [], "selected": [], "is": [], "oos": []}
    for combo in combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in combo])

        def sharpes(rows):
            values = [sharpe_ratio(frame.iloc[rows, j], risk_free_rate) for j in range(k)]
            return np.array([0.0 if v is None else v for v in values])  # undefined (zero variance) -> 0

        sr_is, sr_oos = sharpes(is_rows), sharpes(oos_rows)
        best = int(np.flatnonzero(sr_is == sr_is.max())[0])  # lowest position among ties
        rank = pd.Series(sr_oos).rank(method="average").iloc[best]
        w = rank / (k + 1)
        out["logits"].append(np.log(w / (1 - w)))
        out["selected"].append(best)
        out["is"].append(sr_is[best])
        out["oos"].append(sr_oos[best])
    return {key: np.asarray(v) for key, v in out.items()}


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_obs,n_blocks", [(240, 8), (247, 8), (1003, 12), (120, 12), (1500, 16)])
def test_blocks_partition_the_sample_exactly(n_obs, n_blocks):
    edges = block_boundaries(n_obs, n_blocks)
    sizes = np.diff(edges)
    assert len(edges) == n_blocks + 1 and edges[0] == 0 and edges[-1] == n_obs
    assert sizes.sum() == n_obs and sizes.max() - sizes.min() <= 1
    assert (np.diff(sizes) <= 0).all()  # the longer blocks come first: deterministic
    covered = np.concatenate([np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])])
    np.testing.assert_array_equal(covered, np.arange(n_obs))  # no gap, no overlap
    expected = [len(b) for b in np.array_split(np.arange(n_obs), n_blocks)]
    np.testing.assert_array_equal(sizes, expected)


def test_result_reports_the_blocks_it_used_for_uneven_samples():
    res = probability_of_backtest_overfitting(_matrix(n=1003, k=4), n_blocks=12)
    np.testing.assert_array_equal(res.block_edges, block_boundaries(1003, 12))
    assert res.n_obs == 1003 and res.n_splits == 924 and res.n_configs == 4


# --------------------------------------------------------------------------- #
# Correctness against a brute-force loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_obs", [240, 247])
def test_vectorised_pbo_matches_naive_loop(n_obs):
    frame = _matrix(n=n_obs, k=5, seed=n_obs)
    res = probability_of_backtest_overfitting(frame, n_blocks=8, risk_free_rate=0.02)
    ref = _naive_pbo(frame, 8, risk_free_rate=0.02)

    assert res.n_splits == 70 and res.n_configs == 5
    np.testing.assert_array_equal(res.selected, ref["selected"])
    np.testing.assert_allclose(res.logits, ref["logits"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.is_sharpes, ref["is"], rtol=1e-9)
    np.testing.assert_allclose(res.oos_sharpes, ref["oos"], rtol=1e-9)
    assert res.pbo == pytest.approx(np.mean(ref["logits"] <= 0))
    assert res.prob_oos_loss == pytest.approx(np.mean(ref["oos"] < 0))
    slope, intercept = np.polyfit(ref["is"], ref["oos"], 1)
    assert res.degradation_slope == pytest.approx(slope, rel=1e-9)
    assert res.degradation_intercept == pytest.approx(intercept, rel=1e-9, abs=1e-12)
    w = res.oos_relative_ranks
    assert ((w > 0) & (w < 1)).all()
    np.testing.assert_allclose(res.logits, np.log(w / (1 - w)))
    assert res.selection_counts().sum() == res.n_splits
    assert list(res.selection_counts().index) == list(frame.columns)


def test_complementary_splits_mirror_each_other():
    res = probability_of_backtest_overfitting(_matrix(n=300, k=6, seed=3), n_blocks=10)
    splits = list(combinations(range(10), 5))
    position = {c: i for i, c in enumerate(splits)}
    for i, combo in enumerate(splits):
        mirror = position[tuple(sorted(set(range(10)) - set(combo)))]
        if res.selected[i] == res.selected[mirror]:  # same winner both ways: IS and OOS Sharpes swap
            assert res.is_sharpes[i] == pytest.approx(res.oos_sharpes[mirror], rel=1e-12)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_pbo_of_pure_noise_is_near_one_half():
    # Under the null every configuration is the same noise process: the in-sample winner's
    # out-of-sample rank is uniform, so E[PBO] = 1/2 for an even number of configurations.
    # A single draw is very noisy (sd ~ 0.2), so average over 100 seeded grids.
    pbos = [
        probability_of_backtest_overfitting(_matrix(n=1200, k=10, seed=500 + s), n_blocks=12).pbo
        for s in range(100)
    ]
    assert np.mean(pbos) == pytest.approx(0.5, abs=0.1)


def test_dominant_configuration_gives_low_pbo():
    frame = _matrix(n=1200, k=10, seed=1)
    frame["cfg3"] += 0.002  # daily Sharpe ~0.2: unmistakably better than the rest
    res = probability_of_backtest_overfitting(frame, n_blocks=12)
    assert res.pbo < 0.05
    assert res.selection_counts().idxmax() == "cfg3"
    assert res.prob_oos_loss < 0.05


def test_shared_edge_leaves_pbo_near_one_half():
    # The documented caveat: when every configuration has the same edge, choosing among
    # them is uninformative, so PBO sits near 0.5 although every OOS Sharpe is positive.
    pbos, losses = [], []
    for s in range(30):
        frame = _matrix(n=1200, k=10, seed=900 + s) + 0.001
        res = probability_of_backtest_overfitting(frame, n_blocks=12)
        pbos.append(res.pbo)
        losses.append(res.prob_oos_loss)
    assert np.mean(pbos) == pytest.approx(0.5, abs=0.15)
    assert np.mean(losses) < 0.02


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #


def test_zero_variance_columns_and_all_zero_blocks_are_finite_and_match_naive():
    frame = _matrix(n=240, k=6, seed=11)
    frame["cfg0"] = 0.0  # never invested
    frame["cfg1"] = 0.0004  # constant return: zero variance everywhere
    frame.iloc[:90, 2] = 0.0  # flat for the first three blocks
    frame.iloc[120:150, :] = 0.0  # one block in which nothing moves at all
    with np.errstate(all="raise"):  # no division by zero, no invalid operation
        res = probability_of_backtest_overfitting(frame, n_blocks=8)
    for arr in (res.logits, res.is_sharpes, res.oos_sharpes, res.oos_relative_ranks):
        assert np.isfinite(arr).all()
    assert np.isfinite(res.pbo) and np.isfinite(res.degradation_slope)
    ref = _naive_pbo(frame, 8)
    np.testing.assert_array_equal(res.selected, ref["selected"])
    np.testing.assert_allclose(res.logits, ref["logits"], atol=1e-12)
    np.testing.assert_allclose(res.oos_sharpes, ref["oos"], rtol=1e-9, atol=1e-12)


def test_all_flat_grid_is_defined():
    frame = pd.DataFrame(np.zeros((200, 4)))
    with np.errstate(all="raise"):
        res = probability_of_backtest_overfitting(frame, n_blocks=10)
    assert (res.selected == 0).all()  # every Sharpe ties at 0: lowest column wins
    assert (res.oos_relative_ranks == 0.5).all() and (res.logits == 0).all()
    assert res.pbo == 1.0 and res.prob_oos_loss == 0.0
    assert res.degradation_slope is None and res.degradation_intercept is None


def test_sub_samples_flat_away_from_the_column_mean_have_exactly_zero_sharpe():
    # A step column is constant within each half but not overall: its variance on a half is pure
    # round-off of ``sum of squares - n * mean^2`` (about 3e-10 in standard deviation for these
    # levels, well above the 1e-12 floor) and must not turn into an astronomical Sharpe ratio.
    n = 160
    frame = pd.DataFrame(
        {
            "flat": np.zeros(n),
            "step": np.r_[np.full(n // 2, -0.0137), np.full(n // 2, 0.0177)],
            "step2": np.r_[np.full(n // 2, 0.0033), np.full(n // 2, -0.0151)],
        }
    )
    res = probability_of_backtest_overfitting(frame, n_blocks=8)
    halves = [i for i, c in enumerate(combinations(range(8), 4)) if c in ((0, 1, 2, 3), (4, 5, 6, 7))]
    assert len(halves) == 2
    for i in halves:
        assert res.is_sharpes[i] == 0.0 and res.oos_sharpes[i] == 0.0
        assert res.selected[i] == 0  # three exact ties at zero: the lowest position wins


def test_ties_are_broken_by_lowest_column_position():
    frame = _matrix(n=480, k=5, seed=4)
    frame["cfg1"] += 0.003
    frame["cfg3"] = frame["cfg1"]  # an exact duplicate of the dominant configuration
    res = probability_of_backtest_overfitting(frame, n_blocks=8)
    assert (res.selected == 1).all()
    assert res.selection_counts()["cfg3"] == 0


def test_result_arrays_are_read_only():
    res = probability_of_backtest_overfitting(_matrix(), n_blocks=8)
    with pytest.raises(ValueError):
        res.logits[0] = 1.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_blocks", [7, 1, 0, -2, 22, 2.5])
def test_invalid_block_counts_raise(n_blocks):
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_matrix(), n_blocks=n_blocks)


def test_invalid_matrices_raise():
    frame = _matrix()
    frame.iloc[5, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        probability_of_backtest_overfitting(frame, n_blocks=8)
    frame.iloc[5, 2] = np.inf
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(frame, n_blocks=8)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((240, 3, 2)), n_blocks=8)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(pd.DataFrame({"a": ["x"] * 240, "b": [0.0] * 240}), n_blocks=8)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_matrix(), n_blocks=8, periods_per_year=0)


def test_too_little_data_returns_none():
    assert probability_of_backtest_overfitting(_matrix(k=1), n_blocks=8) is None  # nothing to choose from
    assert probability_of_backtest_overfitting(_matrix(n=119), n_blocks=12) is None  # blocks under 10 sessions
    assert probability_of_backtest_overfitting(_matrix(n=120), n_blocks=12) is not None
    assert probability_of_backtest_overfitting(_matrix(k=2).to_numpy(), n_blocks=8) is not None
