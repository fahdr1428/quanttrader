"""Walk-forward evaluation: chronological separation and leakage, plus PSR/DSR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.config import BacktestConfig
from quantlab.research import (
    WalkForwardConfig,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    make_folds,
    probabilistic_sharpe_ratio,
    walk_forward,
)
from tests.helpers import random_walk

GRID = {"short_window": [10, 20], "long_window": [50, 100]}


def test_folds_are_chronological_and_non_overlapping():
    sessions = pd.bdate_range("2015-01-01", periods=1000)
    folds = make_folds(sessions, WalkForwardConfig(train_sessions=300, test_sessions=100))
    assert len(folds) == 7
    for fold in folds:
        assert fold.train_start < fold.train_end < fold.test_start <= fold.test_end
        assert sessions.get_loc(fold.test_start) == sessions.get_loc(fold.train_end) + 1
    for prev, nxt in zip(folds, folds[1:]):
        assert prev.test_end < nxt.test_start  # test windows never overlap
        assert sessions.get_loc(nxt.test_start) == sessions.get_loc(prev.test_end) + 1  # contiguous


def test_rolling_versus_anchored_training_windows():
    sessions = pd.bdate_range("2015-01-01", periods=800)
    rolling = make_folds(sessions, WalkForwardConfig(300, 100, anchored=False))
    anchored = make_folds(sessions, WalkForwardConfig(300, 100, anchored=True))
    assert all(f.train_sessions == 300 for f in rolling)
    assert all(f.train_start == sessions[0] for f in anchored)
    assert [f.test_start for f in rolling] == [f.test_start for f in anchored]


def test_short_final_test_window_is_dropped():
    sessions = pd.bdate_range("2015-01-01", periods=410)
    folds = make_folds(sessions, WalkForwardConfig(300, 100, min_test_sessions=20))
    assert len(folds) == 1  # the remaining 10 sessions are too few for a second fold


def test_selection_ignores_data_after_training_end(long_walk):
    cfg = WalkForwardConfig(train_sessions=400, test_sessions=200)
    base = walk_forward(long_walk, "ma_crossover", GRID, BacktestConfig(), cfg)
    fold1 = base.folds[0]
    altered = long_walk.copy()
    after = altered.index > fold1.train_end
    altered.loc[after, ["open", "high", "low", "close"]] *= np.linspace(2.0, 0.3, after.sum())[:, None]
    alt = walk_forward(altered, "ma_crossover", GRID, BacktestConfig(), cfg)
    assert base.fold_table.loc[0, "params"] == alt.fold_table.loc[0, "params"]
    assert base.fold_table.loc[0, "IS objective"] == alt.fold_table.loc[0, "IS objective"]
    pd.testing.assert_frame_equal(
        base.in_sample_trials[base.in_sample_trials["fold"] == 1],
        alt.in_sample_trials[alt.in_sample_trials["fold"] == 1],
    )


def test_out_of_sample_equity_does_not_depend_on_later_prices(long_walk):
    cfg = WalkForwardConfig(train_sessions=400, test_sessions=200)
    base = walk_forward(long_walk, "ma_crossover", GRID, BacktestConfig(), cfg)
    cut = base.folds[1].test_end
    altered = long_walk.copy()
    after = altered.index > cut
    altered.loc[after, ["open", "high", "low", "close"]] *= 1.5
    alt = walk_forward(altered, "ma_crossover", GRID, BacktestConfig(), cfg)
    pd.testing.assert_series_equal(base.oos.equity.loc[:cut], alt.oos.equity.loc[:cut])


def test_out_of_sample_portfolio_is_valid_and_covers_only_test_periods(long_walk):
    result = walk_forward(long_walk, "rsi_reversion", {"window": [7, 14], "entry_threshold": [25, 30]},
                          BacktestConfig(), WalkForwardConfig(400, 150))
    assert result.oos.is_valid, result.oos.reconciliation_issues
    first, last = result.folds[0], result.folds[-1]
    assert result.oos.daily.index[0] == first.train_end  # anchor
    assert result.oos.daily.index[1] == first.test_start
    assert result.oos.daily.index[-1] == last.test_end
    assert result.oos_benchmark.daily.index.equals(result.oos.daily.index)
    assert result.total_backtests == result.n_combinations * len(result.folds)
    assert (result.fold_table["Selection data ends"] < result.fold_table["Test start"]).all()


def test_search_space_is_constrained(walk):
    huge = {"short_window": list(range(2, 40)), "long_window": list(range(41, 60))}
    with pytest.raises(ValueError, match="exceed the limit"):
        walk_forward(walk, "ma_crossover", huge, BacktestConfig(), WalkForwardConfig(200, 100))


def test_insufficient_history_for_a_fold_raises(walk):
    with pytest.raises(ValueError, match="not enough"):
        walk_forward(walk.iloc[:250], "ma_crossover", GRID, BacktestConfig(), WalkForwardConfig(300, 100))


def test_probabilistic_sharpe_behaviour():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.002, 0.01, 1000))
    noise = pd.Series(rng.normal(0.0, 0.01, 1000))
    assert probabilistic_sharpe_ratio(good) > 0.99
    assert 0.05 < probabilistic_sharpe_ratio(noise) < 0.95
    assert probabilistic_sharpe_ratio(pd.Series([0.0] * 100)) is None


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0008, 0.01, 750))
    few = deflated_sharpe_ratio(returns, list(rng.normal(0, 0.02, 3)))
    many = deflated_sharpe_ratio(returns, list(rng.normal(0, 0.02, 200)))
    assert many < few <= probabilistic_sharpe_ratio(returns)
    assert expected_max_sharpe([0.1]) is None  # needs at least two trials
