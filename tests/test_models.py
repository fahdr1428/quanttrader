"""Advanced models: logistic regression, purged walk-forward ML, adaptive ensemble.

Includes two honesty checks: on a pure random walk the ML model must show no
skill (an apparent edge on noise would indicate leakage), and on a market with a
planted momentum effect it must detect it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import ml
from quantlab.strategies import AdaptiveTrendEnsemble, WalkForwardLogistic, _deadband_steps
from tests.helpers import make_prices, random_walk


def momentum_market(n: int, seed: int, drift: float, noise: float = 0.01, lookback: int = 63) -> pd.DataFrame:
    """Synthetic market whose expected return is +drift after a rising quarter, -drift after a falling one."""
    rng = np.random.default_rng(seed)
    logp = np.zeros(n)
    for t in range(1, n):
        mu = 0.0 if t <= lookback else (drift if logp[t - 1] > logp[t - 1 - lookback] else -drift)
        logp[t] = logp[t - 1] + mu + rng.normal(0, noise)
    close = 100 * np.exp(logp)
    opens = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.002, n))
    return make_prices(close, opens, start="2010-01-04")


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #


def test_logistic_regression_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40_000, 3))
    true_w, true_b = np.array([1.5, -0.8, 0.0]), 0.3
    y = (rng.uniform(size=len(X)) < 1 / (1 + np.exp(-(X @ true_w + true_b)))).astype(float)
    w, b = ml.fit_logistic(X, y, ridge=0.0)
    np.testing.assert_allclose(w, true_w, atol=0.05)
    assert b == pytest.approx(true_b, abs=0.05)


def test_ridge_shrinks_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2_000, 4))
    y = (X[:, 0] + rng.normal(size=2_000) > 0).astype(float)
    norms = [np.linalg.norm(ml.fit_logistic(X, y, ridge=r)[0]) for r in (0.0, 0.1, 1.0, 10.0)]
    assert norms == sorted(norms, reverse=True)


def test_forward_labels_and_auc():
    close = pd.Series([100.0, 101, 99, 103, 104])
    labels = ml.forward_labels(close, 2)
    assert labels.tolist()[:3] == [0.0, 1.0, 1.0] and labels.iloc[3:].isna().all()
    assert ml.roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert ml.roc_auc(np.array([0.2, 0.9]), np.array([0, 1])) == 1.0


# --------------------------------------------------------------------------- #
# Purged walk-forward fitting
# --------------------------------------------------------------------------- #


def test_every_refit_trains_only_on_known_labels():
    prices = random_walk(n=1200, seed=4)
    horizon = 21
    fit = ml.walk_forward_probabilities(prices, horizon=horizon, train_window=500, retrain_every=50)
    pos = prices.index.get_loc
    assert len(fit.fits) > 5
    for _, row in fit.fits.iterrows():
        # The newest training sample's outcome (horizon sessions later) is known at the refit close.
        assert pos(row["train_end"]) + horizon <= pos(row["fit_date"])
        assert row["n_train"] <= 500


def test_predictions_do_not_change_when_unknown_outcomes_change():
    """Alter only prices that a refit must not see: those after each prediction's date."""
    prices = random_walk(n=1100, seed=6)
    k = 900
    altered = prices.copy()
    altered.iloc[k:, :4] = altered.iloc[k:, :4].to_numpy() * 1.5
    base = ml.walk_forward_probabilities(prices, horizon=21).probabilities
    alt = ml.walk_forward_probabilities(altered, horizon=21).probabilities
    pd.testing.assert_series_equal(base.iloc[:k], alt.iloc[:k])


def test_no_skill_on_a_random_walk():
    prices = random_walk(n=3000, seed=21, drift=0.0)
    model = WalkForwardLogistic()
    signals = model.generate_signals(prices)
    report = ml.classification_report(signals["prob_up_raw"], prices["close"], 21)
    assert report["n"] > 2000
    assert 0.42 < report["auc"] < 0.58, report


def test_detects_a_planted_momentum_effect():
    prices = momentum_market(3000, seed=1, drift=0.001)
    signals = WalkForwardLogistic().generate_signals(prices)
    report = ml.classification_report(signals["prob_up_raw"], prices["close"], 21)
    assert report["auc"] > 0.58, report


def test_ml_strategy_targets_follow_confidence():
    prices = momentum_market(1600, seed=2, drift=0.001)
    signals = WalkForwardLogistic(full_confidence=0.6).generate_signals(prices)
    valid = signals.dropna(subset=["target"])
    assert set(np.round(valid["target"].unique(), 6)) <= {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
    assert (valid.loc[valid["prob_up"] <= 0.45, "target"] <= 0.2 + 1e-9).all()


# --------------------------------------------------------------------------- #
# Adaptive ensemble
# --------------------------------------------------------------------------- #


def test_deadband_quantiser_ignores_noise_near_boundaries():
    raw = np.array([np.nan, 0.49, 0.51, 0.49, 0.52, 0.71, 0.69, 0.0])
    out = _deadband_steps(raw, 0.2)
    assert np.isnan(out[0])
    assert out[1:5].tolist() == [0.4, 0.4, 0.4, 0.4]  # wobbling around 0.5 does not trade
    assert out[5] == pytest.approx(0.8) and out[6] == pytest.approx(0.8)
    assert out[7] == 0.0


def test_ensemble_goes_to_cash_in_a_persistent_decline():
    closes = 100 * np.exp(np.cumsum(np.full(600, -0.002)))
    signals = AdaptiveTrendEnsemble().generate_signals(make_prices(closes))
    assert (signals["target"].dropna().iloc[-100:] == 0).all()
    assert (signals["trend_score"].dropna().iloc[-100:] == 0).all()


def test_volatility_targeting_cuts_exposure_in_turbulent_regimes():
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0008, 0.006, 700)
    wild = rng.normal(0.0008, 0.030, 300)
    closes = 100 * np.exp(np.cumsum(np.r_[calm, wild]))
    signals = AdaptiveTrendEnsemble(target_vol=0.10, pullback_boost=0.0).generate_signals(make_prices(closes))
    calm_scale = signals["vol_scale"].iloc[500:700].mean()
    wild_scale = signals["vol_scale"].iloc[750:1000].mean()
    assert calm_scale > 0.9 and wild_scale < 0.35
    # Position never exceeds what the volatility budget allows (plus one quantisation step).
    valid = signals.dropna(subset=["target"])
    assert (valid["target"] <= valid["vol_scale"] * (1 + 0.25) + 0.2 + 1e-9).all()
