"""Leakage-safe machine learning for return-direction forecasting.

The model answers one question at each close: *what is the probability that the
asset is higher in ``horizon`` sessions than it is now?* Getting this right
without fooling yourself requires three disciplines, all enforced here:

1. **Causal features.** Every feature at session ``t`` uses closes up to ``t``.
2. **Purged labels.** The label for session ``s`` (did price rise over the next
   ``horizon`` sessions?) only becomes known at ``s + horizon``. A model refit
   at session ``r`` therefore trains only on samples ``s <= r - horizon``.
   Training on later samples would leak the future into the past.
3. **Walk-forward fitting.** The model is refit every ``retrain_every`` sessions
   on a rolling window and then frozen until the next refit. Feature scaling
   (mean and standard deviation) is estimated on the training window only.

Predictions made this way are genuinely out-of-sample, so their accuracy can be
scored honestly with :func:`classification_report`.

The classifier is an L2-regularised logistic regression solved by Newton's
method (iteratively reweighted least squares) in NumPy — small enough to read
and verify, with no extra dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab import indicators as ind

FEATURE_DESCRIPTIONS = {
    "ret_1d": "Return over the last session",
    "ret_5d": "Return over the last 5 sessions (one week)",
    "ret_21d": "Return over the last 21 sessions (one month)",
    "ret_63d": "Return over the last 63 sessions (one quarter)",
    "ret_126d": "Return over the last 126 sessions (six months)",
    "ret_252d": "Return over the last 252 sessions (one year)",
    "vol_21d": "Annualised volatility of daily returns, last 21 sessions",
    "vol_ratio": "Short-term (21) / long-term (126) volatility — rising stress when above 1",
    "dist_sma50": "Distance of the close above its 50-session average",
    "dist_sma200": "Distance of the close above its 200-session average",
    "rsi_14": "Wilder RSI(14), centred (RSI/100 - 0.5)",
    "drawdown_252": "Distance below the highest close of the last 252 sessions",
}
FEATURE_SHORT = {
    "ret_1d": "1-day return",
    "ret_5d": "1-week return",
    "ret_21d": "1-month return",
    "ret_63d": "3-month return",
    "ret_126d": "6-month return",
    "ret_252d": "12-month return",
    "vol_21d": "1-month volatility",
    "vol_ratio": "Volatility ratio 21/126",
    "dist_sma50": "Distance from 50-day MA",
    "dist_sma200": "Distance from 200-day MA",
    "rsi_14": "RSI(14)",
    "drawdown_252": "Drawdown from 1-year high",
}
FEATURE_NAMES = tuple(FEATURE_DESCRIPTIONS)
FEATURE_WARMUP = 252  # longest look-back among the features


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Causal feature matrix (one row per session; NaN during warm-up)."""
    close = prices["close"]
    ret = close.pct_change()
    vol21 = ret.rolling(21, min_periods=21).std()
    vol126 = ret.rolling(126, min_periods=126).std()
    feats = {
        "ret_1d": ret,
        "ret_5d": close / close.shift(5) - 1,
        "ret_21d": close / close.shift(21) - 1,
        "ret_63d": close / close.shift(63) - 1,
        "ret_126d": close / close.shift(126) - 1,
        "ret_252d": close / close.shift(252) - 1,
        "vol_21d": vol21 * math.sqrt(252),
        "vol_ratio": vol21 / vol126.where(vol126 > 0),
        "dist_sma50": close / ind.sma(close, 50) - 1,
        "dist_sma200": close / ind.sma(close, 200) - 1,
        "rsi_14": ind.rsi_wilder(close, 14) / 100 - 0.5,
        "drawdown_252": close / close.rolling(252, min_periods=252).max() - 1,
    }
    return pd.DataFrame(feats, index=prices.index)[list(FEATURE_NAMES)]


def forward_labels(close: pd.Series, horizon: int) -> pd.Series:
    """1.0 if the close ``horizon`` sessions ahead is higher, else 0.0; NaN where unknown.

    These labels look into the future by definition. They are only ever used
    (a) for training with a ``horizon``-session purge and (b) for scoring
    predictions after the fact — never as an input to a decision.
    """
    future = close.shift(-horizon)
    label = (future / close - 1 > 0).astype("float64")
    return label.where(future.notna())


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 0.1, max_iter: int = 50,
                 tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by Newton's method.

    Minimises ``mean(log loss) + ridge / 2 * ||w||^2`` (the intercept is not
    penalised). Returns ``(weights, intercept)``.
    """
    n, k = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(k + 1)
    beta[0] = math.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    penalty = np.full(k + 1, ridge)
    penalty[0] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(A @ beta)
        grad = A.T @ (p - y) / n + penalty * beta
        w = p * (1 - p)
        hess = (A * w[:, None]).T @ A / n + np.diag(penalty + 1e-10)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta[1:], float(beta[0])


def predict_proba(X: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return _sigmoid(X @ weights + intercept)


# --------------------------------------------------------------------------- #
# Walk-forward prediction
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardFit:
    probabilities: pd.Series
    fits: pd.DataFrame  # one row per refit: dates, sample size, coefficients


def walk_forward_probabilities(
    prices: pd.DataFrame,
    horizon: int = 21,
    train_window: int = 756,
    retrain_every: int = 21,
    ridge: float = 0.1,
    min_train: int = 252,
) -> WalkForwardFit:
    """Out-of-sample P(up over ``horizon``) for every session where a model exists.

    The refit schedule is anchored to the first session with complete features,
    so truncating the data later never changes earlier predictions.
    """
    features = build_features(prices)
    labels = forward_labels(prices["close"], horizon)
    X_all = features.to_numpy(dtype="float64")
    y_all = labels.to_numpy(dtype="float64")
    n = len(prices)
    valid = np.isfinite(X_all).all(axis=1)
    probs = np.full(n, np.nan)
    fit_rows: list[dict] = []
    if not valid.any():
        return WalkForwardFit(pd.Series(probs, index=prices.index, name="prob_up"),
                              pd.DataFrame(fit_rows))
    first = int(np.argmax(valid))
    start = first + min_train + horizon - 1
    for r in range(start, n, retrain_every):
        hi = r - horizon  # last sample whose label is known at the close of session r
        lo = max(first, hi - train_window + 1)
        idx = np.arange(lo, hi + 1)
        idx = idx[valid[idx] & np.isfinite(y_all[idx])]
        if len(idx) < min_train or len(np.unique(y_all[idx])) < 2:
            continue
        X, y = X_all[idx], y_all[idx]
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        weights, intercept = fit_logistic((X - mu) / sd, y, ridge=ridge)
        stop = min(r + retrain_every, n)
        seg = np.arange(r, stop)
        seg = seg[valid[seg]]
        probs[seg] = predict_proba((X_all[seg] - mu) / sd, weights, intercept)
        in_sample = predict_proba((X - mu) / sd, weights, intercept)
        fit_rows.append(
            {
                "fit_date": prices.index[r],
                "train_start": prices.index[idx[0]],
                "train_end": prices.index[idx[-1]],
                "n_train": len(idx),
                "base_rate": float(y.mean()),
                "in_sample_accuracy": float(((in_sample > 0.5) == (y > 0.5)).mean()),
                "intercept": intercept,
                **{f"w_{name}": float(w) for name, w in zip(FEATURE_NAMES, weights)},
            }
        )
    return WalkForwardFit(pd.Series(probs, index=prices.index, name="prob_up"), pd.DataFrame(fit_rows))


# --------------------------------------------------------------------------- #
# Honest scoring
# --------------------------------------------------------------------------- #


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve via the Mann-Whitney rank statistic (ties averaged)."""
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_report(prob: pd.Series, close: pd.Series, horizon: int) -> dict[str, float | int | None]:
    """Score out-of-sample probabilities against the outcomes that later occurred.

    Overlapping ``horizon``-session labels are strongly autocorrelated, so the
    *effective* number of independent observations is roughly ``n / horizon``;
    the report includes it so accuracy differences are not over-interpreted.
    """
    labels = forward_labels(close, horizon)
    mask = prob.notna() & labels.notna()
    p = prob[mask].to_numpy()
    y = labels[mask].to_numpy()
    n = len(p)
    if n == 0:
        return {"n": 0}
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    base = float(y.mean())
    accuracy = float(((p > 0.5) == (y > 0.5)).mean())
    return {
        "n": n,
        "effective_n": max(1, n // horizon),
        "accuracy": accuracy,
        "base_rate": base,
        "always_up_accuracy": base,
        "accuracy_edge": accuracy - max(base, 1 - base),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_climatology": float(np.mean((base - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))),
        "auc": roc_auc(p, y),
        "mean_probability": float(p.mean()),
    }
