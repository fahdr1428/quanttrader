"""Statistical inference: Sharpe standard errors, HAC statistics, alpha/beta, MinTRL and tail risk.

All data is synthetic and seeded. Where practical each formula is checked against
an independent brute-force (pure-Python loop) computation.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from quantlab.inference import (
    alpha_beta,
    conditional_value_at_risk,
    cornish_fisher_z,
    excess_kurtosis,
    hac_mean_tstat,
    min_track_record_length,
    newey_west_lags,
    newey_west_variance_of_mean,
    omega_ratio,
    sharpe_inference,
    skewness,
    tail_ratio,
    value_at_risk,
)
from quantlab.metrics import daily_risk_free, sharpe_ratio
from quantlab.research import probabilistic_sharpe_ratio

PPY = 252
ND = NormalDist()


def _series(values, start: str = "2020-01-02") -> pd.Series:
    values = np.asarray(values, dtype="float64")
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def _brute_moments(values) -> tuple[float, float, float, float]:
    """Loop implementation: mean, sample sd (ddof=1), population skewness g3 and raw kurtosis g4."""
    xs = [float(v) for v in values]
    n = len(xs)
    mean = sum(xs) / n
    m2 = sum((v - mean) ** 2 for v in xs) / n
    m3 = sum((v - mean) ** 3 for v in xs) / n
    m4 = sum((v - mean) ** 4 for v in xs) / n
    return mean, math.sqrt(m2 * n / (n - 1)), m3 / m2**1.5, m4 / m2**2


def _ar1(n: int, phi: float, seed: int, sigma: float = 0.01) -> np.ndarray:
    eps = np.random.default_rng(seed).normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0] / math.sqrt(1 - phi**2)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out + 0.0004


# --------------------------------------------------------------------------- #
# Sharpe-ratio inference
# --------------------------------------------------------------------------- #

_T, _SIMS = 1000, 2000


def _standardised_shocks(dist: str, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Mean-zero, unit-variance draws."""
    if dist == "normal":
        return rng.standard_normal(shape)
    if dist == "student_t":  # df = 6: excess kurtosis 3
        return rng.standard_t(6, shape) / math.sqrt(6 / 4)
    return 1.0 - rng.exponential(1.0, shape)  # skewness -2, excess kurtosis 6


@pytest.mark.parametrize(("dist", "seed"), [("normal", 101), ("student_t", 202), ("skewed", 303)])
def test_sharpe_standard_error_matches_monte_carlo(dist, seed):
    rng = np.random.default_rng(seed)
    sr_true = 0.15  # per session, about 2.4 annualised
    paths = 0.01 * (sr_true + _standardised_shocks(dist, rng, (_SIMS, _T)))
    empirical_sd = float((paths.mean(axis=1) / paths.std(axis=1, ddof=1)).std(ddof=1))
    analytic = float(np.mean([sharpe_inference(p).se_per_period for p in paths]))
    assert analytic == pytest.approx(empirical_sd, rel=0.07)
    if dist == "skewed":
        # The normal-iid formula (Lo, 2002) ignores the skew and is visibly too small here.
        naive = math.sqrt((1 + sr_true**2 / 2) / (_T - 1))
        assert naive < empirical_sd * 0.93


def test_sharpe_inference_matches_headline_sharpe_with_risk_free():
    r = _series(np.random.default_rng(7).normal(0.0006, 0.012, 1500))
    res = sharpe_inference(r, risk_free_rate=0.04, periods_per_year=PPY)
    assert res is not None and res.n == 1500
    assert res.sharpe_annual == pytest.approx(sharpe_ratio(r, 0.04, PPY), abs=1e-12)
    assert (res.ci_lower + res.ci_upper) / 2 == pytest.approx(res.sharpe_annual, abs=1e-12)
    assert res.ci_upper - res.ci_lower == pytest.approx(2 * ND.inv_cdf(0.975) * res.se_annual, rel=1e-12)
    assert res.se_annual == pytest.approx(res.se_per_period * math.sqrt(PPY), rel=1e-12)
    assert res.sharpe_annual == pytest.approx(res.sharpe_per_period * math.sqrt(PPY), rel=1e-12)
    assert sharpe_inference(r).sharpe_annual > res.sharpe_annual  # excess returns are used

    excess = (r - daily_risk_free(0.04, PPY)).to_numpy()
    mean, sd, g3, g4 = _brute_moments(excess)
    sr = mean / sd
    se = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1 + sr**2 / 2) / (len(excess) - 1))
    assert res.sharpe_per_period == pytest.approx(sr, rel=1e-10)
    assert res.se_per_period == pytest.approx(se, rel=1e-10)
    assert res.skewness == pytest.approx(g3, rel=1e-8, abs=1e-12)
    assert res.kurtosis == pytest.approx(g4, rel=1e-10)
    # Same denominator as the PSR: Phi(SR / SE) is the PSR against zero.
    psr = probabilistic_sharpe_ratio(pd.Series(excess), 0.0)
    assert ND.cdf(res.sharpe_per_period / res.se_per_period) == pytest.approx(psr, abs=1e-12)

    wider = sharpe_inference(r, 0.04, PPY, confidence=0.99)
    assert wider.ci_lower < res.ci_lower and wider.ci_upper > res.ci_upper


# --------------------------------------------------------------------------- #
# Newey-West
# --------------------------------------------------------------------------- #


def _brute_force_newey_west(values, lags: int) -> float:
    xs = [float(v) for v in values]
    t = len(xs)
    mean = sum(xs) / t
    d = [v - mean for v in xs]
    lrv = sum(v * v for v in d) / t
    for lag in range(1, lags + 1):
        gamma = sum(d[i] * d[i - lag] for i in range(lag, t)) / t
        lrv += 2 * (1 - lag / (lags + 1)) * gamma
    return lrv / t


def test_newey_west_lag_zero_is_iid_variance_of_mean():
    x = np.random.default_rng(3).normal(0.0005, 0.01, 400)
    iid_var = np.var(x, ddof=0) / len(x)  # ddof=0 convention
    assert newey_west_variance_of_mean(x, lags=0) == pytest.approx(iid_var, rel=1e-12)
    assert hac_mean_tstat(x, lags=0) == pytest.approx(x.mean() / math.sqrt(iid_var), rel=1e-12)


def test_newey_west_matches_brute_force_and_default_lags():
    x = _ar1(500, 0.3, seed=5)
    for lags in (1, 5, 12):
        assert newey_west_variance_of_mean(x, lags) == pytest.approx(_brute_force_newey_west(x, lags), rel=1e-10)
    assert [newey_west_lags(n) for n in (100, 500, 1000, 2520)] == [4, 5, 6, 8]
    assert newey_west_variance_of_mean(x) == newey_west_variance_of_mean(x, lags=5)
    short = x[:5]
    assert newey_west_variance_of_mean(short, lags=50) == pytest.approx(_brute_force_newey_west(short, 4), rel=1e-10)


def test_hac_standard_error_exceeds_iid_on_autocorrelated_data():
    x = _ar1(5000, 0.3, seed=11)
    hac, iid = newey_west_variance_of_mean(x), newey_west_variance_of_mean(x, lags=0)
    assert math.sqrt(hac) > math.sqrt(iid)
    # The population long-run / iid variance ratio is (1 + phi) / (1 - phi) = 1.86.
    assert 1.4 < hac / iid < 2.2
    assert abs(hac_mean_tstat(x)) < abs(hac_mean_tstat(x, lags=0))
    white = np.random.default_rng(12).normal(0.0, 0.01, 5000)
    assert newey_west_variance_of_mean(white) / newey_west_variance_of_mean(white, lags=0) == pytest.approx(1, abs=0.15)


# --------------------------------------------------------------------------- #
# Alpha / beta
# --------------------------------------------------------------------------- #


def _planted(n: int = 2520, seed: int = 1000, rf: float = 0.02, alpha: float = 0.05, beta: float = 0.6):
    """Ten years of daily returns; residual sd 0.2%/day puts the alpha SE near 1%/yr."""
    rng = np.random.default_rng(seed)
    rf_d = daily_risk_free(rf, PPY)
    bench_excess = rng.normal(0.0003, 0.01, n)
    strat_excess = alpha / PPY + beta * bench_excess + rng.normal(0.0, 0.002, n)
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(strat_excess + rf_d, index=idx), pd.Series(bench_excess + rf_d, index=idx)


def _brute_force_hac_ols(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    u = y - design @ coef
    meat = np.zeros((2, 2))
    for t in range(len(y)):
        meat += u[t] ** 2 * np.outer(design[t], design[t])
    for lag in range(1, lags + 1):
        for t in range(lag, len(y)):
            g = u[t] * u[t - lag] * np.outer(design[t], design[t - lag])
            meat += (1 - lag / (lags + 1)) * (g + g.T)
    bread = np.linalg.inv(design.T @ design)
    return coef, bread @ meat @ bread


def test_alpha_beta_recovers_planted_alpha_and_beta():
    strat, bench = _planted()
    res = alpha_beta(strat, bench, risk_free_rate=0.02, periods_per_year=PPY)
    assert res is not None and res.n == 2520 and res.lags == newey_west_lags(2520)
    assert abs(res.alpha_annual - 0.05) < 2 * res.alpha_se_annual
    assert abs(res.beta - 0.6) < 2 * res.beta_se
    assert res.alpha_se_annual < 0.015 and res.alpha_tstat > 2  # the planted alpha is detected, not just "not rejected"
    assert res.alpha_annual == pytest.approx(res.alpha_per_period * PPY, rel=1e-12)
    assert res.alpha_tstat == pytest.approx(res.alpha_annual / res.alpha_se_annual, rel=1e-12)
    assert res.beta_tstat == pytest.approx(res.beta / res.beta_se, rel=1e-12)
    assert res.beta_tstat > 20 and 0 < res.r_squared < 1


@pytest.mark.parametrize("lags", [0, 4])
def test_alpha_beta_matches_brute_force_sandwich(lags):
    strat, bench = _planted(n=300, seed=8)
    res = alpha_beta(strat, bench, risk_free_rate=0.03, periods_per_year=PPY, lags=lags)
    rf_d = daily_risk_free(0.03, PPY)
    y, x = strat.to_numpy() - rf_d, bench.to_numpy() - rf_d
    coef, cov = _brute_force_hac_ols(y, x, lags)  # lags=0 is White's HC0
    assert res.lags == lags
    assert res.alpha_per_period == pytest.approx(coef[0], rel=1e-9)
    assert res.beta == pytest.approx(coef[1], rel=1e-10)
    assert res.beta == pytest.approx(np.cov(y, x)[0, 1] / np.var(x, ddof=1), rel=1e-10)
    assert res.alpha_se_annual == pytest.approx(math.sqrt(cov[0, 0]) * PPY, rel=1e-9)
    assert res.beta_se == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-9)
    assert res.r_squared == pytest.approx(np.corrcoef(y, x)[0, 1] ** 2, rel=1e-10)


def test_alpha_beta_of_benchmark_against_itself():
    _, bench = _planted()
    res = alpha_beta(bench, bench, risk_free_rate=0.02)
    assert res.beta == pytest.approx(1.0, abs=1e-12)
    assert res.alpha_annual == pytest.approx(0.0, abs=1e-12)
    assert res.r_squared == pytest.approx(1.0, abs=1e-12)
    assert res.alpha_tstat is None and res.beta_tstat is None  # zero residuals: t undefined


def test_alpha_beta_alignment_and_degenerate_cases():
    strat, bench = _planted(n=600, seed=9)
    dirty = pd.concat([strat, pd.Series([0.5], index=[pd.Timestamp("2040-01-02")])])  # date not in benchmark
    dirty.iloc[10] = np.nan
    ref = alpha_beta(strat.drop(strat.index[10]), bench.drop(bench.index[10]))
    res = alpha_beta(dirty, bench)
    assert res.n == 599 and res == ref
    assert alpha_beta(strat.iloc[:29], bench.iloc[:29]) is None
    assert alpha_beta(strat.iloc[:30], bench.iloc[:30]) is not None
    assert alpha_beta(strat, pd.Series(0.001, index=bench.index)) is None  # zero benchmark variance
    flat = alpha_beta(pd.Series(0.0, index=bench.index), bench, risk_free_rate=0.02)
    assert flat.beta == pytest.approx(0.0, abs=1e-12)
    assert flat.alpha_annual == pytest.approx(-daily_risk_free(0.02, PPY) * PPY, rel=1e-9)
    assert flat.r_squared is None and flat.alpha_tstat is None


# --------------------------------------------------------------------------- #
# Minimum track record length
# --------------------------------------------------------------------------- #


def _unit_noise(n: int = 1000, seed: int = 31) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal(n)
    return (z - z.mean()) / z.std(ddof=1)


def test_min_trl_decreases_as_sharpe_rises():
    noise = _unit_noise()
    results = [min_track_record_length(mu + 0.01 * noise) for mu in (0.0003, 0.0006, 0.001, 0.002)]
    sessions = [r.sessions for r in results]
    sharpes = [r.sharpe_annual for r in results]
    assert all(a > b for a, b in zip(sessions, sessions[1:]))
    assert all(a < b for a, b in zip(sharpes, sharpes[1:]))
    assert not results[0].sufficient and results[-1].sufficient
    for r in results:
        assert r.years == pytest.approx(r.sessions / PPY, rel=1e-12) and r.n == 1000


def test_min_trl_matches_formula_and_psr_identity():
    rng = np.random.default_rng(32)
    r = 0.0012 + 0.01 * (1.0 - rng.exponential(1.0, 1500))  # negatively skewed
    res = min_track_record_length(r, risk_free_rate=0.02, target_sharpe_annual=0.5, confidence=0.9)
    mean, sd, g3, g4 = _brute_moments(r - daily_risk_free(0.02, PPY))
    sr, sr_star = mean / sd, 0.5 / math.sqrt(PPY)
    term = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    expected = 1 + term * (ND.inv_cdf(0.9) / (sr - sr_star)) ** 2
    assert res.sessions == pytest.approx(expected, rel=1e-9)
    # At T = MinTRL the PSR against SR* equals the requested confidence.
    assert ND.cdf((sr - sr_star) * math.sqrt(res.sessions - 1) / math.sqrt(term)) == pytest.approx(0.9, abs=1e-12)
    assert res.sessions > min_track_record_length(r, 0.02, PPY, 0.0, 0.9).sessions
    assert res.sessions < min_track_record_length(r, 0.02, PPY, 0.5, 0.95).sessions


def test_min_trl_none_when_sharpe_at_or_below_target():
    noise = _unit_noise()
    assert min_track_record_length(-0.0005 + 0.01 * noise) is None
    good = 0.001 + 0.01 * noise
    observed = sharpe_inference(good).sharpe_annual
    assert min_track_record_length(good, target_sharpe_annual=observed + 0.01) is None
    assert min_track_record_length(good, target_sharpe_annual=observed - 0.01) is not None


# --------------------------------------------------------------------------- #
# VaR, CVaR and distribution statistics
# --------------------------------------------------------------------------- #

# Sorted: -0.05 -0.03 -0.02 -0.015 -0.01 -0.008 -0.005 -0.002 0 0.001 0.003 0.005 0.007 0.01 0.012 0.015 0.018
#         0.02 0.025 0.03
_TWENTY = np.array([
    0.012, -0.03, 0.005, 0.02, -0.008, 0.0, 0.03, -0.015, 0.001, -0.05,
    0.018, -0.002, 0.007, -0.01, 0.025, 0.003, -0.02, 0.015, -0.005, 0.01,
])


@pytest.mark.parametrize(
    ("level", "var", "cvar"),
    [
        (0.95, -0.05 + 0.95 * 0.02, -0.05),  # position 0.05 * 19 = 0.95 between x(0) and x(1)
        (0.90, -0.03 + 0.90 * 0.01, (-0.05 - 0.03) / 2),  # position 1.9
        (0.80, -0.015 + 0.80 * 0.005, (-0.05 - 0.03 - 0.02 - 0.015) / 4),  # position 3.8
    ],
)
def test_historical_var_and_cvar_hand_computed(level, var, cvar):
    assert value_at_risk(_TWENTY, level) == pytest.approx(var, abs=1e-15)
    assert conditional_value_at_risk(_series(_TWENTY), level) == pytest.approx(cvar, abs=1e-15)
    assert conditional_value_at_risk(_TWENTY, level) <= value_at_risk(_TWENTY, level) < 0


def test_normal_and_cornish_fisher_var():
    mean, sd, g3, g4 = _brute_moments(_TWENTY)
    z = ND.inv_cdf(0.05)
    assert value_at_risk(_TWENTY, 0.95, "normal") == pytest.approx(mean + z * sd, rel=1e-12)
    z_cf = z + (z**2 - 1) * g3 / 6 + (z**3 - 3 * z) * (g4 - 3) / 24 - (2 * z**3 - 5 * z) * g3**2 / 36
    assert value_at_risk(_TWENTY, 0.95, "cornish_fisher") == pytest.approx(mean + z_cf * sd, rel=1e-10)
    assert cornish_fisher_z(z, 0.0, 0.0) == z
    # Negative skew fattens the left tail: Cornish-Fisher VaR is the larger loss.
    skewed = 0.0005 + 0.01 * (1.0 - np.random.default_rng(4).exponential(1.0, 2000))
    assert value_at_risk(skewed, method="cornish_fisher") < value_at_risk(skewed, method="normal")


def test_cornish_fisher_equals_normal_without_skew_or_excess_kurtosis():
    # {-a, 0, 0, 0, 0, a}: symmetric (skew 0) with kurtosis n / (2k) = 6 / 2 = 3 exactly.
    sample = 0.0004 + np.tile([-0.02, 0.0, 0.0, 0.0, 0.0, 0.02], 50)
    assert skewness(sample) == pytest.approx(0.0, abs=1e-12)
    assert excess_kurtosis(sample) == pytest.approx(0.0, abs=1e-12)
    for level in (0.9, 0.95, 0.99):
        assert value_at_risk(sample, level, "cornish_fisher") == pytest.approx(
            value_at_risk(sample, level, "normal"), rel=1e-9
        )


def test_skewness_and_kurtosis_conventions():
    x = np.array([0.01, 0.02, 0.03, 0.04, 0.10, -0.03, 0.005, -0.012, 0.0])
    n = len(x)
    _, _, g3, g4 = _brute_moments(x)
    assert skewness(x) == pytest.approx(g3, rel=1e-10)
    assert excess_kurtosis(x) == pytest.approx(g4 - 3, rel=1e-10)
    # pandas reports bias-corrected G1 / G2; convert back to the population g1 / g2.
    s = pd.Series(x)
    assert skewness(x) == pytest.approx(s.skew() * (n - 2) / math.sqrt(n * (n - 1)), rel=1e-10)
    assert excess_kurtosis(x) == pytest.approx((s.kurt() * (n - 2) * (n - 3) / (n - 1) - 6) / (n + 1), rel=1e-10)
    two_point = np.tile([-0.01, 0.01], 100)
    assert skewness(two_point) == pytest.approx(0.0, abs=1e-12)
    assert excess_kurtosis(two_point) == pytest.approx(-2.0, abs=1e-12)


def test_omega_and_tail_ratio_hand_computed():
    assert omega_ratio(_TWENTY) == pytest.approx(0.146 / 0.140, rel=1e-12)
    assert omega_ratio(_TWENTY, threshold=0.005) == pytest.approx(0.097 / 0.191, rel=1e-12)
    # Identity: Omega(tau) = 1 + n (mean - tau) / sum(max(tau - r, 0)).
    assert omega_ratio(_TWENTY, 0.005) == pytest.approx(1 + 20 * (_TWENTY.mean() - 0.005) / 0.191, rel=1e-12)
    upper = 0.025 + 0.05 * (0.03 - 0.025)  # position 0.95 * 19 = 18.05
    assert tail_ratio(_TWENTY) == pytest.approx(upper / 0.031, rel=1e-12)
    assert omega_ratio(np.abs(_TWENTY) + 0.001) is None  # no return below the threshold


# --------------------------------------------------------------------------- #
# Degenerate inputs and argument validation
# --------------------------------------------------------------------------- #

_DEGENERATE = {
    "empty": pd.Series([], dtype="float64"),
    "one": pd.Series([0.01]),
    "two_list": [0.01, -0.02],
    "all_nan": pd.Series([np.nan] * 50),
    "two_after_nan": pd.Series([np.nan, 0.01, np.inf, -0.02, np.nan]),
    "constant": _series(np.full(100, 0.001)),
}


@pytest.mark.parametrize("name", list(_DEGENERATE))
def test_degenerate_inputs_return_none(name):
    r = _DEGENERATE[name]
    assert sharpe_inference(r) is None
    assert sharpe_inference(r, risk_free_rate=0.04) is None
    assert min_track_record_length(r) is None
    assert newey_west_variance_of_mean(r) is None
    assert hac_mean_tstat(r) is None
    assert alpha_beta(r, r) is None
    assert skewness(r) is None
    assert excess_kurtosis(r) is None
    assert value_at_risk(r, method="normal") is None
    assert value_at_risk(r, method="cornish_fisher") is None
    assert tail_ratio(r) is None
    assert omega_ratio(r) is None
    if name == "constant":  # the empirical quantile of a constant series is that constant
        assert value_at_risk(r) == pytest.approx(0.001)
        assert conditional_value_at_risk(r) == pytest.approx(0.001)
    else:
        assert value_at_risk(r) is None
        assert conditional_value_at_risk(r) is None


def test_nans_are_dropped_not_propagated():
    clean = _series(np.random.default_rng(41).normal(0.0005, 0.01, 300))
    dirty = clean.copy()
    dirty.iloc[[3, 50, 200]] = np.nan
    kept = clean.drop(clean.index[[3, 50, 200]])
    assert sharpe_inference(dirty, 0.03) == sharpe_inference(kept, 0.03)
    assert min_track_record_length(dirty) == min_track_record_length(kept)
    assert newey_west_variance_of_mean(dirty) == newey_west_variance_of_mean(kept)
    assert value_at_risk(dirty, method="cornish_fisher") == value_at_risk(kept, method="cornish_fisher")
    assert conditional_value_at_risk(dirty) == conditional_value_at_risk(kept)
    assert tail_ratio(dirty) == tail_ratio(kept) and omega_ratio(dirty) == omega_ratio(kept)


def test_invalid_arguments_raise():
    r = np.random.default_rng(1).normal(0.0, 0.01, 100)
    with pytest.raises(ValueError):
        sharpe_inference(r, confidence=1.0)
    with pytest.raises(ValueError):
        min_track_record_length(r, confidence=0.0)
    with pytest.raises(ValueError):
        value_at_risk(r, level=1.5)
    with pytest.raises(ValueError):
        value_at_risk(r, method="monte_carlo")
    with pytest.raises(ValueError):
        newey_west_variance_of_mean(r, lags=-1)
    with pytest.raises(ValueError):
        tail_ratio(r, q=1.0)


def test_psr_holds_its_size_for_a_rarely_invested_strategy():
    """Regression: with 1% of sessions invested, sample skew is large and its errors move with those of SR,
    so the skew adjustment shrank the standard error and PSR > 95% occurred about 10% of the time under a
    true Sharpe of zero (nominal 5%). The adjustment may now widen the interval but never narrow it."""
    rng = np.random.default_rng(11)
    reps, n = 3000, 1139
    hits = 0
    for _ in range(reps):
        x = np.where(rng.random(n) < 0.01, rng.normal(0.0, 0.012, n), 0.0)
        psr = probabilistic_sharpe_ratio(pd.Series(x))
        hits += psr is not None and psr > 0.95
    assert hits / reps < 0.07, hits / reps


def test_skew_adjustment_never_narrows_the_interval():
    from quantlab.inference import sharpe_variance_term

    for sr, g3, g4 in [(0.05, 10.0, 190.0), (0.2, 3.0, 3.0), (0.1, -2.0, 9.0), (0.0, 0.0, 3.0)]:
        assert sharpe_variance_term(sr, g3, g4) >= 1 + sr**2 / 2
    assert sharpe_variance_term(0.1, -2.0, 9.0) == pytest.approx(1 + 0.2 + 2 * 0.01)  # widening is kept
