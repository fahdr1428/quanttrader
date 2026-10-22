"""PORTFOLIO REVIEW on synthetic holdings: weights, concentration, hypothetical risk, betas, failures listed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantlab.data import TickerNotFoundError
from quantlab.holdings import HYPOTHETICAL, Holding, parse_holdings_csv, review_portfolio
from quantlab.inference import value_at_risk
from tests.technical_fixtures import dataset, loader, ohlcv

N = 300
RNG = np.random.default_rng(5)
BENCH_RET = RNG.normal(0.0003, 0.01, N - 1)


def _closes(returns, level):
    return level * np.r_[1.0, np.cumprod(1 + returns)]


BENCH = ohlcv(_closes(BENCH_RET, 400.0))
DOUBLE = ohlcv(_closes(2 * BENCH_RET, 100.0))          # beta 2 by construction
SAME = ohlcv(_closes(BENCH_RET, 50.0))                 # beta 1 by construction


def _book(**extra):
    entries = {"DBL": dataset("DBL", DOUBLE), "ONE": dataset("ONE", SAME), **extra}
    return loader(entries)


def _quantities(weight_dbl=0.25, total=100_000):
    """Quantities that give the wanted weights at the latest closes."""
    return (weight_dbl * total / DOUBLE["close"].iloc[-1], (1 - weight_dbl) * total / SAME["close"].iloc[-1])


def test_weights_and_concentration_come_from_the_latest_closes():
    q_dbl, q_one = _quantities()
    r = review_portfolio([Holding("DBL", q_dbl, sector="Tech"), Holding("ONE", q_one, cost_basis=40.0, sector="Tech")],
                         _book(), BENCH)
    assert r.holdings.loc["DBL", "weight"] == pytest.approx(0.25) and r.concentration["largest"] == "ONE"
    assert r.concentration["hhi"] == pytest.approx(0.25**2 + 0.75**2)
    assert r.concentration["effective_positions"] == pytest.approx(1 / 0.625)
    assert r.sector_exposure.loc["Tech", "weight"] == pytest.approx(1.0)
    assert r.holdings.loc["ONE", "unrealised_pnl"] == pytest.approx((SAME["close"].iloc[-1] - 40.0) * q_one)
    assert np.isnan(r.holdings.loc["DBL", "unrealised_pnl"]) or r.holdings.loc["DBL", "unrealised_pnl"] is None


def test_risk_is_today_weights_on_past_returns_and_labelled_hypothetical():
    q_dbl, q_one = _quantities()
    r = review_portfolio([Holding("DBL", q_dbl), Holding("ONE", q_one)], _book(), BENCH, lookback=252)
    port = 0.25 * 2 * BENCH_RET[-252:] + 0.75 * BENCH_RET[-252:]
    assert r.risk["sessions"] == 252 and r.risk["label"] == HYPOTHETICAL and HYPOTHETICAL in r.warnings
    assert r.risk["var_95_1d"] == pytest.approx(value_at_risk(port, 0.95))
    assert r.risk["cvar_95_1d"] <= r.risk["var_95_1d"] < 0
    assert r.risk["annualised_volatility"] == pytest.approx(port.std(ddof=1) * np.sqrt(252))
    assert r.holdings.loc["DBL", "beta"] == pytest.approx(2.0) and r.holdings.loc["ONE", "beta"] == pytest.approx(1.0)
    assert r.risk["beta"] == pytest.approx(1.25)
    shock = r.scenarios.loc["Benchmark -10% (beta-implied)"]
    assert shock["portfolio_return"] == pytest.approx(-0.125) and shock["value_change"] == pytest.approx(-12_500)
    assert any(s.startswith("Worst calendar month") for s in r.scenarios.index)
    assert r.correlation.loc["DBL", "ONE"] == pytest.approx(1.0)
    json.dumps(r.to_dict())


def test_failed_holdings_are_listed_and_excluded_never_dropped_silently():
    q_dbl, q_one = _quantities()
    book = _book(GONE=TickerNotFoundError("GONE", "unknown ticker"))
    r = review_portfolio([Holding("DBL", q_dbl), Holding("ONE", q_one), Holding("GONE", 10)], book)
    assert "GONE" in r.data_failures and "not_found" in r.data_failures["GONE"]
    assert "GONE" not in r.holdings.index and any("excluded from every figure" in w for w in r.warnings)
    nothing = review_portfolio([Holding("GONE", 10)], book)
    assert nothing.holdings.empty and nothing.data_failures and "No holding could be evaluated." in nothing.warnings


def test_short_overlap_withholds_risk_figures_and_thin_volume_is_flagged():
    short = ohlcv(_closes(BENCH_RET[:40], 20.0), volumes=np.full(41, 100.0))
    r = review_portfolio([Holding("DBL", 10), Holding("THIN", 1_000)], _book(THIN=dataset("THIN", short)))
    assert "var_95_1d" not in r.risk and r.correlation.empty and any("common to every holding" in w for w in r.warnings)
    assert r.liquidity.loc["THIN", "flag"] == "above 10% of ADV"


def test_holdings_csv_is_validated_row_by_row_and_lots_are_combined():
    content = b"Symbol,Shares,Avg Cost,Sector\nacme,10,100,Tech\nACME,30,120,\nbeta,5,,Energy\n"
    held = {h.ticker: h for h in parse_holdings_csv(content)}
    assert held["ACME"].quantity == 40 and held["ACME"].cost_basis == pytest.approx((10 * 100 + 30 * 120) / 40)
    assert held["ACME"].sector == "Tech" and held["BETA"].cost_basis is None
    with pytest.raises(ValueError, match=r"data row 1: .*positive.*data row 2: quantity 'x'"):
        parse_holdings_csv(b"ticker,quantity\nACME,-5\n\nBETA,x\n")      # blank lines do not shift row numbers
    with pytest.raises(ValueError, match="ambiguous"):
        parse_holdings_csv(b'ticker,quantity\nACME,"10,5"\n')             # a decimal comma is never guessed
    assert parse_holdings_csv(b'ticker,quantity,cost_basis\nACME,"1,250",1.5\n')[0].quantity == 1_250
    with pytest.raises(ValueError, match="missing: quantity"):
        parse_holdings_csv(b"ticker,price\nACME,5\n")
    with pytest.raises(ValueError):
        review_portfolio([], _book())


def test_sectors_group_case_insensitively_and_foreign_quotes_need_an_fx_rate():
    q_dbl, q_one = _quantities()
    r = review_portfolio([Holding("DBL", q_dbl, sector="Technology"), Holding("ONE", q_one, sector="technology ")],
                         _book())
    assert list(r.sector_exposure.index) == ["Technology"] and r.sector_exposure["weight"].iloc[0] == pytest.approx(1)
    book = loader({"DBL": dataset("DBL", DOUBLE), "VOD.L": dataset("VOD.L", SAME)})
    unconverted = review_portfolio([Holding("DBL", q_dbl), Holding("VOD.L", q_one)], book)
    assert any("VOD.L may not be quoted in the account currency" in w for w in unconverted.warnings)
    converted = review_portfolio([Holding("DBL", q_dbl), Holding("VOD.L", q_one, fx_rate=0.5)], book)
    assert not any("account currency" in w for w in converted.warnings)
    assert converted.holdings.loc["VOD.L", "value"] == pytest.approx(0.5 * q_one * SAME["close"].iloc[-1])
    assert "rebalanced daily" in HYPOTHETICAL
