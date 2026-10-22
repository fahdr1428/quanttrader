"""RESEARCH mode report: ten sections, gating on data quality, documented setup rules and confidence rubric."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from quantlab.agent.guardrails import REPORT_SECTIONS, check_report
from quantlab.data import CacheStatus, ProviderConnectionError, SyntheticProvider, load_market_data
from quantlab.fundamentals import RequestThrottle, SecEdgarProvider
from quantlab.research_report import SETUP_RULES, gather_research, research_report
from quantlab.risk import PortfolioState, RiskLimits
from quantlab.simulation import SIM_END
from tests.technical_fixtures import dataset, loader, ohlcv
from tests.test_fundamentals import FakeSec

NOW = datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)
FRESH = datetime(2024, 3, 11, 21, 0, tzinfo=timezone.utc)       # the next business day after the data ends
T = np.arange(800)
ZIGZAG_UP = ohlcv(50 * np.exp(0.001 * T + 0.01 * (-1.0) ** T), volumes=np.full(800, 2e6))   # RSI near 50
BENCH = ohlcv(400 * np.exp(0.0005 * T + 0.005 * (-1.0) ** T))


ACCOUNT = PortfolioState(equity=100_000, cash=100_000, peak_equity=100_000, start_of_day_equity=100_000)


def _load(frame=ZIGZAG_UP, ticker="ACME"):
    """Provider-style datasets (not user uploads), so only the data's age and quality matter."""
    return loader({ticker: dataset(ticker, frame, cache_status=CacheStatus.DOWNLOADED, provider_key="yahoo"),
                   "SPY": dataset("SPY", BENCH, cache_status=CacheStatus.DOWNLOADED, provider_key="yahoo")})


def sec():
    return SecEdgarProvider(user_agent="Test Runner test@example.com", http_get=FakeSec(), use_cache=False,
                            throttle=RequestThrottle(1e6), clock=lambda: NOW, sleep=lambda s: None)


def test_every_condition_holding_gives_a_candidate_with_primary_source_fundamentals():
    report = research_report("ACME", _load(), fundamentals=sec(), portfolio=ACCOUNT, now=FRESH)
    assert report.conclusion.startswith("Candidate long setup") and report.failed_conditions == (), report.failed_conditions
    assert report.confidence == "High" and report.confidence_reasons[0] == "4 of 4 points"
    assert [s.title for s in report.sections] == list(REPORT_SECTIONS)
    md = report.to_markdown()
    assert not check_report(md, [report.to_dict()]).missing_sections         # headings the guardrails expect
    assert "Statements from SEC EDGAR" in md and "Filing 10-K filed 2024-02-15" in md
    assert "ILLUSTRATIVE" in md and "a framework level, not a forecast" in md
    inputs = report.inputs
    assert inputs.sizing.units > 0 and inputs.setup.stop < inputs.setup.entry
    json.dumps(report.to_dict())


def test_simulated_data_never_supports_a_trade_and_caps_confidence():
    ds = load_market_data("SIM-TREND-1", "2014-01-01", "2024-12-31", SyntheticProvider(), today=SIM_END.date())
    report = research_report("SIM-TREND-1", loader({"SIM-TREND-1": ds}), benchmark="SIM-TREND-1", now=NOW)
    assert report.conclusion.startswith("No qualifying setup") and SETUP_RULES[0] in report.failed_conditions
    assert report.confidence == "Low" and "capped at Low: simulated data" in report.confidence_reasons
    assert "SIMULATED price series" in report.to_markdown()


def test_unusable_data_withholds_the_analysis_but_keeps_all_sections():
    report = research_report("GONE", loader({"GONE": ProviderConnectionError("GONE", "offline")}), now=NOW)
    assert report.conclusion.startswith("Analysis withheld") and report.confidence == "Low"
    assert len(report.sections) == 10 and all(s.lines for s in report.sections)
    assert "Not assessed" in report.to_markdown() and report.inputs.sizing is None


def test_a_downtrend_means_avoid_not_short():
    down = ohlcv(80 * np.exp(-0.001 * T + 0.01 * (-1.0) ** T), volumes=np.full(800, 2e6))
    report = research_report("ACME", _load(down), portfolio=ACCOUNT, now=FRESH)
    assert "avoid / reduce" in report.conclusion and "no shorting" in report.conclusion
    assert SETUP_RULES[1] in report.failed_conditions
    stale = research_report("ACME", _load(down), portfolio=ACCOUNT, now=NOW)       # no advice on old data
    assert "avoid" not in stale.conclusion and "cannot support a trade conclusion" in stale.conclusion


def test_failed_risk_limits_remove_the_setup_and_say_do_not_trade():
    tight = RiskLimits(account_equity=1_000, available_cash=10)
    report = research_report("ACME", _load(), limits=tight, portfolio=ACCOUNT, now=FRESH)
    assert SETUP_RULES[5] in report.failed_conditions and report.conclusion == "No qualifying setup."
    assert "do not trade" in report.to_markdown()
    no_account = research_report("ACME", _load(), now=FRESH)                # account limits cannot be confirmed
    assert no_account.failed_conditions == (SETUP_RULES[5],)
    assert "NOT CHECKED (fails)" in no_account.to_markdown() and "No account state" in no_account.to_markdown()


def test_missing_sources_are_reported_not_hidden():
    inputs = gather_research("ACME", loader({"ACME": dataset("ACME", ZIGZAG_UP)}))
    assert "benchmark" in inputs.not_retrieved and "fundamentals" in inputs.not_retrieved
    assert inputs.snapshot.value("rs_63") is None


def test_fundamental_failures_appear_in_the_report(monkeypatch):
    monkeypatch.delenv("QLP_SEC_USER_AGENT", raising=False)
    provider = SecEdgarProvider(user_agent="", http_get=FakeSec(), use_cache=False)
    report = research_report("ACME", _load(), fundamentals=provider, now=NOW)
    md = report.to_markdown()
    assert "Not retrieved — annual statements: SEC EDGAR: SEC EDGAR requires a User-Agent" in md
    assert report.confidence != "High"


def test_data_that_ends_long_before_the_report_date_is_stale_whatever_its_status():
    """A historical dataset is never 'current': no candidate, confidence capped at Low."""
    report = research_report("ACME", _load(), fundamentals=sec(), portfolio=ACCOUNT, now=NOW)
    assert SETUP_RULES[0] in report.failed_conditions and report.confidence == "Low"
    assert any("business days before 2024-06-03" in r for r in report.confidence_reasons)


def test_a_simulated_benchmark_is_not_used_as_market_context_for_a_real_asset():
    ds = load_market_data("SIM-TREND-1", "2014-01-01", "2024-12-31", SyntheticProvider(), today=SIM_END.date())
    entries = {"ACME": dataset("ACME", ZIGZAG_UP, cache_status=CacheStatus.DOWNLOADED, provider_key="yahoo"),
               "SIM-TREND-1": ds}
    report = research_report("ACME", loader(entries), benchmark="SIM-TREND-1", portfolio=ACCOUNT, now=FRESH)
    md = report.to_markdown()
    assert "Benchmark SIM-TREND-1 is SIMULATED" in md and report.inputs.snapshot.value("rs_63") is None


def test_confidence_points_follow_the_rubric_text():
    """No signal agreement point without a signal; no fundamentals point with material quality warnings."""
    down = ohlcv(80 * np.exp(-0.001 * T[:300] + 0.01 * (-1.0) ** T[:300]), volumes=np.full(300, 2e6),
                 start="2023-01-03")
    warm = research_report("ACME", _load(down), strategy_key="ml_logistic", portfolio=ACCOUNT,
                           now=datetime.combine(down.index[-1].date(), datetime.min.time(), tzinfo=timezone.utc))
    assert warm.inputs.signal["position"] is None
    assert "+1 trend and strategy signal agree" not in warm.confidence_reasons
    inputs = gather_research("ACME", _load(), fundamentals=sec(), portfolio=ACCOUNT)
    inputs.statement_warnings.append("total_assets differ from total_liabilities + total_equity by more than 10%")
    from quantlab.research_report import build_research_report

    flagged = build_research_report(inputs, FRESH)
    assert "+1 primary-source fundamentals without quality warnings" not in flagged.confidence_reasons
    trend_line = next(ln.text for s in flagged.sections for ln in s.lines if ln.text.startswith("Trend:"))
    assert trend_line.startswith("Trend: uptrend (uptrend:")
