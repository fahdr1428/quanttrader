"""RESEARCH mode report: ten sections, gating on data quality, documented setup rules and confidence rubric."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from quantlab.agent.guardrails import REPORT_SECTIONS, check_report
from quantlab.data import ProviderConnectionError, SyntheticProvider, load_market_data
from quantlab.fundamentals import RequestThrottle, SecEdgarProvider
from quantlab.research_report import SETUP_RULES, gather_research, research_report
from quantlab.risk import RiskLimits
from quantlab.simulation import SIM_END
from tests.technical_fixtures import dataset, loader, ohlcv
from tests.test_fundamentals import FakeSec

NOW = datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)
T = np.arange(800)
ZIGZAG_UP = ohlcv(50 * np.exp(0.001 * T + 0.01 * (-1.0) ** T), volumes=np.full(800, 2e6))   # RSI near 50
BENCH = ohlcv(400 * np.exp(0.0005 * T + 0.005 * (-1.0) ** T))


def _load(frame=ZIGZAG_UP, ticker="ACME"):
    return loader({ticker: dataset(ticker, frame), "SPY": dataset("SPY", BENCH)})


def sec():
    return SecEdgarProvider(user_agent="Test Runner test@example.com", http_get=FakeSec(), use_cache=False,
                            throttle=RequestThrottle(1e6), clock=lambda: NOW, sleep=lambda s: None)


def test_every_condition_holding_gives_a_candidate_with_primary_source_fundamentals():
    report = research_report("ACME", _load(), fundamentals=sec(), now=NOW)
    assert report.conclusion.startswith("Candidate long setup") and report.failed_conditions == ()
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
    report = research_report("ACME", _load(down), now=NOW)
    assert "avoid / reduce" in report.conclusion and "no shorting" in report.conclusion
    assert SETUP_RULES[1] in report.failed_conditions


def test_failed_risk_limits_remove_the_setup_and_say_do_not_trade():
    tight = RiskLimits(account_equity=1_000, available_cash=10)
    report = research_report("ACME", _load(), limits=tight, now=NOW)
    assert SETUP_RULES[5] in report.failed_conditions and report.conclusion == "No qualifying setup."
    assert "do not trade" in report.to_markdown()


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
