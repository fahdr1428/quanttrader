"""RESEARCH mode: the ten-section report, built deterministically from the engine's results.

Every line is labelled as a **fact** (read from a source), a **calculated** metric
(with its formula available in the inputs), an **interpretation** (a judgement made
by a documented rule) or a **warning**. Nothing here is written by a language model:
the optional AI analyst works from this report's inputs and is checked against them.

Gating. Price data that fails validation withholds the analysis. Stale data, simulated
data or a failed risk check rules out a trade conclusion. A long setup qualifies only
when every condition in :data:`SETUP_RULES` holds; otherwise the conclusion is
"no qualifying setup", with the conditions that failed. The system is long-only, so a
bearish reading means "avoid / reduce", never a short.

Confidence (:data:`CONFIDENCE_RUBRIC`) rates the quality of the evidence behind the
conclusion — including a no-trade conclusion. It is not a probability of profit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

import numpy as np

from quantlab.agent.guardrails import REPORT_SECTIONS
from quantlab.data import DataProviderError, MarketDataset
from quantlab.fundamentals import (
    FundamentalsProvider,
    NotSupportedError,
    StatementPeriod,
    fundamental_metrics,
    statements_quality,
)
from quantlab.provenance import Timeliness
from quantlab.risk import (
    PortfolioState,
    RiskLimits,
    TradeSetup,
    check_limits,
    correlation_matrix,
    gap_risk,
    illustrative_limits,
    risk_reward,
    size_position,
    trade_allowed,
)
from quantlab.serialization import clean_json
from quantlab.signals import current_signal, signal_spec
from quantlab.technicals import TREND_RULE, TechnicalSnapshot, load_validated, snapshot_from_dataset

logger = logging.getLogger(__name__)

STOP_ATR_MULTIPLE = 2.0
REFERENCE_R_MULTIPLE = 2.0
MIN_DOLLAR_VOLUME = 5e6
MAX_STALE_SESSIONS = 5
SETUP_RULES = (
    f"price data passed validation, is not simulated, and is current: its last session is at most "
    f"{MAX_STALE_SESSIONS} business days before the report date and it is not from a stale cache",
    "trend is 'uptrend' (" + TREND_RULE.split(";")[0] + ")",
    "the selected strategy's current signal is long",
    "RSI(14) is below 70 (not stretched)",
    f"20-session average dollar volume is at least {MIN_DOLLAR_VOLUME:,.0f}",
    "the position size is positive and every risk-limit check passes",
)
CONFIDENCE_RUBRIC = (
    "One point each for: validated data with no quality warnings; at least three years (756 sessions) of history; "
    "technical trend and strategy signal pointing the same way (uptrend and long, or downtrend and cash); "
    "fundamentals from the primary source (SEC EDGAR) without quality warnings (missing line items aside). "
    "High = 4 points; Moderate = 2-3; Low = 0-1. Caps: simulated, stale or under one year of data -> Low; "
    "user-supplied price data or no fundamentals -> at most Moderate."
)


@dataclass(frozen=True)
class ReportLine:
    kind: str       # fact | calculated | interpretation | warning
    text: str


@dataclass(frozen=True)
class ReportSection:
    title: str
    lines: tuple[ReportLine, ...]


@dataclass
class ResearchInputs:
    """Everything the report is built from; each part is None with a reason when unavailable."""

    ticker: str
    benchmark: str
    strategy_key: str
    load_status: str
    load_reason: str = ""
    dataset: MarketDataset | None = None
    snapshot: TechnicalSnapshot | None = None
    bench_snapshot: TechnicalSnapshot | None = None
    signal: dict[str, Any] | None = None
    profile: Any = None
    metrics: dict[str, Any] | None = None
    statement_warnings: list[str] = field(default_factory=list)
    statements_source: str | None = None
    events: list[Any] = field(default_factory=list)
    filings: list[Any] = field(default_factory=list)
    gaps: dict[str, Any] | None = None
    setup: TradeSetup | None = None
    sizing: Any = None
    checks: list[Any] = field(default_factory=list)
    reward: dict[str, Any] | None = None
    limits: RiskLimits | None = None
    portfolio: PortfolioState | None = None
    bench_warnings: list[str] = field(default_factory=list)
    not_retrieved: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchReport:
    ticker: str
    as_of: date | None
    generated_at: datetime
    conclusion: str
    failed_conditions: tuple[str, ...]
    confidence: str
    confidence_reasons: tuple[str, ...]
    sections: tuple[ReportSection, ...]
    inputs: ResearchInputs

    def to_markdown(self) -> str:
        label = {"fact": "Fact", "calculated": "Calculated", "interpretation": "Interpretation", "warning": "Warning"}
        parts = [f"# Research report: {self.ticker}", ""]
        for i, section in enumerate(self.sections, 1):
            parts.append(f"## {i}. {section.title}")
            parts.extend(f"- **{label[ln.kind]}:** {ln.text}" for ln in section.lines)
            parts.append("")
        return "\n".join(parts).strip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        return clean_json({
            "ticker": self.ticker, "as_of": self.as_of, "generated_at": self.generated_at,
            "conclusion": self.conclusion, "failed_conditions": list(self.failed_conditions),
            "confidence": self.confidence, "confidence_reasons": list(self.confidence_reasons),
            "confidence_rubric": CONFIDENCE_RUBRIC, "setup_rules": list(SETUP_RULES),
            "sections": [{"title": s.title, "lines": [{"kind": ln.kind, "text": ln.text} for ln in s.lines]}
                         for s in self.sections],
        })


# --------------------------------------------------------------------------- #
# Gathering
# --------------------------------------------------------------------------- #


def gather_research(ticker: str, load_prices: Callable[[str], MarketDataset], *, benchmark: str = "SPY",
                    strategy_key: str = "ma_crossover", fundamentals: FundamentalsProvider | None = None,
                    limits: RiskLimits | None = None, portfolio: PortfolioState | None = None,
                    extra_checks: Any = ()) -> ResearchInputs:
    """Load, validate and compute everything the report needs; failures are recorded, never raised.

    ``portfolio`` is the account the limits are checked against (holdings, today's loss, equity
    peak). Without it the account-dependent limits cannot be checked and the setup fails them.
    """
    outcome = load_validated(ticker, load_prices)
    inputs = ResearchInputs(ticker=ticker.upper(), benchmark=benchmark, strategy_key=strategy_key,
                            load_status=outcome.status, load_reason=outcome.reason,
                            limits=limits or illustrative_limits(), portfolio=portfolio)
    if not outcome.ok:
        return inputs
    inputs.dataset = outcome.dataset
    prices = outcome.dataset.prices
    bench = load_validated(benchmark, load_prices) if benchmark and benchmark.upper() != ticker.upper() else None
    if bench is not None and not bench.ok:
        inputs.not_retrieved["benchmark"] = f"{benchmark}: {bench.reason}"
    bench_ds = bench.dataset if bench is not None and bench.ok else None
    if bench_ds is not None:
        bench_prov = snapshot_from_dataset(bench_ds).provenance
        asset_simulated = outcome.dataset.cache_status.value == "synthetic"
        if bench_prov.timeliness is Timeliness.SIMULATED and not asset_simulated:
            inputs.bench_warnings.append(f"Benchmark {benchmark} is SIMULATED: market context and relative strength "
                                         "against it are not computed for a real asset.")
            bench_ds = None
        elif bench.status in ("stale", "stale_cache"):
            inputs.bench_warnings.append(f"Benchmark {benchmark} data is {bench.status.replace('_', ' ')}: its context "
                                         "may be out of date.")
        if bench_ds is not None and bench_prov.timeliness is Timeliness.USER_SUPPLIED:
            inputs.bench_warnings.append(f"Benchmark {benchmark} is a user-supplied file (not independently verified).")
    inputs.snapshot = snapshot_from_dataset(outcome.dataset, bench_ds)
    if bench_ds is not None:
        inputs.bench_snapshot = snapshot_from_dataset(bench_ds)
    try:
        inputs.signal = current_signal(strategy_key, prices)
    except (ValueError, KeyError) as exc:
        inputs.not_retrieved["signal"] = str(exc)

    simulated = inputs.snapshot.provenance.timeliness is Timeliness.SIMULATED
    if fundamentals is None:
        inputs.not_retrieved["fundamentals"] = "no fundamentals provider configured"
    elif simulated:
        inputs.not_retrieved["fundamentals"] = "not applicable: simulated price series"
    else:
        _gather_fundamentals(inputs, fundamentals)

    snap = inputs.snapshot
    atr_value, close = snap.value("atr_14"), snap.close.value
    if atr_value is not None and close:
        stop = close - STOP_ATR_MULTIPLE * atr_value
        if stop > 0:
            inputs.gaps = gap_risk(prices, (close - stop) / close)
            allowance = (inputs.gaps.get("suggested_gap_allowance_pct") or 0.0) * close
            target = close + REFERENCE_R_MULTIPLE * (close - stop)
            sector = getattr(inputs.profile, "sector", None)
            inputs.setup = TradeSetup(inputs.ticker, entry=close, stop=stop, targets=(target,),
                                      gap_allowance=allowance, sector=sector)
            inputs.sizing = size_position(inputs.setup, inputs.limits, portfolio)
            correlations = None
            if portfolio is not None and portfolio.positions:
                frames = {}
                for t in {p.ticker for p in portfolio.positions}:
                    held = load_validated(t, load_prices)
                    if held.ok:
                        frames[t] = held.dataset.prices
                correlations = correlation_matrix({**frames, inputs.ticker: prices})
            inputs.checks = check_limits(inputs.setup, inputs.sizing, inputs.limits, portfolio, correlations)
            inputs.checks += list(extra_checks)
            inputs.reward = risk_reward(inputs.setup)
    else:
        inputs.not_retrieved["stop"] = "ATR(14) unavailable: no volatility-based stop can be defined"
    return inputs


def _gather_fundamentals(inputs: ResearchInputs, provider: FundamentalsProvider) -> None:
    def attempt(name: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except NotSupportedError as exc:
            inputs.not_retrieved[name] = f"not offered by {provider.name}: {exc.user_message}"
        except DataProviderError as exc:
            inputs.not_retrieved[name] = f"{provider.name}: {exc.user_message}"
        except Exception as exc:  # noqa: BLE001 - one source's malformed data must not take the report down
            logger.warning("Unexpected %s from %s (%s): %s", type(exc).__name__, provider.name, name, exc)
            inputs.not_retrieved[name] = f"{provider.name}: unexpected {type(exc).__name__}; not used"
        return None

    inputs.profile = attempt("profile", lambda: provider.profile(inputs.ticker))
    annual = attempt("annual statements", lambda: provider.statements(inputs.ticker, StatementPeriod.ANNUAL))
    quarterly = attempt("quarterly statements", lambda: provider.statements(inputs.ticker, StatementPeriod.QUARTERLY))
    if annual is not None:
        inputs.metrics = attempt("metrics", lambda: fundamental_metrics(annual, inputs.snapshot.close, ttm=quarterly))
        if inputs.metrics is not None:
            inputs.statements_source = provider.name
            inputs.statement_warnings = statements_quality(annual)
    inputs.events = attempt("events", lambda: provider.events(inputs.ticker, limit=10)) or []
    inputs.filings = attempt("filings", lambda: provider.filings(inputs.ticker, limit=5)) or []


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:+.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _sessions_behind(as_of: date, today: date) -> int:
    """Business days from the last session to ``today`` (holidays ignored: a slight overestimate)."""
    return max(0, int(np.busday_count(as_of, today)) - 1) if today > as_of else 0


def _data_problems(inputs: ResearchInputs, today: date) -> list[str]:
    snap = inputs.snapshot
    problems = []
    if snap.provenance.timeliness is Timeliness.SIMULATED:
        problems.append("simulated data")
    if inputs.load_status in ("stale", "stale_cache"):
        problems.append("stale data" if inputs.load_status == "stale" else "data from a stale cache")
    behind = _sessions_behind(snap.as_of, today)
    if behind > MAX_STALE_SESSIONS:
        problems.append(f"data ending {snap.as_of}, {behind} business days before {today}")
    if inputs.load_status not in ("ok", "ok_with_warnings", "stale", "stale_cache"):
        problems.append(f"data status {inputs.load_status}")
    return problems


def _conditions(inputs: ResearchInputs, today: date) -> list[tuple[str, bool]]:
    snap, sig = inputs.snapshot, inputs.signal or {}
    rsi = snap.value("rsi_14")
    adv = snap.value("avg_dollar_volume_20")
    risk_ok = bool(inputs.sizing is not None and inputs.sizing.units > 0 and trade_allowed(inputs.checks))
    return [
        (SETUP_RULES[0], not _data_problems(inputs, today)),
        (SETUP_RULES[1], snap.trend == "uptrend"),
        (SETUP_RULES[2], sig.get("position") == "long"),
        (SETUP_RULES[3], rsi is not None and rsi < 70),
        (SETUP_RULES[4], adv is not None and adv >= MIN_DOLLAR_VOLUME),
        (SETUP_RULES[5], risk_ok),
    ]


def _confidence(inputs: ResearchInputs, today: date) -> tuple[str, list[str]]:
    snap = inputs.snapshot
    reasons: list[str] = []
    points = 0
    if inputs.load_status == "ok":
        points += 1
        reasons.append("+1 validated data without quality warnings")
    n = inputs.dataset.n_obs
    if n >= 756:
        points += 1
        reasons.append(f"+1 {n} sessions of history")
    position = (inputs.signal or {}).get("position")
    if (snap.trend == "uptrend" and position == "long") or (snap.trend == "downtrend" and position == "cash"):
        points += 1
        reasons.append("+1 trend and strategy signal agree")
    material = [w for w in inputs.statement_warnings if " not reported — " not in w]
    if inputs.statements_source == "SEC EDGAR" and inputs.metrics is not None and not material:
        points += 1
        reasons.append("+1 primary-source fundamentals without quality warnings")
    level = "High" if points >= 4 else "Moderate" if points >= 2 else "Low"
    caps = _data_problems(inputs, today)
    if n < 252:
        caps.append("under one year of data")
    if caps:
        level = "Low"
        reasons.append("capped at Low: " + ", ".join(caps))
    else:
        moderate = []
        if inputs.metrics is None:
            moderate.append("no fundamentals")
        if snap.provenance.timeliness is Timeliness.USER_SUPPLIED:
            moderate.append("user-supplied price data (not independently verified)")
        if moderate and level == "High":
            level = "Moderate"
            reasons.append("capped at Moderate: " + ", ".join(moderate))
    return level, [f"{points} of 4 points"] + reasons


def build_research_report(inputs: ResearchInputs, now: datetime | None = None) -> ResearchReport:
    now = now or datetime.now(timezone.utc)
    sections: dict[str, list[ReportLine]] = {title: [] for title in REPORT_SECTIONS}

    def add(title_index: int, kind: str, text: str) -> None:
        sections[REPORT_SECTIONS[title_index]].append(ReportLine(kind, text))

    if inputs.snapshot is None:
        conclusion = "Analysis withheld: the price data could not be used."
        add(0, "warning", f"{conclusion} {inputs.ticker}: {inputs.load_reason or inputs.load_status}.")
        add(1, "warning", f"Price data status: {inputs.load_status}. {inputs.load_reason}")
        for i in range(2, 9):
            add(i, "warning", "Not assessed: depends on validated price data.")
        add(9, "interpretation", "No conclusion is drawn from data that failed to load or validate.")
        return ResearchReport(inputs.ticker, None, now, conclusion, (), "Low", ("no usable data",),
                              tuple(ReportSection(t, tuple(v)) for t, v in sections.items()), inputs)

    snap, prov = inputs.snapshot, inputs.snapshot.provenance
    today = now.astimezone(timezone.utc).date()
    conditions = _conditions(inputs, today)
    failed = tuple(rule for rule, ok in conditions if not ok)
    data_problems = _data_problems(inputs, today)
    if not failed:
        conclusion = "Candidate long setup: every setup condition holds (a research candidate, not an order)."
    elif data_problems:
        conclusion = ("No qualifying setup: the data cannot support a trade conclusion ("
                      + ", ".join(data_problems) + ").")
    elif snap.trend == "downtrend":
        conclusion = "No qualifying setup — avoid / reduce: the trend is down (long-only system, no shorting)."
    else:
        conclusion = "No qualifying setup."
    confidence, confidence_reasons = _confidence(inputs, today)
    close = snap.close.value

    # 1. Executive summary
    add(0, "interpretation", f"{conclusion} Evidence confidence: {confidence} ({confidence_reasons[0]}).")
    add(0, "fact", f"Latest validated close {_num(close)} on {snap.as_of} (end-of-day, not live).")
    for rule in failed:
        add(0, "interpretation", f"Condition not met: {rule}.")

    # 2. Data freshness and source quality
    add(1, "fact", f"Prices: {prov.source}; {prov.timeliness.label}; sessions {prov.period_start} to "
                   f"{prov.period_end}; retrieved {prov.retrieved_at:%Y-%m-%d %H:%M} UTC; adjustment "
                   f"{prov.adjustment.value}; cache {prov.cache_status}.")
    add(1, "fact", f"Data status: {inputs.load_status}.")
    for w in prov.warnings:
        add(1, "warning", w)
    if prov.timeliness is Timeliness.SIMULATED:
        add(1, "warning", "SIMULATED price series: not market data; no real-market conclusion can be drawn.")
    for what, why in sorted(inputs.not_retrieved.items()):
        add(1, "warning", f"Not retrieved — {what}: {why}.")

    # 3. Market and sector context
    if inputs.bench_snapshot is not None:
        b = inputs.bench_snapshot
        add(2, "calculated", f"Benchmark {inputs.benchmark}: close {_num(b.close.value)} on {b.as_of}; trend {b.trend}; "
                             f"63-session realised volatility {_pct(b.value('realised_vol_63'))} annualised.")
    else:
        add(2, "warning", f"No benchmark context: {inputs.not_retrieved.get('benchmark', 'benchmark not used')}.")
    for w in inputs.bench_warnings:
        add(2, "warning", w)
    rs = ", ".join(f"{lb} sessions {_pct(snap.value(f'rs_{lb}'))}" for lb in (63, 126, 252))
    add(2, "calculated", f"Relative strength vs {inputs.benchmark} (asset minus benchmark return): {rs}.")
    sector = getattr(inputs.profile, "sector", None) or getattr(inputs.profile, "industry", None)
    add(2, "fact" if sector else "warning",
        f"Sector/industry: {sector}." if sector else "Sector not known: sector context and the sector limit are "
                                                     "not applied.")

    # 4. Key quantitative findings
    rules = {part.split(":", 1)[0].strip(): part.strip() for part in TREND_RULE.split(";")}
    add(3, "interpretation", f"Trend: {snap.trend} ({rules.get(snap.trend, TREND_RULE)}).")
    for name, text in (("sma_50", "50-session SMA"), ("sma_200", "200-session SMA")):
        add(3, "calculated", f"{text}: {_num(snap.value(name))}.")
    add(3, "calculated", f"RSI(14): {_num(snap.value('rsi_14'), 1)}; ATR(14): {_num(snap.value('atr_14'))} "
                         f"({_pct(snap.value('atr_pct'))} of the close); 63-session realised volatility "
                         f"{_pct(snap.value('realised_vol_63'))}.")
    add(3, "calculated", f"52-week high {_num(snap.value('high_52w'))}, low {_num(snap.value('low_52w'))}; close "
                         f"{_pct(snap.value('pct_from_high_52w'))} from the high.")
    add(3, "calculated", f"20-session average dollar volume: {_num(snap.value('avg_dollar_volume_20'), 0)}.")
    if inputs.signal is not None:
        sig = inputs.signal
        spec = signal_spec(inputs.strategy_key)
        add(3, "calculated", f"{sig['label']}: position {sig['position']} as of {sig['as_of']}; last change "
                             f"{sig['last_change_date'] or 'none in the data'}.")
        add(3, "fact", f"Signal definition — hypothesis: {spec.hypothesis} Entry: {spec.entry} Exit: {spec.exit}")
    for note in snap.notes:
        add(3, "warning" if "SIMULATED" in note else "fact", note)

    # 5. Fundamental and catalyst analysis
    if inputs.metrics is not None:
        m = inputs.metrics
        add(4, "fact", f"Statements from {inputs.statements_source}.")
        for key in ("revenue_growth_yoy", "operating_margin", "net_margin", "fcf_margin"):
            calc = m[key]
            add(4, "calculated", f"{key}: {_pct(calc.value)}" + (f" ({calc.reason})" if calc.value is None else "")
                + f" — {calc.formula}.")
        for key in ("pe_trailing", "ev_to_ebitda", "net_debt_to_ebitda", "debt_to_equity"):
            calc = m[key]
            add(4, "calculated", f"{key}: {_num(calc.value)}" + (f" ({calc.reason})" if calc.value is None else "")
                + f" — {calc.formula}.")
        for w in inputs.statement_warnings:
            add(4, "warning", w)
    else:
        add(4, "warning", "No fundamental analysis: " + inputs.not_retrieved.get("fundamentals",
                                                                              inputs.not_retrieved.get(
                                                                                  "annual statements", "unavailable"))
            + ".")
    upcoming = [e for e in inputs.events if e.date >= snap.as_of]
    for e in sorted(upcoming, key=lambda e: e.date)[:3]:
        add(4, "fact", f"Upcoming {e.kind} on {e.date} (confirmed: {e.confirmed}): {e.detail}")
    if inputs.events and not upcoming:
        add(4, "fact", "No upcoming event dates in the source's calendar.")
    for f in inputs.filings[:3]:
        add(4, "fact", f"Filing {f.form} filed {f.filed}: {f.url}")

    # 6. Trading thesis and alternative scenarios
    if not failed:
        add(5, "interpretation", "Bullish continuation: price is in an uptrend with the strategy long and room below "
                                 "an overbought reading.")
    else:
        add(5, "interpretation", "No thesis is proposed: the setup conditions are not all met.")
    add(5, "interpretation", "Alternative — trend failure: a close below the 50-session SMA would turn the trend "
                             "'mixed' and remove the setup.")
    add(5, "interpretation", "Alternative — volatility shock: a move beyond the ATR-based stop, or a gap through it.")

    # 7. Entry, exit and position-sizing framework
    if inputs.setup is not None and inputs.sizing is not None:
        s, z = inputs.setup, inputs.sizing
        add(6, "calculated", f"Reference entry {_num(s.entry)} (latest close; a conditional level, not an execution "
                             f"price); volatility stop {_num(s.stop)} = close − {STOP_ATR_MULTIPLE:g} × ATR(14).")
        add(6, "calculated", f"Profit-taking reference {_num(s.targets[0])} = entry + {REFERENCE_R_MULTIPLE:g}R "
                             "(a framework level, not a forecast).")
        add(6, "calculated", f"Size {z.units:g} units, notional {_num(z.notional)}, risk budget {_num(z.risk_budget)}, "
                             f"estimated loss at the stop {_num(z.est_loss_at_stop)} ({_num(z.est_loss_with_gap)} with "
                             f"the gap allowance); binding limit: {z.binding_constraint}.")
        for w in z.warnings:
            add(6, "warning", w)
    else:
        add(6, "warning", "No sizing: " + inputs.not_retrieved.get("stop", "no stop could be defined") + ".")
    if inputs.limits is not None and inputs.limits.illustrative:
        add(6, "warning", "Limits are ILLUSTRATIVE defaults, not the user's: the size is an example only.")

    # 8. Risk/reward and portfolio implications
    if inputs.reward and inputs.reward.get("valid"):
        row = inputs.reward["targets"][0]
        add(7, "calculated", f"Gross R {_num(row['gross_r'])}; net R after costs {_num(row['net_r'])} at the reference "
                             "level. R describes the payoff if the level is reached, not its likelihood.")
    if inputs.gaps is not None:
        g = inputs.gaps
        add(7, "calculated", f"Gap risk: {g.get('gaps_through_stop', 'n/a')} of {g['sessions']} overnight gaps went "
                             f"through a stop this far away; worst gap {_pct(g.get('worst_gap'))}.")
    if inputs.portfolio is None:
        add(7, "warning", "No account state was supplied (holdings, today's loss, equity peak): the limits that "
                          "depend on it cannot be checked, so they fail.")
    else:
        acct = inputs.portfolio
        dd = acct.drawdown()
        add(7, "fact", f"Account state used: equity {_num(acct.equity)}, cash {_num(acct.cash)}, "
                       f"{len(acct.positions)} position(s), loss today {_pct(acct.daily_loss())}, drawdown "
                       f"{'unknown' if dd is None else _pct(dd)}.")
    for c in inputs.checks:
        unchecked = c.detail.startswith("Not checked")
        verdict = "NOT CHECKED (fails)" if unchecked else ("pass" if c.ok else "FAIL")
        add(7, "calculated" if c.ok else "warning", f"Limit check '{c.name}': {verdict} — {c.detail}")
    if inputs.checks and not trade_allowed(inputs.checks):
        add(7, "interpretation", "A limit check failed or could not be confirmed: do not trade.")

    # 9. Evidence that would invalidate the thesis
    if inputs.setup is not None:
        add(8, "interpretation", f"A close below the stop {_num(inputs.setup.stop)}.")
    add(8, "interpretation", "The trend classification leaving 'uptrend', or the strategy signal turning to cash.")
    if inputs.metrics is not None:
        add(8, "interpretation", "Revenue growth turning negative or free cash flow turning negative in the next filing.")

    # 10. Sources, assumptions and outstanding uncertainties
    add(9, "fact", f"Price source: {prov.source}.")
    if inputs.statements_source:
        add(9, "fact", f"Fundamentals source: {inputs.statements_source}.")
    add(9, "fact", f"Setup rules: {'; '.join(SETUP_RULES)}.")
    add(9, "fact", f"Confidence rubric: {CONFIDENCE_RUBRIC}")
    add(9, "interpretation", "Assumptions: fees 0.10% and slippage 0.05% per leg; end-of-day data; long-only, no "
                             "leverage. Past behaviour does not determine future returns.")
    return ResearchReport(inputs.ticker, snap.as_of, now, conclusion, failed, confidence, tuple(confidence_reasons),
                          tuple(ReportSection(t, tuple(v)) for t, v in sections.items()), inputs)


def research_report(ticker: str, load_prices: Callable[[str], MarketDataset], **kwargs: Any) -> ResearchReport:
    """Gather and build in one call (keyword arguments as for :func:`gather_research`, plus ``now``)."""
    now = kwargs.pop("now", None)
    return build_research_report(gather_research(ticker, load_prices, **kwargs), now)
