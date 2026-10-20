"""Research Agent — the seven operating modes on the deterministic engine, plus an optional AI analyst.

Everything above the "AI analyst" heading is computed by the tested engine and works without an
API key. The AI analyst (optional) orchestrates the same tools and its report is checked against
their results before it is shown. Research and simulation only: nothing here can place a real order.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pandas as pd
import streamlit as st

from quantlab.agent.audit import AuditLog
from quantlab.agent.research_tools import build_registry
from quantlab.agent.runner import AgentConfig, run_agent
from quantlab.agent.tools import MODE_DESCRIPTIONS, MODE_INPUTS, Mode
from quantlab.alerts import ALERT_KINDS
from quantlab.holdings import parse_holdings_csv
from quantlab.research_report import research_report
from quantlab.strategies import STRATEGIES
from ui import agent_state as ag
from ui import components as ui
from ui.state import current_data


@st.cache_resource(show_spinner=False)
def _registry():
    return build_registry()


def _md(text: str) -> str:
    """Markdown with '$' escaped, so amounts are never typeset as formulas."""
    return text.replace("$", r"\$")


def _records(rows, columns=None) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.caption("Nothing to show.")
        return
    ui.show_table(frame[columns] if columns else frame)


def _tool(name: str, inputs: dict, ctx, mode: Mode):
    outcome = _registry().execute(name, inputs, ctx, mode)
    if not outcome.ok:
        st.error(outcome.error, icon=":material/error:")
        return None
    return outcome.result


ui.page_header(
    "Research agent",
    "Quantitative research analyst",
    "Seven operating modes on validated data with provenance, documented rules and risk limits enforced in "
    "code. An optional AI analyst works from the same tools; every number it states is checked against them.",
)
st.markdown(ui.pill("Research and simulation only — no live orders, no broker connection", "warn")
            + " " + ui.pill("End-of-day data, not live prices", "info"), unsafe_allow_html=True)

loaded = current_data()
default_ticker = loaded.asset.ticker if loaded is not None else "SPY"
st.session_state.setdefault("ag_ticker", default_ticker)
st.session_state.setdefault("ag_bench", loaded.benchmark_ticker if loaded is not None and
                            loaded.benchmark_dataset is not None else default_ticker if loaded is not None else "SPY")

mode_value = st.segmented_control("Operating mode", [m.value for m in Mode], key="ag_mode",
                                  default=Mode.RESEARCH.value) or Mode.RESEARCH.value
mode = Mode(mode_value)
required = ", ".join(MODE_INPUTS[mode]) or "none"
st.markdown(f"**Active mode: {mode.value}.** {MODE_DESCRIPTIONS[mode]} Required inputs: **{required}**.")

limits = ag.risk_limits_form()
if limits.illustrative:
    st.caption("Risk limits: ILLUSTRATIVE defaults (not yours) — confirm your own under *Risk limits* for real sizing.")
providers, provider_notes = ag.fundamentals_providers()

# --------------------------------------------------------------------------- #
# Mode inputs
# --------------------------------------------------------------------------- #

inputs: dict = {}
c1, c2, c3 = st.columns([2, 2, 2])
benchmark = c3.text_input("Benchmark", key="ag_bench", max_chars=15,
                          help="Context and relative strength are measured against it.").strip().upper() or "SPY"
if mode in (Mode.RESEARCH, Mode.BACKTEST, Mode.PAPER_TRADE):
    inputs["ticker"] = c1.text_input("Ticker", key="ag_ticker", max_chars=15).strip().upper()
if mode in (Mode.RESEARCH, Mode.BACKTEST):
    inputs["strategy"] = c2.selectbox("Strategy", list(STRATEGIES), key="ag_strategy",
                                      format_func=lambda k: STRATEGIES[k].name)
use_fundamentals = True
if mode is Mode.RESEARCH:
    use_fundamentals = st.toggle("Fetch fundamentals, events and news (network)", value=True, key="ag_fund")
    for note in provider_notes:
        st.caption(note)
if mode is Mode.SCAN:
    raw = st.text_area("Universe (tickers separated by commas or spaces, at most 50)", key="ag_universe",
                       value="SPY QQQ IWM", height=80)
    inputs["universe"] = list(dict.fromkeys(t.upper() for t in re.split(r"[\s,;]+", raw) if t))[:50]
    s1, s2, s3 = st.columns(3)
    trend = s1.selectbox("Trend", ["any", "uptrend", "downtrend"], key="ag_scan_trend")
    min_price = s2.number_input("Min price", 0.0, 1e6, 5.0, key="ag_scan_price")
    min_adv = s3.number_input("Min avg dollar volume (millions)", 0.0, 1e6, 5.0, key="ag_scan_adv")
    st.caption("The universe is yours: a list of today's names has survivorship bias.")
if mode is Mode.ALERTS:
    st.caption("Kinds: " + "; ".join(f"**{k}** — {v.summary}" for k, v in ALERT_KINDS.items()))
    rules_frame = st.data_editor(
        pd.DataFrame([{"ticker": st.session_state["ag_ticker"], "kind": "rsi_above", "params_json": '{"level": 70}'}]),
        num_rows="dynamic", key="ag_rules", width="stretch",
        column_config={"kind": st.column_config.SelectboxColumn(options=list(ALERT_KINDS), required=True)},
    )
    inputs["rules"] = [{"ticker": str(r["ticker"]).strip().upper(), "kind": r["kind"],
                        "params_json": (str(r["params_json"]).strip() or None) if pd.notna(r["params_json"]) else None}
                       for _, r in rules_frame.iterrows() if str(r.get("ticker") or "").strip() and r.get("kind")]
if mode is Mode.PORTFOLIO_REVIEW:
    upload = st.file_uploader("Holdings CSV (ticker, quantity[, cost_basis, sector])", type=["csv"], key="ag_hold_csv")
    if upload is not None:
        try:
            st.session_state["ag_holdings"] = parse_holdings_csv(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc), icon=":material/error:")
    inputs["holdings"] = [h.ticker for h in ag.holdings()]
    if ag.holdings():
        _records([h.__dict__ for h in ag.holdings()])

ctx = ag.build_context(limits, benchmark, providers if use_fundamentals else ())

# --------------------------------------------------------------------------- #
# Deterministic engine
# --------------------------------------------------------------------------- #

ui.section("Engine results", "Computed by the tested engine from validated data — no language model involved.")

if mode is Mode.PAPER_TRADE:
    with st.form("ag_paper_form", border=True):
        st.markdown("**Simulated order** (recorded in the local paper account; nothing is sent anywhere)")
        p1, p2, p3, p4 = st.columns(4)
        side = p1.selectbox("Side", ["buy", "sell"])
        quantity = p2.number_input("Quantity", min_value=1.0, value=10.0, step=1.0)
        order_type = p3.selectbox("Type", ["market", "limit"])
        limit_price = p4.number_input("Limit price (limit orders)", min_value=0.0, value=0.0)
        t1, t2, t3, t4 = st.columns(4)
        stop = t1.number_input("Thesis stop", min_value=0.0, value=0.0)
        target = t2.number_input("Thesis target (optional)", min_value=0.0, value=0.0)
        days = t3.number_input("Expected holding sessions", min_value=1, value=20)
        confidence = t4.selectbox("Evidence confidence", ["low", "moderate", "high"], index=0)
        summary = st.text_input("Thesis", max_chars=500)
        invalidation = st.text_input("What would invalidate it", max_chars=500)
        submitted = st.form_submit_button("Record simulated order", type="primary", icon=":material/receipt:")
    if submitted:
        result = _tool("paper_submit_order", {
            "ticker": inputs["ticker"], "side": side, "quantity": quantity, "order_type": order_type,
            "limit_price": limit_price or None, "stop": stop or None, "target": target or None,
            "thesis_summary": summary or None, "entry_rationale": None, "invalidation": invalidation or None,
            "expected_holding_days": int(days), "confidence": confidence}, ctx, mode)
        if result:
            order = result["order"]
            (st.warning if order["status"] == "rejected" else st.success)(
                f"Simulated order {order['id']}: {order['status']}. {order['reason']}".strip())
            _records(result["limit_checks"], ["name", "ok", "value", "limit", "detail"])
    if st.button("Process fills against the latest bars", key="ag_paper_process", icon=":material/sync:"):
        result = _tool("paper_process_fills", {"ticker": inputs["ticker"]}, ctx, mode)
        if result is not None:
            st.caption(f"{len(result['new_fills'])} new simulated fill(s) through {result['through_session']}.")
    account = _tool("paper_account", {}, ctx, mode)
    if account:
        a1, a2 = st.columns(2)
        a1.metric("Simulated cash", f"{account['cash']:,.2f}", border=True)
        a2.metric("Simulated equity", "n/a" if account["equity_latest"] is None else f"{account['equity_latest']:,.2f}",
                  border=True)
        _records(account["positions"])
        if account["open_orders"]:
            _records(account["open_orders"], ["id", "ticker", "side", "quantity", "order_type", "limit_price",
                                              "stop_price", "role", "decision_date"])

elif st.button("Run the engine", type="primary", key="ag_engine", icon=":material/play_arrow:"):
    with st.spinner("Loading validated data and computing…"):
        if mode is Mode.RESEARCH:
            report = research_report(inputs["ticker"], ctx.load_prices, benchmark=benchmark,
                                     strategy_key=inputs["strategy"],
                                     fundamentals=providers[0] if use_fundamentals and providers else None,
                                     limits=limits)
            kind = "good" if report.conclusion.startswith("Candidate") else "warn"
            st.markdown(ui.pill(report.conclusion, kind) + " " + ui.pill(f"Evidence confidence: {report.confidence}",
                                                                         "info"), unsafe_allow_html=True)
            st.markdown(_md(report.to_markdown()))
            st.download_button("Download report (Markdown)", report.to_markdown(),
                               file_name=f"research_{report.ticker}.md", mime="text/markdown")
        elif mode is Mode.SCAN:
            result = _tool("run_scan", {"universe": inputs["universe"], "min_price": min_price,
                                        "min_avg_dollar_volume": min_adv * 1e6, "min_history_sessions": None,
                                        "trend": None if trend == "any" else trend, "rsi_below": None,
                                        "rsi_above": None, "min_rs_126": None, "max_atr_pct": None}, ctx, mode)
            if result:
                st.markdown(f"**Passed:** {', '.join(result['passed']) or 'none'}")
                _records(result["rows"])
                for t, reason in result["failures"].items():
                    st.warning(f"{t}: {reason}", icon=":material/warning:")
                for note in result["notes"]:
                    st.caption(note)
        elif mode is Mode.ALERTS:
            result = _tool("evaluate_alert_rules", {"rules": inputs["rules"]}, ctx, mode) if inputs["rules"] else None
            if result:
                _records([{**e, "rule": e["rule"]["description"]} for e in result["events"]],
                         ["rule", "triggered", "as_of", "value", "data_status", "detail"])
                st.caption(result["note"])
        elif mode is Mode.BACKTEST:
            result = _tool("run_backtest", {"ticker": inputs["ticker"], "strategy": inputs["strategy"],
                                            "params_json": None, "start": None, "end": None, "fee_rate": None,
                                            "slippage_rate": None}, ctx, mode)
            if result:
                _records([{"metric": k, **v} for k, v in result["metrics"].items()])
                for w in result["warnings"]:
                    st.caption(w)
        elif mode is Mode.PORTFOLIO_REVIEW:
            result = _tool("review_portfolio", {}, ctx, mode)
            if result:
                _records(result["holdings"])
                st.json(result["concentration"], expanded=False)
                st.json(result["risk"], expanded=False)
                _records(result["scenarios"])
                _records(result["liquidity"])
                for w in result["warnings"]:
                    st.caption(w)
        elif mode is Mode.POST_TRADE_REVIEW:
            result = _tool("review_paper_trades", {}, ctx, mode)
            if result:
                _records(result["reviews"], ["order_id", "ticker", "entry_date", "exit_date", "outcome",
                                             "realised_r", "mae_r", "mfe_r", "holding_days", "confidence"])
                st.json(result["summary"], expanded=True)

# --------------------------------------------------------------------------- #
# Optional AI analyst
# --------------------------------------------------------------------------- #

ui.section("AI analyst (optional)",
           "Orchestrates the tools above and writes the narrative. Every number and link in its report must come from "
           "a tool result or your inputs; reports that fail the checks are withheld in strict mode.")
key = ag.api_key()
config = AgentConfig.from_environment()
if not key:
    st.info("The AI analyst is off: set `ANTHROPIC_API_KEY` in your environment or in Streamlit secrets to enable "
            "it. Everything above works without it.", icon=":material/key_off:")
else:
    question = st.text_area("Request", key="ag_question", height=90,
                            placeholder="e.g. Is there a qualifying setup? What would invalidate it?")
    strict = st.toggle("Withhold reports that fail verification (strict)", value=True, key="ag_strict")
    st.caption(f"Model: {config.model} · effort {config.effort} · at most {config.max_turns} requests and "
               f"{config.max_tool_calls} tool calls per run · {ag.max_runs_per_session()} runs per session.")
    entered = st.text_input("Access passcode", type="password", key="ag_passcode") if ag.agent_passcode() else ""
    runs = st.session_state.setdefault("ag_runs", 0)
    limit = ag.max_runs_per_session()
    if st.button("Run AI analyst", key="ag_run_ai", icon=":material/smart_toy:"):
        if not ag.passcode_ok(entered):
            st.error("Incorrect passcode.", icon=":material/lock:")
        elif runs >= limit:
            st.error(f"This session has used its {limit} AI analyst runs.", icon=":material/block:")
        else:
            import anthropic

            st.session_state["ag_runs"] = runs + 1
            with st.spinner("The analyst is working (this can take a few minutes)…"):
                st.session_state["ag_last_run"] = run_agent(
                    anthropic.Anthropic(api_key=key), _registry(), ctx, mode, inputs, question,
                    replace(config, strict=strict), AuditLog.from_environment())
    run = st.session_state.get("ag_last_run")
    if run is not None:
        st.markdown(ui.pill(f"Status: {run.status}", "good" if run.status == "completed" else "warn")
                    + " " + ui.pill(f"Mode: {run.mode.value}", "info"), unsafe_allow_html=True)
        if run.detail:
            st.warning(run.detail, icon=":material/info:")
        g = run.guardrails
        if run.withheld:
            st.error("**Report withheld:** it failed verification (see the checks below).", icon=":material/block:")
            with st.expander("Withheld draft — failed verification; do not rely on it"):
                st.markdown(_md(run.final_text))
        elif run.final_text:
            if g is not None and not g.passed:
                st.warning("Shown unverified (strict mode is off): some checks failed.", icon=":material/warning:")
            st.markdown(_md(run.final_text))
        if g is not None:
            with st.expander("Verification checks", expanded=not g.passed, icon=":material/fact_check:"):
                st.markdown(f"{g.numbers_checked} numbers checked against tool results and your inputs.")
                for issue in g.issues():
                    st.markdown(f"- {_md(issue)}")
                if g.passed:
                    st.success("All checks passed.", icon=":material/verified:")
        with st.expander("Tool calls and usage", icon=":material/build:"):
            _records([{"tool": c.name, "ok": c.ok, "error": c.error, "seconds": round(c.duration_s, 2)}
                      for c in run.tool_calls])
            st.caption(f"Models served: {', '.join(run.models_served) or 'none'} · tokens in "
                       f"{run.usage['input_tokens']:,} / out {run.usage['output_tokens']:,} · run {run.run_id}")

ui.disclaimer()
