"""Research Overview — the selected strategy against its benchmark at a glance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import json

from quantlab.evidence import evidence_checklist
from quantlab.config import MIN_OBS_FOR_ANNUALISED
from quantlab.metrics import compute_metrics, exposed_sessions, sharpe_ratio
from quantlab.nulls import exposure_matched_benchmark
from quantlab.research import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from quantlab.robustness import regime_labels, regime_performance
from ui import charts
from ui import components as ui
from ui.state import active_run, permutation_for, trials_for

data = ui.require_data()
spec = ui.require_spec()
run, bench = active_run(data, spec)
report, net = run.report, run.net

ui.page_header(
    "Research overview",
    net.strategy_label,
    f"{run.strategy.family} strategy on {data.asset.ticker}, compared with {data.benchmark_label} over the same "
    "sessions, capital and costs. All figures are net of fees and slippage unless labelled otherwise.",
)
ui.context_strip(data)
ui.integrity_panel(run, data)
st.space("small")

# ---- headline figures -----------------------------------------------------------
bench_only = compute_metrics(bench)
deltas = {
    "total_return": f"Bench {ui.fmt(report.value('benchmark_total_return'), 'pct')}",
    "cagr": f"Bench {ui.fmt(report.value('benchmark_cagr'), 'pct')}",
    "sharpe": f"Bench {ui.fmt(report.value('benchmark_sharpe'), 'ratio')}",
    "max_drawdown": f"Bench {ui.fmt(report.value('benchmark_max_drawdown'), 'pct')}",
    "volatility": f"Bench {ui.fmt(bench_only.value('volatility'), 'pct_abs')}",
    "time_invested": f"Bench {ui.fmt(bench_only.value('time_invested'), 'pct_abs')}",
    "avg_exposure": f"Bench {ui.fmt(bench_only.value('avg_exposure'), 'pct_abs')}",
}
exposure_key = "avg_exposure" if run.strategy.continuous_exposure else "time_invested"
ui.kpi_row(report, ["total_return", "cagr", "sharpe", "max_drawdown", "volatility", exposure_key],
           deltas=deltas)
ui.unreliable_footnote(report, ["total_return", "cagr", "sharpe", "max_drawdown", "volatility"])

# ---- equity curve -----------------------------------------------------------------
ui.section("Equity curve vs benchmark",
           "Both portfolios start with the same capital at the same anchor session and pay the same costs. "
           "The '(gross, no costs)' line shows the strategy without fees or slippage, so its gap to the net line is the "
           "cost drag.")
c1, c2, _ = st.columns([1, 1, 4])
normalise = c1.toggle("Growth of 100", value=False, key="ov_norm")
log_scale = c2.toggle("Log scale", value=False, key="ov_log")
ui.plot(charts.equity_vs_benchmark(net, bench, run.gross, normalise=normalise, log=log_scale,
                                   strategy_name=f"{run.strategy.name} (net)", bench_name=data.benchmark_label))

left, right = st.columns([3, 2], gap="medium")
with left:
    ui.section("Drawdowns", "Percentage below the running peak of each portfolio.")
    ui.plot(charts.drawdowns(net, bench, height=320))
with right:
    ui.section("Key risk indicators")
    rows = [
        ("Annualised volatility", "volatility", None),
        ("Sharpe ratio", "sharpe", "benchmark_sharpe"),
        ("Sharpe 95% confidence interval", "sharpe_ci", None),
        ("Sortino ratio", "sortino", None),
        ("Maximum drawdown", "max_drawdown", "benchmark_max_drawdown"),
        ("Longest drawdown (sessions)", "max_dd_duration", None),
        ("Beta to benchmark", "beta", None),
        ("Correlation with benchmark", "correlation", None),
        ("Tracking error", "tracking_error", None),
        ("Alpha (annualised)", "alpha_annual", None),
        ("Alpha t-statistic (Newey-West)", "alpha_tstat", None),
    ]
    table = []
    for label, key, bkey in rows:
        if key == "sharpe_ci":
            lo, hi = report.get("sharpe_ci_low"), report.get("sharpe_ci_high")
            blo, bhi = bench_only.get("sharpe_ci_low"), bench_only.get("sharpe_ci_high")
            table.append({
                "Indicator": label,
                "Strategy": f"{lo.formatted()} to {hi.formatted()}" + ("" if lo.reliable else " *") if lo else "n/a",
                "Benchmark": f"{blo.formatted()} to {bhi.formatted()}" if blo else "—",
            })
            continue
        m = report.get(key)
        bm = report.get(bkey) if bkey else bench_only.get(key)
        table.append({
            "Indicator": label,
            "Strategy": (m.formatted() + ("" if m.reliable else " *")) if m else "n/a",
            "Benchmark": bm.formatted() if bm and key not in {"beta", "correlation", "tracking_error", "alpha_annual",
                                                              "alpha_tstat"} else "—",
        })
    ui.show_table(pd.DataFrame(table))
    st.caption("\\* small-sample or undefined — hover the metric cards for details.")

# ---- integrity and assumptions ------------------------------------------------------
ui.section("Research integrity")
trials = trials_for(data)
c1, c2, c3 = st.columns(3)
c1.metric("Configurations tried on this sample", len(trials), border=True,
          help="Every distinct strategy/parameter/cost configuration evaluated on this exact dataset and "
               "period in this session, including sensitivity grids. The more you try, the more likely the "
               "best result is luck.")
psr = probabilistic_sharpe_ratio(net.returns, 0.0, spec.config.risk_free_rate, spec.config.periods_per_year)
thin = exposed_sessions(net) < MIN_OBS_FOR_ANNUALISED
thin_note = (f" Only {exposed_sessions(net)} sessions carried market exposure, so this is not statistically "
             "meaningful." if thin else "")
c2.metric("Probabilistic Sharpe (vs 0)" + (" *" if thin else ""), "n/a" if psr is None else f"{psr:.0%}",
          border=True,
          help="Probability that the true Sharpe ratio (of returns in excess of the risk-free rate) is above zero "
               "given sample length, skew and fat tails "
               "(Bailey & López de Prado). Ignores how many configurations were tried." + thin_note)
pp = [t_["sharpe_pp"] for t_ in trials if t_.get("sharpe_pp") is not None]
dsr = (deflated_sharpe_ratio(net.returns, pp, spec.config.risk_free_rate, spec.config.periods_per_year)
       if len(pp) >= 2 else None)
c3.metric("Deflated Sharpe (this session's trials)" + (" *" if thin and dsr is not None else ""),
          "n/a" if dsr is None else f"{dsr:.0%}", border=True,
          help="Probabilistic Sharpe measured against the best Sharpe you would expect from that many "
               "configurations with no real edge. Needs at least two trials. Heuristic: assumes independent trials."
               + thin_note)
st.caption(f"Parameter selection: **{spec.parameter_selection}**")

# ---- evidence checklist ----------------------------------------------------------------
cfg = spec.config
grid = st.session_state.get("sens_result")
selection = (grid.get("selection") if grid and grid.get("key") == spec.strategy_key
             and grid.get("sample") == data.sample_key else None)
wf_inputs = None
wf_stored = st.session_state.get("wf_result")
if wf_stored:
    sig = json.loads(wf_stored["signature"])
    if sig.get("sample") == data.sample_key and sig.get("key") == spec.strategy_key:
        wf = wf_stored["result"]
        wf_inputs = {"oos_sharpe": sharpe_ratio(wf.oos.returns, wf.oos.config.risk_free_rate,
                                                wf.oos.config.periods_per_year),
                     "is_mean": float(wf.fold_table["IS Sharpe"].astype(float).mean()),
                     "oos_mean": float(wf.fold_table["OOS Sharpe"].astype(float).mean())}
perm_key = data.sample_key + spec.params_json() + cfg.model_dump_json()
permutation_p = (permutation_for(data, spec, run).p_value
                 if st.session_state.get("perm_requested") == perm_key else None)
matched = exposure_matched_benchmark(data.prices, net, cfg)
items = evidence_checklist(
    report=report, accounting_valid=net.is_valid,
    data_warnings=len(data.asset.quality.warnings)
    + (len(data.benchmark_dataset.quality.warnings) if data.benchmark_dataset is not None else 0),
    synthetic=ui.is_synthetic(data),
    parameters_prespecified=spec.uses_defaults or spec.declared_preselected,
    n_trials=len(trials), dsr=dsr,
    exposure_matched_cagr=compute_metrics(matched).value("cagr"),
    permutation_p=permutation_p, selection=selection, walk_forward=wf_inputs,
    regimes=regime_performance(net.returns, bench.returns, regime_labels(data.bench_prices["close"], data.eval_start),
                               cfg.periods_per_year, cfg.risk_free_rate),
)
counts = {s: sum(i.status == s for i in items) for s in ("pass", "caution", "fail", "not_run")}
ui.section("Evidence checklist",
           f"{counts['pass']} passed · {counts['caution']} caution · {counts['fail']} failed · {counts['not_run']} "
           "not run yet. There is deliberately no overall score: passing every check would not make a strategy "
           "profitable, and a check that has not been run is never counted as passed.")
ui.evidence_table(items)

with st.expander("Backtest assumptions", icon=":material/rule:"):
    assumptions = {**spec.config.describe(), "Benchmark": f"{data.benchmark_label} (same engine, same costs)",
                   "Final open position": "Marked to market at the final close (not sold)",
                   "Strategy parameters": ", ".join(f"{k}={v}" for k, v in spec.params.items()) or "none"}
    ui.kv_table(assumptions)

ui.disclaimer()
