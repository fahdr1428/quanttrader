"""Research Overview — the selected strategy against its benchmark at a glance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantlab.metrics import compute_metrics
from quantlab.research import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from ui import charts
from ui import components as ui
from ui.state import active_run, trials_for

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
           "The aqua line shows the strategy with zero costs, so the gap to blue is the cost drag.")
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
        ("Sortino ratio", "sortino", None),
        ("Maximum drawdown", "max_drawdown", "benchmark_max_drawdown"),
        ("Longest drawdown (sessions)", "max_dd_duration", None),
        ("Beta to benchmark", "beta", None),
        ("Correlation with benchmark", "correlation", None),
        ("Tracking error", "tracking_error", None),
    ]
    table = []
    for label, key, bkey in rows:
        m = report.get(key)
        bm = report.get(bkey) if bkey else bench_only.get(key)
        table.append({
            "Indicator": label,
            "Strategy": (m.formatted() + ("" if m.reliable else " *")) if m else "n/a",
            "Benchmark": bm.formatted() if bm and key not in {"beta", "correlation", "tracking_error"} else "—",
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
psr = probabilistic_sharpe_ratio(net.returns)
c2.metric("Probabilistic Sharpe (vs 0)", "n/a" if psr is None else f"{psr:.0%}", border=True,
          help="Probability that the true Sharpe ratio is above zero given sample length, skew and fat tails "
               "(Bailey & López de Prado). Ignores how many configurations were tried.")
pp = [t_["sharpe_pp"] for t_ in trials if t_.get("sharpe_pp") is not None]
dsr = deflated_sharpe_ratio(net.returns, pp) if len(pp) >= 2 else None
c3.metric("Deflated Sharpe (this session's trials)", "n/a" if dsr is None else f"{dsr:.0%}", border=True,
          help="Probabilistic Sharpe measured against the best Sharpe you would expect from that many "
               "configurations with no real edge. Needs at least two trials. Heuristic: assumes independent trials.")
st.caption(f"Parameter selection: **{spec.parameter_selection}**")

with st.expander("Backtest assumptions", icon=":material/rule:"):
    assumptions = {**spec.config.describe(), "Benchmark": f"{data.benchmark_label} (same engine, same costs)",
                   "Final open position": "Marked to market at the final close (not sold)",
                   "Strategy parameters": ", ".join(f"{k}={v}" for k, v in spec.params.items()) or "none"}
    ui.kv_table(assumptions)

ui.disclaimer()
