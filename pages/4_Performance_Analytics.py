"""Performance Analytics — returns, risk, drawdowns, trades and costs in depth."""

from __future__ import annotations

import streamlit as st

from quantlab.analytics import monthly_return_table, rolling_sharpe, rolling_volatility
from quantlab.metrics import compute_metrics, drawdown_episodes, period_returns
from quantlab.reporting import metrics_csv
from ui import charts
from ui import components as ui
from ui import theme as t
from ui.state import active_run

data = ui.require_data()
spec = ui.require_spec()
run, bench = active_run(data, spec)
net, report = run.net, run.report
bench_report = compute_metrics(bench)
cfg = spec.config

ui.page_header(
    "Performance analytics",
    "Risk & return diagnostics",
    "Every figure is computed from the daily equity series of the backtest shown. Metrics that rest on too "
    "few observations are marked with an asterisk and explained.",
)
ui.context_strip(data, label=net.strategy_label)
ui.kpi_row(report, ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown"])
ui.unreliable_footnote(report, ["cagr", "volatility", "sharpe", "sortino"])

tab_metrics, tab_risk, tab_periods, tab_trades, tab_costs = st.tabs(
    ["All metrics", "Risk over time", "Period returns", "Trades", "Costs: gross vs net"]
)

with tab_metrics:
    groups = {
        "Returns": ["initial_value", "final_value", "liquidation_value", "total_return", "gross_total_return",
                    "cagr", "gross_cagr", "best_day", "worst_day", "best_month", "worst_month", "best_year",
                    "worst_year"],
        "Risk": ["volatility", "sharpe", "sortino", "max_drawdown", "max_dd_duration", "max_dd_duration_days",
                 "max_dd_peak", "max_dd_trough", "max_dd_recovery", "calmar"],
        "Versus benchmark": ["benchmark_total_return", "excess_return", "benchmark_cagr", "excess_cagr",
                             "benchmark_max_drawdown", "benchmark_sharpe", "beta", "correlation", "tracking_error",
                             "information_ratio"],
        "Activity & costs": ["fills", "position_changes", "round_trips", "open_trades", "time_invested",
                             "avg_exposure", "turnover", "traded_notional", "total_fees", "total_slippage",
                             "cost_pct_capital", "cost_drag"],
        "Trade statistics (closed trades)": ["avg_trade_return", "win_rate", "avg_winner", "avg_loser",
                                             "profit_factor", "avg_holding", "best_trade", "worst_trade"],
        "Sample": ["observations"],
    }
    for name, keys in groups.items():
        if not any(report.get(k) for k in keys):
            continue
        ui.section(name)
        ui.metrics_html_table(report, bench_report if name in {"Returns", "Risk"} else None, keys)
    st.caption("Win rate: share of closed round trips with positive net return. Average winner/loser: mean net "
               "return of closed trades with positive / non-positive net return. Best/worst periods use calendar "
               "days, months and years; the first and last periods may be partial. Trade-level figures describe "
               "individual round trips and are not the same as portfolio-level returns, which also include time "
               "spent in cash.")
    st.download_button("Download all metrics (CSV)", metrics_csv(report), "metrics.csv", "text/csv",
                       icon=":material/download:", on_click="ignore")

with tab_risk:
    ui.section("Drawdowns")
    ui.plot(charts.drawdowns(net, bench), key="pa_dd")
    episodes = drawdown_episodes(net.equity)
    if len(episodes):
        top = episodes.sort_values("depth").head(5).copy()
        top["depth"] = top["depth"] * 100
        ui.show_table(top, column_config={
            "peak": st.column_config.DateColumn("Peak"), "trough": st.column_config.DateColumn("Trough"),
            "recovery": st.column_config.DateColumn("Recovered"),
            "depth": st.column_config.NumberColumn("Depth", format="%.2f%%"),
            "sessions": st.column_config.NumberColumn("Sessions peak→recovery"),
            "calendar_days": st.column_config.NumberColumn("Calendar days"),
            "sessions_to_trough": st.column_config.NumberColumn("Sessions to trough"),
            "ongoing": st.column_config.CheckboxColumn("Still open"),
        })
        st.caption("Five deepest drawdown episodes of the strategy.")

    c1, c2 = st.columns(2, gap="medium")
    n_obs = len(net.returns)
    with c1:
        ui.section("Rolling volatility")
        vol_window = st.segmented_control("Window (sessions)", [21, 63, 126], default=63, key="pa_vol_w") or 63
        if n_obs < vol_window * 2:
            st.info("Not enough observations for a meaningful rolling window.")
        else:
            ui.plot(charts.rolling_metric({
                "Strategy": (rolling_volatility(net.returns, vol_window, cfg.periods_per_year), t.STRATEGY),
                "Benchmark": (rolling_volatility(bench.returns, vol_window, cfg.periods_per_year), t.BENCHMARK),
            }, "Annualised volatility", ".0%"), key="pa_rvol")
    with c2:
        ui.section("Rolling Sharpe ratio")
        sh_window = st.segmented_control("Window (sessions)", [126, 252], default=252, key="pa_sh_w") or 252
        if n_obs < sh_window * 2:
            st.info(f"A rolling {sh_window}-session Sharpe needs at least {sh_window * 2} observations to show "
                    f"more than one independent window; this sample has {n_obs}.")
        else:
            ui.plot(charts.rolling_metric({
                "Strategy": (rolling_sharpe(net.returns, sh_window, cfg.risk_free_rate, cfg.periods_per_year), t.STRATEGY),
                "Benchmark": (rolling_sharpe(bench.returns, sh_window, cfg.risk_free_rate, cfg.periods_per_year), t.BENCHMARK),
            }, "Rolling Sharpe ratio", None, zero_line=True), key="pa_rsh")
            st.caption("Windows with zero volatility (fully in cash) are left blank rather than shown as infinite.")

    ui.section("Position exposure")
    ui.plot(charts.exposure(net), key="pa_exp")

with tab_periods:
    ui.section("Monthly returns (strategy, net)",
               "Calendar-month returns compounded from the daily equity curve; the last column is the calendar year.")
    table = monthly_return_table(net.equity)
    ui.plot(charts.monthly_heatmap(table), key="pa_heat")
    with st.expander("Monthly returns as a table"):
        ui.show_table((table * 100).round(2).reset_index(), column_config={"Year": st.column_config.TextColumn()})
    ui.section("Calendar-year returns", "Partial first and last years are included and labelled by year.")
    ui.plot(charts.annual_returns(period_returns(net.equity, "YE"), period_returns(bench.equity, "YE")), key="pa_ann")

with tab_trades:
    closed = net.trades[net.trades["status"] == "closed"] if len(net.trades) else net.trades
    if closed.empty:
        st.info("No closed round-trip trades in this sample, so there is no trade-return distribution. "
                "(Buy-and-hold has a single open position.)", icon=":material/info:")
    else:
        ui.kpi_row(report, ["round_trips", "win_rate", "avg_trade_return", "avg_winner", "avg_loser", "profit_factor"])
        ui.unreliable_footnote(report, ["win_rate", "avg_trade_return", "avg_winner", "avg_loser", "profit_factor"])
        ui.plot(charts.trade_distribution(net.trades), key="pa_tdist")
        hold = closed["holding_sessions"]
        st.caption(f"Holding period: median {hold.median():.0f} sessions, range {hold.min():.0f}–{hold.max():.0f}.")

with tab_costs:
    ui.section("Gross vs net performance",
               "Gross = identical signals simulated with zero fees and zero slippage. The difference is the cost "
               "drag. Signal timing is unaffected by costs.")
    ui.plot(charts.equity_vs_benchmark(net, bench, run.gross, strategy_name="Strategy (net)",
                                       bench_name=data.benchmark_label), key="pa_gross")
    ui.kpi_row(report, ["gross_total_return", "total_return", "cost_drag", "total_fees", "total_slippage",
                        "turnover"])
    fees, slip = report.value("total_fees"), report.value("total_slippage")
    st.markdown(
        f"Over {int(report.value('fills'))} fills the strategy paid **${fees:,.2f}** in fees and an estimated "
        f"**${slip:,.2f}** in slippage — {report.value('cost_pct_capital'):.2%} of starting capital. Slippage is a "
        "fixed-percentage estimate; real execution costs depend on liquidity, order size and volatility."
    )

ui.disclaimer()
