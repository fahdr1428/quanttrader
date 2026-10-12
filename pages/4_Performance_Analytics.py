"""Performance Analytics — returns, risk, drawdowns, trades and costs in depth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.analytics import monthly_return_table, rolling_sharpe, rolling_volatility
from quantlab.metrics import compute_metrics, drawdown_episodes, period_returns
from quantlab.reporting import metrics_csv
from ui import charts
from ui import components as ui
from ui import theme as t
from quantlab.nulls import SHORT_WINDOW_SESSIONS, exposure_matched_benchmark
from quantlab.robustness import bootstrap_summary, regime_labels, regime_performance
from ui.state import active_run, bootstrap_for, permutation_for

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

# Only the open tab runs, so the bootstrap is computed only when Robustness is viewed.
tab_metrics, tab_risk, tab_periods, tab_trades, tab_costs, tab_robust = st.tabs(
    ["All metrics", "Risk over time", "Period returns", "Trades", "Costs: gross vs net", "Robustness"],
    key="pa_tabs", on_change="rerun",
)

with tab_metrics:
    if tab_metrics.open:
        groups = {
            "Returns": ["initial_value", "final_value", "liquidation_value", "total_return", "gross_total_return",
                        "cagr", "gross_cagr", "best_day", "worst_day", "best_month", "worst_month", "best_year",
                        "worst_year"],
            "Risk": ["volatility", "sharpe", "sortino", "max_drawdown", "max_dd_duration", "max_dd_duration_days",
                     "max_dd_peak", "max_dd_trough", "max_dd_recovery", "calmar"],
            "Statistical inference": ["sharpe", "sharpe_se", "sharpe_ci_low", "sharpe_ci_high", "psr",
                                      "min_track_record_years", "mean_excess_tstat", "observations"],
            "Tail risk": ["skewness", "excess_kurtosis", "var_95", "var_95_cf", "cvar_95", "tail_ratio", "worst_day"],
            "Versus benchmark": ["benchmark_total_return", "excess_return", "benchmark_cagr", "excess_cagr",
                                 "benchmark_max_drawdown", "benchmark_sharpe", "beta", "correlation", "tracking_error",
                                 "information_ratio", "alpha_annual", "alpha_tstat", "beta_tstat"],
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
            ui.metrics_html_table(report, bench_report if name in {"Returns", "Risk", "Statistical inference",
                                                                    "Tail risk"} else None, keys)
        st.caption("Confidence intervals, the probabilistic Sharpe ratio and the minimum track record assume serially "
                   "independent returns and ignore how many configurations were tried (see the deflated Sharpe on the "
                   "Overview). Win rate: share of closed round trips with positive net return. Average winner/loser: mean net "
                   "return of closed trades with positive / non-positive net return. Best/worst periods use calendar "
                   "days, months and years; the first and last periods may be partial. Trade-level figures describe "
                   "individual round trips and are not the same as portfolio-level returns, which also include time "
                   "spent in cash.")
        st.download_button("Download all metrics (CSV)", metrics_csv(report), "metrics.csv", "text/csv",
                           icon=":material/download:", on_click="ignore")

with tab_risk:
    if tab_risk.open:
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
    if tab_periods.open:
        ui.section("Monthly returns (strategy, net)",
                   "Calendar-month returns compounded from the daily equity curve; the last column is the calendar year.")
        table = monthly_return_table(net.equity)
        ui.plot(charts.monthly_heatmap(table), key="pa_heat")
        with st.expander("Monthly returns as a table"):
            ui.show_table((table * 100).round(2).reset_index(), column_config={"Year": st.column_config.TextColumn()})
        ui.section("Calendar-year returns", "Partial first and last years are included and labelled by year.")
        ui.plot(charts.annual_returns(period_returns(net.equity, "YE"), period_returns(bench.equity, "YE")), key="pa_ann")

with tab_trades:
    if tab_trades.open:
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
    if tab_costs.open:
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

with tab_robust:
    if tab_robust.open:
        ui.section("How much of this could be luck?",
                   "A stationary block bootstrap resamples the strategy's daily returns in random-length blocks "
                   "(keeping volatility clustering) thousands of times, together with the benchmark's returns on the "
                   "same days. The spread of outcomes shows how fragile the headline numbers are. It assumes the "
                   "future resembles this sample, so treat the intervals as a minimum level of uncertainty.")
        c1, c2, _ = st.columns([1, 1, 2])
        n_boot = c1.segmented_control("Resampled paths", [500, 1000, 2000], default=1000, key="rb_n") or 1000
        block = c2.number_input("Mean block length (sessions)", 5, 126, 21, 1, key="rb_block",
                                help="Average length of resampled blocks; longer blocks keep more autocorrelation.")
        if len(net.returns) < 20:
            st.info("At least 20 daily returns are needed for a bootstrap.")
        else:
            with st.spinner("Resampling…"):
                samples = bootstrap_for(net.returns, bench.returns, int(n_boot), int(block), cfg.risk_free_rate)
            observed = {"sharpe": report.value("sharpe"), "cagr": report.value("cagr"),
                        "volatility": report.value("volatility"), "max_drawdown": report.value("max_drawdown"),
                        "excess_cagr": report.value("excess_cagr"),
                        "sharpe_difference": (report.value("sharpe") - report.value("benchmark_sharpe"))
                        if report.value("sharpe") is not None and report.value("benchmark_sharpe") is not None else None}
            summary = bootstrap_summary(samples, observed)
            k1, k2, k3, k4 = st.columns(4)
            sh = samples["sharpe"].dropna()
            k1.metric("P(Sharpe > 0)", f"{(sh > 0).mean():.0%}" if len(sh) else "n/a", border=True,
                      help="Share of resampled paths with a positive Sharpe ratio.")
            k2.metric("P(beats benchmark CAGR)", f"{(samples['excess_cagr'] > 0).mean():.0%}", border=True,
                      help="Share of paths where the strategy's annualised return exceeds the benchmark's on the "
                           "same resampled days.")
            if len(sh):
                k3.metric("Sharpe, 90% interval", f"{sh.quantile(0.05):.2f} to {sh.quantile(0.95):.2f}", border=True)
            k4.metric("Bad-case drawdown (5th pct)", f"{samples['max_drawdown'].quantile(0.05):.1%}", border=True,
                      help="One path in twenty had a maximum drawdown at least this deep.")
            shown = summary.drop(columns=["key"]).copy()
            pct_rows = summary["key"].isin(["cagr", "volatility", "max_drawdown", "excess_cagr"])
            for col in ("Observed", "5th percentile", "Median", "95th percentile"):
                shown[col] = [
                    ("n/a" if v is None or pd.isna(v) else (f"{v:+.2%}" if is_pct else f"{v:.2f}"))
                    for v, is_pct in zip(summary[col], pct_rows)
                ]
            shown["P(> 0)"] = ["—" if v is None or pd.isna(v) else f"{v:.0%}" for v in summary["P(> 0)"]]
            ui.show_table(shown)
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                ui.plot(charts.bootstrap_histogram(samples, "sharpe", "Sharpe ratio", report.value("sharpe"),
                                                   "benchmark_sharpe", report.value("benchmark_sharpe")), key="rb_sh")
            with c2:
                ui.plot(charts.bootstrap_histogram(samples, "max_drawdown", "Maximum drawdown",
                                                   report.value("max_drawdown"), "benchmark_max_drawdown",
                                                   report.value("benchmark_max_drawdown"), pct=True), key="rb_dd")
            st.caption(f"{int(n_boot):,} paths, mean block {int(block)} sessions, fixed random seed "
                       "(results are reproducible).")

        ui.section("Timing skill, or just less market exposure?",
                   "A strategy that is often in cash shows smaller drawdowns and lower volatility than buy & hold "
                   "without any timing skill. Two honest comparisons: a portfolio that holds the strategy's average "
                   "exposure at all times (same engine, costs and sessions), and the same positions shifted in time.")
        matched = exposure_matched_benchmark(data.prices, net, cfg)
        matched_report = compute_metrics(matched)
        bench_report_only = compute_metrics(bench)
        rows = []
        for name, rep_ in ((net.strategy_label, report), (matched.strategy_label, matched_report),
                           (data.benchmark_label, bench_report_only)):
            rows.append({"Portfolio": name, "Average exposure": rep_.value("avg_exposure"),
                         "CAGR": rep_.value("cagr"), "Volatility": rep_.value("volatility"),
                         "Sharpe": rep_.value("sharpe"), "Max drawdown": rep_.value("max_drawdown")})
        cmp = pd.DataFrame(rows)
        for col in ("Average exposure", "CAGR", "Volatility", "Max drawdown"):
            cmp[col] = cmp[col].astype(float) * 100
        ui.show_table(cmp, column_config={
            **{c: st.column_config.NumberColumn(format="%.1f%%") for c in
               ("Average exposure", "CAGR", "Volatility", "Max drawdown")},
            "Sharpe": st.column_config.NumberColumn(format="%.2f")})
        st.caption("If the strategy does not beat the exposure-matched portfolio on return and drawdown, its "
                   "advantage over buy & hold comes from holding less risk, not from timing. Scaling exposure does "
                   "not change a Sharpe ratio much, so use the permutation test below for risk-adjusted skill.")
        if st.button("Run timing permutation test (200 shifted paths)", key="perm_run",
                     icon=":material/shuffle:"):
            st.session_state["perm_requested"] = data.sample_key + spec.params_json() + cfg.model_dump_json()
        if st.session_state.get("perm_requested") == data.sample_key + spec.params_json() + cfg.model_dump_json():
            with st.spinner("Re-simulating shifted positions…"):
                perm = permutation_for(data, spec, run)
            p1, p2, p3 = st.columns(3)
            p1.metric("Timing p-value", "n/a" if perm.p_value is None else f"{perm.p_value:.3f}", border=True,
                      help="Share of time-shifted copies of the strategy's own positions that did at least as well "
                           "(Sharpe). Small values mean the timing itself mattered.")
            p2.metric("Actual Sharpe", "n/a" if perm.observed is None else f"{perm.observed:.2f}", border=True)
            finite = perm.null[np.isfinite(perm.null)]
            p3.metric("Median shifted Sharpe", f"{np.median(finite):.2f}" if len(finite) else "n/a", border=True)
            if len(finite):
                ui.plot(charts.null_distribution(perm.null, perm.observed, "Sharpe ratio of shifted positions"),
                        key="perm_hist")
            notes = [perm.note] if perm.note else []
            if len(net.daily) < SHORT_WINDOW_SESSIONS:
                notes.append(f"This window is shorter than {SHORT_WINDOW_SESSIONS} sessions; on short windows "
                             "the test rejects too often on pure noise, so treat a small p-value with extra caution.")
            st.caption("Circular shifts keep the exposure, turnover and holding periods but break the link "
                       "between positions and returns. The test's power is modest: a genuine but weak timing "
                       "effect often goes undetected. " + " ".join(notes))

        ui.section("Performance by market regime",
                   "Each day's return is grouped by the benchmark's state at the previous close: above or below "
                   "its 200-session average (trend), and 63-session volatility above or below its median for the "
                   "sample (volatility — the median uses hindsight, so this split is descriptive only).")
        regimes = regime_labels(data.bench_prices["close"])
        table = regime_performance(net.returns, bench.returns, regimes, cfg.periods_per_year, cfg.risk_free_rate)
        if table.empty:
            st.info("Not enough history to classify regimes (the trend label needs 200 sessions).")
        else:
            ui.plot(charts.regime_bars(table), key="rb_regime")
            fmt = table.copy()
            for col in ("Share of sample", "Strategy return (ann.)", "Benchmark return (ann.)", "Strategy volatility",
                        "Benchmark volatility", "Strategy up-days"):
                fmt[col] = fmt[col] * 100
            ui.show_table(fmt, column_config={
                **{c: st.column_config.NumberColumn(format="%.1f%%") for c in
                   ("Share of sample", "Strategy return (ann.)", "Benchmark return (ann.)", "Strategy volatility",
                    "Benchmark volatility", "Strategy up-days")},
                "Strategy Sharpe": st.column_config.NumberColumn(format="%.2f"),
                "Benchmark Sharpe": st.column_config.NumberColumn(format="%.2f"),
            })
            st.caption("A strategy whose edge comes entirely from one regime depends on that regime recurring.")

ui.disclaimer()
