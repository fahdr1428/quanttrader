"""Strategy Lab — understand, compare and stress-test strategies."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.analytics import (
    MAX_SENSITIVITY_COMBINATIONS,
    compare_strategies,
    parameter_sensitivity,
    selection_tests,
)
from quantlab import ml
from quantlab.research import deflated_sharpe_ratio
from quantlab.strategies import STRATEGIES
from ui import charts
from ui import components as ui
from ui.state import (
    active_run,
    benchmark_for,
    cost_sensitivity_for,
    ml_fit_for,
    register_trials_bulk,
    run_for,
    strategy_params_from_state,
    trials_for,
)

data = ui.require_data()
spec = ui.require_spec()

ui.page_header(
    "Strategy lab",
    "Strategy research",
    "Read how each strategy works and when it fails, compare all of them on identical terms, and test how "
    "fragile the active strategy is to its parameters and to trading costs.",
)
ui.context_strip(data)

# Only the open tab runs: hidden tabs would otherwise backtest every strategy (and
# count each one as a trial) on every interaction.
guide_tab, compare_tab, sens_tab, cost_tab = st.tabs(
    ["Strategy guide", "Compare strategies", "Parameter sensitivity", "Cost sensitivity"],
    key="lab_tabs", on_change="rerun",
)

# --------------------------------------------------------------------------- #
# Guide
# --------------------------------------------------------------------------- #
with guide_tab:
    if guide_tab.open:
        keys = list(STRATEGIES)
        key = st.selectbox("Strategy", keys, index=keys.index(spec.strategy_key), format_func=lambda k: STRATEGIES[k].name,
                           key="lab_guide_strategy")
        cls = STRATEGIES[key]
        doc = cls.doc
        st.markdown(f"#### {cls.name}  ·  <span style='color:#8a8f98;font-size:0.9rem'>{cls.family}</span>",
                    unsafe_allow_html=True)
        st.markdown(doc.summary)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with st.container(border=True):
                st.markdown("**Entry rule**")
                st.markdown(doc.entry)
                st.markdown("**Exit rule**")
                st.markdown(doc.exit)
                st.markdown("**Key assumptions**")
                st.markdown("\n".join(f"- {a}" for a in doc.assumptions))
        with c2:
            with st.container(border=True):
                st.markdown("**When it may fail**")
                st.markdown("\n".join(f"- {f}" for f in doc.failure_modes))
                if doc.references:
                    st.markdown("**Reference**")
                    st.markdown("\n".join(f"- {r}" for r in doc.references))
                st.markdown("**Timing**")
                st.markdown("Signals use closing prices up to and including session *t*; orders fill at the open of "
                            "session *t + 1*, after slippage and fees.")

        current = strategy_params_from_state(key)
        if cls.param_specs:
            st.markdown("**Parameters**")
            rows = []
            for p in cls.param_specs:
                rng = ", ".join(map(str, p.choices)) if p.kind == "choice" else f"{p.min:g} – {p.max:g}"
                rows.append({"Parameter": p.label, "Default": str(p.default), "Current": str(current.get(p.name)),
                             "Allowed": rng, "Meaning": p.help})
            ui.show_table(pd.DataFrame(rows))
        else:
            st.caption("No parameters.")

        run, bench = run_for(data, key, cls.validate_params(current) if cls.is_valid(current) else cls.defaults(),
                             spec.config)
        st.markdown("**Result on the current sample (net of costs)**")
        exposure_key = "avg_exposure" if cls.continuous_exposure else "time_invested"
        ui.kpi_row(run.report, ["total_return", "sharpe", "max_drawdown", "fills", exposure_key, "cost_drag"])
        selection = (
            "Pre-specified defaults (fixed before viewing this sample)" if run.strategy.uses_default_parameters()
            else "Custom parameters — unless they were fixed in advance, treat these results as in-sample."
        )
        st.caption(f"Parameter selection: **{selection}**")

        if key == "ml_logistic":
            ui.section("Model diagnostics (out-of-sample)",
                       "Every prediction below was made before its outcome was known, so these scores are honest. "
                       "Compare the hit rate with the 'always predict up' base rate — beating a coin flip is easy in "
                       "a rising market.")
            model = run.strategy
            hist = data.prices.loc[: data.eval_end]
            fit = ml_fit_for(data, model.params)
            window = run.net.signals.index
            horizon = model.params["horizon"]
            report = ml.classification_report(fit.probabilities.loc[window], hist["close"], horizon, hist["open"])
            traded = ml.classification_report(run.net.signals["prob_up"], hist["close"], horizon, hist["open"])
            st.caption(f"Scored against the tradeable outcome: from the next open to the close {horizon} sessions "
                       "later. Scores below are for the model's raw probabilities"
                       + (f"; the strategy trades on a {model.params['smoothing']}-session smoothed version, whose "
                          f"hit rate is {traded['accuracy']:.1%} and AUC "
                          f"{'n/a' if traded.get('auc') is None else format(traded['auc'], '.3f')}."
                          if model.params["smoothing"] > 1 and traded.get("n") else "."))
            if report.get("n", 0) == 0:
                st.info("No scored predictions yet: outcomes are not known for the evaluation window.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Hit rate", f"{report['accuracy']:.1%}", border=True,
                          delta=f"base rate {max(report['base_rate'], 1 - report['base_rate']):.1%}", delta_color="off",
                          delta_arrow="off", help="Share of predictions on the right side of 50%, versus always "
                                                  "predicting the more common outcome.")
                auc = report["auc"]
                c2.metric("AUC", "n/a" if auc is None else f"{auc:.3f}", border=True,
                          help="Probability that a randomly chosen 'up' period got a higher prediction than a "
                               "randomly chosen 'down' period. 0.5 = no skill.")
                c3.metric("Brier score", f"{report['brier']:.4f}", border=True,
                          delta=f"climatology {report['brier_climatology']:.4f}", delta_color="off", delta_arrow="off",
                          help="Mean squared error of the probabilities; lower is better. 'Climatology' always "
                               "predicts the base rate — a skilful model must beat it.")
                c4.metric("Effective sample", f"≈{report['effective_n']:,}", border=True,
                          help=f"{report['n']:,} scored predictions, but overlapping {model.params['horizon']}-session "
                               "outcomes mean only about this many are independent.")
                edge = report["accuracy_edge"]
                if auc is not None and auc < 0.53 and edge <= 0.01:
                    st.warning("No meaningful out-of-sample skill: the model is not reliably better than the base "
                               "rate on this sample.", icon=":material/insights:")
                elif report["effective_n"] < 60:
                    st.info("Too few independent outcomes to judge skill reliably.", icon=":material/info:")
            fits = fit.fits[~fit.fits["skipped"].astype(bool)] if "skipped" in fit.fits else fit.fits
            n_skipped = len(fit.fits) - len(fits)
            if n_skipped:
                st.caption(f"{n_skipped} scheduled refit(s) were skipped because the training window held only one "
                           "outcome class; the previous model kept predicting until the next successful refit.")
            if len(fits):
                last = fits.iloc[-1]
                weights = {name: last[f"w_{name}"] for name in ml.FEATURE_NAMES}
                c1, c2 = st.columns([3, 2], gap="medium")
                with c1:
                    st.markdown(f"**Latest model** (refit {last['fit_date']:%Y-%m-%d} on {int(last['n_train'])} samples, "
                                f"{last['train_start']:%Y-%m-%d} → {last['train_end']:%Y-%m-%d})")
                    ui.plot(charts.coefficient_bars(weights, ml.FEATURE_SHORT), key="ml_coef")
                    with st.expander("Feature definitions"):
                        ui.kv_table({ml.FEATURE_SHORT[n]: ml.FEATURE_DESCRIPTIONS[n] for n in ml.FEATURE_NAMES})
                with c2:
                    st.markdown("**Coefficient stability across refits**")
                    coefs = fits[[f"w_{n}" for n in ml.FEATURE_NAMES]]
                    latest_sign = np.sign(coefs.iloc[-1])
                    stability = pd.DataFrame({
                        "Feature": [ml.FEATURE_SHORT[n] for n in ml.FEATURE_NAMES],
                        "Mean": coefs.mean().to_numpy(),
                        "Same sign as latest": (np.sign(coefs) == latest_sign).mean().to_numpy(),
                    })
                    ui.show_table(stability.sort_values("Mean", key=abs, ascending=False), column_config={
                        "Mean": st.column_config.NumberColumn(format="%+.3f"),
                        "Same sign as latest": st.column_config.NumberColumn(format="percent")})
                    st.caption(f"{len(fits)} refits. Features whose sign keeps flipping carry no stable "
                               "information — a warning sign of fitting noise.")

        if key == "adaptive_ensemble":
            ui.section("Model diagnostics")
            sig = run.net.signals.dropna(subset=["target"])
            if len(sig):
                c1, c2 = st.columns([2, 3], gap="medium")
                with c1:
                    ui.kv_table({
                        "Average trend score": f"{sig['trend_score'].mean():.0%}",
                        "Average volatility scale": f"{sig['vol_scale'].mean():.2f}",
                        "Realised volatility (median)": f"{sig['realised_vol'].median():.1%}",
                        "Volatility target": f"{run.strategy.params['target_vol']:.0%}",
                        "Sessions with pullback boost": f"{sig['pullback'].mean():.1%}",
                        "Average target weight": f"{sig['target'].mean():.0%}",
                    })
                with c2:
                    ui.plot(charts.weight_distribution(sig["target"]), key="ens_dist")

# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
with compare_tab:
    if compare_tab.open:
        chosen = st.multiselect("Strategies to compare", list(STRATEGIES), default=list(STRATEGIES),
                                format_func=lambda k: STRATEGIES[k].name, key="lab_compare")
        st.caption("Each strategy uses the parameters last set for it in the sidebar (defaults if never changed). "
                   "All runs share the same sessions, capital, costs, timing and benchmark. Every strategy shown "
                   "here counts as a configuration tried on this sample.")
        if not chosen:
            st.info("Select at least one strategy.")
        else:
            runs = []
            with st.spinner("Running backtests…"):
                for k in chosen:
                    params = strategy_params_from_state(k)
                    if not STRATEGIES[k].is_valid(params):
                        st.warning(f"{STRATEGIES[k].name}: invalid parameters in the sidebar; using defaults.")
                        params = STRATEGIES[k].defaults()
                    runs.append(run_for(data, k, STRATEGIES[k].validate_params(params), spec.config)[0])
            bench = run_for(data, spec.strategy_key, spec.params, spec.config, register=False)[1]
            table = compare_strategies(runs)

            exposures = table["Average exposure"].astype(float)
            if exposures.max() - exposures.min() > 0.2:
                st.info(
                    f"**Exposure differs widely** (average exposure {exposures.min():.0%} to {exposures.max():.0%}). "
                    "A strategy that sits in cash more often will mechanically show lower volatility and usually "
                    "lower return; compare risk-adjusted measures and drawdowns, not total return alone.",
                    icon=":material/balance:",
                )

            pct_cols = ["Net total return", "Gross total return", "CAGR (net)", "Volatility", "Max drawdown",
                        "Time invested", "Average exposure", "Turnover (ann.)", "Cost drag", "Excess vs benchmark"]
            display = table.drop(columns=["key"]).copy()
            for col in pct_cols:
                display[col] = display[col].astype(float) * 100
            column_config = {c: st.column_config.NumberColumn(c, format="%.2f%%") for c in pct_cols}
            column_config.update({
                "Sharpe": st.column_config.NumberColumn(format="%.2f", help=ui.GLOSSARY["sharpe"]),
                "Sortino": st.column_config.NumberColumn(format="%.2f", help=ui.GLOSSARY["sortino"]),
                "Calmar": st.column_config.NumberColumn(format="%.2f"),
                "Fees ($)": st.column_config.NumberColumn(format="dollar"),
                "Slippage ($)": st.column_config.NumberColumn(format="dollar"),
                "Fills": st.column_config.NumberColumn(format="%d"),
                "Round trips": st.column_config.NumberColumn(format="%d"),
            })
            ui.show_table(display, column_config=column_config)
            st.caption("Rows are in a fixed order, not ranked. Click a column header to sort by any measure — "
                       "no single number decides which strategy is 'best'.")
            st.download_button("Download comparison (CSV)", table.to_csv(index=False), "strategy_comparison.csv",
                               "text/csv", icon=":material/download:", on_click="ignore")

            ui.section("Equity curves (net of costs)")
            has_bh = any(r.strategy.key == "buy_and_hold" for r in runs)
            bench_name = None if (data.same_asset_benchmark and has_bh) else data.benchmark_label
            ui.plot(charts.comparison_equity(runs, bench, bench_name=bench_name), key="cmp_eq")
            if bench_name is None:
                st.caption("The benchmark is buy & hold of the same asset, so it is the Buy & Hold line.")
            c1, c2 = st.columns([3, 2], gap="medium")
            with c1:
                ui.section("Risk vs return", "Bubble size = average market exposure (holdings ÷ equity).")
                ui.plot(charts.risk_return_scatter(table), key="cmp_scatter")
            with c2:
                ui.section("Risk-adjusted return")
                ui.plot(charts.comparison_bars(table, "Sharpe", ".2f", height=200, title="Sharpe ratio"), key="cmp_sharpe")
                ui.plot(charts.comparison_bars(table, "Max drawdown", ".1%", height=200, title="Maximum drawdown"),
                        key="cmp_mdd")
            c3, c4, c5 = st.columns(3, gap="medium")
            with c3:
                ui.plot(charts.comparison_bars(table, "Volatility", ".1%", height=230, title="Volatility"), key="cmp_vol")
            with c4:
                ui.plot(charts.comparison_bars(table, "Turnover (ann.)", ".0%", height=230, title="Annual turnover"),
                        key="cmp_turn")
            with c5:
                costs = table.assign(**{"Total costs": table["Fees ($)"] + table["Slippage ($)"]})
                ui.plot(charts.comparison_bars(costs, "Total costs", ",.0f", height=230, title="Total costs ($)"),
                        key="cmp_cost")

# --------------------------------------------------------------------------- #
# Parameter sensitivity
# --------------------------------------------------------------------------- #
with sens_tab:
    if sens_tab.open:
        cls = STRATEGIES[spec.strategy_key]
        numeric = [p for p in cls.param_specs if p.kind != "choice"]
        st.warning(
            "**This is in-sample analysis.** Scanning parameters on the same data you evaluate on is the classic "
            "route to overfitting: with enough combinations, something always looks good. Use it to judge "
            "*fragility* (does performance collapse if a parameter moves a little?), and use the Walk-Forward page "
            "for out-of-sample evidence. Every combination is counted as a trial.",
            icon=":material/warning:",
        )
        if not numeric:
            st.info(f"{cls.name} has no numeric parameters to vary.")
        else:
            names = [p.name for p in numeric]
            labels = {p.name: p.label for p in numeric}
            c1, c2, c3 = st.columns(3)
            x_param = c1.selectbox("Vary (x-axis)", names, format_func=labels.get, key="sens_x")
            y_options = ["(none)"] + [n for n in names if n != x_param]
            y_param = c2.selectbox("And (y-axis)", y_options, index=1 if len(y_options) > 1 else 0,
                                   format_func=lambda n: labels.get(n, n), key="sens_y")
            metric_options = {"sharpe": "Sharpe ratio", "cagr": "Annualised return", "total_return": "Total return",
                              "max_drawdown": "Maximum drawdown"}
            metric = c3.selectbox("Metric", list(metric_options), format_func=metric_options.get, key="sens_metric")

            def _values_input(col, name):
                p = next(s for s in numeric if s.name == name)
                default = list(cls.search_space.get(name, ())) or [p.default]
                if spec.params[name] not in default:
                    default = sorted(set(default) | {spec.params[name]})
                text = col.text_input(f"{p.label} values (comma-separated)", ", ".join(f"{v:g}" for v in default),
                                      key=f"sens_vals_{spec.strategy_key}_{name}")
                values = []
                for tok in text.split(","):
                    tok = tok.strip()
                    if tok:
                        try:
                            values.append(p.coerce(float(tok)))
                        except Exception as exc:  # noqa: BLE001
                            col.error(str(exc))
                return sorted(set(values))

            v1, v2 = st.columns(2)
            grid = {x_param: _values_input(v1, x_param)}
            if y_param != "(none)":
                grid[y_param] = _values_input(v2, y_param)
            n_combos = 1
            for vals in grid.values():
                n_combos *= max(len(vals), 1)
            st.caption(f"{n_combos} combinations (limit {MAX_SENSITIVITY_COMBINATIONS}).")
            signature = json.dumps({"sample": data.sample_key, "key": spec.strategy_key, "grid": grid,
                                    "base": {k: v for k, v in spec.params.items() if k not in grid},
                                    "config": spec.config.model_dump()}, sort_keys=True, default=str)
            if st.button("Run sensitivity analysis", type="primary", key="sens_run",
                         disabled=n_combos > MAX_SENSITIVITY_COMBINATIONS or any(not v for v in grid.values())):
                with st.spinner(f"Running {n_combos} backtests…"):
                    frame, matrix = parameter_sensitivity(data.prices, spec.strategy_key, spec.params, grid,
                                                          spec.config, data.eval_start, data.eval_end,
                                                          return_matrix=True)
                    selection = selection_tests(matrix, benchmark_for(data, spec.config).returns, spec.config)
                st.session_state["sens_result"] = {"frame": frame, "grid": grid, "key": spec.strategy_key,
                                                   "sample": data.sample_key, "metric": metric,
                                                   "selection": selection,
                                                   "signature": signature}
                register_trials_bulk(data, spec.strategy_key, spec.config, [
                    {"params": {**spec.params, **{k: r[k] for k in grid}}, "label": f"{cls.name} grid point",
                     "sharpe": r["sharpe"] if pd.notna(r.get("sharpe")) else None}
                    for _, r in frame[frame["valid"]].iterrows()
                ])
            result = st.session_state.get("sens_result")
            if result and result["key"] == spec.strategy_key and result["sample"] == data.sample_key:
                if result.get("signature") != signature:
                    st.info("Settings have changed since these results were produced (grid values, other "
                            "parameters or costs). Re-run to update them.", icon=":material/sync_problem:")
                frame = result["frame"]
                label = metric_options[metric]
                gkeys = list(result["grid"])
                if not frame["valid"].any():
                    st.info("No combination on this grid is valid (for example, the short window must be shorter "
                            "than the long window), so there is nothing to chart. Adjust the values above.")
                elif len(gkeys) == 2:
                    current = (spec.params[gkeys[0]], spec.params[gkeys[1]])
                    ui.plot(charts.sensitivity_heatmap(frame, gkeys[0], gkeys[1], metric, label, current,
                                                       x_label=labels[gkeys[0]], y_label=labels[gkeys[1]]), key="sens_heat")
                    st.caption("Outlined cell = current sidebar parameters (if it lies on the grid). Blank cells are "
                               "invalid combinations (e.g. short window ≥ long window).")
                else:
                    ui.plot(charts.sensitivity_line(frame, gkeys[0], metric, label, spec.params[gkeys[0]],
                                                    x_label=labels[gkeys[0]]), key="sens_line")
                valid = frame[frame["valid"]]
                if len(valid) and valid["sharpe"].notna().any():
                    spread = valid["sharpe"].max() - valid["sharpe"].min()
                    st.markdown(
                        f"Sharpe ranges from **{valid['sharpe'].min():.2f}** to **{valid['sharpe'].max():.2f}** across "
                        f"{len(valid)} valid combinations (spread {spread:.2f}). A narrow, smooth plateau is more "
                        "credible than a single sharp peak."
                    )
                run, _ = active_run(data, spec)
                pp = [t_["sharpe_pp"] for t_ in trials_for(data) if t_.get("sharpe_pp") is not None]
                dsr = (deflated_sharpe_ratio(run.net.returns, pp, spec.config.risk_free_rate, spec.config.periods_per_year)
                       if len(pp) >= 2 else None)
                if dsr is not None:
                    st.metric("Deflated Sharpe of active configuration", f"{dsr:.0%}", border=True,
                              help="Probability that the active configuration's Sharpe exceeds what the best of "
                                   f"{len(pp)} trials would show by chance. Values near or below 50% mean the result "
                                   "is consistent with selection luck.")
                selection = result.get("selection")
                if selection is not None:
                    ui.section("Was the best grid point just lucky?",
                               "Tests that account for every combination on this grid at once. They use only the "
                               "combinations shown here; anything tried earlier or elsewhere is not included.")
                    s1, s2, s3 = st.columns(3)
                    pbo = selection.pbo
                    s1.metric("Probability of backtest overfitting", "n/a" if pbo is None else f"{pbo.pbo:.0%}",
                              border=True,
                              help="Share of in-sample/out-of-sample splits (combinatorially symmetric cross-"
                                   "validation, Bailey et al. 2017) in which the best in-sample combination ranks "
                                   "in the bottom half out of sample. Near 50% or above: choosing among these "
                                   "combinations is not informative. It does not say whether any edge exists.")
                    spa_b, spa_c = selection.spa_vs_benchmark, selection.spa_vs_cash
                    s2.metric("Best vs benchmark: SPA p-value", f"{spa_b.p_value:.3f}", border=True,
                              help="Hansen's test for superior predictive ability: could the best of these "
                                   f"{selection.n_configs} combinations beat {data.benchmark_label} by luck alone? "
                                   "Small values (e.g. below 0.05) mean the outperformance survives the search.")
                    s3.metric("Best vs cash: SPA p-value", f"{spa_c.p_value:.3f}", border=True,
                              help="The same test against holding cash at the risk-free rate.")
                    with st.expander("Per-combination results and details"):
                        rw = selection.stepdown_vs_benchmark
                        detail = pd.DataFrame({
                            "Combination": rw.adjusted_p_values.index,
                            "Mean excess return vs benchmark (ann.)":
                                spa_b.mean_differentials.reindex(rw.adjusted_p_values.index).to_numpy()
                                * spec.config.periods_per_year * 100,
                            "t-statistic": rw.t_statistics.to_numpy(),
                            "Adjusted p-value (Romano-Wolf)": rw.adjusted_p_values.to_numpy(),
                        })
                        ui.show_table(detail, column_config={
                            "Mean excess return vs benchmark (ann.)": st.column_config.NumberColumn(format="%+.2f%%"),
                            "t-statistic": st.column_config.NumberColumn(format="%.2f"),
                            "Adjusted p-value (Romano-Wolf)": st.column_config.NumberColumn(format="%.3f")})
                        if pbo is not None:
                            st.caption(
                                f"PBO used {selection.n_blocks} blocks ({pbo.n_splits:,} splits). In "
                                f"{pbo.prob_oos_loss:.0%} of splits the in-sample winner lost money out of sample"
                                + (f"; out-of-sample Sharpe moved {pbo.degradation_slope:+.2f} per unit of in-sample "
                                   "Sharpe." if pbo.degradation_slope is not None else "."))
                        st.caption("Romano-Wolf p-values control the chance of any false discovery across the "
                                   "grid. All tests use a stationary block bootstrap (mean block "
                                   f"{spa_b.mean_block:.0f} sessions, fixed seed) and assume the return process is "
                                   "stable over the sample.")
                with st.expander("Grid results table"):
                    ui.show_table(frame)

# --------------------------------------------------------------------------- #
# Cost sensitivity
# --------------------------------------------------------------------------- #
with cost_tab:
    if cost_tab.open:
        st.markdown(
            "How much of the result survives higher trading costs? The same signals are re-simulated with fees and "
            "slippage scaled from zero to ten times the sidebar settings. The benchmark is re-run at each cost level."
        )
        with st.spinner("Re-running at each cost level…"):
            frame = cost_sensitivity_for(data, spec)
        ui.plot(charts.cost_sensitivity_chart(frame), key="cost_chart")
        current_bps = (spec.config.fee_rate + spec.config.slippage_rate) * 1e4
        beaten = frame[frame["excess_return"] > 0]
        if len(beaten) == 0:
            st.markdown("The strategy does **not** beat its benchmark at any tested cost level, including zero costs.")
        elif len(beaten) == len(frame):
            st.markdown(f"The strategy beats the benchmark at every tested cost level up to "
                        f"{frame['cost_per_side_bps'].max():.0f} bps per execution.")
        else:
            worst_ok = beaten["cost_per_side_bps"].max()
            st.markdown(f"Excess return over the benchmark stays positive up to about **{worst_ok:.0f} bps** per execution "
                        f"(current setting: {current_bps:.1f} bps).")
        shown = frame.copy()
        for col in ("total_return", "benchmark_total_return", "excess_return"):
            shown[col] = shown[col] * 100
        ui.show_table(
            shown[["multiplier", "cost_per_side_bps", "total_return", "benchmark_total_return", "excess_return",
                   "sharpe", "total_costs"]],
            column_config={
                "multiplier": st.column_config.NumberColumn("Cost multiplier", format="%.1f×"),
                "cost_per_side_bps": st.column_config.NumberColumn("Cost per execution (bps)", format="%.1f"),
                "total_return": st.column_config.NumberColumn("Strategy return", format="%.2f%%"),
                "benchmark_total_return": st.column_config.NumberColumn("Benchmark return", format="%.2f%%"),
                "excess_return": st.column_config.NumberColumn("Excess return", format="%.2f%%"),
                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "total_costs": st.column_config.NumberColumn("Total costs", format="dollar"),
            },
        )

ui.disclaimer()
