"""Strategy Lab — understand, compare and stress-test strategies."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from quantlab.analytics import (
    MAX_SENSITIVITY_COMBINATIONS,
    compare_strategies,
    cost_sensitivity,
    parameter_sensitivity,
)
from quantlab.research import deflated_sharpe_ratio
from quantlab.strategies import STRATEGIES
from ui import charts
from ui import components as ui
from ui.state import active_run, register_trials_bulk, run_for, strategy_params_from_state, trials_for

data = ui.require_data()
spec = ui.require_spec()

ui.page_header(
    "Strategy lab",
    "Strategy research",
    "Read how each strategy works and when it fails, compare all of them on identical terms, and test how "
    "fragile the active strategy is to its parameters and to trading costs.",
)
ui.context_strip(data)

guide_tab, compare_tab, sens_tab, cost_tab = st.tabs(
    ["Strategy guide", "Compare strategies", "Parameter sensitivity", "Cost sensitivity"]
)

# --------------------------------------------------------------------------- #
# Guide
# --------------------------------------------------------------------------- #
with guide_tab:
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
    ui.kpi_row(run.report, ["total_return", "sharpe", "max_drawdown", "fills", "time_invested", "cost_drag"])
    selection = (
        "Pre-specified defaults (fixed before viewing this sample)" if run.strategy.uses_default_parameters()
        else "Custom parameters — unless they were fixed in advance, treat these results as in-sample."
    )
    st.caption(f"Parameter selection: **{selection}**")

# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
with compare_tab:
    chosen = st.multiselect("Strategies to compare", list(STRATEGIES), default=list(STRATEGIES),
                            format_func=lambda k: STRATEGIES[k].name, key="lab_compare")
    st.caption("Each strategy uses the parameters currently set for it in the sidebar (defaults if never changed). "
               "All runs share the same sessions, capital, costs, timing and benchmark.")
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
        bench = run_for(data, spec.strategy_key, spec.params, spec.config)[1]
        table = compare_strategies(runs)

        exposures = table["Time invested"].astype(float)
        if exposures.max() - exposures.min() > 0.2:
            st.info(
                f"**Exposure differs widely** ({exposures.min():.0%} to {exposures.max():.0%} of sessions invested). "
                "A strategy that sits in cash more often will mechanically show lower volatility and usually "
                "lower return; compare risk-adjusted measures and drawdowns, not total return alone.",
                icon=":material/balance:",
            )

        pct_cols = ["Net total return", "Gross total return", "CAGR (net)", "Volatility", "Max drawdown",
                    "Time invested", "Turnover (ann.)", "Cost drag", "Excess vs benchmark"]
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
            ui.section("Risk vs return", "Bubble size = share of sessions invested.")
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
        if st.button("Run sensitivity analysis", type="primary", key="sens_run",
                     disabled=n_combos > MAX_SENSITIVITY_COMBINATIONS or any(not v for v in grid.values())):
            with st.spinner(f"Running {n_combos} backtests…"):
                frame = parameter_sensitivity(data.prices, spec.strategy_key, spec.params, grid, spec.config,
                                              data.eval_start, data.eval_end)
            st.session_state["sens_result"] = {"frame": frame, "grid": grid, "key": spec.strategy_key,
                                               "sample": data.sample_key, "metric": metric}
            register_trials_bulk(data, [
                {"id": f"sens|{spec.strategy_key}|{json.dumps({k: r[k] for k in grid}, sort_keys=True)}|"
                       f"{spec.config.model_dump_json()}",
                 "label": f"{cls.name} grid point", "strategy_key": spec.strategy_key, "sharpe": r["sharpe"],
                 # Per-period (daily) Sharpe, the unit used by the deflated Sharpe ratio.
                 "sharpe_pp": (r["sharpe"] / (252 ** 0.5)) if pd.notna(r.get("sharpe")) else None}
                for _, r in frame[frame["valid"]].iterrows()
            ])
        result = st.session_state.get("sens_result")
        if result and result["key"] == spec.strategy_key and result["sample"] == data.sample_key:
            frame = result["frame"]
            label = metric_options[metric]
            gkeys = list(result["grid"])
            if len(gkeys) == 2:
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
            dsr = deflated_sharpe_ratio(run.net.returns, pp) if len(pp) >= 2 else None
            if dsr is not None:
                st.metric("Deflated Sharpe of active configuration", f"{dsr:.0%}", border=True,
                          help="Probability that the active configuration's Sharpe exceeds what the best of "
                               f"{len(pp)} trials would show by chance. Values near or below 50% mean the result "
                               "is consistent with selection luck.")
            with st.expander("Grid results table"):
                ui.show_table(frame)

# --------------------------------------------------------------------------- #
# Cost sensitivity
# --------------------------------------------------------------------------- #
with cost_tab:
    st.markdown(
        "How much of the result survives higher trading costs? The same signals are re-simulated with fees and "
        "slippage scaled from zero to ten times the sidebar settings. The benchmark is re-run at each cost level."
    )
    with st.spinner("Re-running at each cost level…"):
        frame = cost_sensitivity(data.prices, spec.strategy, spec.config, data.eval_start, data.eval_end,
                                 benchmark_prices=data.bench_prices)
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
