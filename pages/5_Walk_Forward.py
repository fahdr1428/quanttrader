"""Walk-Forward Research — chronological out-of-sample evaluation."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from quantlab.metrics import compute_metrics
from quantlab.reporting import equity_curve_csv
from quantlab.research import (
    MAX_WALK_FORWARD_COMBINATIONS,
    OBJECTIVES,
    WalkForwardConfig,
    make_folds,
    probabilistic_sharpe_ratio,
    walk_forward,
)
from quantlab.strategies import STRATEGIES
from ui import charts
from ui import components as ui

data = ui.require_data()
spec = ui.require_spec()

ui.page_header(
    "Walk-forward research",
    "Out-of-sample evaluation",
    "Parameters are chosen using only a training window, then judged on the unseen window that follows. "
    "Repeating this through time gives an honest, chronological estimate of how a parameter-selection "
    "process would have performed.",
)
ui.context_strip(data)

with st.expander("How the procedure works", icon=":material/help:", expanded=False):
    st.markdown(
        """
1. The evaluation period is split into consecutive **folds**: a training window followed immediately by a
   test window. Data are never shuffled; test windows never overlap and always come after their training data.
2. In each fold, every parameter combination in a small grid is backtested on the **training window only**.
   Prices after the training end are cut off *before* signals are computed, so the test period cannot leak
   into the choice.
3. The best combination by the chosen objective is applied to the **test window**. Indicators may use history
   from before the test start (information a trader would have had), never after it.
4. The test windows are stitched into one continuous out-of-sample portfolio with the normal engine, costs
   and next-open execution. That curve — not the training results — is the evidence.

**Rolling** training windows have a fixed length and forget old data; **anchored** windows always start at the
beginning and grow. Even walk-forward results are only one historical path: they reduce, but do not remove,
the risk of overfitting (the grid itself and the strategy idea were still chosen by a human who has seen
market history).
"""
    )

# ---- configuration ---------------------------------------------------------------
ui.section("Configuration")
keys = [k for k, cls in STRATEGIES.items() if cls.param_specs]
default_key = spec.strategy_key if spec.strategy_key in keys else keys[0]
c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1.3])
key = c1.selectbox("Strategy", keys, index=keys.index(default_key), format_func=lambda k: STRATEGIES[k].name,
                   key="wf_strategy")
train = c2.number_input("Training sessions", 60, 2520, 504, 21, key="wf_train",
                        help="504 sessions ≈ 2 years.")
test = c3.number_input("Test sessions", 20, 1260, 126, 21, key="wf_test", help="126 sessions ≈ 6 months.")
mode = c4.selectbox("Window", ["Rolling", "Anchored"], key="wf_mode")
objective = c5.selectbox("Select parameters by", list(OBJECTIVES), format_func=OBJECTIVES.get, key="wf_objective")

cls = STRATEGIES[key]
grid: dict[str, list] = {}
grid_cols = st.columns(max(len(cls.param_specs), 1))
for col, p in zip(grid_cols, cls.param_specs):
    if p.kind == "choice":
        chosen = col.multiselect(p.label, list(p.choices), default=[p.default], key=f"wf_grid_{key}_{p.name}")
        grid[p.name] = chosen or [p.default]
        continue
    defaults = list(cls.search_space.get(p.name, ())) or [p.default]
    text = col.text_input(f"{p.label} candidates", ", ".join(f"{v:g}" for v in defaults),
                          key=f"wf_grid_{key}_{p.name}", help="Comma-separated values to search.")
    values = []
    for tok in text.split(","):
        if tok.strip():
            try:
                values.append(p.coerce(float(tok)))
            except Exception as exc:  # noqa: BLE001
                col.error(str(exc))
    grid[p.name] = sorted(set(values)) or [p.default]

try:
    wf_cfg = WalkForwardConfig(train_sessions=int(train), test_sessions=int(test), anchored=mode == "Anchored",
                               objective=objective)
    folds = make_folds(data.eval_prices.index, wf_cfg)
    combos = cls.parameter_grid({k: tuple(v) for k, v in grid.items()})
    config_error = None
except ValueError as exc:
    wf_cfg, folds, combos, config_error = None, [], [], str(exc)

if config_error:
    st.error(config_error)
else:
    n_backtests = len(combos) * len(folds)
    st.caption(f"{len(combos)} valid parameter combinations × {len(folds)} folds = **{n_backtests} in-sample "
               f"backtests** (limit {MAX_WALK_FORWARD_COMBINATIONS} combinations).")
    if not folds:
        st.warning(f"The evaluation window has {len(data.eval_prices)} sessions — not enough for one fold of "
                   f"{int(train)} training + at least 20 test sessions. Shorten the windows or extend the date range.")
    if key == "ml_logistic" and combos and folds:
        st.caption("The ML model retrains itself many times inside every backtest, so this run takes longer "
                   "(roughly a few seconds per hundred backtests).")
    if len(combos) > 20:
        st.warning("A large grid raises the chance that the in-sample winner is just the luckiest configuration. "
                   "Keep the search space small and economically motivated.", icon=":material/warning:")

signature = json.dumps({"sample": data.sample_key, "key": key, "grid": grid, "train": int(train), "test": int(test),
                        "mode": mode, "objective": objective, "config": spec.config.model_dump()},
                       sort_keys=True, default=str)
run_disabled = bool(config_error) or not folds or not combos or len(combos) > MAX_WALK_FORWARD_COMBINATIONS
if st.button("Run walk-forward analysis", type="primary", disabled=run_disabled, key="wf_run"):
    with st.spinner(f"Running {len(combos) * len(folds)} in-sample backtests and the out-of-sample simulation…"):
        try:
            result = walk_forward(data.prices, key, grid, spec.config, wf_cfg, data.eval_start, data.eval_end,
                                  benchmark_prices=data.bench_prices)
            st.session_state["wf_result"] = {"signature": signature, "result": result}
        except ValueError as exc:
            st.error(str(exc))

stored = st.session_state.get("wf_result")
if not stored:
    ui.empty_state("No walk-forward run yet",
                   "Configure the windows and parameter grid above, then press <b>Run walk-forward analysis</b>.")
    ui.disclaimer()
    st.stop()
if stored["signature"] != signature:
    st.info("Settings have changed since the results below were produced. Re-run to update them.",
            icon=":material/sync_problem:")

wf = stored["result"]
oos_report = compute_metrics(wf.oos, benchmark=wf.oos_benchmark)

ui.section("Out-of-sample results",
           f"{wf.folds[0].test_start:%Y-%m-%d} → {wf.folds[-1].test_end:%Y-%m-%d} · stitched test windows only · "
           "net of costs")
st.markdown(ui.pill("OUT-OF-SAMPLE", "good") + " " + ui.pill(
    "Accounting reconciled" if wf.oos.is_valid else "Accounting FAILED", "good" if wf.oos.is_valid else "bad"),
    unsafe_allow_html=True)
ui.kpi_row(oos_report, ["total_return", "cagr", "sharpe", "max_drawdown", "benchmark_total_return", "excess_return"],
           labels={"total_return": "OOS total return", "cagr": "OOS CAGR", "sharpe": "OOS Sharpe",
                   "max_drawdown": "OOS drawdown"})
ui.unreliable_footnote(oos_report, ["total_return", "cagr", "sharpe", "max_drawdown"])
ui.plot(charts.walk_forward_equity(wf), key="wf_eq")

psr = probabilistic_sharpe_ratio(wf.oos.returns, 0.0, wf.oos.config.risk_free_rate, wf.oos.config.periods_per_year)
is_mean = wf.fold_table["IS Sharpe"].astype(float).mean()
oos_mean = wf.fold_table["OOS Sharpe"].astype(float).mean()
c1, c2, c3 = st.columns(3)
c1.metric("Mean in-sample Sharpe (training)", f"{is_mean:.2f}", border=True,
          help="Average Sharpe of the winning configuration on its own training window. Optimistic by construction.")
c2.metric("Mean out-of-sample Sharpe (test)", f"{oos_mean:.2f}", border=True,
          help="Average Sharpe in the following, unseen test windows.")
c3.metric("Probabilistic Sharpe of OOS returns", "n/a" if psr is None else f"{psr:.0%}", border=True,
          help="Probability the true out-of-sample Sharpe exceeds zero, given sample length, skew and kurtosis.")
if pd.notna(is_mean) and pd.notna(oos_mean):
    if is_mean > 0 and oos_mean <= 0:
        st.caption(f"The edge disappeared out of sample: mean Sharpe fell from {is_mean:+.2f} in training to "
                   f"{oos_mean:+.2f} in testing — the typical signature of fitting noise in the training window.")
    elif is_mean > 0:
        st.caption(f"Out-of-sample Sharpe retained {oos_mean / is_mean:.0%} of the in-sample level. Some decay is "
                   "normal; a large drop suggests the training-window optimum was partly noise.")

ui.section("Folds", "Training figures are in-sample and shown only to diagnose decay. Dates are exact sessions.")
table = wf.fold_table.drop(columns=["params"]).copy()
pct_cols = ["IS total return", "OOS total return", "OOS max drawdown", "Benchmark OOS return", "OOS excess return"]
for col in pct_cols:
    table[col] = table[col].astype(float) * 100
ui.show_table(table, column_config={
    **{c: st.column_config.NumberColumn(c, format="%.2f%%") for c in pct_cols},
    **{c: st.column_config.DateColumn(c, width=104) for c in ("Train start", "Train end", "Test start", "Test end",
                                                             "Selection data ends")},
    "IS objective": st.column_config.NumberColumn(f"IS {OBJECTIVES[wf.config.objective]}", format="%.3f"),
    "IS Sharpe": st.column_config.NumberColumn(format="%.2f"),
    "OOS Sharpe": st.column_config.NumberColumn(format="%.2f"),
    "Selected parameters": st.column_config.TextColumn(width="large"),
})
for note in wf.notes:
    st.caption(note)

c1, c2 = st.columns([3, 2], gap="medium")
with c1:
    ui.section("In-sample vs out-of-sample Sharpe by fold")
    ui.plot(charts.is_vs_oos(wf.fold_table), key="wf_isoos")
with c2:
    ui.section("Parameter stability")
    counts = wf.fold_table["Selected parameters"].value_counts().rename_axis("Parameters").reset_index(name="Folds")
    ui.show_table(counts)
    st.caption("If the selected parameters jump around between folds, the 'optimal' setting is probably noise.")

with st.expander("Out-of-sample trade ledger"):
    ui.show_table(wf.oos.ledger)
with st.expander("All in-sample trials"):
    ui.show_table(wf.in_sample_trials)

cols = st.columns(3)
cols[0].download_button("Fold table (CSV)", wf.fold_table.drop(columns=["params"]).to_csv(index=False),
                        "walk_forward_folds.csv", "text/csv", icon=":material/download:", on_click="ignore",
                        width="stretch")
cols[1].download_button("OOS equity curve (CSV)", equity_curve_csv(wf.oos, wf.oos_benchmark),
                        "walk_forward_oos_equity.csv", "text/csv", icon=":material/download:", on_click="ignore",
                        width="stretch")
cols[2].download_button("In-sample trials (CSV)", wf.in_sample_trials.to_csv(index=False),
                        "walk_forward_trials.csv", "text/csv", icon=":material/download:", on_click="ignore",
                        width="stretch")

ui.disclaimer()
