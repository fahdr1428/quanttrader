"""Research Notes — experiment tracking, comparison, notes and export."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from quantlab.config import storage_mode
from quantlab.experiments import ExperimentRecord, NoteRecord, SensitiveContentError
from quantlab.metrics import format_value
from ui import components as ui
from ui.state import active_run, current_data, current_spec, get_store, restore_configuration, trials_for

ui.page_header(
    "Research notes",
    "Experiments & research log",
    "Record what you expected, what you tested and what you concluded — with the exact data, parameters and "
    "methodology version — so results can be compared honestly and reproduced later.",
)

store = get_store()
st.space("small")
if storage_mode() == "sqlite":
    st.markdown(ui.pill("Storage: local SQLite file (persists between runs)", "good"), unsafe_allow_html=True)
else:
    st.markdown(ui.pill("Storage: this browser session only", "warn"), unsafe_allow_html=True)
    st.caption("Records live in memory and are private to your session; nobody else using this app can see them, "
               "and they disappear when the session ends. Export them as JSON to keep them. When running locally, "
               "start the app with `QLP_STORAGE=sqlite` to persist records to `data/quant_lab_research.db`.")

KEY_METRICS = {
    "total_return": ("Total return", "pct"), "cagr": ("CAGR", "pct"), "sharpe": ("Sharpe", "ratio"),
    "max_drawdown": ("Max drawdown", "pct"), "volatility": ("Volatility", "pct_abs"),
    "excess_return": ("Excess vs benchmark", "pct"), "fills": ("Fills", "int"),
    "time_invested": ("Time invested", "pct_abs"), "cost_drag": ("Cost drag", "pct"),
}


def _error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return str(exc)


save_tab, exp_tab, notes_tab, io_tab = st.tabs(["Save experiment", "Saved experiments", "Notes", "Import / export"])

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
with save_tab:
    data, spec = current_data(), current_spec()
    if data is None or not data.is_usable or spec is None:
        ui.empty_state("Nothing to save yet", "Load data and configure a strategy first; the current backtest is "
                                              "what gets recorded.")
    else:
        run, _ = active_run(data, spec)
        report = run.report
        st.markdown(f"**Recording:** {run.net.strategy_label} on **{data.asset.ticker}**, "
                    f"{data.eval_start:%Y-%m-%d} → {data.eval_end:%Y-%m-%d}, benchmark {data.benchmark_label}.")
        preview = {label: format_value(report.value(k), kind) for k, (label, kind) in KEY_METRICS.items()}
        ui.show_table(pd.DataFrame([preview]))
        st.caption(f"Parameter selection: {spec.parameter_selection} · configurations tried on this sample this "
                   f"session: {len(trials_for(data))} · data fingerprint {data.fingerprint[:16]}")
        with st.form("save_experiment", clear_on_submit=False):
            name = st.text_input("Experiment name *", max_chars=120,
                                 placeholder="e.g. SPY 20/50 trend filter, 2015–2025")
            hypothesis = st.text_area("Hypothesis (what did you expect, and why?)", max_chars=5000,
                                      placeholder="e.g. A trend filter reduces drawdowns versus buy-and-hold at the "
                                                  "cost of lower total return.")
            observations = st.text_area("Observations", max_chars=5000)
            limitations = st.text_area("Limitations", max_chars=5000,
                                       placeholder="e.g. single asset, one historical path, parameters are defaults")
            conclusion = st.text_area("Conclusion", max_chars=5000)
            st.caption("Do not enter passwords, API keys or personal information — text that looks like a "
                       "credential is refused.")
            if st.form_submit_button("Save experiment", type="primary"):
                try:
                    record = ExperimentRecord(
                        name=name, hypothesis=hypothesis, observations=observations, limitations=limitations,
                        conclusion=conclusion, ticker=data.asset.ticker, benchmark=data.benchmark_ticker,
                        start_date=str(data.requested_start), end_date=str(data.requested_end),
                        data_source=data.asset.source_name, price_adjustment=data.asset.adjustment,
                        data_fingerprint=data.fingerprint, last_observation=str(data.asset.last_date.date()),
                        strategy_key=spec.strategy_key, strategy_label=run.net.strategy_label, params=spec.params,
                        config={**spec.config.model_dump(), "warmup_days": data.warmup_days,
                                "evaluation_start": str(data.eval_start.date()),
                                "evaluation_end": str(data.eval_end.date()),
                                "accounting_reconciled": run.net.is_valid},
                        metrics=report.to_dict(), parameter_selection=spec.parameter_selection,
                        session_trials=len(trials_for(data)),
                    )
                    new_id = store.save_experiment(record)
                    st.session_state["exp_table_version"] = st.session_state.get("exp_table_version", 0) + 1
                    st.success(f"Saved experiment #{new_id}.", icon=":material/check_circle:")
                except (ValidationError, SensitiveContentError, ValueError) as exc:
                    st.error(_error_text(exc), icon=":material/error:")

# --------------------------------------------------------------------------- #
# Saved experiments
# --------------------------------------------------------------------------- #
with exp_tab:
    records = store.list_experiments()
    if not records:
        ui.empty_state("No saved experiments", "Save the current backtest from the <b>Save experiment</b> tab.")
    else:
        rows = []
        for r in records:
            row = {"ID": r.id, "Name": r.name, "Saved (UTC)": r.created_at.replace("T", " ")[:16], "Ticker": r.ticker,
                   "Period": f"{r.start_date} → {r.end_date}", "Strategy": r.strategy_label}
            for k, (label, kind) in KEY_METRICS.items():
                row[label] = format_value(r.metrics.get(k), kind)
            row["Parameter selection"] = r.parameter_selection
            row["Trials"] = r.session_trials
            row["Methodology"] = r.methodology_version
            rows.append(row)
        frame = pd.DataFrame(rows)
        # The table key changes whenever records are added or deleted, so a stale row
        # selection can never point past the end of the new table.
        version = st.session_state.setdefault("exp_table_version", 0)
        selection = st.dataframe(frame, hide_index=True, width="stretch", on_select="rerun",
                                 selection_mode="multi-row", key=f"exp_table_{version}")
        picked = ([int(frame.iloc[i]["ID"]) for i in selection.selection.rows if 0 <= i < len(frame)]
                  if selection else [])
        st.caption("Select one row to inspect, edit or restore it; select several to compare side by side.")
        methodologies = {r.methodology_version for r in records if r.id in picked}
        if len(methodologies) > 1:
            st.warning("The selected experiments were produced under different methodology versions; their "
                       "numbers are not directly comparable.", icon=":material/warning:")

        if len(picked) >= 2:
            chosen = [store.get_experiment(i) for i in picked]
            compare = pd.DataFrame({
                f"#{r.id} {r.name}": {
                    "Ticker / period": f"{r.ticker} {r.start_date} → {r.end_date}",
                    "Strategy": r.strategy_label,
                    "Data fingerprint": r.data_fingerprint[:16],
                    **{label: format_value(r.metrics.get(k), kind) for k, (label, kind) in KEY_METRICS.items()},
                    "Fee / slippage": f"{r.config.get('fee_rate', 0):.3%} / {r.config.get('slippage_rate', 0):.3%}",
                    "Parameter selection": r.parameter_selection,
                } for r in chosen
            })
            ui.section("Side-by-side comparison")
            st.dataframe(compare, width="stretch")
            if len({r.data_fingerprint for r in chosen}) > 1:
                st.caption("Note: these experiments used different datasets (fingerprints differ).")
        elif len(picked) == 1:
            r = store.get_experiment(picked[0])
            ui.section(f"#{r.id} · {r.name}")
            c1, c2 = st.columns([3, 2], gap="medium")
            with c1:
                st.markdown(f"**Hypothesis.** {r.hypothesis or '—'}")
                with st.form(f"edit_{r.id}"):
                    obs = st.text_area("Observations", r.observations, max_chars=5000)
                    lim = st.text_area("Limitations", r.limitations, max_chars=5000)
                    con = st.text_area("Conclusion", r.conclusion, max_chars=5000)
                    if st.form_submit_button("Update notes"):
                        try:
                            store.update_conclusions(r.id, obs, lim, con)
                            st.success("Updated.")
                        except SensitiveContentError as exc:
                            st.error(str(exc))
            with c2:
                st.button("Restore this configuration", type="primary", on_click=restore_configuration, args=(r,),
                          key=f"restore_{r.id}", icon=":material/settings_backup_restore:",
                          help="Loads the ticker, dates, strategy, parameters and costs into the sidebar and "
                               "re-downloads the data. Compare the data fingerprint to detect revisions.")
                st.download_button("Download record (JSON)", json.dumps(r.model_dump(), indent=2, default=str),
                                   f"experiment_{r.id}.json", "application/json", on_click="ignore",
                                   icon=":material/download:")
                confirm = st.checkbox("I want to delete this experiment", key=f"confirm_del_{r.id}")
                if st.button("Delete", disabled=not confirm, key=f"del_{r.id}", icon=":material/delete:"):
                    store.delete_experiment(r.id)
                    st.session_state["exp_table_version"] = st.session_state.get("exp_table_version", 0) + 1
                    st.rerun()
            with st.expander("Full record (configuration, metrics, provenance)"):
                st.json(r.model_dump(), expanded=False)

# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
with notes_tab:
    experiments = {r.id: r.name for r in store.list_experiments()}
    with st.form("new_note", clear_on_submit=True):
        title = st.text_input("Title *", max_chars=200)
        body = st.text_area("Note", max_chars=20000, height=140)
        c1, c2 = st.columns(2)
        tags = c1.text_input("Tags", max_chars=200, placeholder="momentum, costs")
        link = c2.selectbox("Link to experiment", [None, *experiments],
                            format_func=lambda i: "— none —" if i is None else f"#{i} {experiments[i]}")
        if st.form_submit_button("Add note", type="primary"):
            try:
                store.add_note(NoteRecord(title=title, body=body, tags=tags, experiment_id=link))
                st.success("Note added.")
            except (ValidationError, SensitiveContentError) as exc:
                st.error(_error_text(exc))
    notes = store.list_notes()
    if not notes:
        st.caption("No notes yet.")
    for note in notes:
        linked = f" · linked to #{note.experiment_id}" if note.experiment_id else ""
        with st.expander(f"{note.title}  ·  {note.updated_at.replace('T', ' ')[:16]} UTC{linked}"):
            with st.form(f"note_{note.id}"):
                new_title = st.text_input("Title", note.title, max_chars=200)
                new_body = st.text_area("Note", note.body, max_chars=20000, height=140)
                new_tags = st.text_input("Tags", note.tags, max_chars=200)
                c1, c2 = st.columns(2)
                if c1.form_submit_button("Save changes"):
                    try:
                        store.update_note(note.id, new_title, new_body, new_tags)
                        st.rerun()
                    except (ValidationError, SensitiveContentError) as exc:
                        st.error(_error_text(exc))
                if c2.form_submit_button("Delete note"):
                    store.delete_note(note.id)
                    st.rerun()

# --------------------------------------------------------------------------- #
# Import / export
# --------------------------------------------------------------------------- #
with io_tab:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.section("Export")
        st.download_button("All experiments & notes (JSON)", store.export_json(), "quant_lab_research.json",
                           "application/json", icon=":material/download:", on_click="ignore", width="stretch")
        st.download_button("Experiments table (CSV)", store.export_csv(), "quant_lab_experiments.csv", "text/csv",
                           icon=":material/download:", on_click="ignore", width="stretch")
        st.caption("Exports contain only what you entered plus backtest configuration and metrics — no "
                   "credentials or personal data are collected by this app.")
    with c2:
        ui.section("Import")
        upload = st.file_uploader("Quant Lab Pro research export (JSON)", type=["json"], key="import_json")
        if upload is not None and st.button("Import file", key="do_import"):
            try:
                n_exp, n_notes = store.import_json(upload.getvalue().decode("utf-8"))
                st.session_state["exp_table_version"] = st.session_state.get("exp_table_version", 0) + 1
                st.success(f"Imported {n_exp} new experiment(s) and {n_notes} new note(s); records already "
                           "present were skipped.")
            except (ValueError, ValidationError, UnicodeDecodeError) as exc:
                st.error(f"Import failed: {_error_text(exc)}")

ui.disclaimer()
