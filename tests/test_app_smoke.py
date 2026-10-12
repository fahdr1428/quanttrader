"""UI smoke tests: every page renders and the main workflows run end to end.

Uses Streamlit's headless AppTest harness with a synthetic dataset injected
into the session (clearly labelled as synthetic), so no network is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.app_fixtures import synthetic_research_data

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = [
    "pages/1_Research_Overview.py",
    "pages/2_Strategy_Lab.py",
    "pages/3_Backtesting.py",
    "pages/4_Performance_Analytics.py",
    "pages/5_Walk_Forward.py",
    "pages/6_Data_Explorer.py",
    "pages/7_Research_Notes.py",
]

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = synthetic_research_data()
    return _DATA


def _app(page: str | None = None, with_data: bool = True) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    if with_data:
        at.session_state["research_data"] = _data()
    at.session_state["load_attempted"] = True  # never auto-download in tests
    at.run()
    if page:
        at.switch_page(page).run()
    return at


def _open_tab(at: AppTest, key: str, label: str) -> AppTest:
    """Select a lazily rendered tab (tabs only run their content while open)."""
    at.session_state[key] = label
    return at.run()


def _sensitivity_page() -> AppTest:
    return _open_tab(_app("pages/2_Strategy_Lab.py"), "lab_tabs", "Parameter sensitivity")


def _click_in_tab(at: AppTest, button_key: str, tab_key: str, label: str) -> AppTest:
    """Click a button inside a lazy tab. (AppTest does not carry the selected tab between runs the way the
    browser does, so the selection is re-applied before the rerun.)"""
    at.button(key=button_key).click()
    at.session_state[tab_key] = label
    return at.run()


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app(page)
    _assert_clean(at)
    assert not at.error, [e.value for e in at.error]


def test_overview_metrics_come_from_engine():
    at = _app()
    labels = {m.label: m.value for m in at.metric}
    assert "Total return" in labels and labels["Total return"] not in {"", "n/a"}
    assert "Configurations tried on this sample" in labels


@pytest.mark.parametrize("page", ["pages/1_Research_Overview.py", "pages/3_Backtesting.py", "pages/6_Data_Explorer.py"])
def test_pages_show_empty_state_without_data(page):
    at = _app(page, with_data=False)
    _assert_clean(at)
    assert any("No market data loaded" in m.value for m in at.markdown)


def test_switching_strategy_updates_results():
    at = _app()
    at.selectbox(key="w_strategy").set_value("rsi_reversion").run()
    _assert_clean(at)
    assert any("RSI Mean Reversion" in m.value for m in at.markdown)


def test_invalid_parameters_are_blocked_with_a_message():
    at = _app()
    at.number_input(key="p_ma_crossover_short_window").set_value(80).run()
    _assert_clean(at)
    assert any("short window must be shorter" in e.value for e in at.sidebar.error)
    assert any("Fix the strategy parameters" in w.value for w in at.warning)


def test_walk_forward_runs_end_to_end():
    at = _app("pages/5_Walk_Forward.py")
    at.number_input(key="wf_train").set_value(252).run()
    at.number_input(key="wf_test").set_value(126).run()
    at.button(key="wf_run").click().run()
    _assert_clean(at)
    labels = [m.label for m in at.metric]
    assert "OOS Sharpe" in labels and "Mean out-of-sample Sharpe (test)" in labels


def test_parameter_sensitivity_runs_and_counts_trials():
    at = _sensitivity_page()
    _click_in_tab(at, "sens_run", "lab_tabs", "Parameter sensitivity")
    _assert_clean(at)
    assert len(at.session_state["trials"]) > 5


def test_sensitivity_trials_include_non_grid_parameters():
    """Regression: grid points were keyed by the grid values only, so a second grid run
    with a different non-grid parameter was silently dropped, and the sidebar run of a
    configuration that lies on the grid was counted twice."""
    at = _sensitivity_page()
    before = len(at.session_state["trials"])
    _click_in_tab(at, "sens_run", "lab_tabs", "Parameter sensitivity")
    _assert_clean(at)
    grid_points = len(at.session_state["sens_result"]["frame"].query("valid"))
    # The sidebar configuration (20/50 SMA) lies on the default grid and is already counted.
    assert len(at.session_state["trials"]) - before == grid_points - 1

    next(s for s in at.sidebar.selectbox if s.label == "Average type").set_value("EMA")
    _open_tab(at, "lab_tabs", "Parameter sensitivity")
    before = len(at.session_state["trials"])
    _click_in_tab(at, "sens_run", "lab_tabs", "Parameter sensitivity")
    _assert_clean(at)
    assert len(at.session_state["trials"]) - before == grid_points - 1


def test_sensitivity_grid_with_no_valid_combination_does_not_crash():
    at = _sensitivity_page()
    next(t for t in at.text_input if t.key == "sens_vals_ma_crossover_short_window").input("100, 150")
    _open_tab(at, "lab_tabs", "Parameter sensitivity")
    next(t for t in at.text_input if t.key == "sens_vals_ma_crossover_long_window").input("50")
    _open_tab(at, "lab_tabs", "Parameter sensitivity")
    _click_in_tab(at, "sens_run", "lab_tabs", "Parameter sensitivity")
    _assert_clean(at)
    assert any("No combination on this grid is valid" in i.value for i in at.info)


def test_walk_forward_runs_are_counted_as_trials():
    at = _app("pages/5_Walk_Forward.py")
    at.button(key="wf_run").click().run()
    _assert_clean(at)
    assert len(at.session_state["wf_trials"]) == 1
    next(s for s in at.selectbox if s.key == "wf_objective").set_value("cagr").run()
    at.button(key="wf_run").click().run()
    _assert_clean(at)
    assert len(at.session_state["wf_trials"]) == 2
    metric = next(m for m in at.metric if m.label == "Deflated Sharpe across those runs")
    assert metric.value != "n/a"


def test_synthetic_demo_loads_offline_and_is_labelled_everywhere(monkeypatch):
    import ui.state

    def no_network(*args, **kwargs):
        raise AssertionError("the synthetic demo must not touch the network")

    monkeypatch.setattr(ui.state, "_fetch_yahoo", no_network)
    at = _app(with_data=False)
    at.session_state["data_error"] = {"message": "Yahoo Finance is unreachable.", "detail": None, "kind": "network"}
    at.run()
    at.button(key="demo_empty").click().run()
    _assert_clean(at)
    data = at.session_state["research_data"]
    assert data.asset.provider_key == "synthetic" and data.asset.ticker == "SIM-RANDOM-1"
    for page in PAGES:
        at.switch_page(page).run()
        _assert_clean(at)
        assert not at.error, (page, [e.value for e in at.error])
        if page != "pages/7_Research_Notes.py":
            assert any("SYNTHETIC" in m.value for m in at.markdown), page


LAB_TABS = ["Strategy guide", "Compare strategies", "Parameter sensitivity", "Cost sensitivity"]
ANALYTICS_TABS = ["All metrics", "Risk over time", "Period returns", "Trades", "Costs: gross vs net", "Robustness"]


@pytest.mark.parametrize("page,key,label", [("pages/2_Strategy_Lab.py", "lab_tabs", t) for t in LAB_TABS]
                         + [("pages/4_Performance_Analytics.py", "pa_tabs", t) for t in ANALYTICS_TABS])
def test_every_tab_renders(page, key, label):
    at = _open_tab(_app(page), key, label)
    _assert_clean(at)
    assert not at.error, [e.value for e in at.error]


def test_hidden_tabs_and_no_op_settings_do_not_add_trials():
    """Regression: opening Strategy Lab ran (and counted) all seven strategies in the hidden Compare tab,
    and changing capital (which does not change returns with fractional shares) added trials."""
    at = _app("pages/2_Strategy_Lab.py")
    assert len(at.session_state["trials"]) == 1
    for capital in (20_000.0, 50_000.0):
        at.number_input(key="w_capital").set_value(capital).run()
    assert len(at.session_state["trials"]) == 1
    _open_tab(at, "lab_tabs", "Compare strategies")
    assert len(at.session_state["trials"]) == 7  # viewing the comparison does count every strategy shown


def test_custom_parameters_survive_switching_strategy():
    at = _app("pages/2_Strategy_Lab.py")
    at.number_input(key="p_ma_crossover_short_window").set_value(10).run()
    at.selectbox(key="w_strategy").set_value("rsi_reversion").run()
    at.selectbox(key="w_strategy").set_value("ma_crossover").run()
    _assert_clean(at)
    assert at.number_input(key="p_ma_crossover_short_window").value == 10
    assert at.session_state["run_spec"].params["short_window"] == 10


def test_guide_follows_the_sidebar_strategy():
    at = _app("pages/2_Strategy_Lab.py")
    at.selectbox(key="w_strategy").set_value("rsi_reversion").run()
    assert at.selectbox(key="lab_guide_strategy").value == "rsi_reversion"


def test_stale_sensitivity_results_are_flagged():
    at = _sensitivity_page()
    _click_in_tab(at, "sens_run", "lab_tabs", "Parameter sensitivity")
    assert not any("Settings have changed" in i.value for i in at.info)
    at.number_input(key="w_fee").set_value(1.0)
    _open_tab(at, "lab_tabs", "Parameter sensitivity")
    assert any("Settings have changed" in i.value for i in at.info)


def test_walk_forward_settings_survive_navigation():
    at = _app("pages/5_Walk_Forward.py")
    at.number_input(key="wf_test").set_value(63).run()
    at.button(key="wf_run").click().run()
    at.switch_page("pages/1_Research_Overview.py").run()
    at.switch_page("pages/5_Walk_Forward.py").run()
    _assert_clean(at)
    assert at.number_input(key="wf_test").value == 63
    assert not any("Settings have changed" in i.value for i in at.info)


def test_deleting_a_selected_experiment_does_not_crash():
    from quantlab.experiments import ExperimentRecord

    at = _app("pages/7_Research_Notes.py")
    store = at.session_state["session_store"]
    for name in ("First", "Second"):
        store.save_experiment(ExperimentRecord(
            name=name, ticker="SYNTH", start_date="2020-01-01", end_date="2021-01-01", data_source="Synthetic",
            price_adjustment="as_provided", data_fingerprint="0" * 64, last_observation="2021-01-01",
            strategy_key="ma_crossover", strategy_label="MA", params={}, config={}, metrics={},
            parameter_selection="defaults"))
    at.run()
    oldest = store.list_experiments()[-1]  # row 1 of the table (newest first)

    def select_row_1():  # AppTest does not round-trip dataframe selections, so re-apply before each run
        at.session_state[f"exp_table_{at.session_state['exp_table_version']}"] = {
            "selection": {"rows": [1], "columns": [], "cells": []}}

    select_row_1()
    at.run()
    at.checkbox(key=f"confirm_del_{oldest.id}").check()
    select_row_1()
    at.run()
    at.button(key=f"del_{oldest.id}").click()
    select_row_1()
    at.run()  # the rerun after deleting must not index the now one-row table with the old selection
    _assert_clean(at)
    assert [r.name for r in store.list_experiments()] == ["Second"]


def test_timing_permutation_test_runs_from_the_robustness_tab():
    at = _open_tab(_app("pages/4_Performance_Analytics.py"), "pa_tabs", "Robustness")
    assert any("exposure-matched" in str(v) for t in at.dataframe for v in t.value.iloc[:, 0])
    _click_in_tab(at, "perm_run", "pa_tabs", "Robustness")
    _assert_clean(at)
    p_value = next(m for m in at.metric if m.label == "Timing p-value").value
    assert p_value == "n/a" or 0 < float(p_value) <= 1


def test_save_experiment_and_add_note():
    at = _app("pages/7_Research_Notes.py")
    next(t for t in at.text_input if t.label == "Experiment name *").input("Synthetic smoke test")
    next(b for b in at.button if b.label == "Save experiment").click().run()
    _assert_clean(at)
    assert any("Saved experiment #1" in s.value for s in at.success)
    store = at.session_state["session_store"]
    assert len(store.list_experiments()) == 1

    next(t for t in at.text_input if t.label == "Title *").input("Follow-up idea")
    next(b for b in at.button if b.label == "Add note").click().run()
    _assert_clean(at)
    assert len(store.list_notes()) == 1


def test_credentials_are_refused_in_the_ui():
    at = _app("pages/7_Research_Notes.py")
    next(t for t in at.text_input if t.label == "Experiment name *").input("api_key = hunter2")
    next(b for b in at.button if b.label == "Save experiment").click().run()
    _assert_clean(at)
    assert any("credential" in e.value for e in at.error)
    assert at.session_state["session_store"].list_experiments() == []


def test_restore_configuration_sets_sidebar_state():
    def script():
        import streamlit as st

        from quantlab.experiments import ExperimentRecord
        from ui.state import restore_configuration

        record = ExperimentRecord(
            name="restore me", ticker="QQQ", benchmark="SPY", start_date="2018-01-02", end_date="2022-12-30",
            data_source="Yahoo Finance (via yfinance)", price_adjustment="price_only", data_fingerprint="x",
            last_observation="2022-12-30", strategy_key="rsi_reversion", strategy_label="RSI",
            params={"window": 7, "entry_threshold": 25.0, "exit_threshold": 60.0},
            config={"initial_capital": 50_000, "fee_rate": 0.002, "slippage_rate": 0.001,
                    "risk_free_rate": 0.02, "fractional_shares": False, "warmup_days": 300},
            metrics={},
        )
        if "restored" not in st.session_state:
            restore_configuration(record)
            st.session_state["restored"] = True

    at = AppTest.from_function(script).run()
    _assert_clean(at)
    ss = at.session_state
    assert ss["w_ticker"] == "QQQ" and ss["w_bench_mode"] == "Another ticker" and ss["w_bench_ticker"] == "SPY"
    assert ss["w_strategy"] == "rsi_reversion" and ss["p_rsi_reversion_window"] == 7
    assert ss["w_fee"] == pytest.approx(0.2) and ss["w_slip"] == pytest.approx(0.1)
    assert ss["w_fractional"] is False and ss["w_adjustment"] == "price_only" and ss["w_warmup"] == 300
    assert ss["pending_load"] is True


def test_sqlite_storage_mode_persists_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("QLP_STORAGE", "sqlite")
    monkeypatch.setenv("QLP_DATA_DIR", str(tmp_path))
    from ui.state import _file_store

    _file_store.clear()
    try:
        at = _app("pages/7_Research_Notes.py")
        assert any("local SQLite file" in m.value for m in at.markdown)
        next(t for t in at.text_input if t.label == "Experiment name *").input("Persisted")
        next(b for b in at.button if b.label == "Save experiment").click().run()
        _assert_clean(at)
        from quantlab.experiments import ExperimentStore

        reopened = ExperimentStore.open_file(tmp_path / "quant_lab_research.db")
        assert [r.name for r in reopened.list_experiments()] == ["Persisted"]
    finally:
        _file_store.clear()


@pytest.mark.parametrize("strategy_key", ["adaptive_ensemble", "ml_logistic"])
@pytest.mark.parametrize("page", ["pages/1_Research_Overview.py", "pages/2_Strategy_Lab.py",
                                  "pages/3_Backtesting.py", "pages/4_Performance_Analytics.py"])
def test_advanced_models_render_on_every_analysis_page(strategy_key, page):
    at = _app()
    at.selectbox(key="w_strategy").set_value(strategy_key).run()
    at.switch_page(page).run()
    _assert_clean(at)
    assert not at.error, [e.value for e in at.error]


def test_ml_diagnostics_are_shown():
    at = _app()
    at.selectbox(key="w_strategy").set_value("ml_logistic").run()
    at.switch_page("pages/2_Strategy_Lab.py").run()
    _assert_clean(at)
    labels = {m.label for m in at.metric}
    assert {"Hit rate", "AUC", "Brier score"} <= labels


def test_walk_forward_with_the_ensemble():
    at = _app("pages/5_Walk_Forward.py")
    at.selectbox(key="wf_strategy").set_value("adaptive_ensemble").run()
    at.number_input(key="wf_train").set_value(504).run()
    at.number_input(key="wf_test").set_value(126).run()
    at.button(key="wf_run").click().run()
    _assert_clean(at)
    assert "OOS Sharpe" in [m.label for m in at.metric]
