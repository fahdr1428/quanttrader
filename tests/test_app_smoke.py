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
    at = _app("pages/2_Strategy_Lab.py")
    at.button(key="sens_run").click().run()
    _assert_clean(at)
    assert len(at.session_state["trials"]) > 5


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
