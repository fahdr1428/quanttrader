"""Experiment store: persistence, schema, notes, import/export and secret refusal."""

from __future__ import annotations

import json

import pytest

from quantlab.experiments import (
    SCHEMA_VERSION,
    ExperimentRecord,
    ExperimentStore,
    NoteRecord,
)


def record(**overrides) -> ExperimentRecord:
    base = dict(
        name="MA trend on synthetic data",
        hypothesis="Trend filters reduce drawdowns.",
        ticker="SYN",
        benchmark="SYN",
        start_date="2020-01-01",
        end_date="2023-12-31",
        data_source="test",
        price_adjustment="total_return",
        data_fingerprint="abc123",
        last_observation="2023-12-29",
        strategy_key="ma_crossover",
        strategy_label="Moving-Average Trend (SMA 20/50)",
        params={"short_window": 20, "long_window": 50, "ma_type": "SMA"},
        config={"initial_capital": 10_000, "fee_rate": 0.001},
        metrics={"sharpe": 0.5, "total_return": 0.12, "sortino": float("nan")},
        parameter_selection="Defaults (pre-specified)",
        session_trials=3,
    )
    base.update(overrides)
    return ExperimentRecord(**base)


def test_save_and_load_round_trip():
    store = ExperimentStore.open_memory()
    exp_id = store.save_experiment(record())
    loaded = store.get_experiment(exp_id)
    assert loaded.name == "MA trend on synthetic data"
    assert loaded.params == {"short_window": 20, "long_window": 50, "ma_type": "SMA"}
    assert loaded.metrics["sharpe"] == 0.5
    assert loaded.metrics["sortino"] is None  # NaN stored as JSON null, not invalid JSON
    assert loaded.methodology_version and loaded.app_version


def test_file_database_persists_across_connections(tmp_path):
    path = tmp_path / "research.db"
    store = ExperimentStore.open_file(path)
    store.save_experiment(record())
    store.conn.close()
    reopened = ExperimentStore.open_file(path)
    assert len(reopened.list_experiments()) == 1
    assert reopened.schema_version() == SCHEMA_VERSION


def test_update_and_delete():
    store = ExperimentStore.open_memory()
    exp_id = store.save_experiment(record())
    store.update_conclusions(exp_id, "Fewer trades than expected", "One asset only", "Not robust")
    assert store.get_experiment(exp_id).conclusion == "Not robust"
    store.delete_experiment(exp_id)
    assert store.get_experiment(exp_id) is None


def test_notes_crud_and_link_survives_experiment_deletion():
    store = ExperimentStore.open_memory()
    exp_id = store.save_experiment(record())
    note_id = store.add_note(NoteRecord(title="Idea", body="Try 12-1 momentum", tags="momentum", experiment_id=exp_id))
    store.update_note(note_id, "Idea v2", "Try 12-1 momentum with skip=21", "momentum")
    assert store.list_notes()[0].title == "Idea v2"
    store.delete_experiment(exp_id)
    assert store.list_notes()[0].experiment_id is None  # ON DELETE SET NULL
    store.delete_note(note_id)
    assert store.list_notes() == []


def test_export_import_round_trip():
    source = ExperimentStore.open_memory()
    exp_id = source.save_experiment(record())
    source.add_note(NoteRecord(title="Linked", experiment_id=exp_id))
    payload = source.export_json()
    json.loads(payload)  # strictly valid JSON

    target = ExperimentStore.open_memory()
    assert target.import_json(payload) == (1, 1)
    imported = target.list_experiments()[0]
    assert imported.params == record().params
    assert target.list_notes()[0].experiment_id == imported.id


def test_import_rejects_foreign_files():
    store = ExperimentStore.open_memory()
    with pytest.raises(ValueError):
        store.import_json('{"hello": "world"}')
    with pytest.raises(ValueError):
        store.import_json("not json")


def test_csv_export_flattens_metrics():
    store = ExperimentStore.open_memory()
    store.save_experiment(record())
    frame = store.experiments_frame()
    assert {"m_sharpe", "m_total_return", "strategy", "data_fingerprint"} <= set(frame.columns)
    assert "m_sharpe" in store.export_csv().splitlines()[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("hypothesis", "use api_key = 12345 for the vendor"),
        ("observations", "token: abcdef"),
        ("name", "sk-abcdefghijklmnopqrstuvwxyz"),
    ],
)
def test_credentials_are_refused(field, value):
    with pytest.raises(ValueError, match="credential"):
        record(**{field: value})


def test_sessions_are_isolated():
    a, b = ExperimentStore.open_memory(), ExperimentStore.open_memory()
    a.save_experiment(record())
    assert b.list_experiments() == []
