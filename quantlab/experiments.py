"""Experiment tracking and research notes, persisted in SQLite.

Two storage modes (see :func:`quantlab.config.storage_mode`):

* ``sqlite`` — a file database under ``data/`` for running the app on your own
  machine. Records survive restarts.
* ``session`` — an in-memory database owned by one browser session. Used for
  public deployments so that one visitor can never read another visitor's
  notes. Records disappear when the session ends; export them as JSON to keep
  them and import the file later.

All SQL uses parameter binding (never string formatting) and schema changes are
applied through numbered migrations recorded in ``schema_version``.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantlab.config import APP_VERSION, METHODOLOGY_VERSION
from quantlab.serialization import clean_json

SCHEMA_VERSION = 3

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS experiments (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        name                TEXT    NOT NULL,
        created_at          TEXT    NOT NULL,
        hypothesis          TEXT    NOT NULL DEFAULT '',
        ticker              TEXT    NOT NULL,
        benchmark           TEXT    NOT NULL DEFAULT '',
        start_date          TEXT    NOT NULL,
        end_date            TEXT    NOT NULL,
        data_source         TEXT    NOT NULL,
        price_adjustment    TEXT    NOT NULL,
        data_fingerprint    TEXT    NOT NULL,
        last_observation    TEXT    NOT NULL,
        strategy_key        TEXT    NOT NULL,
        strategy_label      TEXT    NOT NULL,
        params_json         TEXT    NOT NULL,
        config_json         TEXT    NOT NULL,
        metrics_json        TEXT    NOT NULL,
        parameter_selection TEXT    NOT NULL DEFAULT '',
        session_trials      INTEGER NOT NULL DEFAULT 0,
        observations        TEXT    NOT NULL DEFAULT '',
        limitations         TEXT    NOT NULL DEFAULT '',
        conclusion          TEXT    NOT NULL DEFAULT '',
        methodology_version TEXT    NOT NULL,
        app_version         TEXT    NOT NULL
    );
    CREATE TABLE IF NOT EXISTS notes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        title         TEXT NOT NULL,
        body          TEXT NOT NULL DEFAULT '',
        tags          TEXT NOT NULL DEFAULT '',
        experiment_id INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
    """,
    # Paper trading (quantlab.paper). Every row is simulated: the CHECK constraints make a
    # row that claims otherwise impossible to store.
    2: """
    CREATE TABLE IF NOT EXISTS paper_accounts (
        account       TEXT PRIMARY KEY,
        starting_cash REAL NOT NULL CHECK (starting_cash > 0),
        created_at    TEXT NOT NULL,
        simulated     INTEGER NOT NULL DEFAULT 1 CHECK (simulated = 1)
    );
    CREATE TABLE IF NOT EXISTS paper_orders (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        account         TEXT    NOT NULL REFERENCES paper_accounts(account),
        client_order_id TEXT,
        ticker          TEXT    NOT NULL,
        side            TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
        quantity        REAL    NOT NULL CHECK (quantity > 0),
        order_type      TEXT    NOT NULL CHECK (order_type IN ('market', 'limit', 'stop')),
        limit_price     REAL,
        stop_price      REAL,
        status          TEXT    NOT NULL CHECK (status IN ('open', 'filled', 'cancelled', 'rejected')),
        role            TEXT    NOT NULL DEFAULT 'entry',
        parent_id       INTEGER REFERENCES paper_orders(id),
        created_at      TEXT    NOT NULL,
        decision_date   TEXT    NOT NULL,
        thesis_json     TEXT    NOT NULL DEFAULT '',
        note            TEXT    NOT NULL DEFAULT '',
        reason          TEXT    NOT NULL DEFAULT '',
        simulated       INTEGER NOT NULL DEFAULT 1 CHECK (simulated = 1),
        UNIQUE (account, client_order_id)
    );
    CREATE TABLE IF NOT EXISTS paper_fills (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id        INTEGER NOT NULL REFERENCES paper_orders(id),
        account         TEXT    NOT NULL,
        ticker          TEXT    NOT NULL,
        side            TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
        quantity        REAL    NOT NULL CHECK (quantity > 0),
        price           REAL    NOT NULL CHECK (price > 0),
        reference_price REAL    NOT NULL,
        fee             REAL    NOT NULL CHECK (fee >= 0),
        session         TEXT    NOT NULL,
        filled_at       TEXT    NOT NULL,
        simulated       INTEGER NOT NULL DEFAULT 1 CHECK (simulated = 1)
    );
    CREATE INDEX IF NOT EXISTS idx_paper_orders_account ON paper_orders(account, ticker, status);
    CREATE INDEX IF NOT EXISTS idx_paper_fills_account ON paper_fills(account, ticker);
    """,
    # The last session evaluated per ticker: paper orders are filled forward only, never retroactively.
    3: """
    CREATE TABLE IF NOT EXISTS paper_progress (
        account TEXT NOT NULL,
        ticker  TEXT NOT NULL,
        through TEXT NOT NULL,
        PRIMARY KEY (account, ticker)
    );
    """,
}

# Strings that look like credentials are refused so they never end up in a
# research record or an exported file.
_SENSITIVE_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token|bearer)\b\s*[:=]"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
]


class SensitiveContentError(ValueError):
    """Raised when free text appears to contain a credential."""


def check_not_sensitive(text: str, field_name: str) -> str:
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text or ""):
            raise SensitiveContentError(
                f"The {field_name} field looks like it contains a credential (API key, token or "
                "password). Research records must never store secrets — please remove it."
            )
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    name: str = Field(min_length=1, max_length=120)
    created_at: str = Field(default_factory=_utc_now)
    hypothesis: str = Field(default="", max_length=5000)
    ticker: str = Field(min_length=1, max_length=20)
    benchmark: str = Field(default="", max_length=40)
    start_date: str
    end_date: str
    data_source: str
    price_adjustment: str
    data_fingerprint: str
    last_observation: str
    strategy_key: str
    strategy_label: str
    params: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    parameter_selection: str = ""
    session_trials: int = 0
    observations: str = Field(default="", max_length=5000)
    limitations: str = Field(default="", max_length=5000)
    conclusion: str = Field(default="", max_length=5000)
    methodology_version: str = METHODOLOGY_VERSION
    app_version: str = APP_VERSION

    @field_validator("name", "hypothesis", "observations", "limitations", "conclusion")
    @classmethod
    def _no_secrets(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        return check_not_sensitive(value.strip() if isinstance(value, str) else value, info.field_name)


class NoteRecord(BaseModel):
    id: int | None = None
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(default="", max_length=20000)
    tags: str = Field(default="", max_length=200)
    experiment_id: int | None = None
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    @field_validator("title", "body", "tags")
    @classmethod
    def _no_secrets(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        return check_not_sensitive(value, info.field_name)


class ExperimentStore:
    def __init__(self, connection: sqlite3.Connection, persistent: bool):
        self.conn = connection
        self.persistent = persistent
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # ---- construction ---------------------------------------------------------
    @classmethod
    def open_file(cls, path: Path | str) -> "ExperimentStore":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit may rerun the script on another thread.
        return cls(sqlite3.connect(path, check_same_thread=False), persistent=True)

    @classmethod
    def open_memory(cls) -> "ExperimentStore":
        return cls(sqlite3.connect(":memory:", check_same_thread=False), persistent=False)

    def _migrate(self) -> None:
        with self.conn:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = row["v"] or 0
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema v{current} is newer than this app supports (v{SCHEMA_VERSION})."
                )
            for version in range(current + 1, SCHEMA_VERSION + 1):
                self.conn.executescript(_MIGRATIONS[version])
                self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    def schema_version(self) -> int:
        return int(self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0])

    # ---- experiments ------------------------------------------------------------
    def save_experiment(self, record: ExperimentRecord) -> int:
        data = record.model_dump(exclude={"id"})
        row = {
            **{k: v for k, v in data.items() if k not in ("params", "config", "metrics")},
            "params_json": json.dumps(clean_json(data["params"]), sort_keys=True),
            "config_json": json.dumps(clean_json(data["config"]), sort_keys=True),
            "metrics_json": json.dumps(clean_json(data["metrics"]), sort_keys=True),
        }
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        with self.conn:
            cursor = self.conn.execute(
                f"INSERT INTO experiments ({columns}) VALUES ({placeholders})", tuple(row.values())
            )
        return int(cursor.lastrowid)

    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        data = dict(row)
        data["params"] = json.loads(data.pop("params_json"))
        data["config"] = json.loads(data.pop("config_json"))
        data["metrics"] = json.loads(data.pop("metrics_json"))
        return ExperimentRecord.model_validate(data)

    def get_experiment(self, experiment_id: int) -> ExperimentRecord | None:
        row = self.conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        return None if row is None else self._row_to_record(row)

    def list_experiments(self) -> list[ExperimentRecord]:
        rows = self.conn.execute("SELECT * FROM experiments ORDER BY created_at DESC, id DESC").fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_conclusions(
        self, experiment_id: int, observations: str, limitations: str, conclusion: str
    ) -> None:
        for name, value in (("observations", observations), ("limitations", limitations), ("conclusion", conclusion)):
            check_not_sensitive(value, name)
        with self.conn:
            self.conn.execute(
                "UPDATE experiments SET observations = ?, limitations = ?, conclusion = ? WHERE id = ?",
                (observations, limitations, conclusion, experiment_id),
            )

    def delete_experiment(self, experiment_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))

    def experiments_frame(self) -> pd.DataFrame:
        """Flat table (one row per experiment, metrics as columns) for comparison and CSV export."""
        records = self.list_experiments()
        rows = []
        for r in records:
            row = {
                "id": r.id,
                "name": r.name,
                "created_at": r.created_at,
                "ticker": r.ticker,
                "benchmark": r.benchmark,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "strategy": r.strategy_label,
                "parameters": json.dumps(r.params, sort_keys=True),
                "parameter_selection": r.parameter_selection,
                "session_trials": r.session_trials,
                "data_source": r.data_source,
                "price_adjustment": r.price_adjustment,
                "data_fingerprint": r.data_fingerprint[:16],
                "methodology_version": r.methodology_version,
                "hypothesis": r.hypothesis,
                "conclusion": r.conclusion,
            }
            for key, value in r.metrics.items():
                row[f"m_{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    # ---- notes ------------------------------------------------------------------
    def add_note(self, note: NoteRecord) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO notes (title, body, tags, experiment_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (note.title, note.body, note.tags, note.experiment_id, note.created_at, note.updated_at),
            )
        return int(cursor.lastrowid)

    def update_note(self, note_id: int, title: str, body: str, tags: str) -> None:
        validated = NoteRecord(title=title, body=body, tags=tags)
        with self.conn:
            self.conn.execute(
                "UPDATE notes SET title = ?, body = ?, tags = ?, updated_at = ? WHERE id = ?",
                (validated.title, validated.body, validated.tags, _utc_now(), note_id),
            )

    def delete_note(self, note_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def list_notes(self) -> list[NoteRecord]:
        rows = self.conn.execute("SELECT * FROM notes ORDER BY updated_at DESC, id DESC").fetchall()
        return [NoteRecord.model_validate(dict(r)) for r in rows]

    # ---- import / export -----------------------------------------------------------
    def export_json(self) -> str:
        payload = {
            "format": "quant-lab-pro/research-export",
            "schema_version": SCHEMA_VERSION,
            "exported_at": _utc_now(),
            "app_version": APP_VERSION,
            "experiments": [r.model_dump() for r in self.list_experiments()],
            "notes": [n.model_dump() for n in self.list_notes()],
        }
        return json.dumps(clean_json(payload), indent=2)

    def export_csv(self) -> str:
        buffer = io.StringIO()
        self.experiments_frame().to_csv(buffer, index=False)
        return buffer.getvalue()

    def import_json(self, text: str) -> tuple[int, int]:
        """Import experiments and notes from :meth:`export_json` output.

        Returns the numbers of experiments and notes *added*. Records already in
        the store (same creation time, name, data fingerprint, strategy and
        parameters; for notes, same title, creation time and body) are skipped,
        so importing the same file twice does not duplicate anything.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != "quant-lab-pro/research-export":
            raise ValueError("This file is not a Quant Lab Pro research export.")
        id_map: dict[int, int] = {}
        experiments = [ExperimentRecord.model_validate(e) for e in payload.get("experiments", [])]
        notes = [NoteRecord.model_validate(n) for n in payload.get("notes", [])]

        def identity(r: ExperimentRecord) -> tuple:
            return (r.created_at, r.name, r.data_fingerprint, r.strategy_key, json.dumps(r.params, sort_keys=True))

        existing = {identity(r): r.id for r in self.list_experiments()}
        existing_notes = {(n.title, n.created_at, n.body) for n in self.list_notes()}
        added_exp = added_notes = 0
        for record in experiments:
            old_id = record.id
            new_id = existing.get(identity(record))
            if new_id is None:
                new_id = self.save_experiment(record)
                existing[identity(record)] = new_id
                added_exp += 1
            if old_id is not None:
                id_map[old_id] = new_id
        for note in notes:
            if (note.title, note.created_at, note.body) in existing_notes:
                continue
            note.experiment_id = id_map.get(note.experiment_id) if note.experiment_id else None
            self.add_note(note)
            existing_notes.add((note.title, note.created_at, note.body))
            added_notes += 1
        return added_exp, added_notes
