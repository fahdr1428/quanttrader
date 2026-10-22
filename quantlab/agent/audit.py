"""Append-only audit log of agent runs.

Every model request, tool call (inputs, outcome, duration) and guardrail result
is recorded with the run ID, so a report can be traced back to the exact data
it was built from. Secrets are redacted before anything is written.

By default the log is kept in memory for the session only — on a public
deployment, one visitor's questions are never written to the server's disk.
With ``QLP_STORAGE=sqlite`` (local use) it is also appended to
``<QLP_DATA_DIR>/agent_audit/<date>.jsonl``, which ``.gitignore`` excludes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantlab.agent.guardrails import redact_secrets
from quantlab.serialization import clean_json

logger = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, directory: Path | str | None = None, max_events: int = 5_000):
        self.directory = Path(directory) if directory is not None else None
        self.events: list[dict[str, Any]] = []
        self.max_events = max_events

    @classmethod
    def from_environment(cls) -> AuditLog:
        if os.environ.get("QLP_STORAGE", "session").lower() == "sqlite":
            from quantlab.config import data_dir

            return cls(data_dir() / "agent_audit")
        return cls()

    def record(self, run_id: str, event: str, **payload: Any) -> dict[str, Any]:
        entry = _redact({"ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "event": event,
                         **clean_json(payload)})
        line = json.dumps(entry, sort_keys=True, default=str)
        self.events.append(entry)
        del self.events[:-self.max_events]
        if self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                with open(self.directory / f"{entry['ts'][:10]}.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:  # the audit file is a convenience; the in-memory log still holds the event
                logger.warning("Could not write the agent audit log: %s", exc)
        return entry

    def for_run(self, run_id: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["run_id"] == run_id]


def _redact(value: Any) -> Any:
    """Redact every string inside the entry (before serialising, so escaping can never be corrupted)."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {redact_secrets(str(k)): _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value
