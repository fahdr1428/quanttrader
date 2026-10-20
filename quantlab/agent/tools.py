"""The agent's tools: strict schemas, per-mode access and a hard execution boundary.

A tool is a deterministic function of the research engine plus a JSON schema the
model must satisfy. Three properties are enforced here rather than requested in a
prompt:

* **No execution tools.** The registry refuses any tool that could act on a real
  account (see :func:`quantlab.agent.guardrails.is_execution_tool`); only
  ``paper_*`` tools that are explicitly simulated may record orders.
* **Mode scoping.** Each operating mode exposes only its own tools, so a RESEARCH
  run cannot submit paper orders and a SCAN run cannot read holdings.
* **Validated inputs.** Inputs are checked against the schema before the handler
  runs (strict tool use guarantees this on the API side; the check here also
  protects direct callers and tests).

Handlers receive a :class:`ToolContext` holding everything that is *not* the
model's to choose — the data loaders, the user's risk limits, the holdings the
user supplied, the paper book — so the model can never widen its own limits.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from quantlab.agent.guardrails import is_execution_tool
from quantlab.serialization import clean_json


class Mode(str, Enum):
    RESEARCH = "RESEARCH"
    SCAN = "SCAN"
    ALERTS = "ALERTS"
    BACKTEST = "BACKTEST"
    PAPER_TRADE = "PAPER TRADE"
    PORTFOLIO_REVIEW = "PORTFOLIO REVIEW"
    POST_TRADE_REVIEW = "POST-TRADE REVIEW"


# What each mode needs before it runs (shown to the user, checked before any model call).
MODE_INPUTS: dict[Mode, tuple[str, ...]] = {
    Mode.RESEARCH: ("ticker",),
    Mode.SCAN: ("universe",),
    Mode.ALERTS: ("rules",),
    Mode.BACKTEST: ("ticker", "strategy"),
    Mode.PAPER_TRADE: ("ticker",),
    Mode.PORTFOLIO_REVIEW: ("holdings",),
    Mode.POST_TRADE_REVIEW: (),
}

MODE_DESCRIPTIONS: dict[Mode, str] = {
    Mode.RESEARCH: "Investigate one security: data quality, context, fundamentals, technicals, catalysts, a thesis "
                   "with invalidation, and a risk-sized framework — or a no-trade conclusion.",
    Mode.SCAN: "Screen a universe you define with documented criteria.",
    Mode.ALERTS: "Check whether specified conditions hold on the latest end-of-day data (nothing is sent anywhere).",
    Mode.BACKTEST: "Evaluate a fully specified strategy on history, in and out of sample, with costs.",
    Mode.PAPER_TRADE: "Simulate orders against later daily bars and track hypothetical performance. No real orders.",
    Mode.PORTFOLIO_REVIEW: "Assess the holdings you supply: concentration, correlation, drawdown and scenario risk.",
    Mode.POST_TRADE_REVIEW: "Compare paper trades' original theses with what happened and look for systematic errors.",
}


@dataclass
class ToolContext:
    """Everything a tool may use that the model must not choose."""

    load_prices: Callable[[str], Any]                     # ticker -> quantlab.data.MarketDataset (validated)
    risk_limits: Any = None                               # quantlab.risk.RiskLimits (user-configured)
    fundamentals: tuple[Any, ...] = ()                    # quantlab.fundamentals providers, in order of preference
    portfolio: Any = None                                 # quantlab.risk.PortfolioState
    holdings: tuple[Any, ...] = ()                        # quantlab.holdings.Holding, as supplied by the user
    paper_book: Any = None                                # quantlab.paper.PaperBook
    benchmark: str = "SPY"
    now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    extras: dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    """A tool could not produce a result; the message is shown to the model as an error result."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], dict[str, Any]]
    modes: frozenset[Mode]
    simulated: bool = False

    def api_definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema,
                "strict": True}


@dataclass
class ToolOutcome:
    name: str
    inputs: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None
    error: str | None
    duration_s: float


def obj(properties: dict[str, Any]) -> dict[str, Any]:
    """A strict object schema: every property required, nothing extra (use a null union for optional)."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def optional(schema: dict[str, Any]) -> dict[str, Any]:
    kind = schema.get("type")
    types = kind if isinstance(kind, list) else [kind]
    return {**schema, "type": [*types, "null"]}


def validate(value: Any, schema: dict[str, Any], path: str = "input") -> list[str]:
    """Minimal JSON-schema check for the subset the tools use (type, enum, properties, required,
    additionalProperties, items, minimum/maximum, minItems/maxItems)."""
    errors: list[str] = []
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds] if kinds else []

    def is_kind(v: Any, k: str) -> bool:
        return {"null": v is None, "boolean": isinstance(v, bool),
                "integer": isinstance(v, int) and not isinstance(v, bool),
                "number": isinstance(v, int | float) and not isinstance(v, bool),
                "string": isinstance(v, str), "array": isinstance(v, list), "object": isinstance(v, dict)}[k]

    if kinds and not any(is_kind(value, k) for k in kinds):
        return [f"{path}: expected {' or '.join(kinds)}, got {type(value).__name__}"]
    if value is None:
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required")
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{k}: not allowed" for k in value if k not in props)
        for k, v in value.items():
            if k in props:
                errors.extend(validate(v, props[k], f"{path}.{k}"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: at most {schema['maxItems']} item(s)")
        if "items" in schema:
            for i, v in enumerate(value):
                errors.extend(validate(v, schema["items"], f"{path}[{i}]"))
    return errors


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]):
        self.tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            if is_execution_tool(tool.name, tool.description, tool.simulated):
                raise ValueError(f"tool {tool.name!r} could act on a real account; only simulated paper_* tools "
                                 "are allowed")
            schema = tool.input_schema
            if schema.get("type") != "object" or schema.get("additionalProperties") is not False \
                    or set(schema.get("required", [])) != set(schema.get("properties", {})):
                raise ValueError(f"tool {tool.name!r}: strict schemas need every property required and no extras")
            self.tools[tool.name] = tool

    def for_mode(self, mode: Mode) -> list[Tool]:
        return [t for t in self.tools.values() if mode in t.modes]

    def execute(self, name: str, inputs: Any, context: ToolContext, mode: Mode) -> ToolOutcome:
        started = time.perf_counter()

        def done(ok: bool, result: dict[str, Any] | None = None, error: str | None = None) -> ToolOutcome:
            return ToolOutcome(name, inputs if isinstance(inputs, dict) else {}, ok, result, error,
                               time.perf_counter() - started)

        tool = self.tools.get(name)
        if tool is None or mode not in tool.modes:
            return done(False, error=f"Tool {name!r} is not available in {mode.value} mode.")
        problems = validate(inputs, tool.input_schema)
        if problems:
            return done(False, error="Invalid input: " + "; ".join(problems[:5]))
        try:
            return done(True, clean_json(tool.handler(context, inputs)))
        except ToolError as exc:
            return done(False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a data-source failure must reach the model as an error, not crash the run
            from quantlab.data import DataProviderError

            if isinstance(exc, DataProviderError):
                return done(False, error=f"Data unavailable: {getattr(exc, 'user_message', str(exc))}")
            if isinstance(exc, ValueError):
                return done(False, error=f"Could not compute: {exc}")
            return done(False, error=f"Internal error in {name} ({type(exc).__name__}); the result is unavailable.")


def missing_inputs(mode: Mode, inputs: dict[str, Any]) -> list[str]:
    return [k for k in MODE_INPUTS[mode] if inputs.get(k) in (None, "", [], ())]
