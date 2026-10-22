"""Run the research agent: a tool-use loop on the Claude API with guardrails around it.

The loop is written out (rather than using the SDK's beta tool runner) so that
every step can be budgeted, audited and checked:

1. the active mode and its required inputs are validated before any model call;
2. the model sees only the tools of that mode, and only the user's limits;
3. each tool call is validated, executed by the deterministic engine and logged;
4. the final text is checked against the tool results (numbers, links,
   sections, language) before it is shown — and in strict mode, withheld if it
   fails.

Refusals, truncated output and exhausted budgets end the run with an explicit
status; nothing is silently retried with a different prompt. Requests use the
server-side refusal fallback (``fallbacks="default"``) on models that support it.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from quantlab.agent.audit import AuditLog
from quantlab.agent.guardrails import GuardrailReport, check_report, redact_secrets
from quantlab.agent.tools import MODE_DESCRIPTIONS, MODE_INPUTS, Mode, ToolContext, ToolRegistry, missing_inputs
from quantlab.config import PROJECT_ROOT

DEFAULT_MODEL = "claude-opus-5-5"
# Models that accept the server-side refusal fallback (fallbacks="default").
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
ROLE_PATH = PROJECT_ROOT / "docs" / "agent_role.md"


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    effort: str = "high"            # low | medium | high | xhigh | max
    max_turns: int = 14             # model requests per run
    max_tool_calls: int = 40
    max_tokens: int = 16_000        # per request (non-streaming)
    strict: bool = True             # withhold the report when a guardrail fails

    @classmethod
    def from_environment(cls) -> AgentConfig:
        return cls(model=os.environ.get("QLP_AGENT_MODEL", DEFAULT_MODEL),
                   effort=os.environ.get("QLP_AGENT_EFFORT", "high"))


@dataclass
class ToolCallRecord:
    name: str
    inputs: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None
    error: str | None
    duration_s: float


@dataclass
class AgentRun:
    run_id: str
    mode: Mode
    question: str
    status: str = "pending"         # completed | refused | truncated | budget_exhausted | error
    final_text: str = ""
    withheld: bool = False
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    guardrails: GuardrailReport | None = None
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0,
                                                            "cache_read_input_tokens": 0,
                                                            "cache_creation_input_tokens": 0})
    models_served: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    detail: str = ""

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        return [c.result for c in self.tool_calls if c.ok and c.result is not None]


def role_prompt() -> str:
    return ROLE_PATH.read_text(encoding="utf-8")


def _limits_summary(context: ToolContext) -> str:
    limits = context.risk_limits
    if limits is None:
        return "No risk limits configured: position sizing is unavailable in this run."
    data = limits.model_dump() if hasattr(limits, "model_dump") else dict(limits)
    label = " (ILLUSTRATIVE — the user did not supply them; say so)" if data.get("illustrative") else ""
    return "Risk limits" + label + ": " + json.dumps(data, sort_keys=True, default=str)


def opening_message(mode: Mode, inputs: dict[str, Any], question: str, context: ToolContext) -> str:
    required = ", ".join(MODE_INPUTS[mode]) or "none"
    if mode is Mode.RESEARCH:
        shape = "Write the full research report with the ten required section headings, in order."
    else:
        shape = ("Write a concise structured answer for this mode; use the report headings that apply and say which "
                 "were not applicable.")
    return (f"Active mode: {mode.value} — {MODE_DESCRIPTIONS[mode]}\n"
            f"Required inputs: {required}\n"
            f"Inputs supplied: {json.dumps(inputs, sort_keys=True, default=str)}\n"
            f"{_limits_summary(context)}\n"
            f"Benchmark: {context.benchmark}\n"
            f"Current time (UTC): {context.now().isoformat(timespec='minutes')}\n\n"
            f"Request: {question.strip() or '(no further instructions)'}\n\n"
            f"State the active mode and inputs first. Use the tools for every number. {shape}")


def _usage_add(run: AgentRun, usage: Any) -> None:
    for key in run.usage:
        value = getattr(usage, key, None)
        if isinstance(value, int):
            run.usage[key] += value


def run_agent(client: Any, registry: ToolRegistry, context: ToolContext, mode: Mode, inputs: dict[str, Any],
              question: str, config: AgentConfig | None = None, audit: AuditLog | None = None) -> AgentRun:
    """Run one agent session to completion and return what happened (never raises for API failures)."""
    config = config or AgentConfig()
    audit = audit or AuditLog()
    run = AgentRun(run_id=uuid.uuid4().hex[:12], mode=mode, question=question)

    missing = missing_inputs(mode, inputs)
    if missing:
        run.status, run.detail = "error", f"{mode.value} mode needs: {', '.join(missing)}."
        audit.record(run.run_id, "rejected", mode=mode.value, reason=run.detail)
        return run

    tools = registry.for_mode(mode)
    system = [{"type": "text", "text": role_prompt(), "cache_control": {"type": "ephemeral"}}]
    opening = opening_message(mode, inputs, question, context)
    messages: list[dict[str, Any]] = [{"role": "user", "content": opening}]
    audit.record(run.run_id, "start", mode=mode.value, inputs=inputs, question=question, model=config.model,
                 tools=[t.name for t in tools])

    request: dict[str, Any] = {
        "model": config.model, "max_tokens": config.max_tokens, "system": system,
        "tools": [t.api_definition() for t in tools],
        "thinking": {"type": "adaptive"}, "output_config": {"effort": config.effort},
    }
    if config.model in _FALLBACK_MODELS:
        request.update(betas=[_FALLBACK_BETA], fallbacks="default")

    import anthropic

    for _turn in range(config.max_turns):
        try:
            response = client.beta.messages.create(messages=messages, **request)
        except anthropic.AuthenticationError:
            run.status, run.detail = "error", "The API key was rejected. Check ANTHROPIC_API_KEY."
        except anthropic.PermissionDeniedError:
            run.status, run.detail = "error", "The API key does not have access to this model."
        except anthropic.RateLimitError:
            run.status, run.detail = "error", "Rate limited by the API. Wait a minute and try again."
        except anthropic.BadRequestError as exc:
            run.status, run.detail = "error", f"The API rejected the request: {redact_secrets(str(exc.message))}"
        except anthropic.APIStatusError as exc:
            run.status, run.detail = "error", f"API error {exc.status_code}. Try again later."
        except anthropic.APIConnectionError:
            run.status, run.detail = "error", "Could not reach the API (network)."
        if run.status == "error":
            audit.record(run.run_id, "api_error", detail=run.detail)
            return run

        _usage_add(run, getattr(response, "usage", None))
        served = getattr(response, "model", None)
        if served and served not in run.models_served:
            run.models_served.append(served)
        run.stop_reason = response.stop_reason
        audit.record(run.run_id, "response", stop_reason=response.stop_reason, model=served,
                     request_id=getattr(response, "_request_id", None),
                     blocks=[getattr(b, "type", "?") for b in response.content])

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            run.status = "refused"
            run.detail = "The model declined this request" + (f" ({category})." if category else ".")
            return run
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "max_tokens":
            run.status, run.detail = "truncated", "The answer hit the length limit and is incomplete."
            run.final_text = _text_of(response)
            break
        if response.stop_reason == "pause_turn":
            continue
        if response.stop_reason != "tool_use":
            run.status, run.final_text = "completed", _text_of(response)
            break

        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if len(run.tool_calls) >= config.max_tool_calls:
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": "Tool-call budget exhausted for this run. Write the answer with the "
                                           "results you have, and say what could not be checked."})
                continue
            outcome = registry.execute(block.name, block.input, context, mode)
            run.tool_calls.append(ToolCallRecord(outcome.name, outcome.inputs, outcome.ok, outcome.result,
                                                 outcome.error, outcome.duration_s))
            audit.record(run.run_id, "tool", name=outcome.name, inputs=outcome.inputs, ok=outcome.ok,
                         error=outcome.error, duration_s=round(outcome.duration_s, 3),
                         result_keys=sorted(outcome.result) if outcome.result else None)
            content = json.dumps(outcome.result, sort_keys=True, default=str) if outcome.ok else outcome.error
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content or "",
                            **({} if outcome.ok else {"is_error": True})})
        messages.append({"role": "user", "content": results})
    else:
        run.status = "budget_exhausted"
        run.detail = f"The run stopped after {config.max_turns} model requests without a final answer."

    if run.final_text:
        # Grounding sources: tool results, tool errors (the model may quote them) and the user's request, but not
        # the clock line, whose digits are not facts the report could state.
        request_text = "\n".join(ln for ln in opening.splitlines() if not ln.startswith("Current time"))
        errors = [c.error for c in run.tool_calls if c.error]
        run.guardrails = check_report(run.final_text, run.tool_results, user_inputs=[request_text, *errors],
                                      require_sections=mode is Mode.RESEARCH,
                                      simulated_context=mode in (Mode.PAPER_TRADE, Mode.POST_TRADE_REVIEW,
                                                                 Mode.BACKTEST))
        run.withheld = config.strict and not run.guardrails.passed
        audit.record(run.run_id, "guardrails", **run.guardrails.to_dict(), withheld=run.withheld)
    audit.record(run.run_id, "end", status=run.status, usage=run.usage, models=run.models_served,
                 tool_calls=len(run.tool_calls))
    return run


def _text_of(response: Any) -> str:
    return "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
