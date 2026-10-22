"""The agent loop, with a scripted fake client standing in for the API (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest

from quantlab.agent.audit import AuditLog
from quantlab.agent.guardrails import REPORT_SECTIONS
from quantlab.agent.runner import AgentConfig, run_agent
from quantlab.agent.tools import Mode, Tool, ToolContext, ToolError, ToolRegistry, obj, optional


def _usage():
    return SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def text(body):
    return SimpleNamespace(type="text", text=body)


def tool_use(name, inputs, tid="t1"):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=inputs)


def response(stop, *blocks, model="claude-opus-5-5", details=None):
    return SimpleNamespace(stop_reason=stop, content=list(blocks), usage=_usage(), model=model,
                           _request_id="req_test", stop_details=details)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot: the runner appends later
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _price_tool(ctx, inputs):
    if inputs["ticker"] == "BAD":
        raise ToolError("Price data for BAD failed validation; analysis withheld.")
    return {"ticker": inputs["ticker"], "last_close": 101.25, "as_of": "2024-05-31", "sma_200": 95.5,
            "source": "Synthetic demo data (simulated)"}


def _paper_tool(ctx, inputs):
    return {"order_id": 1, "status": "accepted (simulated)"}


REGISTRY = ToolRegistry([
    Tool("get_price_summary", "Latest validated end-of-day price data for a ticker",
         obj({"ticker": {"type": "string"}, "window": optional({"type": "integer", "minimum": 1})}),
         _price_tool, frozenset(Mode)),
    Tool("paper_submit_order", "Record a simulated paper order (no real order is sent)",
         obj({"ticker": {"type": "string"}}), _paper_tool, frozenset({Mode.PAPER_TRADE}), simulated=True),
])
CTX = ToolContext(load_prices=lambda t: None)


def _report(body):
    return "\n".join(f"## {i}. {s}\n{body if i == 1 else 'Not applicable.'}" for i, s in enumerate(REPORT_SECTIONS, 1))


def test_a_research_run_calls_tools_and_passes_its_report_through_the_guardrails():
    client = FakeClient(
        response("tool_use", text("Checking the data first."), tool_use("get_price_summary", {"ticker": "SIM", "window": None})),
        response("end_turn", text(_report("Close 101.25 on 2024-05-31, above the 200-day average of 95.5. "
                                          "No qualifying setup."))),
    )
    audit = AuditLog()
    run = run_agent(client, REGISTRY, CTX, Mode.RESEARCH, {"ticker": "SIM"}, "Research SIM", AgentConfig(), audit)
    assert run.status == "completed" and not run.withheld and run.guardrails.passed
    assert [c.name for c in run.tool_calls] == ["get_price_summary"] and run.tool_calls[0].ok
    assert run.usage["input_tokens"] == 200 and run.models_served == ["claude-opus-5-5"]
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert all(t["strict"] for t in first["tools"])
    assert "Role: AI quantitative finance" in first["system"][0]["text"]
    assert "Active mode: RESEARCH" in first["messages"][0]["content"]
    # The tool result went back in a single user message, matched to the tool_use id.
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"][0]["tool_use_id"] == "t1"
    events = [e["event"] for e in audit.for_run(run.run_id)]
    assert events[0] == "start" and "tool" in events and events[-1] == "end"


def test_a_report_with_invented_numbers_is_withheld_in_strict_mode():
    client = FakeClient(
        response("tool_use", tool_use("get_price_summary", {"ticker": "SIM", "window": None})),
        response("end_turn", text(_report("Buy at 101.25 with a target of $135 (+33%)."))),
    )
    run = run_agent(client, REGISTRY, CTX, Mode.RESEARCH, {"ticker": "SIM"}, "", AgentConfig(strict=True))
    assert run.withheld and not run.guardrails.passed
    assert {n.text for n in run.guardrails.ungrounded_numbers} >= {"$135", "+33%"}
    relaxed = FakeClient(*[response("tool_use", tool_use("get_price_summary", {"ticker": "SIM", "window": None})),
                           response("end_turn", text(_report("Target $135.")))])
    assert not run_agent(relaxed, REGISTRY, CTX, Mode.RESEARCH, {"ticker": "SIM"}, "", AgentConfig(strict=False)).withheld


def test_tools_are_scoped_to_the_mode_and_cannot_be_called_outside_it():
    """A RESEARCH run never sees the paper-order tool, and a call to it anyway is refused as an error result."""
    client = FakeClient(
        response("tool_use", tool_use("paper_submit_order", {"ticker": "SIM"})),
        response("end_turn", text(_report("No qualifying setup."))),
    )
    run = run_agent(client, REGISTRY, CTX, Mode.RESEARCH, {"ticker": "SIM"}, "", AgentConfig())
    assert [t["name"] for t in client.calls[0]["tools"]] == ["get_price_summary"]
    assert not run.tool_calls[0].ok and "not available in RESEARCH mode" in run.tool_calls[0].error
    paper = FakeClient(response("end_turn", text("Simulated only.")))
    run_agent(paper, REGISTRY, CTX, Mode.PAPER_TRADE, {"ticker": "SIM"}, "", AgentConfig())
    assert {t["name"] for t in paper.calls[0]["tools"]} == {"get_price_summary", "paper_submit_order"}


def test_failed_data_reaches_the_model_as_an_error_not_as_numbers():
    client = FakeClient(
        response("tool_use", tool_use("get_price_summary", {"ticker": "BAD", "window": None})),
        response("end_turn", text(_report("Data for BAD failed validation; no conclusion."))),
    )
    run = run_agent(client, REGISTRY, CTX, Mode.RESEARCH, {"ticker": "BAD"}, "", AgentConfig())
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "failed validation" in result["content"]
    assert run.tool_results == []


def test_invalid_tool_input_is_rejected_before_the_handler_runs():
    client = FakeClient(response("tool_use", tool_use("get_price_summary", {"ticker": "SIM", "window": -3, "x": 1})),
                        response("end_turn", text(_report("No qualifying setup."))))
    run = run_agent(client, REGISTRY, CTX, Mode.RESEARCH, {"ticker": "SIM"}, "", AgentConfig())
    assert not run.tool_calls[0].ok and "below the minimum" in run.tool_calls[0].error and "not allowed" in run.tool_calls[0].error


def test_missing_mode_inputs_stop_the_run_before_any_model_call():
    client = FakeClient()
    run = run_agent(client, REGISTRY, CTX, Mode.RESEARCH, {"ticker": ""}, "Research", AgentConfig())
    assert run.status == "error" and "needs: ticker" in run.detail and client.calls == []


def test_refusals_truncation_and_budgets_end_the_run_with_an_explicit_status():
    refused = run_agent(FakeClient(response("refusal", details=SimpleNamespace(category="cyber"))), REGISTRY, CTX,
                        Mode.RESEARCH, {"ticker": "SIM"}, "", AgentConfig())
    assert refused.status == "refused" and "cyber" in refused.detail and refused.final_text == ""
    truncated = run_agent(FakeClient(response("max_tokens", text("Partial"))), REGISTRY, CTX, Mode.SCAN,
                          {"universe": ["SIM"]}, "", AgentConfig())
    assert truncated.status == "truncated" and truncated.final_text == "Partial"
    loop = FakeClient(*[response("tool_use", tool_use("get_price_summary", {"ticker": "SIM", "window": None}, f"t{i}"))
                        for i in range(3)])
    exhausted = run_agent(loop, REGISTRY, CTX, Mode.RESEARCH, {"ticker": "SIM"}, "", AgentConfig(max_turns=3))
    assert exhausted.status == "budget_exhausted" and len(loop.calls) == 3


def test_tool_call_budget_returns_errors_instead_of_running_more_tools():
    many = response("tool_use", *[tool_use("get_price_summary", {"ticker": "SIM", "window": None}, f"t{i}")
                                  for i in range(5)])
    client = FakeClient(many, response("end_turn", text("Done.")))
    run = run_agent(client, REGISTRY, CTX, Mode.SCAN, {"universe": ["SIM"]}, "", AgentConfig(max_tool_calls=2))
    results = client.calls[1]["messages"][-1]["content"]
    assert len(run.tool_calls) == 2 and sum(bool(r.get("is_error")) for r in results) == 3


def test_api_errors_become_a_status_and_never_leak_the_key():
    import anthropic

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    auth = anthropic.AuthenticationError("bad key sk-ant-api03-SECRETSECRETSECRET",
                                         response=httpx2.Response(401, request=request), body=None)
    run = run_agent(FakeClient(auth), REGISTRY, CTX, Mode.RESEARCH, {"ticker": "SIM"}, "", AgentConfig())
    assert run.status == "error" and "rejected" in run.detail and "sk-ant" not in run.detail


def test_models_without_server_side_fallback_do_not_send_the_parameter():
    client = FakeClient(response("end_turn", text("Done.")))
    run_agent(client, REGISTRY, CTX, Mode.SCAN, {"universe": ["SIM"]}, "", AgentConfig(model="claude-haiku-5-5"))
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


def test_registry_refuses_execution_tools_and_loose_schemas():
    with pytest.raises(ValueError, match="real account"):
        ToolRegistry([Tool("place_order", "Send an order to the broker", obj({"ticker": {"type": "string"}}),
                           _price_tool, frozenset(Mode))])
    with pytest.raises(ValueError, match="strict"):
        ToolRegistry([Tool("get_x", "x", {"type": "object", "properties": {"a": {"type": "string"}}, "required": []},
                           _price_tool, frozenset(Mode))])


def test_numbers_quoted_from_a_tool_error_are_grounded():
    """Reporting a data failure faithfully must not get the report withheld."""
    client = FakeClient(
        response("tool_use", tool_use("get_price_summary", {"ticker": "BAD", "window": None})),
        response("end_turn", text(_report("BAD failed validation; analysis withheld, as the tool reported."))),
    )

    def failing(ctx, inputs):
        raise ToolError("Only 300 sessions, fewer than the 504 training sessions needed.")

    registry = ToolRegistry([Tool("get_price_summary", "Latest validated end-of-day price data for a ticker",
                                  obj({"ticker": {"type": "string"}, "window": optional({"type": "integer"})}),
                                  failing, frozenset(Mode))])
    quoting = FakeClient(
        response("tool_use", tool_use("get_price_summary", {"ticker": "BAD", "window": None})),
        response("end_turn", text(_report("Only 300 sessions are available, fewer than the 504 needed."))),
    )
    assert not run_agent(client, registry, CTX, Mode.RESEARCH, {"ticker": "BAD"}, "", AgentConfig()).withheld
    assert not run_agent(quoting, registry, CTX, Mode.RESEARCH, {"ticker": "BAD"}, "", AgentConfig()).withheld
