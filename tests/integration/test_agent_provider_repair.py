"""Verify agent repair budgets over real ChatGroq parsing and mocked Groq HTTP errors."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import respx

from research_agent.agents.planner import run_planner
from research_agent.agents.researcher import run_researcher_task
from research_agent.config.settings import Settings
from research_agent.llm.groq_client import create_llm
from research_agent.tools.registry import create_default_registry
from tests.integration.test_groq_client import URL, completion
from tests.unit.test_report_generator import _make_plan


def test_model_call_spacing_waits_only_for_same_model_and_reports_no_content() -> None:
    from research_agent.graph.events import event_scope
    from research_agent.llm.reliable import _model_pacer

    clock = [0.0]
    sleeps = []
    events = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    _model_pacer.cache_clear()
    with (
        patch("research_agent.llm.reliable.time.monotonic", side_effect=lambda: clock[0]),
        patch("research_agent.llm.reliable.time.sleep", side_effect=sleep),
        event_scope(events.append),
    ):
        _model_pacer("model-a").wait(60)
        clock[0] = 10
        _model_pacer("model-b").wait(60)
        _model_pacer("model-a").wait(60)
        clock[0] = 125
        _model_pacer("model-a").wait(60)
    assert sleeps == [50]
    waits = [event for event in events if event["event_type"] == "llm_wait"]
    assert len(waits) == 1 and waits[0]["duration_seconds"] == 50
    _model_pacer.cache_clear()


@respx.mock
def test_native_rejection_repairs_field_types_without_echoing_rejected_content() -> None:
    model = create_llm("planner", Settings(groq_api_key="test", structured_repair_attempts=1))
    invalid = _make_plan().model_dump()
    invalid["research_tasks"][0]["search_queries"] = "private-rejected-text"
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                400,
                json={
                    "error": {
                        "code": "json_validate_failed",
                        "message": "private-body",
                        "failed_generation": json.dumps(invalid),
                    }
                },
            ),
            httpx.Response(200, json=completion({"content": _make_plan().model_dump_json()})),
        ]
    )
    assert run_planner("Query", "Objective", model).research_tasks
    assert route.call_count == 2
    correction = json.loads(route.calls.last.request.content)["messages"][-1]["content"]
    assert "research_tasks.0.search_queries" in correction
    assert "list_type" in correction
    assert "private-rejected-text" not in correction and "private-body" not in correction


def tool_message(name: str, arguments: str) -> dict:
    """Return a provider-format tool call without bypassing the ChatGroq parser."""
    return completion(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
    )


@respx.mock
def test_planner_repairs_provider_tool_use_failed_once_without_echoing_body() -> None:
    model = create_llm("planner", Settings(groq_api_key="test", structured_repair_attempts=1))
    model.structured_output_method = "function_calling"  # Exercise legacy tool-output repair.
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                400,
                json={
                    "error": {
                        "code": "tool_use_failed",
                        "message": "dummy-private-provider-body",
                        "failed_generation": "private",
                    }
                },
            ),
            httpx.Response(200, json=tool_message("ResearchPlan", _make_plan().model_dump_json())),
        ]
    )
    plan = run_planner("Query", "Objective", model)
    assert plan.research_tasks
    assert route.call_count == 2
    correction = json.loads(route.calls.last.request.content)["messages"][-1]["content"]
    assert "dummy-private" not in correction
    assert "Output repair" in correction


@respx.mock
def test_researcher_repairs_provider_tool_use_failed_then_executes_real_tool() -> None:
    model = create_llm("researcher", Settings(groq_api_key="test", structured_repair_attempts=1))
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                400, json={"error": {"code": "tool_use_failed", "message": "private-body"}}
            ),
            httpx.Response(200, json=tool_message("calculator", '{"expression":"15+27"}')),
            httpx.Response(200, json=completion({"content": "42"})),
        ]
    )
    registry = create_default_registry()
    result = run_researcher_task("task", "Calculate", [], model, registry)
    assert route.call_count == 3
    assert registry.get_call_count("calculator") == 1
    assert result["tool_calls"][0].success
    assert "42" in result["tool_results"][0].output


@respx.mock
def test_researcher_malformed_provider_output_stops_at_repair_limit() -> None:
    model = create_llm("researcher", Settings(groq_api_key="test", structured_repair_attempts=1))
    route = respx.post(URL).respond(
        400, json={"error": {"code": "tool_use_failed", "message": "private-body"}}
    )
    registry = create_default_registry()
    result = run_researcher_task("task", "Calculate", [], model, registry)
    assert route.call_count == 2
    assert not result["evidence"] and not result["tool_calls"]
    assert result["error_details"] == ["Malformed tool calls exhausted the repair budget"]
