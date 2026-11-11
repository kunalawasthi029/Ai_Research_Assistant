"""Exercise real ChatGroq serialization, parsing and retries against mocked HTTP."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, model_validator

from research_agent.config.settings import Settings
from research_agent.llm.groq_client import create_llm
from research_agent.llm.retry import GroqAPIError

URL = "https://api.groq.com/openai/v1/chat/completions"


def completion(message: dict, tokens: int = 12) -> dict:
    """Build a provider-compatible test response without contacting Groq."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "llama-3.1-8b-instant",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": tokens - 5, "total_tokens": tokens},
    }


class Answer(BaseModel):
    """Small structured contract for provider-parsing tests."""

    answer: int


@respx.mock
def test_current_model_uses_native_schema_and_does_not_bind_tools() -> None:
    llm = create_llm("planner", Settings(groq_api_key="test"))
    route = respx.post(URL).respond(
        200, json=completion({"content": '{"answer": 4}', "reasoning": "private"})
    )
    answer = llm.with_structured_output(Answer).invoke("2+2")
    assert answer.answer == 4
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in body


def test_strict_schema_closes_nested_reports_without_changing_the_local_contract() -> None:
    from research_agent.llm.reliable import _closed_schema
    from research_agent.schemas.report import Report

    original = Report.model_json_schema()
    wire = _closed_schema(Report)
    assert wire is not None
    for node in [wire, *wire["$defs"].values()]:
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
    assert (
        wire["properties"]["recommendations"]["anyOf"]
        == original["properties"]["recommendations"]["anyOf"]
    )
    assert Report.model_json_schema() == original


@respx.mock
def test_arbitrary_mapping_schema_keeps_its_entries_in_best_effort_mode() -> None:
    class MappingAnswer(BaseModel):
        entries: dict[str, str]

    llm = create_llm("planner", Settings(groq_api_key="test"))
    route = respx.post(URL).respond(
        200, json=completion({"content": '{"entries":{"fact":"kept"}}'})
    )
    result = llm.with_structured_output(MappingAnswer).invoke("test")
    assert result.entries == {"fact": "kept"}
    assert (
        "strict"
        not in json.loads(route.calls.last.request.content)["response_format"]["json_schema"]
    )


@respx.mock
def test_native_schema_agent_repair_stays_bounded() -> None:
    from research_agent.llm.structured import invoke_structured

    llm = create_llm("planner", Settings(groq_api_key="test", structured_repair_attempts=1))
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(200, json=completion({"content": '{"answer": "bad"}'})),
            httpx.Response(200, json=completion({"content": '{"answer": 4}'})),
        ]
    )
    assert invoke_structured(llm, Answer, [HumanMessage(content="2+2")]).answer == 4
    assert route.call_count == 2


@respx.mock
def test_native_provider_json_rejection_gets_one_safe_repair() -> None:
    from research_agent.llm.structured import invoke_structured

    llm = create_llm("planner", Settings(groq_api_key="test", structured_repair_attempts=1))
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                400,
                json={
                    "error": {"code": "json_validate_failed", "message": "private-provider-body"}
                },
            ),
            httpx.Response(200, json=completion({"content": '{"answer": 4}'})),
        ]
    )
    assert invoke_structured(llm, Answer, [HumanMessage(content="2+2")]).answer == 4
    assert route.call_count == 2
    assert (
        "private-provider-body"
        not in json.loads(route.calls[1].request.content)["messages"][-1]["content"]
    )


@respx.mock
def test_native_invalid_request_is_not_repaired() -> None:
    from research_agent.llm.structured import invoke_structured

    llm = create_llm("planner", Settings(groq_api_key="test", structured_repair_attempts=1))
    route = respx.post(URL).respond(
        400, json={"error": {"code": "invalid_parameter", "message": "private"}}
    )
    with pytest.raises(GroqAPIError):
        invoke_structured(llm, Answer, [HumanMessage(content="2+2")])
    assert route.call_count == 1


@respx.mock
def test_error_event_contains_status_without_provider_body() -> None:
    from research_agent.graph.events import event_scope

    llm = create_llm("supervisor", Settings(groq_api_key="test-secret", llm_max_retries=0))
    respx.post(URL).mock(
        return_value=httpx.Response(413, json={"error": {"message": "private-prompt-and-key"}})
    )
    events = []
    with event_scope(events.append), pytest.raises(GroqAPIError):
        llm.invoke([HumanMessage(content="test")])
    event = next(e for e in events if e["event_type"] == "llm_error")
    assert event["http_status"] == 413
    assert event["error_type"] == "APIStatusError"
    assert "private-prompt-and-key" not in json.dumps(events)
    assert "test-secret" not in json.dumps(events)


@respx.mock
def test_unexpected_structured_function_name_gets_one_safe_agent_repair() -> None:
    from research_agent.llm.structured import invoke_structured

    wrong = structured('{"answer": 4}')
    wrong["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "private-wrong-function"
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(200, json=wrong),
            httpx.Response(200, json=structured('{"answer": 4}')),
        ]
    )
    llm = create_llm("planner", Settings(groq_api_key="test-secret", structured_repair_attempts=1))
    llm.structured_output_method = "function_calling"  # Explicit legacy protocol regression.
    result = invoke_structured(llm, Answer, [HumanMessage(content="2+2")])
    assert result.answer == 4
    assert route.call_count == 2
    body = json.loads(route.calls[1].request.content)
    assert "Answer" in body["messages"][-1]["content"]
    assert "private-wrong-function" not in body["messages"][-1]["content"]


@respx.mock
def test_application_key_error_inside_schema_is_not_misclassified_as_model_output() -> None:
    class BrokenSchema(BaseModel):
        answer: int

        @model_validator(mode="after")
        def application_bug(self):
            raise KeyError("application bug")

    response = structured('{"answer": 4}')
    response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "BrokenSchema"
    route = respx.post(URL).respond(200, json=response)
    llm = create_llm("planner", Settings(groq_api_key="test"))
    with pytest.raises(KeyError, match="application bug"):
        llm.with_structured_output(BrokenSchema, method="function_calling").invoke("test")
    assert route.call_count == 1


def structured(arguments: str) -> dict:
    """Return a schema tool call in the provider wire format."""
    return completion(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "Answer", "arguments": arguments},
                }
            ],
        }
    )


@respx.mock
def test_sdk_retries_honors_retry_after_and_logs_usage(caplog: pytest.LogCaptureFixture) -> None:
    llm = create_llm("supervisor", Settings(groq_api_key="unit-test-secret", llm_max_retries=1))
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                429, headers={"retry-after": "0.01"}, json={"error": {"message": "limited"}}
            ),
            httpx.Response(
                200, json=completion({"content": "OK", "reasoning_content": "private analysis"})
            ),
        ]
    )
    with patch("groq._base_client.time.sleep") as sleep, caplog.at_level("INFO"):
        result = llm.invoke([HumanMessage(content="test")])
    assert result.content == "OK"
    assert "reasoning_content" not in result.additional_kwargs
    assert route.call_count == 2
    sleep.assert_called_once_with(0.01)
    assert "total_tokens=12" in caplog.text
    assert "unit-test-secret" not in caplog.text
    assert llm.request_timeout == 30


@respx.mock
def test_auth_error_is_not_retried_or_echoed() -> None:
    llm = create_llm("supervisor", Settings(groq_api_key="unit-test-secret"))
    route = respx.post(URL).respond(401, json={"error": {"message": "unit-test-secret"}})
    with pytest.raises(GroqAPIError) as error:
        llm.invoke("test")
    assert "unit-test-secret" not in str(error.value)
    assert route.call_count == 1


@respx.mock
def test_rate_limit_retry_budget_terminates() -> None:
    llm = create_llm("supervisor", Settings(groq_api_key="test", llm_max_retries=1))
    route = respx.post(URL).respond(429, json={"error": {"message": "limited"}})
    with patch("groq._base_client.time.sleep"), pytest.raises(GroqAPIError, match="RateLimitError"):
        llm.invoke("test")
    assert route.call_count == 2


@respx.mock
def test_request_timeout_is_bounded_and_sanitized() -> None:
    llm = create_llm(
        "supervisor", Settings(groq_api_key="test", llm_max_retries=0, llm_timeout_seconds=1)
    )
    route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("provider body must not escape"))
    with pytest.raises(GroqAPIError, match="APITimeoutError") as error:
        llm.invoke("test")
    assert "provider body" not in str(error.value)
    assert route.call_count == 1


def test_concurrency_wait_is_bounded() -> None:
    from research_agent.llm.reliable import _call_slots

    llm = create_llm("supervisor", Settings(groq_api_key="test", llm_max_concurrency=1))
    llm.slot_timeout = 0.01
    slot = _call_slots(1)
    assert slot.acquire(timeout=0.01)
    try:
        with pytest.raises(GroqAPIError, match="concurrency wait timed out"):
            llm.invoke("test")
    finally:
        slot.release()


@respx.mock
def test_structured_schema_failure_is_repaired_with_bounded_retry() -> None:
    llm = create_llm("planner", Settings(groq_api_key="test"))
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(200, json=structured('{"answer": "bad"}')),
            httpx.Response(200, json=structured('{"answer": 42}')),
        ]
    )
    result = llm.with_structured_output(Answer, method="function_calling").invoke("Return answer")
    assert result.answer == 42
    assert route.call_count == 2
    body = json.loads(route.calls.last.request.content)
    assert "required fields" in body["messages"][-1]["content"]


@respx.mock
def test_structured_repair_exhaustion_is_sanitized() -> None:
    llm = create_llm("planner", Settings(groq_api_key="test", structured_repair_attempts=1))
    route = respx.post(URL).respond(200, json=structured('{"answer": "invalid"}'))
    with pytest.raises(GroqAPIError, match="bounded repair"):
        llm.with_structured_output(Answer, method="function_calling").invoke("test")
    assert route.call_count == 2


@respx.mock
def test_real_tool_call_parsing_execution_and_response_roundtrip() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    llm = create_llm("researcher", Settings(groq_api_key="test"))
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=completion(
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "add-1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a": 15, "b": 27}'},
                            }
                        ],
                    }
                ),
            ),
            httpx.Response(200, json=completion({"content": "42"})),
        ]
    )
    bound = llm.bind_tools([add])
    messages = [HumanMessage(content="Add 15 and 27 using add")]
    response = bound.invoke(messages)
    call = response.tool_calls[0]
    answer = add.invoke(call["args"])
    final = bound.invoke(
        [*messages, response, ToolMessage(content=str(answer), tool_call_id=call["id"])]
    )
    assert answer == 42
    assert final.content == "42"
    assert route.call_count == 2
    assert json.loads(route.calls.last.request.content)["messages"][-1]["role"] == "tool"
