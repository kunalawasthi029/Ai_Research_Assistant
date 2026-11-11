"""Unit tests for FakeChatModel and FakeStructuredOutput."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from tests.fake_llm import FakeChatModel


class TestFakeChatModel:
    """Tests for the deterministic test LLM."""

    def test_returns_responses_in_order(self) -> None:
        llm = FakeChatModel(
            responses=[
                AIMessage(content="first"),
                AIMessage(content="second"),
            ]
        )
        r1 = llm.invoke([HumanMessage(content="q1")])
        r2 = llm.invoke([HumanMessage(content="q2")])
        assert r1.content == "first"
        assert r2.content == "second"

    def test_records_invocations(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content="ok")])
        llm.invoke([HumanMessage(content="hello")])
        assert len(llm.invocations) == 1
        assert llm.invocations[0][0].content == "hello"

    def test_exhausted_responses_raises(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content="only one")])
        llm.invoke([HumanMessage(content="q1")])
        with pytest.raises(IndexError, match="exhausted"):
            llm.invoke([HumanMessage(content="q2")])

    def test_bind_tools_returns_self(self) -> None:
        llm = FakeChatModel(responses=[])
        result = llm.bind_tools([])
        assert result is llm

    def test_tool_calls_in_response(self) -> None:
        """Verify FakeChatModel can return AIMessages with tool_calls."""
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "web_search",
                    "args": {"query": "quantum computing"},
                }
            ],
        )
        llm = FakeChatModel(responses=[tool_call_msg])
        result = llm.invoke([HumanMessage(content="search for quantum computing")])
        assert result.tool_calls[0]["name"] == "web_search"
        assert result.tool_calls[0]["args"]["query"] == "quantum computing"


class TestFakeStructuredOutput:
    """Tests for the structured output wrapper."""

    def test_parses_json_response(self) -> None:
        class Answer(BaseModel):
            text: str
            score: float

        json_response = json.dumps({"text": "hello", "score": 0.9})
        llm = FakeChatModel(responses=[AIMessage(content=json_response)])
        structured = llm.with_structured_output(Answer)
        result = structured.invoke("test question")
        assert isinstance(result, Answer)
        assert result.text == "hello"
        assert result.score == 0.9

    def test_invalid_json_raises(self) -> None:
        class Answer(BaseModel):
            text: str

        llm = FakeChatModel(responses=[AIMessage(content="not json")])
        structured = llm.with_structured_output(Answer)
        with pytest.raises(ValueError, match="not valid JSON"):
            structured.invoke("test")

    def test_schema_validation_error(self) -> None:
        class StrictSchema(BaseModel):
            value: int

        # Valid JSON but wrong type for schema
        llm = FakeChatModel(responses=[AIMessage(content=json.dumps({"value": "not_an_int"}))])
        structured = llm.with_structured_output(StrictSchema)
        with pytest.raises(Exception):  # Pydantic ValidationError
            structured.invoke("test")
