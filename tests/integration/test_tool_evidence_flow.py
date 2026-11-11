"""Exercise normalized tool results through the researcher into the SQLite evidence store."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from research_agent.agents.researcher import run_researcher_task
from research_agent.config.settings import Settings
from research_agent.persistence.models import RunRow
from research_agent.service.research_service import ResearchService
from research_agent.tools.registry import create_default_registry
from tests.fake_llm import FakeChatModel


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    """Build an observable model-selected tool call."""
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


def test_search_sources_flow_through_real_registry_researcher_and_sqlite() -> None:
    model = FakeChatModel(
        responses=[tool_call("web_search", {"query": "test"}), AIMessage(content="Finished")]
    )
    rows = [
        {"title": "Source A", "href": "https://example.org/a", "body": "First retrieved fact"},
        {"title": "Source B", "href": "https://example.org/b", "body": "Second retrieved fact"},
    ]
    registry = create_default_registry()
    with patch("ddgs.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.return_value = rows
        result = run_researcher_task("task", "Research test", ["test"], model, registry)
    assert len(result["evidence"]) == 2
    assert result["tool_results"][0].evidence_ids == [
        item.evidence_id for item in result["evidence"]
    ]
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        service = ResearchService(settings=Settings(groq_api_key="dummy"), db_conn=connection)
        service._run_repo.create(RunRow(id="run", thread_id="thread", query="test"))
        service._persist_results("run", result)
        stored = service.get_evidence("run")
        assert {item.url for item in stored} == {row["href"] for row in rows}
        assert {item.title for item in stored} == {"Source A", "Source B"}
        assert all(item.tool_name == "web_search" for item in stored)
    finally:
        connection.close()


def test_provider_failure_is_feedback_and_consumes_budget_without_evidence() -> None:
    model = FakeChatModel(
        responses=[tool_call("web_search", {"query": "test"}), AIMessage(content="Try later")]
    )
    registry = create_default_registry(call_limit=1)
    with patch("ddgs.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.side_effect = RuntimeError(
            "private-key-from-provider"
        )
        result = run_researcher_task("task", "Research test", ["test"], model, registry)
    assert not result["evidence"]
    assert not result["tool_calls"][0].success
    assert registry.get_call_count("web_search") == 1
    assert not registry.can_call("web_search")
    feedback = [message for message in model.invocations[-1] if isinstance(message, ToolMessage)]
    assert "Search provider unavailable" in feedback[0].content
    assert "private-key" not in str(result)


def test_unstructured_mock_text_is_not_treated_as_citable_evidence() -> None:
    model = FakeChatModel(
        responses=[tool_call("web_search", {"query": "test"}), AIMessage(content="Done")]
    )
    registry = create_default_registry()
    with patch.object(
        type(registry.get_tool("web_search")), "invoke", return_value="BLOCKED: do not cite"
    ):
        result = run_researcher_task("task", "Research test", ["test"], model, registry)
    assert not result["evidence"]
    assert not result["tool_calls"][0].success
