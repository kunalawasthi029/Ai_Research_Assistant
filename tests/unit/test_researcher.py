"""Unit tests for the Researcher agent."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from research_agent.agents.researcher import run_researcher_task
from research_agent.tools.registry import ToolRegistry
from tests.fake_llm import FakeChatModel


def test_last_allowed_tool_result_gets_one_terminal_assessment() -> None:
    llm = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"id": "last", "name": "web_search", "args": {"query": "test"}}],
            ),
            AIMessage(content="TASK_COMPLETE"),
        ]
    )
    result = run_researcher_task(
        "t1", "Find evidence", [], llm, _make_registry_with_mock_tool(), max_tool_calls=1
    )
    assert result["status"] == "done"
    assert len(result["tool_calls"]) == 1
    assert len(llm.invocations) == 2
    assert "New breakthroughs" in str(llm.invocations[-1])
    assert "tool budget is closed" in llm.invocations[-1][-1].content


def test_terminal_assessment_cannot_execute_another_tool_or_retry() -> None:
    call = AIMessage(
        content="", tool_calls=[{"id": "call", "name": "web_search", "args": {"query": "test"}}]
    )
    llm = FakeChatModel(responses=[call, call])
    result = run_researcher_task(
        "t1", "Find evidence", [], llm, _make_registry_with_mock_tool(), max_tool_calls=1
    )
    assert result["status"] == "budget_exhausted"
    assert len(result["tool_calls"]) == 1
    assert len(llm.invocations) == 2


def test_original_source_urls_reach_researcher_even_when_task_omits_them() -> None:
    llm = FakeChatModel(responses=[AIMessage(content="No tools requested")])
    objective = "Compare https://example.org/primary-one and https://example.org/primary-two."
    run_researcher_task(
        "t1",
        "Compare the documented properties",
        [],
        llm,
        _make_registry_with_mock_tool(),
        research_objective=objective,
    )
    prompt = llm.invocations[0][-1].content
    assert '"provided_source_urls"' in prompt
    assert "https://example.org/primary-one" in prompt
    assert "https://example.org/primary-two" in prompt
    assert objective in prompt


def _make_registry_with_mock_tool() -> ToolRegistry:
    """Create a registry with a mock web_search tool."""
    registry = ToolRegistry()
    mock_tool = MagicMock()
    mock_tool.name = "web_search"
    mock_tool.description = "Search the web"
    mock_tool.invoke.return_value = '{"status":"success","items":[{"url":"https://example.com/qc","title":"Quantum Computing Advances","content":"New breakthroughs"}]}'
    registry.register(mock_tool, call_limit=10)
    return registry


class TestRunResearcherTask:
    """Tests for the researcher agent's tool-calling loop."""

    def test_no_tool_calls(self) -> None:
        """When the model doesn't request tools, return empty results."""
        llm = FakeChatModel(
            responses=[
                AIMessage(content="I have enough information from context."),
            ]
        )
        registry = _make_registry_with_mock_tool()

        result = run_researcher_task(
            task_id="t1",
            task_description="Test task",
            search_queries=["test"],
            llm=llm,
            registry=registry,
            max_tool_calls=5,
        )

        assert result["evidence"] == []
        assert result["tool_calls"] == []

    def test_single_tool_call(self) -> None:
        """Model requests one tool call, then finishes."""
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
        final_msg = AIMessage(content="Found relevant results.")

        llm = FakeChatModel(responses=[tool_call_msg, final_msg])
        registry = _make_registry_with_mock_tool()

        result = run_researcher_task(
            task_id="t1",
            task_description="Research quantum computing",
            search_queries=["quantum computing"],
            llm=llm,
            registry=registry,
            max_tool_calls=5,
        )

        assert len(result["evidence"]) == 1
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0].success is True
        assert result["tool_calls"][0].tool_name == "web_search"

    def test_tool_budget_enforced(self) -> None:
        """Tool calls stop when budget is reached."""
        # Model always wants to call a tool
        responses = []
        for i in range(5):
            responses.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"call_{i}",
                            "name": "web_search",
                            "args": {"query": f"search {i}"},
                        }
                    ],
                )
            )
        # Final message if budget is hit
        responses.append(AIMessage(content="Budget reached."))

        llm = FakeChatModel(responses=responses)
        registry = _make_registry_with_mock_tool()

        result = run_researcher_task(
            task_id="t1",
            task_description="Extensive research",
            search_queries=["test"],
            llm=llm,
            registry=registry,
            max_tool_calls=3,  # Only allow 3
        )

        assert len(result["tool_calls"]) == 3

    def test_tool_failure_fed_back(self) -> None:
        """When a tool fails, the error is fed back to the model."""
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "web_search",
                    "args": {"query": "test"},
                }
            ],
        )
        final_msg = AIMessage(content="Tool failed, moving on.")

        llm = FakeChatModel(responses=[tool_call_msg, final_msg])

        # Create a registry where the tool raises an error
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        mock_tool.description = "Search"
        mock_tool.invoke.side_effect = RuntimeError("Network timeout")
        registry.register(mock_tool)

        result = run_researcher_task(
            task_id="t1",
            task_description="Test",
            search_queries=["test"],
            llm=llm,
            registry=registry,
            max_tool_calls=5,
        )

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0].success is False
        assert "RuntimeError" in result["tool_calls"][0].error
        assert len(result["error_details"]) == 1

    def test_unknown_tool_handled(self) -> None:
        """When model requests a non-existent tool, error is fed back."""
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "nonexistent_tool",
                    "args": {},
                }
            ],
        )
        final_msg = AIMessage(content="OK, no tool available.")

        llm = FakeChatModel(responses=[tool_call_msg, final_msg])
        registry = _make_registry_with_mock_tool()

        result = run_researcher_task(
            task_id="t1",
            task_description="Test",
            search_queries=["test"],
            llm=llm,
            registry=registry,
            max_tool_calls=5,
        )

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0].success is False
