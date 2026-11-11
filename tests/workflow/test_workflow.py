"""Workflow tests: end-to-end graph execution with FakeChatModel.

These tests compile the real StateGraph and run it with pre-scripted
LLM responses — no API calls, but real graph routing and state updates.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from research_agent.graph.state import create_initial_state
from research_agent.graph.workflow import build_graph
from research_agent.tools.registry import ToolRegistry
from tests.fake_llm import FakeChatModel

# ──────────────────── Shared fixtures ────────────────────


def _make_plan_json(task_id: str = "t1") -> str:
    return json.dumps(
        {
            "research_objective": "Test research",
            "sub_questions": [
                {
                    "question_id": "q1",
                    "question": "What is X?",
                    "priority": 1,
                }
            ],
            "research_tasks": [
                {
                    "task_id": task_id,
                    "question_id": "q1",
                    "description": "Search for X",
                    "search_queries": ["X research"],
                    "required_evidence": "Articles about X",
                    "completion_criteria": "At least 1 source",
                    "priority": 1,
                }
            ],
        }
    )


def _make_supervisor_decision_json(action: str, target: str, junction: str) -> str:
    return json.dumps(
        {
            "decision_type": junction,
            "selected_action": action,
            "reasoning": f"Decided to {action}",
            "target_node": target,
        }
    )


def _make_analysis_json() -> str:
    return json.dumps(
        {
            "summary": "Evidence shows X is significant.",
            "findings": [
                {
                    "finding_id": "f1",
                    "research_task_id": "t1",
                    "claim": "X has been demonstrated",
                    "supporting_evidence": ["ev1"],
                    "source_url": "https://example.com/x",
                    "source_title": "Source about X",
                    "source_type": "web",
                    "relevance_score": 0.9,
                    "confidence": 0.85,
                    "verification_status": "verified",
                }
            ],
            "gap_report": {
                "has_gaps": False,
                "coverage_score": 0.9,
            },
            "total_evidence_count": 1,
            "unique_source_count": 1,
        }
    )


def _make_report_json() -> str:
    now = datetime.now(UTC).isoformat()
    return json.dumps(
        {
            "title": "Report on X",
            "executive_summary": "X is important and well-researched.",
            "research_objective": "Understand X",
            "methodology": "Systematic search and analysis.",
            "key_findings": ["X is significant", "X has multiple applications"],
            "detailed_analysis": [
                {
                    "heading": "Overview of X",
                    "content": "X has been studied extensively. " * 20 + "[ref1]",
                    "evidence_ids": ["ev1"],
                }
            ],
            "limitations": ["Limited to English sources"],
            "conclusion": "X is a promising area of research.",
            "references": [
                {
                    "ref_id": "ref1",
                    "evidence_id": "ev1",
                    "url": "https://example.com/x",
                    "title": "Source about X",
                    "source_type": "web",
                    "accessed_at": now,
                }
            ],
        }
    )


def _make_validation_result_json(status: str = "pass", score: float = 0.9) -> str:
    return json.dumps(
        {
            "status": status,
            "layer": "semantic",
            "scores": [
                {"criterion": criterion, "score": score, "passed": status == "pass"}
                for criterion in (
                    "relevance",
                    "evidence_alignment",
                    "completeness",
                    "logical_consistency",
                    "objectivity",
                )
            ],
            "issues": [],
            "feedback": "All good." if status == "pass" else "Needs work.",
            "overall_score": score,
        }
    )


def _make_mock_registry() -> ToolRegistry:
    registry = ToolRegistry()
    mock_tool = MagicMock()
    mock_tool.name = "web_search"
    mock_tool.description = "Search the web"
    mock_tool.invoke.return_value = '{"status":"success","items":[{"url":"https://example.com/x","title":"Source about X","content":"X has been demonstrated"}]}'
    registry.register(mock_tool, call_limit=20)
    return registry


def _build_happy_path_llms() -> dict[str, FakeChatModel]:
    """Build FakeChatModels for a happy-path run.

    Flow: planner → supervisor(approve) → researcher(1 tool call, then done)
    → analyst → supervisor(sufficient) → report_generator → validator
    → supervisor(accept) → finalize
    """
    return {
        "planner": FakeChatModel(
            responses=[
                AIMessage(content=_make_plan_json()),
            ]
        ),
        "supervisor": FakeChatModel(
            responses=[
                # plan_review: approve
                AIMessage(
                    content=_make_supervisor_decision_json("approve", "researcher", "plan_review")
                ),
                # evidence_review: sufficient
                AIMessage(
                    content=_make_supervisor_decision_json(
                        "sufficient", "report_generator", "evidence_review"
                    )
                ),
                # final_review: accept
                AIMessage(
                    content=_make_supervisor_decision_json("accept", "finalize", "final_review")
                ),
            ]
        ),
        "researcher": FakeChatModel(
            responses=[
                # Tool call
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "web_search",
                            "args": {"query": "X research"},
                        }
                    ],
                ),
                # Done
                AIMessage(content="Found relevant results about X."),
            ]
        ),
        "analyst": FakeChatModel(
            responses=[
                AIMessage(content=_make_analysis_json()),
            ]
        ),
        "report": FakeChatModel(
            responses=[
                AIMessage(content=_make_report_json()),
            ]
        ),
        "validator": FakeChatModel(
            responses=[
                AIMessage(content=_make_validation_result_json()),
            ]
        ),
    }


# ──────────────────── Tests ────────────────────


class TestHappyPath:
    """Test: query in → report out, no loops."""

    def test_end_to_end(self) -> None:
        llms = _build_happy_path_llms()
        registry = _make_mock_registry()

        compiled = build_graph(
            planner_llm=llms["planner"],
            researcher_llm=llms["researcher"],
            analyst_llm=llms["analyst"],
            report_llm=llms["report"],
            supervisor_llm=llms["supervisor"],
            validator_llm=llms["validator"],
            registry=registry,
        )

        initial = create_initial_state(
            research_id="test-run",
            thread_id="test-thread",
            query="What is X?",
        )

        final_state = compiled.invoke(initial)

        # Verify completion
        assert final_state["execution_status"] == "completed"

        # Verify report was generated
        assert final_state["report"] is not None
        assert final_state["report"].title == "Report on X"

        # Verify evidence was collected
        assert len(final_state["evidence"]) >= 1
        assert (
            final_state["report"].references[0].evidence_id
            == final_state["evidence"][0].evidence_id
        )
        assert final_state["report"].references[0].url == final_state["evidence"][0].url

        # Verify tool calls were made
        assert len(final_state["tool_calls"]) >= 1

        # Verify decisions were recorded
        assert len(final_state["agent_decisions"]) == 3  # 3 supervisor junctions


class TestMaxIterations:
    """Test: research iteration cap terminates the run."""

    def test_research_cap_terminates(self) -> None:
        """When supervisor always says needs_research, the cap forces report generation."""
        # Supervisor: approve plan, then keep saying needs_research at evidence_review
        # After MAX_RESEARCH_ITERATIONS, routing forces report_generator
        supervisor_responses = [
            # plan_review: approve
            AIMessage(
                content=_make_supervisor_decision_json("approve", "researcher", "plan_review")
            ),
            # evidence_review round 1: needs_research
            AIMessage(
                content=_make_supervisor_decision_json(
                    "needs_research", "researcher", "evidence_review"
                )
            ),
            # evidence_review round 2: needs_research
            AIMessage(
                content=_make_supervisor_decision_json(
                    "needs_research", "researcher", "evidence_review"
                )
            ),
            # evidence_review round 3: needs_research (but cap=3, so forced to report)
            AIMessage(
                content=_make_supervisor_decision_json(
                    "needs_research", "researcher", "evidence_review"
                )
            ),
            # final_review: accept (after report)
            AIMessage(content=_make_supervisor_decision_json("accept", "finalize", "final_review")),
        ]

        # Researcher: always does one tool call per round
        researcher_responses = []
        for i in range(4):  # up to 4 rounds
            researcher_responses.extend(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": f"call_{i}",
                                "name": "web_search",
                                "args": {"query": f"search {i}"},
                            }
                        ],
                    ),
                    AIMessage(content=f"Found results round {i}."),
                ]
            )

        llms = {
            "planner": FakeChatModel(
                responses=[
                    AIMessage(content=_make_plan_json(task_id)) for task_id in ("t1", "t2", "t3")
                ]
            ),
            "supervisor": FakeChatModel(responses=supervisor_responses),
            "researcher": FakeChatModel(responses=researcher_responses),
            "analyst": FakeChatModel(
                responses=[
                    AIMessage(content=_make_analysis_json()),
                    AIMessage(content=_make_analysis_json()),
                    AIMessage(content=_make_analysis_json()),
                ]
            ),
            "report": FakeChatModel(responses=[AIMessage(content=_make_report_json())]),
            "validator": FakeChatModel(
                responses=[AIMessage(content=_make_validation_result_json())]
            ),
        }

        compiled = build_graph(
            planner_llm=llms["planner"],
            researcher_llm=llms["researcher"],
            analyst_llm=llms["analyst"],
            report_llm=llms["report"],
            supervisor_llm=llms["supervisor"],
            validator_llm=llms["validator"],
            registry=_make_mock_registry(),
        )

        initial = create_initial_state(
            research_id="iter-test",
            thread_id="iter-thread",
            query="What is X?",
        )

        final_state = compiled.invoke(initial)
        assert final_state["execution_status"] == "best_effort"
        assert final_state["research_iteration"] >= 3
        assert len(final_state["tool_calls"]) == 3
        assert {task.task_id for task in final_state["research_plan"].research_tasks} == {
            "t1",
            "t2",
            "t3",
        }


class TestValidationLoop:
    """Test: validation failure triggers revision loop."""

    def test_revision_loop(self) -> None:
        """Supervisor asks for revision after validation failure, then accepts."""
        supervisor_responses = [
            # plan_review: approve
            AIMessage(
                content=_make_supervisor_decision_json("approve", "researcher", "plan_review")
            ),
            # evidence_review: sufficient
            AIMessage(
                content=_make_supervisor_decision_json(
                    "sufficient", "report_generator", "evidence_review"
                )
            ),
            # final_review 1: needs_revision
            AIMessage(
                content=_make_supervisor_decision_json(
                    "needs_revision", "report_generator", "final_review"
                )
            ),
            # final_review 2: accept
            AIMessage(content=_make_supervisor_decision_json("accept", "finalize", "final_review")),
        ]

        llms = {
            "planner": FakeChatModel(responses=[AIMessage(content=_make_plan_json())]),
            "supervisor": FakeChatModel(responses=supervisor_responses),
            "researcher": FakeChatModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": "web_search",
                                "args": {"query": "X research"},
                            }
                        ],
                    ),
                    AIMessage(content="Found results."),
                ]
            ),
            "analyst": FakeChatModel(responses=[AIMessage(content=_make_analysis_json())]),
            "report": FakeChatModel(
                responses=[
                    AIMessage(content=_make_report_json()),  # v1 (will fail validation)
                    AIMessage(content=_make_report_json()),  # v2 (revised)
                ]
            ),
            "validator": FakeChatModel(
                responses=[
                    AIMessage(content=_make_validation_result_json("needs_revision", 0.4)),
                    AIMessage(content=_make_validation_result_json("pass", 0.9)),
                ]
            ),
        }

        compiled = build_graph(
            planner_llm=llms["planner"],
            researcher_llm=llms["researcher"],
            analyst_llm=llms["analyst"],
            report_llm=llms["report"],
            supervisor_llm=llms["supervisor"],
            validator_llm=llms["validator"],
            registry=_make_mock_registry(),
        )

        initial = create_initial_state(
            research_id="loop-test",
            thread_id="loop-thread",
            query="What is X?",
        )

        final_state = compiled.invoke(initial)
        assert final_state["execution_status"] == "completed"
        assert final_state["revision_count"] == 1  # one revision after the initial version
        # Decisions: plan_review, evidence_review, final_review x2
        assert len(final_state["agent_decisions"]) == 4
