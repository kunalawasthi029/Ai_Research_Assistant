"""Unit tests for the Planner agent."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from research_agent.agents.planner import run_planner
from research_agent.schemas.plan import ResearchPlan
from tests.fake_llm import FakeChatModel


def _make_plan_response() -> str:
    """Create a valid ResearchPlan JSON response."""
    return json.dumps(
        {
            "research_objective": "Understand quantum computing advances",
            "sub_questions": [
                {
                    "question_id": "q1",
                    "question": "What are the latest quantum computing breakthroughs?",
                    "priority": 1,
                    "expected_source_types": ["web", "arxiv"],
                },
                {
                    "question_id": "q2",
                    "question": "Who are the key players in quantum computing?",
                    "priority": 2,
                    "expected_source_types": ["web", "wikipedia"],
                },
            ],
            "research_tasks": [
                {
                    "task_id": "t1",
                    "question_id": "q1",
                    "description": "Search for recent quantum computing breakthroughs",
                    "search_queries": ["quantum computing breakthroughs 2024"],
                    "required_evidence": "Recent papers or articles",
                    "expected_source_types": ["arxiv", "web"],
                    "completion_criteria": "At least 3 sources found",
                    "priority": 1,
                },
                {
                    "task_id": "t2",
                    "question_id": "q2",
                    "description": "Identify major quantum computing companies and labs",
                    "search_queries": ["quantum computing companies"],
                    "required_evidence": "Company or lab names with their contributions",
                    "expected_source_types": ["web"],
                    "completion_criteria": "At least 5 entities identified",
                    "priority": 2,
                },
            ],
            "scope": "balanced",
            "estimated_complexity": "moderate",
        }
    )


class TestRunPlanner:
    """Tests for the planner agent."""

    def test_creates_valid_plan(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_plan_response())])
        plan = run_planner(
            query="What are the latest advances in quantum computing?",
            objective="Research quantum computing advances",
            llm=llm,
        )
        assert isinstance(plan, ResearchPlan)
        assert len(plan.sub_questions) == 2
        assert len(plan.research_tasks) == 2
        assert plan.scope == "balanced"

    def test_plan_has_search_queries(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_plan_response())])
        plan = run_planner("test query", "test objective", llm)
        for task in plan.research_tasks:
            assert len(task.search_queries) >= 1

    def test_plan_tasks_link_to_questions(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_plan_response())])
        plan = run_planner("test query", "test objective", llm)
        question_ids = {q.question_id for q in plan.sub_questions}
        for task in plan.research_tasks:
            assert task.question_id in question_ids

    def test_revision_includes_feedback(self) -> None:
        """When feedback is provided, the prompt includes it."""
        llm = FakeChatModel(responses=[AIMessage(content=_make_plan_response())])

        previous_plan = ResearchPlan(
            research_objective="Old objective",
            sub_questions=[
                {
                    "question_id": "q1",
                    "question": "Old question",
                    "priority": 1,
                }
            ],
            research_tasks=[
                {
                    "task_id": "t1",
                    "question_id": "q1",
                    "description": "Old task",
                    "search_queries": ["old"],
                    "required_evidence": "old",
                    "completion_criteria": "old",
                    "priority": 1,
                }
            ],
        )

        _ = run_planner(
            query="test",
            objective="test",
            llm=llm,
            feedback="Add more sub-questions about applications",
            previous_plan=previous_plan,
        )

        # Check that the LLM received the feedback in its messages
        assert len(llm.invocations) == 1
        messages = llm.invocations[0]
        user_msg = messages[-1].content
        assert "Revision Request" in user_msg
        assert "Add more sub-questions" in user_msg
