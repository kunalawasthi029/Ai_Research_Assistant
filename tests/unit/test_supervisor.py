"""Unit tests for the Supervisor agent."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from research_agent.agents.supervisor import (
    LEGAL_TRANSITIONS,
    _validate_decision,
    run_supervisor,
)
from research_agent.schemas.decisions import SupervisorDecision
from tests.fake_llm import FakeChatModel
from tests.unit.test_report_generator import _make_plan


class TestLegalTransitions:
    """Tests for the transition table."""

    def test_plan_review_has_approve_and_revise(self) -> None:
        assert "approve" in LEGAL_TRANSITIONS["plan_review"]
        assert "revise" in LEGAL_TRANSITIONS["plan_review"]

    def test_evidence_review_has_sufficient_and_needs_research(self) -> None:
        assert "sufficient" in LEGAL_TRANSITIONS["evidence_review"]
        assert "needs_research" in LEGAL_TRANSITIONS["evidence_review"]

    def test_final_review_has_all_actions(self) -> None:
        tr = LEGAL_TRANSITIONS["final_review"]
        assert "accept" in tr
        assert "needs_revision" in tr
        assert "needs_research" in tr
        assert "accept_best_effort" in tr


class TestValidateDecision:
    """Tests for decision validation against legal transitions."""

    def test_valid_decision_passes(self) -> None:
        decision = SupervisorDecision(
            decision_type="plan_review",
            selected_action="approve",
            reasoning="Plan is good",
            target_node="researcher",
        )
        result = _validate_decision(decision, "plan_review")
        assert result.selected_action == "approve"
        assert result.target_node == "researcher"

    def test_corrects_wrong_target_node(self) -> None:
        decision = SupervisorDecision(
            decision_type="plan_review",
            selected_action="approve",
            reasoning="Plan is good",
            target_node="wrong_node",  # Wrong target
        )
        result = _validate_decision(decision, "plan_review")
        assert result.selected_action == "approve"
        assert result.target_node == "researcher"  # Corrected

    def test_invalid_action_falls_back(self) -> None:
        decision = SupervisorDecision(
            decision_type="plan_review",
            selected_action="invalid_action",
            reasoning="Whatever",
            target_node="somewhere",
        )
        result = _validate_decision(decision, "plan_review")
        assert result.selected_action == "revise"
        assert result.target_node == "planner"
        assert "Fallback" in result.reasoning


class TestRunSupervisor:
    """Tests for the full supervisor agent execution."""

    def test_plan_review_approve(self) -> None:
        response_json = json.dumps(
            {
                "decision_type": "plan_review",
                "selected_action": "approve",
                "reasoning": "The plan covers all aspects.",
                "target_node": "researcher",
                "context": {},
            }
        )
        llm = FakeChatModel(responses=[AIMessage(content=response_json)])

        state = {
            "research_query": "quantum computing",
            "research_iteration": 0,
            "revision_count": 0,
            "plan_revision_count": 0,
            "research_plan": _make_plan(),
            "evidence": [],
            "findings": [],
            "analyst_summary": None,
            "validation_result": None,
        }

        decision = run_supervisor(state, "plan_review", llm)
        assert decision.selected_action == "approve"
        assert decision.target_node == "researcher"

    def test_evidence_review_needs_research(self) -> None:
        response_json = json.dumps(
            {
                "decision_type": "evidence_review",
                "selected_action": "needs_research",
                "reasoning": "Coverage is only 40%.",
                "target_node": "researcher",
                "context": {},
            }
        )
        llm = FakeChatModel(responses=[AIMessage(content=response_json)])

        state = {
            "research_query": "AI safety",
            "research_iteration": 1,
            "revision_count": 0,
            "plan_revision_count": 0,
            "evidence": [],
            "findings": [],
            "analyst_summary": None,
            "validation_result": None,
        }

        decision = run_supervisor(state, "evidence_review", llm)
        assert decision.selected_action == "needs_research"
        assert decision.target_node == "researcher"

    def test_unknown_junction_raises(self) -> None:
        llm = FakeChatModel(responses=[])
        with pytest.raises(ValueError, match="Unknown junction"):
            run_supervisor({}, "invalid_junction", llm)
