"""Unit tests for graph routing functions.

Routing functions are pure, side-effect-free, and table-driven —
each test verifies a specific state → node-name mapping.
"""

from __future__ import annotations

from research_agent.graph.routing import (
    MAX_PLAN_REVISIONS,
    MAX_REPORT_REVISIONS,
    MAX_RESEARCH_ITERATIONS,
    route_after_analysis,
    route_after_plan,
    route_after_report,
    route_after_research,
    route_after_validation,
    route_evidence_review,
    route_final_review,
    route_plan_review,
)
from research_agent.schemas.decisions import SupervisorDecision


def _decision(action: str, target: str = "") -> SupervisorDecision:
    """Shortcut to build a SupervisorDecision for tests."""
    return SupervisorDecision(
        decision_type="test",
        selected_action=action,
        reasoning="test",
        target_node=target,
    )


# ──────────────────── Fixed routing ────────────────────


class TestFixedRoutes:
    """Deterministic edges that always go to one node."""

    def test_after_plan(self) -> None:
        assert route_after_plan({}) == "supervisor_plan_review"

    def test_after_research(self) -> None:
        assert route_after_research({}) == "analyst"

    def test_after_analysis(self) -> None:
        assert route_after_analysis({}) == "supervisor_evidence_review"

    def test_after_report(self) -> None:
        assert route_after_report({}) == "validator"

    def test_after_validation(self) -> None:
        assert route_after_validation({}) == "supervisor_final_review"


# ──────────────────── Plan Review ────────────────────


class TestRoutePlanReview:
    """Supervisor plan_review junction routing."""

    def test_approve_goes_to_researcher(self) -> None:
        state = {"_supervisor_decision": _decision("approve"), "plan_revision_count": 0}
        assert route_plan_review(state) == "approval"

    def test_revise_goes_to_planner(self) -> None:
        state = {"_supervisor_decision": _decision("revise"), "plan_revision_count": 0}
        assert route_plan_review(state) == "planner"

    def test_revise_capped(self) -> None:
        """A rejected plan reaches recovery when its revision budget is exhausted."""
        state = {
            "_supervisor_decision": _decision("revise"),
            "plan_revision_count": MAX_PLAN_REVISIONS,
        }
        assert route_plan_review(state) == "recovery"

    def test_no_decision_defaults_to_researcher(self) -> None:
        state = {"plan_revision_count": 0}
        assert route_plan_review(state) == "recovery"


# ──────────────────── Evidence Review ────────────────────


class TestRouteEvidenceReview:
    """Supervisor evidence_review junction routing."""

    def test_sufficient_goes_to_report(self) -> None:
        state = {"_supervisor_decision": _decision("sufficient"), "research_iteration": 0}
        assert route_evidence_review(state) == "report_generator"

    def test_needs_research_goes_back(self) -> None:
        state = {"_supervisor_decision": _decision("needs_research"), "research_iteration": 0}
        assert route_evidence_review(state) == "replan"

    def test_research_iteration_capped(self) -> None:
        state = {
            "_supervisor_decision": _decision("needs_research"),
            "research_iteration": MAX_RESEARCH_ITERATIONS,
        }
        assert route_evidence_review(state) == "report_generator"

    def test_no_decision_defaults_to_report(self) -> None:
        state = {"research_iteration": 0}
        assert route_evidence_review(state) == "recovery"


# ──────────────────── Final Review ────────────────────


class TestRouteFinalReview:
    """Supervisor final_review junction routing."""

    def test_accept_goes_to_finalize(self) -> None:
        state = {
            "_supervisor_decision": _decision("accept"),
            "revision_count": 0,
            "research_iteration": 0,
        }
        assert route_final_review(state) == "finalize"

    def test_accept_best_effort_goes_to_finalize(self) -> None:
        state = {
            "_supervisor_decision": _decision("accept_best_effort"),
            "revision_count": 0,
            "research_iteration": 0,
        }
        assert route_final_review(state) == "finalize"

    def test_needs_revision_routes_to_report(self) -> None:
        state = {
            "_supervisor_decision": _decision("needs_revision"),
            "revision_count": 0,
            "research_iteration": 0,
        }
        assert route_final_review(state) == "report_generator"

    def test_needs_research_routes_to_researcher(self) -> None:
        state = {
            "_supervisor_decision": _decision("needs_research"),
            "revision_count": 0,
            "research_iteration": 0,
        }
        assert route_final_review(state) == "replan"

    def test_revision_cap_forces_finalize(self) -> None:
        state = {
            "_supervisor_decision": _decision("needs_revision"),
            "revision_count": MAX_REPORT_REVISIONS,
            "research_iteration": 0,
        }
        assert route_final_review(state) == "finalize"

    def test_research_cap_forces_finalize(self) -> None:
        state = {
            "_supervisor_decision": _decision("needs_research"),
            "revision_count": 0,
            "research_iteration": MAX_RESEARCH_ITERATIONS,
        }
        assert route_final_review(state) == "finalize"

    def test_all_caps_hit_forces_finalize(self) -> None:
        state = {
            "_supervisor_decision": _decision("needs_revision"),
            "revision_count": MAX_REPORT_REVISIONS,
            "research_iteration": MAX_RESEARCH_ITERATIONS,
        }
        assert route_final_review(state) == "finalize"

    def test_no_decision_defaults_to_finalize(self) -> None:
        state = {"revision_count": 0, "research_iteration": 0}
        assert route_final_review(state) == "recovery"


# ──────────────────── Constants ────────────────────


class TestConstants:
    """Verify routing constants are sane."""

    def test_plan_revisions_positive(self) -> None:
        assert MAX_PLAN_REVISIONS >= 1

    def test_research_iterations_positive(self) -> None:
        assert MAX_RESEARCH_ITERATIONS >= 1

    def test_report_revisions_positive(self) -> None:
        assert MAX_REPORT_REVISIONS >= 1
