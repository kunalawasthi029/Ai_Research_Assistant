"""Route observed supervisor decisions through configured finite workflow budgets."""

from __future__ import annotations

from typing import Any

MAX_PLAN_REVISIONS = 2
MAX_RESEARCH_ITERATIONS = 3
MAX_REPORT_REVISIONS = 2


def route_after_plan(state: dict[str, Any]) -> str:
    """Review a newly emitted plan."""
    return "supervisor_plan_review"


def route_after_research(state: dict[str, Any]) -> str:
    """Analyze actual retrieved evidence."""
    return "analyst"


def route_after_analysis(state: dict[str, Any]) -> str:
    """Ask the supervisor to inspect the analysis and gaps."""
    return "supervisor_evidence_review"


def route_after_report(state: dict[str, Any]) -> str:
    """Run the deterministic validator before the semantic validator."""
    return "validator"


def route_after_validation(state: dict[str, Any]) -> str:
    """Review the combined validation outcome."""
    return "supervisor_final_review"


def route_plan_review(state: dict[str, Any]) -> str:
    """Never silently approve a rejected plan because its revision budget ran out."""
    decision = state.get("_supervisor_decision")
    if decision is None:
        return "recovery"
    if decision.selected_action == "revise":
        cap = int(state.get("limits", {}).get("max_plan_revisions", MAX_PLAN_REVISIONS))
        return "planner" if state.get("plan_revision_count", 0) < cap else "recovery"
    return "approval" if decision.selected_action == "approve" else "recovery"


def route_evidence_review(state: dict[str, Any]) -> str:
    """Route additional research through fresh planning, not completed task IDs."""
    decision = state.get("_supervisor_decision")
    if decision is None:
        return "recovery"
    cap = int(state.get("limits", {}).get("max_research_iterations", MAX_RESEARCH_ITERATIONS))
    if decision.selected_action == "needs_research" and state.get("research_iteration", 0) < cap:
        return "replan"
    return "report_generator"


def route_final_review(state: dict[str, Any]) -> str:
    """Choose revision/replanning or terminate honestly when the requested budget is exhausted."""
    decision = state.get("_supervisor_decision")
    if decision is None:
        return "recovery"
    limits = state.get("limits", {})
    if decision.selected_action == "needs_revision":
        return (
            "report_generator"
            if state.get("revision_count", 0)
            < int(limits.get("max_validation_revisions", MAX_REPORT_REVISIONS))
            else "finalize"
        )
    if decision.selected_action == "needs_research":
        return (
            "replan"
            if state.get("research_iteration", 0)
            < int(limits.get("max_research_iterations", MAX_RESEARCH_ITERATIONS))
            else "finalize"
        )
    return (
        "finalize" if decision.selected_action in {"accept", "accept_best_effort"} else "recovery"
    )
