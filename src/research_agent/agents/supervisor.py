"""Supervisor agent: makes structured routing decisions at workflow junctions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.llm.context import untrusted_block
from research_agent.llm.structured import invoke_structured
from research_agent.schemas.decisions import SupervisorDecision

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "supervisor.md"

# Legal transitions from each junction
LEGAL_TRANSITIONS: dict[str, dict[str, str]] = {
    "plan_review": {
        "approve": "researcher",
        "revise": "planner",
    },
    "evidence_review": {
        "sufficient": "report_generator",
        "needs_research": "researcher",
    },
    "final_review": {
        "accept": "finalize",
        "needs_revision": "report_generator",
        "needs_research": "researcher",
        "accept_best_effort": "finalize",
    },
}


def _load_prompt() -> str:
    """Load the supervisor prompt from file."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_state_summary(state: dict[str, Any], junction: str) -> str:
    """Build a concise state summary for the supervisor's decision context."""
    summary_parts = [
        f"Junction: {junction}",
        f"Research Query: {state.get('research_query', 'N/A')}",
        f"Research Iteration: {state.get('research_iteration', 0)}",
        f"Revision Count: {state.get('revision_count', 0)}",
        f"Plan Revision Count: {state.get('plan_revision_count', 0)}",
    ]

    if junction == "plan_review":
        plan = state.get("research_plan")
        if plan:
            summary_parts.append(untrusted_block("PLAN", plan.model_dump(mode="json")))

    elif junction == "evidence_review":
        summary_parts.append(f"Evidence collected: {len(state.get('evidence', []))}")
        summary_parts.append(f"Findings: {len(state.get('findings', []))}")
        analyst = state.get("analyst_summary")
        if analyst:
            summary_parts.append(
                untrusted_block("GAPS", analyst.gap_report.model_dump(mode="json"))
            )
            summary_parts.append(f"Coverage score: {analyst.gap_report.coverage_score}")
            if analyst.gap_report.has_gaps:
                summary_parts.append("Evidence has unresolved gaps.")

    elif junction == "final_review":
        vr = state.get("validation_result")
        if vr:
            summary_parts.append(untrusted_block("VALIDATION", vr.model_dump(mode="json")))
            summary_parts.append(f"Validation status: {vr.status}")
            summary_parts.append(f"Overall score: {vr.overall_score}")
            if vr.issues:
                summary_parts.append(f"Issues: {len(vr.issues)}")
            if vr.unsupported_claims:
                summary_parts.append(f"Unsupported claims: {len(vr.unsupported_claims)}")

    return "\n".join(summary_parts)


def run_supervisor(
    state: dict[str, Any],
    junction: str,
    llm: Any,
) -> SupervisorDecision:
    """Execute the supervisor agent for a specific junction.

    Args:
        state: Current research state
        junction: Which junction (plan_review, evidence_review, final_review)
        llm: LLM instance (ChatGroq or FakeChatModel)

    Returns:
        A validated SupervisorDecision with a legal target node.
    """
    if junction not in LEGAL_TRANSITIONS:
        raise ValueError(f"Unknown junction: {junction}. Valid: {list(LEGAL_TRANSITIONS.keys())}")

    system_prompt = _load_prompt()
    state_summary = _build_state_summary(state, junction)
    legal_actions = list(LEGAL_TRANSITIONS[junction].keys())

    user_message = (
        f"Make a routing decision for the following state:\n\n"
        f"{state_summary}\n\n"
        f"Legal actions at this junction: {legal_actions}\n"
        f"Respond with your decision."
    )

    decision = invoke_structured(
        llm,
        SupervisorDecision,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
    )

    # Validate the decision against legal transitions
    decision = _validate_decision(decision, junction)
    if (
        junction == "plan_review"
        and decision.selected_action == "approve"
        and state.get("research_plan") is None
    ):
        decision = decision.model_copy(
            update={
                "selected_action": "revise",
                "target_node": "planner",
                "reasoning": "A valid plan is required before research.",
            }
        )
    if (
        junction == "evidence_review"
        and decision.selected_action == "sufficient"
        and not state.get("evidence")
    ):
        decision = decision.model_copy(
            update={
                "selected_action": "needs_research",
                "target_node": "researcher",
                "reasoning": "Retrieved evidence is required before reporting.",
            }
        )
    validation = state.get("validation_result")
    if (
        junction == "final_review"
        and decision.selected_action == "accept"
        and (validation is None or validation.status != "pass")
    ):
        decision = decision.model_copy(
            update={
                "selected_action": "needs_research"
                if validation and validation.status == "needs_research"
                else "needs_revision",
                "target_node": "researcher"
                if validation and validation.status == "needs_research"
                else "report_generator",
                "reasoning": "Report lacks a passing validation result; further work is required.",
            }
        )

    logger.info(
        "Supervisor decision at %s: action=%s target=%s",
        junction,
        decision.selected_action,
        decision.target_node,
    )
    return decision.model_copy(
        update={
            "context": {
                "research_iteration": state.get("research_iteration", 0),
                "revision_count": state.get("revision_count", 0),
                "plan_revision_count": state.get("plan_revision_count", 0),
                "evidence_count": len(state.get("evidence", [])),
                "validation_status": validation.status if validation else "unassessed",
            }
        }
    )


def _validate_decision(decision: SupervisorDecision, junction: str) -> SupervisorDecision:
    """Ensure the decision uses a legal action and correct target node.

    If the LLM outputs an invalid action, fall back to a safe default.
    """
    legal = LEGAL_TRANSITIONS[junction]
    allowed_context = {
        "evidence_count",
        "research_iteration",
        "revision_count",
        "plan_revision_count",
        "validation_status",
    }
    decision = decision.model_copy(
        update={
            "context": {
                key: value
                for key, value in decision.context.items()
                if key in allowed_context and isinstance(value, (str, int, float, bool))
            }
        }
    )

    if decision.selected_action in legal:
        # Ensure target_node matches the action's defined target
        expected_target = legal[decision.selected_action]
        if decision.target_node != expected_target or decision.decision_type != junction:
            logger.warning(
                "Supervisor target or junction mismatch: action=%s expected_target=%s. Correcting.",
                decision.selected_action,
                expected_target,
            )
            decision = SupervisorDecision(
                decision_type=junction,
                selected_action=decision.selected_action,
                reasoning=decision.reasoning,
                target_node=expected_target,
                context=decision.context,
            )
        return decision

    # Invalid action — pick a safe default
    logger.warning(
        "Supervisor chose an invalid action at junction '%s'. Legal: %s. Requesting more work.",
        junction,
        list(legal.keys()),
    )

    default_action = {
        "plan_review": "revise",
        "evidence_review": "needs_research",
        "final_review": "needs_revision",
    }[junction]
    default_target = legal[default_action]

    return SupervisorDecision(
        decision_type=junction,
        selected_action=default_action,
        reasoning=f"Fallback: invalid model action; requesting {default_action}.",
        target_node=default_target,
        context=decision.context,
    )
