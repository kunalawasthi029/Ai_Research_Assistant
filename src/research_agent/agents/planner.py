"""Planner agent: decomposes research queries into structured plans."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.llm.context import research_time_context, untrusted_block
from research_agent.llm.structured import ContractError, invoke_structured
from research_agent.schemas.evidence import GapReport
from research_agent.schemas.plan import ResearchPlan

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.md"


def _load_prompt() -> str:
    """Load the planner prompt from file."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def run_planner(
    query: str,
    objective: str,
    llm: Any,
    feedback: str = "",
    previous_plan: ResearchPlan | None = None,
    gap_report: GapReport | None = None,
    completed_task_ids: list[str] | None = None,
    preferences: dict[str, Any] | None = None,
) -> ResearchPlan:
    """Execute the planner agent to create or revise a research plan.

    Args:
        query: The user's research query
        objective: The research objective
        llm: LLM instance
        feedback: Supervisor feedback for plan revision (empty on first pass)
        previous_plan: Previous plan to revise (None on first pass)

    Returns:
        A validated ResearchPlan.
    """
    system_prompt = _load_prompt() + "\n\n" + research_time_context()

    user_parts = [
        f"Research Query: {query}",
        f"Research Objective: {objective}",
    ]
    if preferences:
        user_parts.append(untrusted_block("RESEARCH_PREFERENCES", preferences))
        user_parts.append(
            "Use the requested depth and scope to guide coverage. Prioritize preferred source types when suitable; they are preferences, not guarantees of availability."
        )

    if feedback:
        user_parts.append("\n## Revision Request\n" + untrusted_block("FEEDBACK", feedback[:5000]))
    if previous_plan:
        user_parts.append(untrusted_block("PREVIOUS_PLAN", previous_plan.model_dump(mode="json")))
    if gap_report:
        user_parts.append(untrusted_block("GAPS", gap_report.model_dump(mode="json")))
        user_parts.append(
            "Create targeted additional work addressing these gaps. Use new task IDs; do not repeat prior completed tasks."
        )

    user_message = "\n".join(user_parts) + "\n\nCreate a structured research plan."

    def validate(plan: ResearchPlan) -> ResearchPlan:
        if gap_report and gap_report.has_gaps:
            old_ids = set(completed_task_ids or [])
            if previous_plan:
                old_ids.update(task.task_id for task in previous_plan.research_tasks)
            if any(task.task_id in old_ids for task in plan.research_tasks):
                raise ContractError("Additional research tasks require new task IDs.")
        if preferences and preferences.get("scope") in {"narrow", "balanced", "broad"}:
            plan.scope = preferences["scope"]
        return plan.model_copy(
            update={
                "research_tasks": [
                    task.model_copy(update={"status": "pending"})
                    for task in sorted(plan.research_tasks, key=lambda task: task.priority)
                ]
            }
        )

    plan = invoke_structured(
        llm,
        ResearchPlan,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
        validate=validate,
    )

    logger.info(
        "Planner created plan: %d sub-questions, %d tasks, complexity=%s",
        len(plan.sub_questions),
        len(plan.research_tasks),
        plan.estimated_complexity,
    )

    return plan
