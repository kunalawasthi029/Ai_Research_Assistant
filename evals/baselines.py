"""Controlled baselines sharing the production generator and initial state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.agents.report_generator import run_report_generator
from research_agent.schemas.report import Report


def run_baseline_b0(query: str, llm: Any) -> Report:
    """One generation attempt; no retrieval, schema repair or validation loop."""
    report = llm.with_structured_output(Report, repair_attempts=0).invoke(
        [
            SystemMessage(
                content="Write a factual research report using only model knowledge. "
                "Return the Report schema. Do not claim to have retrieved sources. "
                "Set references=[] and every section's evidence_ids=[]. "
                "State uncertainty and knowledge limitations explicitly."
            ),
            HumanMessage(content=f"Research query: {query}"),
        ]
    )
    if report.references or any(section.evidence_ids for section in report.detailed_analysis):
        raise ValueError("B0 invented retrieved provenance")
    report.generated_at = datetime.now(UTC)
    report.version = 1
    report.research_objective = query
    model_name = getattr(llm, "model_name", "")
    report.model_used = model_name if isinstance(model_name, str) else ""
    return cast(Report, report)


def run_baseline_b1(state: dict[str, Any], llm: Any) -> Report:
    """One production generation from shared preparation, without quality feedback.

    B1/B2 have identical bounded schema/provenance repair in the generator.
    The common planner/retrieval/analyst cost is charged to both arms separately.
    """
    if not state.get("research_plan") or not state.get("analyst_summary"):
        raise ValueError("Shared preparation is incomplete")
    return run_report_generator(
        objective=state["research_objective"],
        plan=state["research_plan"],
        analysis=state["analyst_summary"],
        evidence=state["evidence"],
        llm=llm,
        preferences=state.get("research_preferences"),
    )
