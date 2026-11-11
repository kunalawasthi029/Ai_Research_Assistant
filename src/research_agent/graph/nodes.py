"""Graph node functions that wrap agent modules.

Each node reads from ResearchState, calls the corresponding agent, and returns
a state update dict. Nodes never import FastAPI or Streamlit.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from research_agent.agents.analyst import run_analyst
from research_agent.agents.planner import run_planner
from research_agent.agents.report_generator import run_report_generator
from research_agent.agents.researcher import run_researcher_task
from research_agent.agents.supervisor import run_supervisor
from research_agent.agents.validator import (
    combine_validation_results,
    run_deterministic_validator,
    run_semantic_validator,
)
from research_agent.schemas.decisions import AgentDecision
from research_agent.schemas.evidence import GapReport, Source
from research_agent.schemas.plan import ResearchPlan
from research_agent.schemas.report import Reference, Report, ReportSection
from research_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ──────────────────── Planner Node ────────────────────


def planner_node(state: dict[str, Any], *, llm: Any) -> dict[str, Any]:
    """Node: run the Planner agent to create or revise a research plan."""
    query = state["research_query"]
    objective = state["research_objective"]
    plan_revisions = state.get("plan_revision_count", 0)

    # Get feedback from last supervisor decision if revising
    feedback = ""
    decisions = state.get("agent_decisions", [])
    if state.get("research_plan") is not None and decisions:
        last = decisions[-1]
        if last.decision_type == "plan_review" and last.selected_action == "revise":
            feedback = last.reasoning

    previous_plan = state.get("research_plan")

    plan = run_planner(
        query=query,
        objective=objective,
        llm=llm,
        feedback=feedback,
        previous_plan=previous_plan,
        preferences=state.get("research_preferences"),
    )

    return {
        "research_plan": plan,
        "task_queue": list(plan.research_tasks),
        "plan_revision_count": plan_revisions + (1 if previous_plan else 0),
        "current_agent": "planner",
        "execution_status": "planned",
        "timestamps": {**state.get("timestamps", {}), "planned_at": _now_iso()},
    }


# ──────────────────── Supervisor Node ────────────────────


def _supervisor_node(state: dict[str, Any], *, junction: str, llm: Any) -> dict[str, Any]:
    """Generic supervisor node for any junction."""
    decision = run_supervisor(state=state, junction=junction, llm=llm)

    agent_decision = AgentDecision(
        decision_id=str(uuid.uuid4()),
        agent="supervisor",
        decision_type=junction,
        selected_action=decision.selected_action,
        reasoning=decision.reasoning,
        state_snapshot={
            "research_iteration": state.get("research_iteration", 0),
            "revision_count": state.get("revision_count", 0),
            "evidence_count": len(state.get("evidence", [])),
        },
    )

    warnings = []
    limits = state.get("limits", {})
    if decision.selected_action == "needs_research" and state.get("research_iteration", 0) >= int(
        limits.get("max_research_iterations", 3)
    ):
        warnings.append("Research iteration limit reached while more evidence was requested")
    if decision.selected_action == "needs_revision" and state.get("revision_count", 0) >= int(
        limits.get("max_validation_revisions", 2)
    ):
        warnings.append("Report revision limit reached while a revision was requested")
    if decision.selected_action == "revise" and state.get("plan_revision_count", 0) >= int(
        limits.get("max_plan_revisions", 2)
    ):
        warnings.append("Plan revision limit reached while a revision was requested")

    return {
        "_supervisor_decision": decision,
        "current_agent": "supervisor",
        "agent_decisions": [agent_decision],
        "warnings": warnings,
    }


def supervisor_plan_review_node(state: dict[str, Any], *, llm: Any) -> dict[str, Any]:
    """Node: supervisor reviews the plan."""
    return _supervisor_node(state, junction="plan_review", llm=llm)


def supervisor_evidence_review_node(state: dict[str, Any], *, llm: Any) -> dict[str, Any]:
    """Node: supervisor reviews evidence sufficiency."""
    return _supervisor_node(state, junction="evidence_review", llm=llm)


def supervisor_final_review_node(state: dict[str, Any], *, llm: Any) -> dict[str, Any]:
    """Node: supervisor makes final accept/revise/research decision."""
    return _supervisor_node(state, junction="final_review", llm=llm)


# ──────────────────── Researcher Node ────────────────────


def researcher_node(state: dict[str, Any], *, llm: Any, registry: ToolRegistry) -> dict[str, Any]:
    """Run one queued task so each task's normalized result can be checkpointed."""
    task_queue = state.get("task_queue", [])
    plan = state.get("research_plan")

    if not task_queue and plan:
        task_queue = list(plan.research_tasks)
    task_queue = [
        task for task in task_queue if task.task_id not in state.get("completed_tasks", [])
    ]
    reservations: dict[str, int] = {}
    for call in state.get("tool_calls", []):
        reservations[call.tool_name] = reservations.get(call.tool_name, 0) + 1
    registry.restore_counts(reservations)

    all_evidence = []
    all_tool_calls = []
    all_tool_results = []
    all_errors = []
    completed = []

    statuses: dict[str, str] = {}
    for task in task_queue[:1]:
        if task.task_id in state.get("completed_tasks", []):
            continue

        result = run_researcher_task(
            task_id=task.task_id,
            task_description=task.description,
            search_queries=task.search_queries,
            llm=llm,
            registry=registry,
            max_tool_calls=registry.max_task_calls,
            required_evidence=task.required_evidence,
            completion_criteria=task.completion_criteria,
            expected_source_types=task.expected_source_types,
            research_objective=state.get("research_objective", "")
            + "\n"
            + state.get("research_query", ""),
        )

        all_evidence.extend(result["evidence"])
        all_tool_calls.extend(result["tool_calls"])
        all_tool_results.extend(result["tool_results"])
        all_errors.extend(result["error_details"])
        complete = result["status"] == "done" and bool(result["evidence"])
        statuses[task.task_id] = "completed" if complete else "partial"
        if complete:
            completed.append(task.task_id)
        if result["status"] != "done":
            all_errors.append(f"Task {task.task_id} ended with {result['status']}")

    remaining = task_queue[1:]
    iteration = state.get("research_iteration", 0) + (0 if remaining else 1)

    return {
        "evidence": all_evidence,
        "sources": [
            Source(
                source_id=item.evidence_id,
                evidence_id=item.evidence_id,
                url=item.url,
                title=item.title,
                source_type=item.source_type,
                retrieved_at=item.retrieved_at,
            )
            for item in all_evidence
        ],
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "error_details": all_errors,
        "warnings": ["A research task stopped before observed completion"]
        if any(status == "partial" for status in statuses.values())
        else [],
        "completed_tasks": completed,
        "research_iteration": iteration,
        "current_agent": "researcher",
        "execution_status": "researching",
        "task_queue": remaining,
        "active_task": task_queue[0] if task_queue else None,
        "research_plan": plan.model_copy(
            update={
                "research_tasks": [
                    task.model_copy(update={"status": statuses.get(task.task_id, task.status)})
                    for task in plan.research_tasks
                ]
            }
        )
        if plan
        else None,
        "timestamps": {**state.get("timestamps", {}), "researched_at": _now_iso()},
    }


# ──────────────────── Analyst Node ────────────────────


def analyst_node(state: dict[str, Any], *, llm: Any) -> dict[str, Any]:
    """Node: run the Analyst agent to synthesize evidence."""
    evidence = state.get("evidence", [])
    objective = state["research_objective"]
    plan = state.get("research_plan")
    task_ids = [t.task_id for t in plan.research_tasks] if plan else []

    analysis = run_analyst(
        evidence=evidence,
        research_objective=objective,
        task_ids=task_ids,
        llm=llm,
    )

    return {
        "analyst_summary": analysis,
        "findings": analysis.findings,
        "current_agent": "analyst",
        "execution_status": "analyzed",
        "timestamps": {**state.get("timestamps", {}), "analyzed_at": _now_iso()},
    }


# ──────────────────── Report Generator Node ────────────────────


def report_generator_node(state: dict[str, Any], *, llm: Any) -> dict[str, Any]:
    """Node: generate or revise the research report."""
    objective = state["research_objective"]
    plan = state["research_plan"]
    analysis = state["analyst_summary"]
    evidence = state.get("evidence", [])
    revision_count = state.get("revision_count", 0) + (1 if state.get("report") else 0)

    # Collect validation feedback from previous iterations
    feedback = ""
    vr = state.get("validation_result")
    if vr and vr.status != "pass":
        feedback = vr.feedback

    report = run_report_generator(
        objective=objective,
        plan=plan,
        analysis=analysis,
        evidence=evidence,
        llm=llm,
        validation_feedback=feedback,
        version=revision_count + 1,
        preferences=state.get("research_preferences"),
    )

    return {
        "report": report,
        "revision_count": revision_count,
        "current_agent": "report_generator",
        "execution_status": "report_generated",
        "timestamps": {**state.get("timestamps", {}), "report_at": _now_iso()},
    }


# ──────────────────── Validator Node ────────────────────


def validator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Node: deterministic checks before spending a semantic-model call."""
    report = state["report"]
    evidence = state.get("evidence", [])
    plan = state.get("research_plan")

    evidence_ids = [ev.evidence_id for ev in evidence]
    plan_task_ids = [t.task_id for t in plan.research_tasks] if plan else []

    # Layer A: deterministic (no LLM needed)
    det_result = run_deterministic_validator(
        report=report,
        evidence_ids=evidence_ids,
        plan_task_ids=plan_task_ids,
        evidence=evidence,
    )
    return {
        "deterministic_validation": det_result,
        "current_agent": "validator",
        "validation_history": [{"report_version": report.version, "result": det_result}],
    }


def semantic_validator_node(state: dict[str, Any], *, validator_llm: Any) -> dict[str, Any]:
    """Node: semantic evidence assessment and combined actionable feedback."""
    report = state["report"]
    evidence = state.get("evidence", [])
    objective = state["research_objective"]
    det_result = state["deterministic_validation"]
    thresholds = {
        "semantic_threshold": float(
            state.get("limits", {}).get("semantic_validation_threshold", 0.6)
        ),
        "criterion_threshold": float(
            state.get("limits", {}).get("semantic_criterion_threshold", 0.6)
        ),
    }

    # Layer B: semantic (uses a separate LLM)
    sem_result = run_semantic_validator(
        report=report,
        evidence=evidence,
        research_objective=objective,
        llm=validator_llm,
        research_plan=state.get("research_plan"),
        **thresholds,
    )

    # Combine both layers
    combined = combine_validation_results(det_result, sem_result, **thresholds)

    feedback_entry = (
        f"Validation v{report.version}: "
        f"status={combined.status} score={combined.overall_score:.2f} "
        f"feedback={combined.feedback[:200]}"
    )

    return {
        "validation_result": combined,
        "semantic_validation": sem_result,
        "validation_history": [
            {"report_version": report.version, "result": sem_result},
            {"report_version": report.version, "result": combined},
        ],
        "validation_feedback": [feedback_entry],
        "current_agent": "validator",
        "execution_status": "validated",
        "timestamps": {**state.get("timestamps", {}), "validated_at": _now_iso()},
    }


# ──────────────────── Finalize Node ────────────────────


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    """Node: mark the research run as complete."""
    result = state.get("validation_result")
    decision = state.get("_supervisor_decision")
    warnings = state.get("warnings", [])
    passed = (
        result is not None
        and result.status == "pass"
        and not warnings
        and not state.get("failure_node")
        and decision is not None
        and decision.selected_action == "accept"
    )
    report = state.get("report")
    if not passed and report:
        report = report.model_copy(
            update={
                "limitations": list(
                    dict.fromkeys(
                        [
                            *report.limitations,
                            *warnings,
                            "Best-effort result: full workflow completion criteria were not met.",
                        ]
                    )
                )
            }
        )
    return {
        "execution_status": "completed" if passed else "best_effort" if report else "incomplete",
        "report": report,
        "current_agent": "finalize",
        "timestamps": {**state.get("timestamps", {}), "completed_at": _now_iso()},
    }


def replan_node(state: dict[str, Any], *, llm: Any) -> dict[str, Any]:
    """Turn actual gaps/validation feedback into fresh executable tasks and retain prior work."""
    previous = state["research_plan"]
    analysis = state.get("analyst_summary")
    decision = state.get("_supervisor_decision")
    feedback = decision.reasoning if decision else "Additional evidence is required"
    validation = state.get("validation_result")
    if validation and validation.status != "pass":
        feedback += "\n" + validation.feedback
    gaps = (
        analysis.gap_report.model_copy(deep=True)
        if analysis
        else GapReport(has_gaps=True, coverage_score=0)
    )
    gaps.has_gaps = True
    extra = run_planner(
        query=state["research_query"],
        objective=state["research_objective"],
        llm=llm,
        feedback=feedback,
        previous_plan=previous,
        gap_report=gaps,
        completed_task_ids=state.get("completed_tasks", []),
        preferences=state.get("research_preferences"),
    )
    questions = {question.question_id: question for question in previous.sub_questions}
    for question in extra.sub_questions:
        questions.setdefault(question.question_id, question)
    merged = ResearchPlan.model_validate(
        {
            **previous.model_dump(),
            "sub_questions": list(questions.values()),
            "research_tasks": [*previous.research_tasks, *extra.research_tasks],
        }
    )
    return {
        "research_plan": merged,
        "task_queue": list(extra.research_tasks),
        "current_agent": "planner",
        "execution_status": "replanned",
    }


def recovery_node(state: dict[str, Any]) -> dict[str, Any]:
    """Return an honest partial artifact without asking a model to fabricate missing research."""
    reason = f"Execution could not complete at {state.get('failure_node') or 'a workflow limit'}."
    previous = state.get("report")
    if previous:
        report = previous.model_copy(
            update={
                "limitations": list(
                    dict.fromkeys(
                        [
                            *previous.limitations,
                            reason,
                            "This report has not met all completion criteria.",
                        ]
                    )
                )
            }
        )
    else:
        refs = [
            Reference(
                ref_id=f"ref{index}",
                evidence_id=item.evidence_id,
                url=item.url,
                title=item.title,
                source_type=item.source_type,
                accessed_at=item.retrieved_at,
            )
            for index, item in enumerate(state.get("evidence", []), 1)
        ]
        report = Report(
            title="Incomplete research",
            executive_summary="Research did not complete; no validated conclusions are available.",
            research_objective=state.get("research_objective")
            or state.get("research_query")
            or "Invalid research input",
            methodology="Bounded research workflow; execution stopped before completion.",
            key_findings=["No validated conclusions are available."],
            detailed_analysis=[
                ReportSection(
                    heading="Execution status",
                    content=reason
                    + " Retrieved references, if any, are supplied for inspection and do not establish factual support.",
                )
            ],
            limitations=[
                reason,
                *state.get("warnings", []),
                "Partial output; no factual accuracy or completeness claim.",
            ],
            conclusion="Further research and validation are required.",
            references=refs,
            version=0,
        )
    return {
        "report": report,
        "execution_status": "best_effort" if previous else "incomplete",
        "current_agent": "recovery",
        "warnings": [reason],
    }
