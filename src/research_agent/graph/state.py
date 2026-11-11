"""LangGraph state definition with typed fields and reducers."""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict

from research_agent.schemas.decisions import (
    AgentDecision,
    SupervisorDecision,
    TokenUsage,
    ToolCallRecord,
    ToolResult,
)
from research_agent.schemas.evidence import AnalysisResult, Evidence, Finding, Source
from research_agent.schemas.plan import ResearchPlan, ResearchTask
from research_agent.schemas.report import Report
from research_agent.schemas.validation import ValidationResult


class ResearchState(TypedDict):
    """Central state for the research workflow graph.

    Fields use two strategies:
    - Annotated with operator.add: values APPEND across node updates (evidence, calls, etc.)
    - No annotation: values OVERWRITE on each node update (findings, active_task, report, etc.)

    This distinction matters: append-fields accumulate across the entire run,
    while overwrite-fields reflect the current stage's output.
    """

    # --- Identity ---
    research_id: str
    thread_id: str

    # --- Input ---
    research_query: str
    research_objective: str
    research_preferences: dict[str, str | list[str]]

    # --- Planning ---
    research_plan: ResearchPlan | None
    plan_revision_count: int
    limits: dict[str, int | float | bool]
    deadline_at: float
    approval_granted: bool
    warnings: Annotated[list[str], operator.add]
    _supervisor_decision: SupervisorDecision | None
    failure_node: str

    # --- Task Execution ---
    active_task: ResearchTask | None
    task_queue: list[ResearchTask]
    completed_tasks: Annotated[list[str], operator.add]

    # --- Tool Tracking (append) ---
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]

    # --- Evidence and sources append; latest analysis findings overwrite ---
    evidence: Annotated[list[Evidence], operator.add]
    findings: list[Finding]
    sources: Annotated[list[Source], operator.add]

    # --- Analysis ---
    analyst_summary: AnalysisResult | None

    # --- Report ---
    report: Report | None

    # --- Validation ---
    validation_result: ValidationResult | None
    deterministic_validation: ValidationResult | None
    semantic_validation: ValidationResult | None
    validation_history: Annotated[list[dict[str, object]], operator.add]
    validation_feedback: Annotated[list[str], operator.add]

    # --- Control Flow ---
    revision_count: int
    research_iteration: int
    execution_status: str
    current_agent: str

    # --- Decisions (append) ---
    agent_decisions: Annotated[list[AgentDecision], operator.add]

    # --- Error Handling (append) ---
    error_details: Annotated[list[str], operator.add]

    # --- Messages (LangChain message reducer) ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Metadata ---
    timestamps: dict[str, str]
    token_usage: Annotated[list[TokenUsage], operator.add]


def create_initial_state(
    research_id: str,
    thread_id: str,
    query: str,
    objective: str = "",
) -> ResearchState:
    """Create a fresh initial state for a new research run.

    This factory ensures all fields have valid initial values,
    preventing KeyError from missing fields during node execution.
    """
    return ResearchState(
        research_id=research_id,
        thread_id=thread_id,
        research_query=query,
        research_objective=objective or query,
        research_preferences={},
        research_plan=None,
        plan_revision_count=0,
        limits={},
        deadline_at=0.0,
        approval_granted=False,
        warnings=[],
        _supervisor_decision=None,
        failure_node="",
        active_task=None,
        task_queue=[],
        completed_tasks=[],
        tool_calls=[],
        tool_results=[],
        evidence=[],
        findings=[],
        sources=[],
        analyst_summary=None,
        report=None,
        validation_result=None,
        deterministic_validation=None,
        semantic_validation=None,
        validation_history=[],
        validation_feedback=[],
        revision_count=0,
        research_iteration=0,
        execution_status="initialized",
        current_agent="",
        agent_decisions=[],
        error_details=[],
        messages=[],
        timestamps={},
        token_usage=[],
    )
