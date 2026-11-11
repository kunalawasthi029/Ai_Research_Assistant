"""Pydantic schemas for agent decisions and execution tracking."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """Structured routing decision made by the Supervisor agent."""

    decision_type: str = Field(description="plan_review|evidence_review|final_review")
    selected_action: str = Field(
        description="The chosen action: approve|revise|more_research|rewrite|accept_best_effort"
    )
    reasoning: str = Field(
        max_length=500,
        description="Concise observable decision summary; never private reasoning or chain of thought",
    )
    target_node: str = Field(description="The graph node to route to next")
    context: dict[str, object] = Field(
        default_factory=dict,
        description="Relevant state fields that influenced this decision",
    )


class AgentDecision(BaseModel):
    """Record of any decision made during execution."""

    decision_id: str = Field(description="Unique identifier")
    agent: str = Field(description="Which agent made this decision")
    decision_type: str = Field(description="Type of decision")
    selected_action: str = Field(description="What was decided")
    reasoning: str = Field(description="Why")
    state_snapshot: dict[str, object] = Field(
        default_factory=dict,
        description="Relevant state fields at decision time",
    )
    result: str | None = Field(default=None, description="Outcome of the decision")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCallRecord(BaseModel):
    """Record of a tool call made during research."""

    call_id: str = Field(description="Unique identifier")
    tool_name: str = Field(description="Name of the tool called")
    arguments: dict[str, object] = Field(description="Arguments passed to the tool")
    task_id: str = Field(description="Research task this call belongs to")
    success: bool = Field(description="Whether the call succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_seconds: float | None = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolResult(BaseModel):
    """Result returned from a tool execution."""

    call_id: str = Field(description="Matches ToolCallRecord.call_id")
    tool_name: str = Field(description="Name of the tool")
    output: str = Field(description="Tool output as string")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Evidence records created from this result"
    )


class TokenUsage(BaseModel):
    """Token usage record for an LLM call."""

    agent: str = Field(description="Which agent")
    model: str = Field(description="Model used")
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_seconds: float | None = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
