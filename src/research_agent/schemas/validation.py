"""Pydantic schemas for validation results and rubric scores."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ValidationStatus = Literal["pass", "needs_revision", "needs_research", "fail"]


class ValidationScore(BaseModel):
    """Score for a single validation criterion."""

    criterion: str = Field(description="What was evaluated")
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    passed: bool = Field(description="Whether this criterion passed")
    details: str = Field(default="", description="Explanation of the score")


class ValidationIssue(BaseModel):
    """A specific issue found during validation."""

    issue_type: str = Field(
        description="missing_section|empty_content|broken_citation|unsupported_claim"
        "|low_relevance|duplicate_content|schema_error|incomplete_coverage"
    )
    severity: Literal["critical", "major", "minor"] = Field(description="critical|major|minor")
    description: str = Field(description="Human-readable issue description")
    location: str | None = Field(
        default=None, description="Where in the report the issue was found"
    )
    suggestion: str = Field(default="", description="How to fix this issue")


class ValidationResult(BaseModel):
    """Complete validation result from both layers."""

    status: ValidationStatus = Field(description="pass|needs_revision|needs_research|fail")
    layer: Literal["deterministic", "semantic", "combined"] = Field(
        description="deterministic|semantic|combined"
    )
    scores: list[ValidationScore] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    broken_citations: list[str] = Field(default_factory=list)
    feedback: str = Field(default="", description="Actionable feedback for revision")
    recommendations: list[str] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0, description="Aggregate score")


class DeterministicValidationInput(BaseModel):
    """Input for the deterministic (Layer A) validator."""

    report_json: dict[str, object] = Field(description="The report as a dict")
    evidence_ids: list[str] = Field(description="All available evidence IDs")
    plan_task_ids: list[str] = Field(description="All planned task IDs")


class SemanticValidationInput(BaseModel):
    """Input for the semantic (Layer B) validator."""

    report_text: str = Field(description="The report as Markdown text")
    evidence_texts: list[str] = Field(description="Evidence content for comparison")
    research_objective: str = Field(description="The original research objective")
