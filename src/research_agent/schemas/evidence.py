"""Pydantic schemas for evidence, sources, and findings."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(UTC)


class Evidence(BaseModel):
    """A single piece of evidence retrieved by a tool."""

    evidence_id: str = Field(description="Unique identifier (UUID)")
    research_task_id: str = Field(description="ID of the research task that requested this")
    tool_name: str = Field(description="Which tool retrieved this evidence")
    url: str = Field(description="Source URL")
    title: str = Field(description="Source title")
    content: str = Field(description="Extracted text or snippet")
    source_type: str = Field(description="web|wikipedia|arxiv|calculated")
    retrieved_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, object] = Field(default_factory=dict)


class Source(BaseModel):
    """A verified source reference for citation."""

    source_id: str = Field(description="Unique identifier")
    evidence_id: str = Field(description="ID of the evidence this source comes from")
    url: str = Field(description="Canonical URL")
    title: str = Field(description="Source title")
    source_type: str = Field(description="web|wikipedia|arxiv|calculated")
    is_accessible: bool | None = Field(
        default=None, description="Whether URL reachability was checked; None means unassessed"
    )
    retrieved_at: datetime = Field(default_factory=_utcnow)


class Finding(BaseModel):
    """A structured finding extracted from evidence by the Analyst."""

    finding_id: str = Field(description="Unique identifier")
    research_task_id: str = Field(description="Which research task this answers")
    claim: str = Field(
        min_length=1, max_length=4000, description="The factual claim or key finding"
    )
    supporting_evidence: list[str] = Field(
        min_length=1, description="Evidence IDs that support this finding"
    )
    source_url: str = Field(description="Primary source URL")
    source_title: str = Field(description="Primary source title")
    source_type: str = Field(description="web|wikipedia|arxiv|calculated")
    collected_at: datetime = Field(default_factory=_utcnow)
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance to the research query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this finding")
    verification_status: Literal["unverified", "verified", "contradicted", "uncertain"] = Field(
        default="unverified", description="unverified|verified|contradicted|uncertain"
    )


class GapReport(BaseModel):
    """Analyst's report on what evidence is missing or insufficient."""

    has_gaps: bool = Field(description="Whether significant gaps exist")
    missing_topics: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    low_confidence_areas: list[str] = Field(default_factory=list)
    suggested_tasks: list[str] = Field(
        default_factory=list,
        description="Additional research tasks to fill gaps",
    )
    coverage_score: float = Field(
        ge=0.0, le=1.0, description="How well evidence covers the research plan"
    )


class AnalysisResult(BaseModel):
    """Complete output of the Analyst agent."""

    summary: str = Field(max_length=8000, description="Executive summary of findings")
    findings: list[Finding] = Field(max_length=40, description="Structured findings")
    gap_report: GapReport = Field(description="Evidence gap analysis")
    total_evidence_count: int = Field(ge=0)
    unique_source_count: int = Field(ge=0)
