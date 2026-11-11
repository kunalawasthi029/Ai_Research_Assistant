"""Pydantic schemas for research reports."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    """A section within the research report."""

    heading: str = Field(description="Section heading")
    content: str = Field(min_length=1, description="Section content in Markdown")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Evidence IDs cited in this section"
    )


class Reference(BaseModel):
    """A citation reference in the report."""

    ref_id: str = Field(description="Reference identifier used in the report text")
    evidence_id: str = Field(description="Links to an Evidence record")
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="Additional excerpt IDs from the exact same source URL; omit the primary ID",
    )
    url: str
    title: str
    source_type: str
    accessed_at: datetime

    @property
    def all_evidence_ids(self) -> list[str]:
        return [self.evidence_id, *self.supporting_evidence_ids]


class Report(BaseModel):
    """Structured research report produced by the Report Generator."""

    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Brief summary of key findings")
    research_objective: str = Field(description="The research objective this report addresses")
    methodology: str = Field(description="How the research was conducted")

    key_findings: list[str] = Field(min_length=1, description="Top-level key findings")

    detailed_analysis: list[ReportSection] = Field(
        min_length=1, description="Detailed analysis sections"
    )

    comparative_insights: list[str] | None = Field(
        default=None, description="Comparative analysis across sources (when relevant)"
    )
    recommendations: list[str] | None = Field(
        default=None, description="Actionable recommendations (when relevant)"
    )

    limitations: list[str] = Field(min_length=1, description="Known limitations of this research")
    conclusion: str = Field(description="Final conclusion")

    references: list[Reference] = Field(
        default_factory=list,
        description="All cited references; empty only when evidence is unavailable",
    )

    # Metadata
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_used: str = Field(default="", description="Model that generated this report")
    version: int = Field(
        default=1, ge=0, description="Report revision number; zero is a recovery artifact"
    )
