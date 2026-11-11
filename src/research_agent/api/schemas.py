"""API request/response schemas (Pydantic models for the HTTP layer)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, field_validator

from research_agent.config.settings import Settings
from research_agent.schemas.report import Report
from research_agent.schemas.validation import ValidationResult


class ResearchOptions(BaseModel):
    """Per-run limits may lower server caps, never enlarge them."""

    model_config = ConfigDict(extra="forbid")
    max_research_iterations: int | None = Field(default=None, ge=1, le=10, strict=True)
    max_tool_calls_per_task: int | None = Field(default=None, ge=1, le=20, strict=True)
    max_tool_calls_per_run: int | None = Field(default=None, ge=1, le=1000, strict=True)
    max_validation_revisions: int | None = Field(default=None, ge=0, le=10, strict=True)
    overall_run_timeout_seconds: int | None = Field(default=None, ge=1, le=3600, strict=True)
    human_approval_enabled: StrictBool | None = None
    depth: Literal["brief", "standard", "deep"] | None = None
    scope: Literal["narrow", "balanced", "broad"] | None = None
    source_preferences: list[Literal["web", "academic", "wikipedia"]] | None = Field(
        default=None, max_length=3
    )

    @field_validator("source_preferences")
    @classmethod
    def unique_sources(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and len(values) != len(set(values)):
            raise ValueError("Source preferences must be unique")
        return values

    def effective(self, settings: Settings) -> Settings:
        overrides = self.model_dump(exclude_none=True)
        preferences = {
            key: overrides.pop(key)
            for key in ("depth", "scope", "source_preferences")
            if key in overrides
        }
        if any(
            value > getattr(settings, name)
            for name, value in overrides.items()
            if name != "human_approval_enabled"
        ):
            raise ValueError("Requested limits exceed server configuration")
        if "depth" in preferences:
            overrides["research_depth"] = preferences["depth"]
        if "scope" in preferences:
            overrides["research_scope"] = preferences["scope"]
        if "source_preferences" in preferences:
            overrides["source_preferences"] = preferences["source_preferences"]
        if overrides.get("research_depth", settings.research_depth) == "brief":
            for key, cap in (
                ("max_research_iterations", 1),
                ("max_tool_calls_per_task", 3),
                ("max_validation_revisions", 1),
            ):
                overrides[key] = min(overrides.get(key, getattr(settings, key)), cap)
        return settings.model_copy(update=overrides)


# ──────────────────── Requests ────────────────────


class ResearchRequest(BaseModel):
    """POST /research — start a new research run."""

    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=4000)]
    objective: Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)] = ""
    options: ResearchOptions = Field(default_factory=ResearchOptions)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: StrictBool | None = None


class ResearchResultResponse(BaseModel):
    run_id: str
    status: str
    thread_id: str | None = None
    report: Report | None = None
    validation_result: ValidationResult | None = None
    evidence_count: int = 0
    tool_call_count: int = 0
    elapsed_seconds: float | None = None


# ──────────────────── Responses ────────────────────


class RunSummary(BaseModel):
    """Brief run info for list endpoints."""

    run_id: str
    query: str
    status: str
    created_at: str
    elapsed_seconds: float | None = None
    evidence_count: int = 0
    tool_call_count: int = 0


class RunDetail(BaseModel):
    """Full run info for detail endpoint."""

    run_id: str
    thread_id: str
    query: str
    objective: str
    status: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    total_tool_calls: int = 0
    total_tokens: int = 0
    elapsed_seconds: float | None = None
    evidence_count: int = 0
    job_active: bool = False
    report_version: int | None = None
    research_plan: dict[str, Any] | None = None
    active_task: dict[str, Any] | None = None
    current_agent: str = ""
    revision_count: int = 0
    research_iteration: int = 0
    validation_result: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    research_preferences: dict[str, Any] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    """Report data in the API response."""

    run_id: str
    version: int
    report_json: dict[str, Any]
    report_markdown: str
    created_at: str


class EvidenceResponse(BaseModel):
    """Evidence data in the API response."""

    evidence_id: str
    task_id: str
    tool_name: str
    url: str
    title: str
    source_type: str
    retrieved_at: str
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ResearchStarted(BaseModel):
    """Response when a research run is accepted."""

    run_id: str
    status: str = "accepted"
    message: str = "Research run accepted"


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"
    version: str = "0.1.0"
    active_jobs: int = 0


class ErrorResponse(BaseModel):
    """Uniform error response."""

    error: str
    detail: str = ""
    status_code: int = 500
    request_id: str = ""
