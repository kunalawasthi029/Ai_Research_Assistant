"""Versioned HTTP contract over the reusable service and bounded job dispatcher."""

from __future__ import annotations

import json
import re
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from research_agent.api.dependencies import get_jobs, get_settings
from research_agent.api.events import stream_run_events
from research_agent.api.jobs import JobCapacityError, JobConflictError, JobManager
from research_agent.api.schemas import (
    EvidenceResponse,
    HealthResponse,
    ReportResponse,
    ResearchRequest,
    ResearchResultResponse,
    ResearchStarted,
    ResumeRequest,
    RunDetail,
    RunSummary,
    ToolDescription,
)
from research_agent.config.settings import Settings
from research_agent.persistence.models import RunRow
from research_agent.service.research_service import ResearchService
from research_agent.tools.registry import create_default_registry

router = APIRouter()


def _ensure_run(service: ResearchService, run_id: str) -> Any:
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


def _effective(payload: ResearchRequest, settings: Settings) -> Settings:
    try:
        return payload.options.effective(settings)
    except ValueError:
        raise HTTPException(400, "Requested options exceed server limits") from None


def _submit(
    payload: ResearchRequest, request: Request, settings: Settings, jobs: JobManager
) -> tuple[RunRow, Future[dict[str, Any]]]:
    effective = _effective(payload, settings)
    try:
        return jobs.submit(payload.query, payload.objective, effective, request.state.request_id)
    except JobCapacityError:
        raise HTTPException(
            503, "Research capacity unavailable", headers={"Retry-After": "5"}
        ) from None


@router.get("/health", response_model=HealthResponse)
def health_check(jobs: JobManager = Depends(get_jobs)) -> HealthResponse:
    return HealthResponse(active_jobs=jobs.active_count)


@router.get("/tools", response_model=list[ToolDescription])
def get_tools(settings: Settings = Depends(get_settings)) -> list[ToolDescription]:
    registry = create_default_registry(settings=settings)
    return [
        ToolDescription(**item, parameters=registry.get_tool(item["name"]).args)
        for item in registry.get_tool_descriptions()
    ]


@router.post("/research", status_code=202, response_model=ResearchStarted)
def start_research(
    payload: ResearchRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
) -> ResearchStarted:
    run, _ = _submit(payload, request, settings, jobs)
    return ResearchStarted(run_id=run.id)


@router.post("/research/sync", response_model=ResearchResultResponse, deprecated=True)
def run_research_sync(
    payload: ResearchRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
) -> dict[str, Any]:
    """Compatibility route; shares identities, capacity and cancellation with async dispatch."""
    _, future = _submit(payload, request, settings, jobs)
    try:
        return future.result(
            timeout=settings.overall_run_timeout_seconds
            + settings.api_shutdown_timeout_seconds
            + 10
        )
    except FutureTimeoutError:
        raise HTTPException(504, "Research request timed out") from None


@router.get("/research", response_model=list[RunSummary])
def list_runs(
    limit: int = Query(default=20, ge=1, le=100), settings: Settings = Depends(get_settings)
) -> list[RunSummary]:
    with ResearchService(settings) as service:
        return [
            RunSummary(
                run_id=run.id,
                query=run.query,
                status=run.status,
                created_at=run.created_at,
                elapsed_seconds=run.elapsed_seconds,
                evidence_count=service._evidence_repo.count_for_run(run.id),
                tool_call_count=run.total_tool_calls,
            )
            for run in service.list_runs(limit)
        ]


def _json_value(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


@router.get("/research/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str, settings: Settings = Depends(get_settings), jobs: JobManager = Depends(get_jobs)
) -> RunDetail:
    with ResearchService(settings) as service:
        run = _ensure_run(service, run_id)
        state = service.inspect_checkpoint(run_id)
        report = service.get_report(run_id)
        validations = [row for row in service.get_validations(run_id) if row.layer == "combined"]
        validation = (
            json.loads(max(validations, key=lambda row: row.report_version).result_json)
            if validations
            else None
        )
        active = jobs.contains(run_id)
        elapsed = run.elapsed_seconds
        if active and run.status == "running":
            started = service._event_repo.latest(run_id, "status_change")
            if started and json.loads(started.data_json).get("status") == "running":
                elapsed = (elapsed or 0) + max(
                    0,
                    (
                        datetime.now(UTC) - datetime.fromisoformat(started.created_at)
                    ).total_seconds(),
                )
        config = json.loads(run.config_json)
        return RunDetail(
            run_id=run.id,
            thread_id=run.thread_id,
            query=run.query,
            objective=run.objective,
            status=run.status,
            created_at=run.created_at,
            updated_at=run.updated_at,
            completed_at=run.completed_at,
            total_tool_calls=run.total_tool_calls,
            total_tokens=run.total_tokens,
            elapsed_seconds=elapsed,
            evidence_count=service._evidence_repo.count_for_run(run.id),
            job_active=active,
            report_version=report.version if report else None,
            research_plan=_json_value(state.get("research_plan")),
            active_task=_json_value(state.get("active_task")),
            current_agent=state.get("current_agent", ""),
            revision_count=state.get("revision_count", 0),
            research_iteration=state.get("research_iteration", 0),
            validation_result=validation,
            warnings=state.get("warnings", []),
            research_preferences=state.get("research_preferences")
            or {
                "depth": config.get("research_depth", settings.research_depth),
                "scope": config.get("research_scope", settings.research_scope),
                "source_preferences": config.get("source_preferences", settings.source_preferences),
            },
        )


@router.get("/research/{run_id}/report", response_model=ReportResponse)
def get_report(
    run_id: str, format: Literal["json", "md"] = "json", settings: Settings = Depends(get_settings)
) -> Any:
    with ResearchService(settings) as service:
        _ensure_run(service, run_id)
        report = service.get_report(run_id)
        if report is None:
            raise HTTPException(404, "Report not available")
        if format == "md":
            return PlainTextResponse(
                report.report_markdown,
                media_type="text/markdown",
                headers={"Content-Disposition": 'attachment; filename="research-report.md"'},
            )
        return ReportResponse(
            run_id=run_id,
            version=report.version,
            report_json=json.loads(report.report_json),
            report_markdown=report.report_markdown,
            created_at=report.created_at,
        )


@router.get("/research/{run_id}/evidence", response_model=list[EvidenceResponse], deprecated=True)
@router.get("/research/{run_id}/sources", response_model=list[EvidenceResponse])
def get_sources(run_id: str, settings: Settings = Depends(get_settings)) -> list[EvidenceResponse]:
    with ResearchService(settings) as service:
        _ensure_run(service, run_id)
        return [
            EvidenceResponse(
                evidence_id=row.id,
                task_id=row.task_id,
                tool_name=row.tool_name,
                url=row.url,
                title=row.title,
                content=row.content,
                source_type=row.source_type,
                retrieved_at=row.retrieved_at,
                metadata=json.loads(row.metadata_json),
                findings=json.loads(row.findings_json),
                source=json.loads(row.source_json),
            )
            for row in service.get_evidence(run_id)
        ]


@router.get("/research/{run_id}/stream", deprecated=True)
@router.get(
    "/research/{run_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def stream_research(
    run_id: str,
    request: Request,
    after: int = Query(default=0, ge=0, le=9223372036854775807),
    follow: bool = True,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    with ResearchService(settings) as service:
        _ensure_run(service, run_id)
    if last_event_id is not None:
        if (
            not re.fullmatch(r"[0-9]{1,19}", last_event_id)
            or int(last_event_id) > 9223372036854775807
        ):
            raise HTTPException(400, "Invalid event cursor")
        after = max(after, int(last_event_id))
    return StreamingResponse(
        stream_run_events(
            run_id,
            settings,
            after=after,
            disconnected=request.is_disconnected,
            request_id=request.state.request_id,
            follow=follow,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/research/{run_id}/resume", status_code=202, response_model=ResearchStarted)
def resume_research(
    run_id: str,
    payload: ResumeRequest,
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
) -> ResearchStarted:
    with ResearchService(settings) as service:
        _ensure_run(service, run_id)
    try:
        jobs.resume(run_id, payload.approved)
    except JobConflictError:
        raise HTTPException(409, "Run cannot resume") from None
    except JobCapacityError:
        raise HTTPException(
            503, "Research capacity unavailable", headers={"Retry-After": "5"}
        ) from None
    return ResearchStarted(run_id=run_id, message="Research continuation accepted")
