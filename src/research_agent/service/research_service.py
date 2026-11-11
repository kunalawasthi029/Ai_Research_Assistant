"""Research service: orchestrates graph runs and persists results.

Pure Python — no FastAPI or Streamlit imports.
This is the single entry point for running research from any consumer
(CLI, API, tests).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from threading import Event
from typing import Any

from research_agent.config.settings import Settings, get_settings
from research_agent.execution.isolation import WorkerError, WorkerTimeoutError, run_isolated
from research_agent.graph.nodes import recovery_node
from research_agent.graph.runtime import checkpoint_state, graph_worker
from research_agent.graph.state import create_initial_state
from research_agent.persistence.database import get_connection, init_db
from research_agent.persistence.models import (
    EventRow,
    EvidenceRow,
    ReportRow,
    RunRow,
    ValidationRow,
)
from research_agent.persistence.repositories import (
    EventRepository,
    EvidenceRepository,
    ReportRepository,
    RunRepository,
    ValidationRepository,
)

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class ResearchService:
    """Orchestrates research runs: build graph, invoke, persist.

    Usage:
        service = ResearchService(settings)
        result = service.run_research("What is quantum computing?")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        db_conn: sqlite3.Connection | None = None,
        cancellation_event: Event | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cancellation_event = cancellation_event
        self._owns_connection = db_conn is None
        self._conn = db_conn if db_conn is not None else get_connection(self.settings.database_path)
        init_db(self._conn)

        self._run_repo = RunRepository(self._conn)
        self._event_repo = EventRepository(self._conn)
        self._evidence_repo = EvidenceRepository(self._conn)
        self._report_repo = ReportRepository(self._conn)
        self._validation_repo = ValidationRepository(self._conn)

    def close(self) -> None:
        """Close a service-owned connection on its owning thread."""
        if self._owns_connection:
            self._conn.close()

    def __enter__(self) -> ResearchService:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def run_research(
        self,
        query: str,
        objective: str = "",
        use_checkpointer: bool = True,
    ) -> dict[str, Any]:
        """Execute a complete research run.

        Args:
            query: The research question
            objective: Optional refined objective (defaults to query)
            use_checkpointer: Whether to use SQLite checkpointing

        Returns:
            Dict with: run_id, status, report (if successful), evidence_count,
                       tool_call_count, elapsed_seconds
        """
        run = self.create_run(query, objective)
        return self.execute_run(run.id, use_checkpointer=use_checkpointer)

    def create_run(
        self, query: str, objective: str = "", *, request_id: str = "", origin: str = "direct"
    ) -> RunRow:
        """Persist the single identity before a caller can accept/dispatch execution."""
        run = RunRow(
            id=str(uuid.uuid4()),
            thread_id=str(uuid.uuid4()),
            query=query,
            objective=objective or query,
            status="queued",
            config_json=json.dumps({**self.settings.model_dump(), "execution_origin": origin}),
        )
        self._run_repo.create(run)
        self._log_event(
            run.id, "status_change", data={"status": "queued", "request_id": request_id}
        )
        return run

    def execute_run(self, run_id: str, *, use_checkpointer: bool = True) -> dict[str, Any]:
        """Execute a persisted queued run exactly once per successful atomic claim."""
        run = self.get_run(run_id)
        if run is None or not self._run_repo.claim(run_id, ("queued",)):
            raise ValueError("Run is not available for execution")
        thread_id, query, objective = run.thread_id, run.query, run.objective
        self._log_event(run_id, "status_change", data={"status": "running"})

        start_time = time.monotonic()

        try:
            result = self._execute_graph(
                run_id=run_id,
                thread_id=thread_id,
                query=query,
                objective=objective,
                use_checkpointer=use_checkpointer,
            )

            elapsed = time.monotonic() - start_time

            # Persist results
            self._persist_results(run_id, result)

            # Update run status
            status = result.get(
                "execution_status", "best_effort" if result.get("report") else "incomplete"
            )
            self._run_repo.update_status(
                run_id=run_id,
                status=status,
                completed_at=_utcnow_iso()
                if status not in {"awaiting_approval", "interrupted"}
                else None,
                elapsed_seconds=elapsed,
            )

            self._log_event(
                run_id,
                "status_change",
                data={
                    "status": status,
                    "elapsed_seconds": elapsed,
                },
            )

            return {
                "run_id": run_id,
                "thread_id": thread_id,
                "status": status,
                "report": result.get("report"),
                "validation_result": result.get("validation_result"),
                "evidence_count": len(result.get("evidence", [])),
                "tool_call_count": len(result.get("tool_calls", [])),
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error("Research run %s failed (%s)", run_id, type(e).__name__)

            self._run_repo.update_status(
                run_id=run_id,
                status="failed",
                completed_at=_utcnow_iso(),
                elapsed_seconds=elapsed,
            )
            self._log_event(
                run_id,
                "error",
                data={
                    "error": type(e).__name__,
                    "elapsed_seconds": elapsed,
                },
            )

            return {
                "run_id": run_id,
                "thread_id": thread_id,
                "status": "failed",
                "error": type(e).__name__,
                "elapsed_seconds": elapsed,
            }

    def _execute_graph(
        self,
        run_id: str,
        thread_id: str,
        query: str,
        objective: str,
        use_checkpointer: bool,
    ) -> dict[str, Any]:
        """Build and invoke the LangGraph graph."""
        return self._run_worker(run_id, thread_id, query, objective, use_checkpointer)

    def _run_worker(
        self,
        run_id: str,
        thread_id: str,
        query: str,
        objective: str,
        use_checkpointer: bool,
        *,
        resume: bool = False,
        approved: bool | None = None,
        remaining_seconds: float | None = None,
        settings: Settings | None = None,
    ) -> dict[str, Any]:
        """Own live execution separately from the application's SQLite connection."""
        settings = settings or self.settings
        remaining = (
            remaining_seconds
            if remaining_seconds is not None
            else settings.overall_run_timeout_seconds
        )
        counts: dict[str, int] = {}
        for event in self.get_events(run_id):
            if event.event_type == "tool_started":
                name = json.loads(event.data_json).get("tool", "")
                counts[name] = counts.get(name, 0) + 1

        def publish(event: dict[str, Any]) -> None:
            self._handle_worker_event(run_id, event)

        try:
            result: dict[str, Any] = run_isolated(
                graph_worker,
                (
                    settings,
                    run_id,
                    thread_id,
                    query,
                    objective,
                    use_checkpointer,
                    resume,
                    approved,
                    remaining,
                    counts,
                ),
                remaining,
                publish,
                cancelled=self.cancellation_event.is_set if self.cancellation_event else None,
            )
            return result
        except (WorkerTimeoutError, WorkerError) as error:
            state = checkpoint_state(settings, thread_id) if use_checkpointer else {}
            if not state:
                state = dict(create_initial_state(run_id, thread_id, query, objective))
            state.update(
                {
                    "failure_node": "worker_deadline"
                    if isinstance(error, WorkerTimeoutError)
                    else "worker_exit",
                    "error_details": [type(error).__name__],
                }
            )
            state.update(recovery_node(state))
            state["execution_status"] = "interrupted"
            publish({"event_type": "worker_stopped", "error_type": type(error).__name__})
            return state

    def inspect_checkpoint(self, run_id: str) -> dict[str, Any]:
        """Inspect this run's checkpoint without contacting an LLM or executing a tool."""
        run = self.get_run(run_id)
        if run is None:
            raise ValueError("Research run not found")
        public = json.loads(run.config_json)
        settings = Settings(
            **{
                **self.settings.model_dump(),
                **public,
                "groq_api_key": self.settings.groq_api_key,
                "tavily_api_key": self.settings.tavily_api_key,
            }
        )
        return checkpoint_state(settings, run.thread_id)

    def resume_research(self, run_id: str, approved: bool | None = None) -> dict[str, Any]:
        """Resume the existing thread; do not create another run or repeat accepted planning."""
        run, settings, remaining = self.resume_requirements(run_id, approved)
        if not self._run_repo.claim(run_id, (run.status,)):
            raise ValueError("Run is already executing")
        self._log_event(run_id, "status_change", data={"status": "running"})
        started = time.monotonic()
        result = self._run_worker(
            run_id,
            run.thread_id,
            run.query,
            run.objective,
            True,
            resume=True,
            approved=approved,
            remaining_seconds=remaining,
            settings=settings,
        )
        self._persist_results(run_id, result)
        status = result.get("execution_status", "incomplete")
        elapsed = (run.elapsed_seconds or 0) + time.monotonic() - started
        self._run_repo.update_status(
            run_id,
            status,
            completed_at=_utcnow_iso()
            if status not in {"awaiting_approval", "interrupted"}
            else None,
            elapsed_seconds=elapsed,
        )
        self._log_event(run_id, "status_change", data={"status": status})
        return {
            "run_id": run_id,
            "thread_id": run.thread_id,
            "status": status,
            "report": result.get("report"),
            "validation_result": result.get("validation_result"),
            "evidence_count": len(result.get("evidence", [])),
            "tool_call_count": len(result.get("tool_calls", [])),
            "elapsed_seconds": elapsed,
        }

    def resume_requirements(
        self, run_id: str, approved: bool | None = None
    ) -> tuple[RunRow, Settings, float]:
        """Validate a resume before reserving job capacity; do not mutate state."""
        run = self.get_run(run_id)
        if run is None or run.status not in {"awaiting_approval", "interrupted"}:
            raise ValueError("Run is not awaiting a resumable continuation")
        if run.status == "awaiting_approval" and approved is None:
            raise ValueError("An explicit plan approval or rejection is required")
        if not self.inspect_checkpoint(run_id):
            raise ValueError("Run has no saved checkpoint")
        settings = Settings(
            **{
                **self.settings.model_dump(),
                **json.loads(run.config_json),
                "groq_api_key": self.settings.groq_api_key,
                "tavily_api_key": self.settings.tavily_api_key,
            }
        )
        remaining = settings.overall_run_timeout_seconds - (run.elapsed_seconds or 0)
        if remaining <= 0:
            raise ValueError("The run's total execution budget is exhausted")
        return run, settings, remaining

    def _persist_results(self, run_id: str, state: dict[str, Any]) -> None:
        """Persist evidence, report, and validation from the final state."""
        evidence_list = state.get("evidence", [])
        findings = state.get("findings", [])
        sources = {item.evidence_id: item for item in state.get("sources", [])}
        if evidence_list:
            self._evidence_repo.create_many(
                [
                    EvidenceRow(
                        id=ev.evidence_id,
                        run_id=run_id,
                        task_id=ev.research_task_id,
                        tool_name=ev.tool_name,
                        url=ev.url,
                        title=ev.title,
                        content=ev.content,
                        source_type=ev.source_type,
                        metadata_json=json.dumps(ev.metadata, default=str),
                        retrieved_at=ev.retrieved_at.isoformat(),
                        findings_json=json.dumps(
                            [
                                item.model_dump(mode="json")
                                for item in findings
                                if ev.evidence_id in item.supporting_evidence
                            ]
                        ),
                        source_json=sources[ev.evidence_id].model_dump_json()
                        if ev.evidence_id in sources
                        else "{}",
                    )
                    for ev in evidence_list
                ]
            )
        report = state.get("report")
        if report:
            self._report_repo.create(
                ReportRow(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    version=report.version,
                    report_json=report.model_dump_json(),
                    report_markdown=self._render_markdown(report),
                )
            )
        history = state.get("validation_history", [])
        if "validation_history" not in state and state.get("validation_result"):
            history = [
                {
                    "report_version": report.version if report else 1,
                    "result": state["validation_result"],
                }
            ]
        for entry in history:
            result = entry["result"]
            self._validation_repo.create(
                ValidationRow(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    report_version=entry["report_version"],
                    layer=result.layer,
                    result_json=result.model_dump_json(),
                )
            )
        self._refresh_metrics(run_id, state)

    def _refresh_metrics(self, run_id: str, state: dict[str, Any] | None = None) -> None:
        """Keep observed usage even when a worker dies before its next checkpoint."""
        state = state or {}
        run = self.get_run(run_id)
        if run is None:
            return
        events = self.get_events(run_id)
        calls = sum(event.event_type == "tool_started" for event in events)
        tokens = sum(
            int(json.loads(event.data_json).get("total_tokens", 0) or 0)
            for event in events
            if event.event_type == "llm_call"
        )
        self._run_repo.update_status(
            run_id,
            run.status,
            total_tool_calls=max(run.total_tool_calls, calls, len(state.get("tool_calls", []))),
            total_tokens=max(
                run.total_tokens,
                tokens,
                sum(item.total_tokens for item in state.get("token_usage", [])),
            ),
        )

    def _handle_worker_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Persist normalized state separately from public metadata events."""
        if event["event_type"] == "state_snapshot":
            self._persist_results(run_id, event["state"])
            return
        self._log_event(
            run_id,
            event["event_type"],
            agent=event.get("agent", ""),
            node=event.get("node", ""),
            event_id=event.get("event_id"),
            data={
                key: value
                for key, value in event.items()
                if key not in {"event_type", "agent", "node", "event_id"}
            },
        )
        if event["event_type"] in {"tool_started", "llm_call"}:
            self._refresh_metrics(run_id)

    def _render_markdown(self, report: Any) -> str:
        """Render a Report schema into Markdown."""
        parts = [
            f"# {report.title}\n",
            f"## Executive Summary\n{report.executive_summary}\n",
            f"## Research Objective\n{report.research_objective}\n",
            f"## Methodology\n{report.methodology}\n",
            "## Key Findings\n",
        ]
        for i, kf in enumerate(report.key_findings, 1):
            parts.append(f"{i}. {kf}\n")

        parts.append("\n## Detailed Analysis\n")
        for section in report.detailed_analysis:
            parts.append(f"### {section.heading}\n{section.content}\n")

        if report.comparative_insights:
            parts.append("## Comparative Insights\n")
            for ci in report.comparative_insights:
                parts.append(f"- {ci}\n")

        if report.recommendations:
            parts.append("## Recommendations\n")
            for rec in report.recommendations:
                parts.append(f"- {rec}\n")

        parts.append("\n## Limitations\n")
        for lim in report.limitations:
            parts.append(f"- {lim}\n")

        parts.append(f"\n## Conclusion\n{report.conclusion}\n")

        parts.append("\n## References\n")
        for ref in report.references:
            parts.append(
                f"- [{ref.ref_id}] [{ref.title}]({ref.url}) (evidence: {', '.join(ref.all_evidence_ids)})\n"
            )

        return "\n".join(parts)

    def _log_event(
        self,
        run_id: str,
        event_type: str,
        agent: str = "",
        node: str = "",
        data: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> None:
        """Persist an event to the events table."""
        event = EventRow(
            id=event_id or str(uuid.uuid4()),
            run_id=run_id,
            event_type=event_type,
            agent=agent,
            node=node,
            data_json=json.dumps(data or {}, default=str),
        )
        self._event_repo.create(event)

    # ──────────────────── Query Methods ────────────────────

    def get_run(self, run_id: str) -> RunRow | None:
        """Get a run by ID."""
        return self._run_repo.get(run_id)

    def list_runs(self, limit: int = 50) -> list[RunRow]:
        """List recent runs."""
        return self._run_repo.list_all(limit)

    def get_evidence(self, run_id: str) -> list[EvidenceRow]:
        """Get all evidence for a run."""
        return self._evidence_repo.list_for_run(run_id)

    def get_report(self, run_id: str) -> ReportRow | None:
        """Get the latest report for a run."""
        return self._report_repo.get_latest(run_id)

    def get_reports(self, run_id: str) -> list[ReportRow]:
        """Return every persisted report version in order."""
        return self._report_repo.list_for_run(run_id)

    def get_events(self, run_id: str) -> list[EventRow]:
        """Get all events for a run."""
        return self._event_repo.list_for_run(run_id)

    def get_validations(self, run_id: str) -> list[ValidationRow]:
        """Get all validations for a run."""
        return self._validation_repo.list_for_run(run_id)
