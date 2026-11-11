"""Bounded single-instance API dispatch; SQLite connections stay on their owning thread."""

from __future__ import annotations

import json
import logging
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_agent.config.settings import Settings
from research_agent.persistence.models import RunRow
from research_agent.service.research_service import ResearchService

logger = logging.getLogger(__name__)


class JobConflictError(ValueError):
    """A run already has an active job or is not resumable."""


class JobCapacityError(RuntimeError):
    """The bounded dispatcher is full or shutting down."""


class InstanceLease:
    """Prevent a second API process from treating another server's jobs as abandoned."""

    def __init__(self, database_path: str) -> None:
        if database_path == ":memory:":
            raise ValueError("The API requires a persistent database file")
        path = Path(database_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            file = resources.enter_context(open(str(path) + ".api.lock", "a+b"))
            file.seek(0, 2)
            if not file.tell():
                file.write(b"0")
                file.flush()
            file.seek(0)
            try:
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
            except OSError:
                raise RuntimeError("An API instance already owns this database") from None
            self._resources = resources.pop_all()

    def close(self) -> None:
        self._resources.close()


@dataclass
class Job:
    settings: Settings
    stop: threading.Event
    future: Future[dict[str, Any]] | None = None


class JobManager:
    """Own one bounded executor and cancel its actual isolated graph workers on shutdown."""

    def __init__(self, settings: Settings) -> None:
        if settings.checkpoint_db_path == ":memory:":
            raise ValueError("The API requires persistent checkpoints")
        self.settings = settings
        # One synchronous model call at a time per graph; cap all worker processes.
        self.max_workers = min(settings.api_max_workers, settings.llm_max_concurrency)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._closing = False
        self._lease = InstanceLease(settings.database_path)
        try:
            self._recover_abandoned()
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="research-api"
            )
        except Exception:
            self._lease.close()
            raise

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def contains(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._jobs

    def _check_capacity(self) -> None:
        if self._closing or len(self._jobs) >= self.max_workers + self.settings.api_max_queued_jobs:
            raise JobCapacityError("Research capacity is unavailable")

    def submit(
        self, query: str, objective: str, settings: Settings, request_id: str = ""
    ) -> tuple[RunRow, Future[dict[str, Any]]]:
        with self._lock:
            self._check_capacity()
            with ResearchService(settings) as service:
                run = service.create_run(query, objective, request_id=request_id, origin="api")
            return run, self._dispatch(run.id, Job(settings, threading.Event()), resume=False)

    def resume(self, run_id: str, approved: bool | None) -> Future[dict[str, Any]]:
        with self._lock:
            if run_id in self._jobs:
                raise JobConflictError("Run already has an active job")
            self._check_capacity()
            with ResearchService(self.settings) as service:
                try:
                    run, settings, _ = service.resume_requirements(run_id, approved)
                except ValueError:
                    raise JobConflictError(
                        "Run requires a saved checkpoint, an available execution budget and an explicit approval decision when paused"
                    ) from None
                config = {**json.loads(run.config_json), "execution_origin": "api"}
                service._run_repo.update_config(run_id, json.dumps(config))
            return self._dispatch(
                run_id, Job(settings, threading.Event()), resume=True, approved=approved
            )

    def _dispatch(
        self, run_id: str, job: Job, *, resume: bool, approved: bool | None = None
    ) -> Future[dict[str, Any]]:
        self._jobs[run_id] = job
        try:
            future = self._executor.submit(self._execute, run_id, job, resume, approved)
            job.future = future
            future.add_done_callback(lambda completed: self._forget(run_id, job))
            return future
        except Exception:
            self._jobs.pop(run_id, None)
            self._mark_stopped(run_id, "failed")
            raise JobCapacityError("Research could not be dispatched") from None

    def _execute(
        self, run_id: str, job: Job, resume: bool, approved: bool | None
    ) -> dict[str, Any]:
        try:
            with ResearchService(job.settings, cancellation_event=job.stop) as service:
                if job.stop.is_set():
                    self._mark_stopped(run_id, "interrupted")
                    return {"run_id": run_id, "status": "interrupted"}
                return (
                    service.resume_research(run_id, approved)
                    if resume
                    else service.execute_run(run_id)
                )
        except Exception as error:
            status = "interrupted" if job.stop.is_set() else "failed"
            logger.error(
                "research_job run_id=%s status=%s error_type=%s",
                run_id,
                status,
                type(error).__name__,
            )
            self._mark_stopped(run_id, status)
            return {"run_id": run_id, "status": status}

    def _mark_stopped(self, run_id: str, status: str) -> None:
        try:
            with ResearchService(self.settings) as service:
                run = service.get_run(run_id)
                if run and run.status in {"queued", "running"}:
                    service._run_repo.update_status(
                        run_id,
                        status,
                        completed_at=datetime.now(UTC).isoformat() if status == "failed" else None,
                    )
                    service._log_event(run_id, "status_change", data={"status": status})
        except Exception as error:
            logger.error(
                "research_job_persistence run_id=%s error_type=%s", run_id, type(error).__name__
            )

    def _forget(self, run_id: str, job: Job) -> None:
        with self._lock:
            if self._jobs.get(run_id) is job:
                self._jobs.pop(run_id)
            if self._closing and not self._jobs:
                self._lease.close()

    def _recover_abandoned(self) -> None:
        with ResearchService(self.settings) as service:
            for run in service._run_repo.active():
                if json.loads(run.config_json).get("execution_origin") != "api":
                    continue
                elapsed = run.elapsed_seconds or 0.0
                events = service.get_events(run.id)
                starts = [
                    event
                    for event in events
                    if event.event_type == "status_change"
                    and json.loads(event.data_json).get("status") == "running"
                ]
                if run.status == "running" and starts:
                    # Charge the unclosed slice conservatively, including server downtime.
                    started = datetime.fromisoformat(starts[-1].created_at)
                    cap = json.loads(run.config_json).get(
                        "overall_run_timeout_seconds", self.settings.overall_run_timeout_seconds
                    )
                    elapsed = min(
                        float(cap), elapsed + max(0, (datetime.now(UTC) - started).total_seconds())
                    )
                service._run_repo.update_status(run.id, "interrupted", elapsed_seconds=elapsed)
                service._log_event(run.id, "server_restart", data={"status": "interrupted"})

    def close(self) -> None:
        with self._lock:
            self._closing = True
            jobs = list(self._jobs.items())
            for _, job in jobs:
                job.stop.set()
        futures = []
        for run_id, job in jobs:
            if job.future is not None:
                futures.append(job.future)
                if job.future.cancel():
                    self._mark_stopped(run_id, "interrupted")
        if futures:
            wait(futures, timeout=self.settings.api_shutdown_timeout_seconds)
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            if not self._jobs:
                self._lease.close()
            # If cleanup is delayed, retain the lease until the last callback completes.
            # A replacement server must not recover jobs still owned by this process.
