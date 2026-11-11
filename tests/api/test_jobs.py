"""Job identities, capacity, resume/restart and real subprocess shutdown through HTTP."""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

from research_agent.api.app import create_app
from research_agent.config.settings import Settings
from research_agent.graph.nodes import recovery_node
from research_agent.graph.state import create_initial_state
from research_agent.persistence.models import RunRow
from research_agent.service.research_service import ResearchService
from tests.api.test_routes import parse_sse, wait_idle
from tests.worker_fixtures import blocked_graph


def test_202_is_readable_while_running_and_capacity_is_bounded(
    settings: Settings,
    scripted_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = scripted_worker
    settings = settings.model_copy(update={"api_max_workers": 2, "llm_max_concurrency": 1})
    started, release = threading.Event(), threading.Event()

    def blocked(
        service: ResearchService,
        run_id: str,
        thread_id: str,
        query: str,
        objective: str,
        use_checkpointer: bool,
        **options: Any,
    ) -> dict[str, Any]:
        started.set()
        deadline = time.monotonic() + 10
        while not release.is_set() and time.monotonic() < deadline:
            if service.cancellation_event and service.cancellation_event.is_set():
                state = dict(create_initial_state(run_id, thread_id, query, objective))
                state.update(recovery_node(state))
                state["execution_status"] = "interrupted"
                return state
            release.wait(0.01)
        return runner(service, run_id, thread_id, query, objective, use_checkpointer, **options)

    monkeypatch.setattr(ResearchService, "_run_worker", blocked)
    app = create_app(settings)
    with TestClient(app) as client:
        first = client.post("/api/v1/research", json={"query": "What is X?"})
        assert first.status_code == 202 and started.wait(5)
        first_id = first.json()["run_id"]
        assert not release.is_set()
        assert client.get(f"/api/v1/research/{first_id}").json()["status"] == "running"
        assert client.get("/api/v1/health").status_code == 200
        second = client.post("/api/v1/research", json={"query": "What is X?"})
        assert second.status_code == 202
        second_id = second.json()["run_id"]
        assert client.get(f"/api/v1/research/{second_id}").json()["status"] == "queued"
        full = client.post("/api/v1/research", json={"query": "What is X?"})
        assert full.status_code == 503 and full.headers["retry-after"] == "5"
        assert len(client.get("/api/v1/research").json()) == 2
        release.set()
        assert wait_idle(client, first_id)["status"] == "completed"
        assert wait_idle(client, second_id)["status"] == "completed"
        with ResearchService(settings) as service:
            assert service.get_evidence(first_id)[0].id != service.get_evidence(second_id)[0].id
        assert app.state.jobs.active_count == 0


@pytest.mark.parametrize("approved", [True, False])
def test_http_approval_resume_keeps_identity_and_checkpoint_across_server_restart(
    settings: Settings,
    scripted_worker: Any,
    approved: bool,
) -> None:
    _, models = scripted_worker
    with TestClient(create_app(settings)) as client:
        accepted = client.post(
            "/api/v1/research",
            json={"query": "What is X?", "options": {"human_approval_enabled": True}},
        )
        run_id = accepted.json()["run_id"]
        paused = wait_idle(client, run_id)
        assert paused["status"] == "awaiting_approval"
        thread_id = paused["thread_id"]
        stream = parse_sse(client.get(f"/api/v1/research/{run_id}/events").text)
        assert stream[-1]["data"]["status"] == "awaiting_approval"
        cursor = max(int(row["id"]) for row in stream if "id" in row)
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/v1/research/{run_id}").json()["status"] == "awaiting_approval"
        assert client.post(f"/api/v1/research/{run_id}/resume", json={}).status_code == 409
        resumed = client.post(f"/api/v1/research/{run_id}/resume", json={"approved": approved})
        assert resumed.status_code == 202 and resumed.json()["run_id"] == run_id
        result = wait_idle(client, run_id)
        assert result["thread_id"] == thread_id
        assert result["status"] == ("completed" if approved else "incomplete")
        assert len(client.get("/api/v1/research").json()) == 1
        assert len(models[run_id]["planner"].invocations) == 1
        tail = parse_sse(
            client.get(
                f"/api/v1/research/{run_id}/events", headers={"Last-Event-ID": str(cursor)}
            ).text
        )
        assert all(int(row["id"]) > cursor for row in tail if "id" in row)


def test_duplicate_resume_is_rejected_and_no_second_execution_is_created(
    settings: Settings,
    scripted_worker: Any,
) -> None:
    with TestClient(create_app(settings)) as client:
        run_id = client.post(
            "/api/v1/research",
            json={"query": "What is X?", "options": {"human_approval_enabled": True}},
        ).json()["run_id"]
        wait_idle(client, run_id)
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(
                pool.map(
                    lambda _: (
                        client.post(
                            f"/api/v1/research/{run_id}/resume", json={"approved": True}
                        ).status_code
                    ),
                    range(2),
                )
            )
        assert sorted(statuses) == [202, 409]
        assert wait_idle(client, run_id)["status"] == "completed"
        assert len(client.get("/api/v1/research").json()) == 1


def test_startup_marks_abandoned_jobs_without_changing_paused_or_completed_runs(
    settings: Settings,
) -> None:
    with ResearchService(settings) as service:
        for status in ("running", "queued", "awaiting_approval", "completed"):
            service._run_repo.create(
                RunRow(
                    id=status,
                    thread_id=status,
                    query="Query",
                    status=status,
                    config_json=json.dumps({**settings.model_dump(), "execution_origin": "api"}),
                )
            )
        service._log_event("running", "status_change", data={"status": "running"})
        service._run_repo.create(
            RunRow(
                id="cli",
                thread_id="cli",
                query="Query",
                status="running",
                config_json=settings.model_dump_json(),
            )
        )
    with TestClient(create_app(settings)) as client:
        for run_id in ("running", "queued"):
            assert client.get(f"/api/v1/research/{run_id}").json()["status"] == "interrupted"
        assert (
            client.get("/api/v1/research/awaiting_approval").json()["status"] == "awaiting_approval"
        )
        assert client.get("/api/v1/research/completed").json()["status"] == "completed"
        assert client.get("/api/v1/research/cli").json()["status"] == "running"
        assert client.post("/api/v1/research/queued/resume", json={}).status_code == 409


def test_only_one_api_owner_can_recover_a_database(settings: Settings) -> None:
    with (
        TestClient(create_app(settings)),
        pytest.raises(RuntimeError, match="already owns"),
        TestClient(create_app(settings)),
    ):
        pass
    with TestClient(create_app(settings)) as replacement:
        assert replacement.get("/api/v1/health").status_code == 200


def test_shutdown_cancels_a_real_blocked_process_and_leaves_no_owned_children(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("research_agent.service.research_service.graph_worker", blocked_graph)
    before = {child.pid for child in multiprocessing.active_children()}
    app = create_app(settings)
    with TestClient(app) as client:
        run_id = client.post("/api/v1/research", json={"query": "What is X?"}).json()["run_id"]
        manager = app.state.jobs
        deadline = time.monotonic() + 12
        ready = False
        while time.monotonic() < deadline:
            with ResearchService(settings) as service:
                ready = any(
                    event.event_type == "worker_ready" for event in service.get_events(run_id)
                )
            if ready:
                break
            time.sleep(0.03)
        assert ready, "The actual subprocess must begin before testing shutdown cancellation"
        queued_id = client.post("/api/v1/research", json={"query": "Another query"}).json()[
            "run_id"
        ]
        assert client.get(f"/api/v1/research/{queued_id}").json()["status"] == "queued"
    assert manager.active_count == 0
    assert {child.pid for child in multiprocessing.active_children()} <= before
    with ResearchService(settings) as service:
        assert service.get_run(run_id).status == "interrupted"
        assert service.get_run(queued_id).status == "interrupted"
        assert any(
            "WorkerCancelledError" in event.data_json for event in service.get_events(run_id)
        )


def test_missing_dependencies_and_generic_exceptions_remain_sanitized(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = client.app.state.jobs
    client.app.state.jobs = None
    assert client.get("/api/v1/health").status_code == 503
    client.app.state.jobs = manager

    class BrokenService:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("dummy-private-provider-body")

    monkeypatch.setattr("research_agent.api.routes.ResearchService", BrokenService)
    response = client.get("/api/v1/research", headers={"Origin": "http://allowed.test"})
    assert response.status_code == 500
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "dummy-private" not in response.text
    assert response.headers["access-control-allow-origin"] == "http://allowed.test"
