"""Exercise the required HTTP contract against real service persistence."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from research_agent.config.settings import Settings
from research_agent.service.research_service import ResearchService


def wait_idle(client: TestClient, run_id: str, timeout: float = 10) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/research/{run_id}")
        assert response.status_code == 200, response.text
        value = response.json()
        if not value["job_active"] and value["status"] not in {"queued", "running"}:
            return value
        time.sleep(0.02)
    pytest.fail("API job did not reach an inactive status")


def parse_sse(text: str) -> list[dict[str, Any]]:
    rows = []
    for block in text.strip().split("\n\n"):
        row = {}
        for line in block.splitlines():
            key, _, value = line.partition(": ")
            if key == "data":
                row[key] = json.loads(value)
            elif key in {"id", "event"}:
                row[key] = value
        if row:
            rows.append(row)
    return rows


def test_create_status_events_sources_and_report_use_one_persisted_id(
    client: TestClient, settings: Settings
) -> None:
    accepted = client.post(
        "/api/v1/research", json={"query": "What is X?"}, headers={"X-Request-ID": "create-123"}
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run_id"]
    assert accepted.headers["x-request-id"] == "create-123"
    detail = wait_idle(client, run_id)
    assert detail["status"] == "completed"
    assert detail["research_plan"]["research_tasks"]
    assert detail["validation_result"]["status"] == "pass"
    sources = client.get(f"/api/v1/research/{run_id}/sources").json()
    assert len(sources) == detail["evidence_count"] == 1
    assert sources[0]["findings"][0]["supporting_evidence"] == [sources[0]["evidence_id"]]
    report = client.get(f"/api/v1/research/{run_id}/report?format=json").json()
    assert report["run_id"] == run_id
    assert report["report_json"]["references"][0]["evidence_id"] == sources[0]["evidence_id"]
    markdown = client.get(f"/api/v1/research/{run_id}/report?format=md")
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "https://example.com/x" in markdown.text
    events_response = client.get(f"/api/v1/research/{run_id}/events")
    assert events_response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(events_response.text)
    assert {
        "node_started",
        "tool_started",
        "source_found",
        "validation",
        "decision",
        "done",
    }.issubset({row["event"] for row in events})
    cursors = [int(row["id"]) for row in events if "id" in row]
    assert cursors == sorted(set(cursors))
    tail = parse_sse(
        client.get(
            f"/api/v1/research/{run_id}/events", headers={"Last-Event-ID": str(cursors[-1])}
        ).text
    )
    assert not any("id" in row for row in tail)
    history = client.get("/api/v1/research").json()
    assert len(history) == 1 and history[0]["run_id"] == run_id
    assert history[0]["evidence_count"] == 1
    with ResearchService(settings) as service:
        assert len(service.list_runs()) == 1
        assert "dummy-private-key" not in service.get_run(run_id).config_json
        assert len(service.get_validations(run_id)) == 3


def test_health_tools_openapi_and_compatibility(client: TestClient) -> None:
    assert client.get("/api/v1/health").json()["status"] == "ok"
    tools = client.get("/api/v1/tools").json()
    assert len(tools) == 6
    assert {"calculator", "web_search", "source_verifier"}.issubset({row["name"] for row in tools})
    assert all(row["parameters"] for row in tools)
    paths = client.get("/openapi.json").json()["paths"]
    for suffix in (
        "/health",
        "/tools",
        "/research",
        "/research/{run_id}",
        "/research/{run_id}/events",
        "/research/{run_id}/report",
        "/research/{run_id}/sources",
        "/research/{run_id}/resume",
    ):
        assert "/api/v1" + suffix in paths
    result = client.post("/api/v1/research/sync", json={"query": "What is X?"}).json()
    assert result["status"] == "completed"
    run_id = result["run_id"]
    assert client.get(f"/api/v1/research/{run_id}").status_code == 200
    assert (
        client.get(f"/api/v1/research/{run_id}/evidence").json()
        == client.get(f"/api/v1/research/{run_id}/sources").json()
    )
    assert "done" in client.get(f"/api/v1/research/{run_id}/stream").text


@pytest.mark.parametrize(
    "body",
    [
        {"query": "  "},
        {"query": "x" * 4001},
        {"query": "Valid query", "objective": "x" * 4001},
        {"query": "Valid query", "unknown": "private-payload"},
        {"query": "Valid query", "options": {"max_research_iterations": 0}},
        {"query": "Valid query", "options": {"max_research_iterations": True}},
        {"query": "Valid query", "options": {"database_path": "private-payload"}},
        {"query": "Valid query", "options": {"human_approval_enabled": "true"}},
    ],
)
def test_invalid_requests_are_sanitized_and_create_no_run(client: TestClient, body: Any) -> None:
    response = client.post("/api/v1/research", json=body)
    assert response.status_code == 422
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert response.json()["error"] == "ValidationError"
    assert "private-payload" not in response.text and '"input"' not in response.text
    assert client.get("/api/v1/research").json() == []


@pytest.mark.parametrize(
    "path", ["/research?limit=0", "/research?limit=101", "/research/missing/report?format=pdf"]
)
def test_query_bounds(client: TestClient, path: str) -> None:
    assert client.get("/api/v1" + path).status_code == 422


@pytest.mark.parametrize(
    "suffix,method",
    [("", "GET"), ("/report", "GET"), ("/sources", "GET"), ("/events", "GET"), ("/resume", "POST")],
)
def test_missing_resources_have_uniform_404(client: TestClient, suffix: str, method: str) -> None:
    response = client.request(
        method, "/api/v1/research/missing" + suffix, **({"json": {}} if method == "POST" else {})
    )
    assert response.status_code == 404
    assert response.json()["status_code"] == 404
    assert response.json()["request_id"]


def test_cors_request_ids_and_sanitized_dependency_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = client.get(
        "/api/v1/health", headers={"Origin": "http://allowed.test", "X-Request-ID": "read-123"}
    )
    assert allowed.headers["access-control-allow-origin"] == "http://allowed.test"
    assert allowed.headers["x-request-id"] == "read-123"
    assert "access-control-allow-credentials" not in allowed.headers
    rejected = client.get("/api/v1/health", headers={"Origin": "https://unlisted.example"})
    assert "access-control-allow-origin" not in rejected.headers
    preflight = client.options(
        "/api/v1/research",
        headers={"Origin": "http://allowed.test", "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 200 and preflight.headers["x-request-id"]
    denied = client.options(
        "/api/v1/research",
        headers={"Origin": "https://unlisted.example", "Access-Control-Request-Method": "POST"},
    )
    assert denied.status_code == 400 and denied.json()["error"] == "CorsRejected"
    assert denied.json()["request_id"] == denied.headers["x-request-id"]
    response = client.get("/api/v1/health", headers={"X-Request-ID": "invalid request id"})
    assert response.headers["x-request-id"] != "invalid request id"

    class BrokenService:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise sqlite3.OperationalError("private-database-path")

    monkeypatch.setattr("research_agent.api.routes.ResearchService", BrokenService)
    response = client.get("/api/v1/research", headers={"Origin": "http://allowed.test"})
    assert response.status_code == 503 and "private-database-path" not in response.text
    assert response.headers["access-control-allow-origin"] == "http://allowed.test"


def test_options_cannot_enlarge_server_caps_and_are_persisted(
    client: TestClient, settings: Settings
) -> None:
    response = client.post(
        "/api/v1/research", json={"query": "What is X?", "options": {"max_research_iterations": 4}}
    )
    assert response.status_code == 400
    assert client.get("/api/v1/research").json() == []
    response = client.post(
        "/api/v1/research", json={"query": "What is X?", "options": {"max_research_iterations": 1}}
    )
    run_id = response.json()["run_id"]
    wait_idle(client, run_id)
    with ResearchService(settings) as service:
        assert json.loads(service.get_run(run_id).config_json)["max_research_iterations"] == 1


def test_cursor_bounds_and_invalid_resume_body_are_rejected(client: TestClient) -> None:
    run_id = client.post("/api/v1/research", json={"query": "What is X?"}).json()["run_id"]
    wait_idle(client, run_id)
    for cursor in ("-1", "private-value", "9" * 30):
        response = client.get(
            f"/api/v1/research/{run_id}/events", headers={"Last-Event-ID": cursor}
        )
        assert response.status_code == 400 and "private-value" not in response.text
    assert client.get(f"/api/v1/research/{run_id}/events?after=-1").status_code == 422
    assert (
        client.post(f"/api/v1/research/{run_id}/resume", json={"approved": "yes"}).status_code
        == 422
    )


def test_legacy_empty_configuration_uses_the_injected_checkpoint_path(
    client: TestClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_agent.persistence.models import RunRow

    seen = []

    def checkpoint(configured: Settings, thread_id: str) -> dict[str, Any]:
        seen.append(configured.checkpoint_db_path)
        return {}

    with ResearchService(settings) as service:
        service._run_repo.create(
            RunRow(id="legacy", thread_id="thread", query="Query", config_json="{}")
        )
    monkeypatch.setattr("research_agent.service.research_service.checkpoint_state", checkpoint)
    assert client.get("/api/v1/research/legacy").status_code == 200
    assert seen == [settings.checkpoint_db_path]
