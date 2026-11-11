"""HTTP client contracts and strict frontend/core import boundaries."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import pytest
import respx
from api_client import APIClient, APIError, parse_sse
from live_state import LiveState


def test_backend_auth_key_is_a_header_not_a_url_parameter() -> None:
    with respx.mock as mock:
        route = mock.get("https://backend.test/api/v1/tools").respond(200, json=[])
        with APIClient("https://backend.test/api/v1", api_key="server-only-test-key") as client:
            assert client.get_tools() == []
        assert route.calls.last.request.headers["X-API-Key"] == "server-only-test-key"
        assert "server-only-test-key" not in str(route.calls.last.request.url)


def test_deployed_client_refuses_to_send_key_to_changed_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ui_state

    monkeypatch.setattr(
        ui_state,
        "configuration",
        lambda name: {
            "APP_API_KEY": "server-only-test-key",
            "RESEARCH_API_URL": "https://backend.test/api/v1",
        }.get(name, ""),
    )
    # Exercise the client policy outside Streamlit's session-only cache wrapper.
    with pytest.raises(APIError, match="managed by the deployment"):
        ui_state.get_client.__wrapped__("https://untrusted.test/api/v1")


def test_every_frontend_file_stays_outside_the_core_and_has_no_fake_progress() -> None:
    root = Path(__file__).resolve().parents[2] / "frontend"
    forbidden = {"langgraph", "langchain_core", "langchain_groq", "research_agent", "sqlite3"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"progress", "run_research_sync", "sleep"}


def test_async_sources_report_resume_and_finite_events_use_versioned_contract() -> None:
    base = "http://api.test/api/v1"
    with respx.mock(base_url=base) as mock:
        mock.get("/health").respond(200, json={"status": "ok"})
        mock.get("/tools").respond(200, json=[{"name": "web_search"}])
        start = mock.post("/research").respond(202, json={"run_id": "run-1"})
        mock.get("/research").respond(200, json=[{"run_id": "run-1"}])
        mock.get("/research/run-1").respond(200, json={"status": "running"})
        mock.get("/research/run-1/report").respond(200, json={"report_json": {"title": "Report"}})
        mock.get("/research/run-1/sources").respond(200, json=[{"evidence_id": "e1"}])
        events = mock.get("/research/run-1/events").respond(
            200, text='id: 7\nevent: tool_started\ndata: {"data":{"tool":"web_search"}}\n\n'
        )
        resume = mock.post("/research/run-1/resume").respond(202, json={"run_id": "run-1"})
        with APIClient(base) as client:
            assert client.health_check()["status"] == "ok"
            assert client.get_tools()[0]["name"] == "web_search"
            assert (
                client.start_research("A query", "Instructions", options={"depth": "brief"})[
                    "run_id"
                ]
                == "run-1"
            )
            assert json.loads(start.calls[0].request.content)["options"]["depth"] == "brief"
            assert client.list_runs()[0]["run_id"] == "run-1"
            assert client.get_run("run-1")["status"] == "running"
            assert client.get_report("run-1")["report_json"]["title"] == "Report"
            assert client.get_sources("run-1")[0]["evidence_id"] == "e1"
            assert client.get_events("run-1", 6)[0]["event_id"] == 7
            assert dict(events.calls[0].request.url.params) == {"after": "6", "follow": "false"}
            client.resume_research("run-1", False)
            assert json.loads(resume.calls[0].request.content) == {"approved": False}


@pytest.mark.parametrize("status", [400, 404, 409, 422, 500, 503])
def test_http_errors_never_echo_provider_bodies(status: int) -> None:
    with respx.mock(base_url="http://api.test/api/v1") as mock:
        mock.get("/health").respond(
            status, text="dummy-private-provider-body", headers={"X-Request-ID": "request-1"}
        )
        with APIClient("http://api.test/api/v1") as client, pytest.raises(APIError) as error:
            client.health_check()
        assert error.value.status_code == status and error.value.request_id == "request-1"
        assert "dummy-private" not in str(error.value)


def test_network_and_malformed_responses_are_safe() -> None:
    with respx.mock(base_url="http://api.test/api/v1") as mock:
        route = mock.get("/health").mock(side_effect=httpx.ConnectError("dummy-private"))
        with APIClient("http://api.test/api/v1") as client:
            with pytest.raises(APIError, match="Cannot reach"):
                client.health_check()
            route.respond(200, text="not JSON")
            with pytest.raises(APIError, match="invalid response"):
                client.health_check()
            route.respond(200, json=[])
            with pytest.raises(APIError, match="invalid response"):
                client.health_check()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:password@host/api",
        "http://[invalid",
        "http://host:abc/api",
        "http://host/api?key=private",
        "http://host/api#fragment",
    ],
)
def test_invalid_api_urls_fail_locally(url: str) -> None:
    with pytest.raises(APIError):
        APIClient(url)


def test_sse_frames_multiline_comments_and_invalid_payloads() -> None:
    rows = list(
        parse_sse(
            [
                ": keepalive",
                "id: 12",
                "event: tool_started",
                'data: {"agent":"researcher",',
                'data: "data":{"tool":"web_search"}}',
                "",
            ]
        )
    )
    assert rows == [
        {
            "event_id": 12,
            "event_type": "tool_started",
            "data": {"agent": "researcher", "data": {"tool": "web_search"}},
        }
    ]
    for lines in (["data: broken", ""], ["id: -1"], ["data: []", ""]):
        with pytest.raises(APIError):
            list(parse_sse(lines))
    assert list(parse_sse(["data: {}"]))[0]["event_type"] == "message"


def test_event_projection_is_bounded_and_deduplicates_measured_usage() -> None:
    state = LiveState()
    events = [
        {
            "event_type": "llm_call",
            "event_id": index,
            "data": {"agent": "researcher", "data": {"total_tokens": 2, "duration_seconds": 0.5}},
        }
        for index in range(1, 251)
    ]
    state.ingest(events)
    state.ingest(events)
    assert len(state.events) == 200 and state.cursor == 250
    assert state.tokens == 500 and state.model_seconds == 125
