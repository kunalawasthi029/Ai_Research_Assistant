"""SSE cursors, lifecycle statuses and sanitized handlers against actual SQLite data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request

from research_agent.api.errors import generic_exception_handler, value_error_handler
from research_agent.api.events import _format_sse, stream_run_events
from research_agent.config.settings import Settings
from research_agent.persistence.models import EventRow, RunRow
from research_agent.service.research_service import ResearchService


@pytest.fixture
def event_settings(tmp_path: Path) -> Settings:
    return Settings(groq_api_key="dummy", database_path=str(tmp_path / "events.sqlite"))


def test_format_sse_has_id_and_json() -> None:
    message = _format_sse("tool_started", {"tool": "web_search"}, event_id=23)
    assert message.startswith("id: 23\nevent: tool_started\n")
    assert json.loads(message.split("data: ")[1]) == {"tool": "web_search"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    ["completed", "failed", "best_effort", "incomplete", "interrupted", "awaiting_approval"],
)
async def test_stream_closes_for_each_inactive_status(
    event_settings: Settings, status: str
) -> None:
    with ResearchService(event_settings) as service:
        service._run_repo.create(RunRow(id="run", thread_id="thread", query="Query", status=status))
        service._event_repo.create(
            EventRow(
                id="event", run_id="run", event_type="decision", data_json='{"action":"accept"}'
            )
        )
    messages = [item async for item in stream_run_events("run", event_settings)]
    assert "connected" in messages[0] and "done" in messages[-1]
    assert sum("event: decision" in item for item in messages) == 1
    assert f'"status": "{status}"' in messages[-1]


@pytest.mark.asyncio
async def test_stream_drains_pages_and_resumes_without_replaying_history(
    event_settings: Settings,
) -> None:
    with ResearchService(event_settings) as service:
        service._run_repo.create(
            RunRow(id="run", thread_id="thread", query="Query", status="completed")
        )
        service.conn.executemany(
            "INSERT INTO events (id,run_id,event_type,created_at) VALUES (?, 'run', 'tool_started', '2024-01-01T00:00:00+00:00')",
            [(str(index),) for index in range(205)],
        )
        service.conn.commit()
    messages = [item async for item in stream_run_events("run", event_settings)]
    assert sum("event: tool_started" in item for item in messages) == 205
    tail = [item async for item in stream_run_events("run", event_settings, after=200)]
    assert sum("event: tool_started" in item for item in tail) == 5
    assert "id: 201" in tail[1]


@pytest.mark.asyncio
async def test_stream_missing_timeout_disconnect_and_storage_failure(
    event_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert any(
        "NotFound" in row
        for row in [item async for item in stream_run_events("missing", event_settings)]
    )
    with ResearchService(event_settings) as service:
        service._run_repo.create(
            RunRow(id="run", thread_id="thread", query="Query", status="running")
        )
    messages = [
        item
        async for item in stream_run_events("run", event_settings, timeout=0.03, poll_interval=0.01)
    ]
    assert "event: timeout" in messages[-1]

    async def disconnected() -> bool:
        return True

    assert (
        len(
            [
                item
                async for item in stream_run_events(
                    "run", event_settings, disconnected=disconnected
                )
            ]
        )
        == 1
    )

    def broken(*args: Any) -> Any:
        raise RuntimeError("dummy-private-provider-body")

    monkeypatch.setattr("research_agent.api.events.read_batch", broken)
    messages = [
        item async for item in stream_run_events("run", event_settings, request_id="read-error")
    ]
    assert "Unavailable" in messages[-1] and "read-error" in messages[-1]
    assert "dummy-private" not in "".join(messages)


@pytest.mark.asyncio
async def test_handlers_never_echo_exception_bodies() -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/test",
            "headers": [],
            "state": {"request_id": "request-1"},
        }
    )
    for handler, error, status in [
        (generic_exception_handler, RuntimeError("dummy-private"), 500),
        (value_error_handler, ValueError("dummy-private"), 400),
    ]:
        response = await handler(request, error)
        value = json.loads(response.body)
        assert response.status_code == value["status_code"] == status
        assert value["request_id"] == "request-1"
        assert "dummy-private" not in str(value)
