"""Cursor-based SSE from persisted execution events, without shared SQLite connections."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from research_agent.config.settings import Settings
from research_agent.service.research_service import ResearchService

STOP_STATUSES = {
    "completed",
    "failed",
    "partial",
    "best_effort",
    "incomplete",
    "interrupted",
    "awaiting_approval",
}


def read_batch(settings: Settings, run_id: str, after: int) -> tuple[Any, list[Any]]:
    with ResearchService(settings) as service:
        return service.get_run(run_id), service._event_repo.page(run_id, after)


async def stream_run_events(
    run_id: str,
    settings: Settings,
    *,
    after: int = 0,
    poll_interval: float = 0.25,
    timeout: float | None = None,
    disconnected: Any = None,
    request_id: str = "",
    follow: bool = True,
) -> AsyncGenerator[str, None]:
    deadline = time.monotonic() + (
        timeout if timeout is not None else settings.api_event_stream_timeout_seconds
    )
    cursor = after
    yield _format_sse("connected", {"run_id": run_id, "after": cursor})
    while time.monotonic() < deadline:
        if disconnected and await disconnected():
            return
        try:
            run, events = await asyncio.to_thread(read_batch, settings, run_id, cursor)
        except Exception:
            yield _format_sse(
                "error",
                {
                    "error": "Unavailable",
                    "detail": "Research storage is unavailable.",
                    "status_code": 503,
                    "request_id": request_id,
                },
            )
            return
        if run is None:
            yield _format_sse(
                "error",
                {
                    "error": "NotFound",
                    "detail": "Run not found",
                    "status_code": 404,
                    "request_id": request_id,
                },
            )
            return
        for sequence, event in events:
            cursor = sequence
            yield _format_sse(
                event.event_type,
                {
                    "event_id": event.id,
                    "run_id": run_id,
                    "agent": event.agent,
                    "node": event.node,
                    "data": json.loads(event.data_json),
                    "timestamp": event.created_at,
                },
                event_id=cursor,
            )
        if not follow:
            yield _format_sse(
                "snapshot",
                {
                    "run_id": run_id,
                    "status": run.status,
                    "after": cursor,
                    "has_more": len(events) == 200,
                },
            )
            return
        if run.status in STOP_STATUSES and len(events) < 200:
            yield _format_sse(
                "done",
                {
                    "run_id": run_id,
                    "status": run.status,
                    "after": cursor,
                    "elapsed_seconds": run.elapsed_seconds,
                },
            )
            return
        if len(events) < 200:
            await asyncio.sleep(poll_interval)
    yield _format_sse("timeout", {"run_id": run_id, "after": cursor})


def _format_sse(event_type: str, data: dict[str, Any], event_id: int | None = None) -> str:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return prefix + f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"
