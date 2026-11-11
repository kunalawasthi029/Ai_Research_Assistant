"""Emit observable graph events without exposing prompts or private model reasoning."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph.config import get_stream_writer

_sink: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "research_event_sink", default=None
)


@contextmanager
def event_scope(observer: Callable[[dict[str, Any]], None] | None = None) -> Iterator[None]:
    """Carry the parent's stream sink and metrics observer into nested tool graphs."""
    parent = _sink.get()
    if parent is None:
        try:
            parent = get_stream_writer()
        except (RuntimeError, KeyError):
            parent = None

    def publish(event: dict[str, Any]) -> None:
        if observer:
            observer(event)
        if parent:
            parent(event)

    token = _sink.set(publish)
    try:
        yield
    finally:
        _sink.reset(token)


def emit_event(event_type: str, **data: Any) -> None:
    """Write a custom stream event when executing inside a graph; otherwise do nothing."""
    writer = _sink.get()
    if writer is None:
        try:
            writer = get_stream_writer()
        except (RuntimeError, KeyError):
            return
    writer(
        {
            "event_type": event_type,
            "event_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            **data,
        }
    )
