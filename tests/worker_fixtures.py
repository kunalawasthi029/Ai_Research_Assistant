"""Lightweight spawn targets: do not import the application at process startup."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from typing import Any


def blocked(channel: Any) -> None:
    """A blocked native-call stand-in that must be killed, not merely abandoned in a thread."""
    channel.put(("event", {"ready": True}))
    time.sleep(60)


def blocked_graph(channel: Any, *arguments: Any) -> None:
    """An actual API graph worker that stays blocked until its owner cancels it."""
    channel.put(("event", {"event_type": "worker_ready"}))
    time.sleep(60)


def private_error(channel: Any) -> None:
    """Never send exception bodies back across the application boundary."""
    raise ValueError("dummy-private-provider-body")


def nested_block(marker: str) -> None:
    """A descendant simulating a tool running inside an overall research worker."""
    Path(marker).write_text("running")
    time.sleep(60)


def root_with_child(channel: Any, marker: str) -> None:
    """Start a real descendant and publish readiness without abandoning it deliberately."""
    child = multiprocessing.get_context("spawn").Process(target=nested_block, args=(marker,))
    child.start()
    while not Path(marker).is_file():
        time.sleep(0.01)
    channel.put(("event", {"child_pid": child.pid}))
    child.join()


def blocked_tool(channel: Any, name: str, arguments: dict[str, Any], settings: Any) -> None:
    """Mark that execution began and simulate a blocked provider/DNS call."""
    Path(settings.database_path + ".started").write_text("started")
    time.sleep(60)
