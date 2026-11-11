"""Run trusted workers with a wall-clock deadline and terminate their owned process tree."""

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any


class WorkerTimeoutError(TimeoutError):
    """A worker exceeded its allotted wall-clock execution budget."""


class WorkerError(RuntimeError):
    """A trusted worker exited without a valid result; provider bodies are suppressed."""


class WorkerCancelledError(WorkerError):
    """The owning application requested cancellation and worker-tree cleanup."""


def _entry(
    target: Callable[..., Any], arguments: tuple[Any, ...], channel: Any, new_session: bool
) -> None:
    """Send results/errors through a local trusted queue, never via stdout."""
    if os.name != "nt" and new_session:
        os.setsid()  # type: ignore[attr-defined]
    try:
        result = target(channel, *arguments)
        channel.put(("result", result))
    except Exception as error:
        channel.put(("error", type(error).__name__))


def _stop(process: Any, new_session: bool) -> None:
    """Stop only this still-running owned worker and its descendants."""
    if not process.is_alive():
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
    elif new_session:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]
    else:
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def run_isolated(
    target: Callable[..., Any],
    arguments: tuple[Any, ...],
    timeout: float,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Any:
    """Enforce a process deadline including startup; nested workers share POSIX sessions."""
    if timeout <= 0:
        raise WorkerTimeoutError("Execution budget exhausted")
    if cancelled and cancelled():
        raise WorkerCancelledError("Execution was cancelled")
    context = multiprocessing.get_context("spawn")
    channel = context.Queue(maxsize=128)
    new_session = multiprocessing.current_process().name == "MainProcess"
    process = context.Process(target=_entry, args=(target, arguments, channel, new_session))
    deadline = time.monotonic() + timeout
    process.start()
    finished = False
    try:
        while True:
            if cancelled and cancelled():
                raise WorkerCancelledError("Execution was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeoutError("Worker exceeded its wall-clock budget")
            try:
                kind, value = channel.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if not process.is_alive():
                    raise WorkerError("Worker exited before returning a result") from None
                continue
            if kind == "event":
                if on_event:
                    on_event(value)
            elif kind == "result":
                finished = True
                return value
            else:
                raise WorkerError(f"Worker failed ({value})") from None
    finally:
        if finished:
            process.join(timeout=0.2)
        _stop(process, new_session)
        if not process.is_alive():
            process.close()
        channel.close()
        channel.cancel_join_thread()
