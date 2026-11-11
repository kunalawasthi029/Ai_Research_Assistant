"""Verify actual process termination and credential-free service recovery with SQLite state."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from research_agent.config.settings import Settings
from research_agent.execution.isolation import WorkerError, WorkerTimeoutError, run_isolated
from research_agent.service.research_service import ResearchService
from research_agent.tools.registry import create_default_registry
from research_agent.tools.results import ToolOutput
from tests.worker_fixtures import blocked, blocked_tool, private_error, root_with_child


def test_overall_timeout_stops_descendant_tool_process(tmp_path: Path) -> None:
    events = []
    with pytest.raises(WorkerTimeoutError):
        run_isolated(
            root_with_child, (str(tmp_path / "child.ready"),), timeout=6, on_event=events.append
        )
    assert events
    pid = events[0]["child_pid"]
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=True,
        )
        assert f'"{pid}"' not in result.stdout
    else:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False
        )
        assert not result.stdout.strip() or result.stdout.strip().startswith("Z")


def checkpoint_then_block(
    channel: Any,
    settings: Settings,
    run_id: str,
    thread_id: str,
    query: str,
    objective: str,
    use_checkpointer: bool,
    *options: Any,
) -> None:
    """Commit a real completed task, then simulate a blocked next model request."""
    from research_agent.graph.checkpointing import create_checkpointer
    from research_agent.graph.state import create_initial_state
    from tests.workflow.test_lifecycle import make_graph

    with create_checkpointer(settings.checkpoint_db_path) as saver:
        graph = make_graph(checkpointer=saver, settings=settings)
        graph.invoke(
            create_initial_state(run_id, thread_id, query, objective),
            {"configurable": {"thread_id": thread_id}},
            interrupt_after=["researcher"],
        )
    channel.put(("event", {"event_type": "checkpoint_test_ready"}))
    time.sleep(60)


def test_service_timeout_preserves_committed_evidence_and_exhausted_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research_agent.service.research_service.graph_worker", checkpoint_then_block
    )
    settings = Settings(
        groq_api_key="dummy",
        overall_run_timeout_seconds=15,
        database_path=str(tmp_path / "runs.sqlite"),
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    service = ResearchService(settings)
    try:
        result = service.run_research("What is X?")
        assert result["status"] == "interrupted"
        assert result["report"].references
        assert result["evidence_count"] == 1
        assert len(service.get_evidence(result["run_id"])) == 1
        assert service.inspect_checkpoint(result["run_id"])["completed_tasks"] == ["t1"]
        assert any(
            event.event_type == "checkpoint_test_ready"
            for event in service.get_events(result["run_id"])
        )
        with pytest.raises(ValueError, match="budget is exhausted"):
            service.resume_research(result["run_id"])
    finally:
        service.conn.close()


def test_blocked_worker_is_terminated_and_has_no_living_child() -> None:
    before = {child.pid for child in multiprocessing.active_children()}
    observed = []
    started = time.monotonic()
    with pytest.raises(WorkerTimeoutError):
        run_isolated(blocked, (), timeout=10, on_event=observed.append)
    assert observed == [{"ready": True}]
    assert time.monotonic() - started < 18
    assert {child.pid for child in multiprocessing.active_children()} <= before


def test_worker_errors_suppress_sensitive_exception_details() -> None:
    with pytest.raises(WorkerError) as error:
        run_isolated(private_error, (), timeout=10)
    assert "ValueError" in str(error.value)
    assert "dummy-private" not in str(error.value)


def test_tool_timeout_kills_real_worker_and_counts_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("research_agent.tools.registry._tool_worker", blocked_tool)
    settings = Settings(
        groq_api_key="dummy", per_tool_timeout_seconds=10, database_path=str(tmp_path / "marker")
    )
    registry = create_default_registry(settings=settings, isolate_calls=True)
    result = ToolOutput.model_validate_json(registry.execute("calculator", {"expression": "2+3"}))
    assert Path(settings.database_path + ".started").is_file()
    assert result.status == "error"
    assert result.error == "Tool wall-clock timeout"
    assert registry.get_call_count("calculator") == 1


def test_real_service_worker_recovers_invalid_input_without_llm_calls(tmp_path: Path) -> None:
    settings = Settings(
        groq_api_key="dummy",
        database_path=str(tmp_path / "runs.sqlite"),
        checkpoint_db_path=str(tmp_path / "cp.sqlite"),
        overall_run_timeout_seconds=15,
    )
    service = ResearchService(settings)
    try:
        result = service.run_research("")
        assert result["status"] == "incomplete"
        assert result["report"].references == []
        assert service.get_run(result["run_id"]).status == "incomplete"
        events = service.get_events(result["run_id"])
        assert any(event.event_type == "node_error" for event in events)
        assert not any(event.event_type in {"llm_call", "tool_started"} for event in events)
        state = service.inspect_checkpoint(result["run_id"])
        assert state["execution_status"] == "incomplete"
        assert service.get_report(result["run_id"]).version == 0
    finally:
        service.conn.close()
