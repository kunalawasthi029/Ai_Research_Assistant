"""Execute reporting, revisions, CLI and replay persistence with real temporary SQLite files."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from research_agent import cli
from research_agent.config.settings import Settings
from research_agent.graph.checkpointing import create_checkpointer
from research_agent.graph.runtime import consume_graph
from research_agent.graph.state import create_initial_state
from research_agent.persistence.database import _SCHEMA_SQL, get_connection, init_db
from research_agent.persistence.models import RunRow
from research_agent.reporting.deterministic import validate_report
from research_agent.schemas.evidence import Evidence
from research_agent.service.research_service import ResearchService
from tests.fake_llm import FakeChatModel
from tests.workflow.test_lifecycle import make_graph
from tests.workflow.test_workflow import (
    _build_happy_path_llms,
    _make_report_json,
    _make_supervisor_decision_json,
    _make_validation_result_json,
)


def settings_for(tmp_path: Path, **options: Any) -> Settings:
    return Settings(
        groq_api_key="dummy-secret-not-persisted",
        database_path=str(tmp_path / "runs.sqlite"),
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        **options,
    )


def install_local_models(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, models: Any
) -> list[dict[str, Any]]:
    """Replace model/network execution only; retain real graph, service, events and SQLite."""
    snapshots = []

    def local_worker(
        service: ResearchService,
        run_id: str,
        thread_id: str,
        query: str,
        objective: str,
        use_checkpointer: bool,
        **options: Any,
    ) -> dict[str, Any]:
        with create_checkpointer(settings.checkpoint_db_path) as saver:
            graph = make_graph(models, settings=settings, checkpointer=saver)

            def publish(event: dict[str, Any]) -> None:
                service._handle_worker_event(run_id, event)
                if event["event_type"] == "state_snapshot":
                    snapshots.append(event["state"])
                    if event["state"].get("report"):
                        # A separate live reader sees the report before execution returns.
                        with sqlite3.connect(settings.database_path) as reader:
                            assert (
                                reader.execute(
                                    "SELECT COUNT(*) FROM reports WHERE run_id=?", (run_id,)
                                ).fetchone()[0]
                                > 0
                            )

            value = (
                Command(resume=options.get("approved"))
                if options.get("resume")
                else create_initial_state(run_id, thread_id, query, objective)
            )
            return consume_graph(graph, value, {"configurable": {"thread_id": thread_id}}, publish)

    monkeypatch.setattr(ResearchService, "_run_worker", local_worker)
    return snapshots


def test_every_revision_and_validation_layer_survives_restart_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for(tmp_path)
    models = _build_happy_path_llms()
    models["report"] = FakeChatModel(
        responses=[AIMessage(content=_make_report_json()) for _ in range(2)]
    )
    models["validator"] = FakeChatModel(
        responses=[
            AIMessage(content=_make_validation_result_json("needs_revision", 0.4)),
            AIMessage(content=_make_validation_result_json()),
        ]
    )
    models["supervisor"].responses.insert(
        -1,
        AIMessage(
            content=_make_supervisor_decision_json(
                "needs_revision", "report_generator", "final_review"
            )
        ),
    )
    snapshots = install_local_models(monkeypatch, settings, models)
    with ResearchService(settings) as service:
        result = service.run_research("What is X?")
        assert result["status"] == "completed"
        run_id = result["run_id"]
        for _ in range(2):
            service._persist_results(run_id, snapshots[-1])
        assert [row.version for row in service.get_reports(run_id)] == [1, 2]
        rows = service.get_validations(run_id)
        assert {(row.report_version, row.layer) for row in rows} == {
            (version, layer)
            for version in (1, 2)
            for layer in ("deterministic", "semantic", "combined")
        }
        assert len(rows) == 6
        assert (
            json.loads(
                next(
                    row.result_json
                    for row in rows
                    if row.report_version == 1 and row.layer == "combined"
                )
            )["status"]
            != "pass"
        )
        ev = service.get_evidence(run_id)[0]
        assert json.loads(ev.findings_json)[0]["supporting_evidence"] == [ev.id]
        assert json.loads(ev.source_json)["is_accessible"] is None
        assert "dummy-secret-not-persisted" not in service.get_run(run_id).config_json
        assert all(event.event_type != "state_snapshot" for event in service.get_events(run_id))
        assert "Revision Required" in str(models["report"].invocations[1])
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        service.conn.execute("SELECT 1")
    with ResearchService(settings) as reopened:
        assert [row.version for row in reopened.get_reports(run_id)] == [1, 2]
        assert len(reopened.get_validations(run_id)) == 6
        records = [
            Evidence(
                evidence_id=row.id,
                research_task_id=row.task_id,
                tool_name=row.tool_name,
                url=row.url,
                title=row.title,
                content=row.content,
                source_type=row.source_type,
                retrieved_at=row.retrieved_at,
                metadata=json.loads(row.metadata_json),
            )
            for row in reopened.get_evidence(run_id)
        ]
        report = json.loads(reopened.get_report(run_id).report_json)
        assert validate_report(report, evidence=records, plan_task_ids=["t1"]).status == "pass"
        report["references"][0]["url"] = "https://invented.example/changed"
        reopened.conn.execute(
            "UPDATE reports SET report_json=? WHERE run_id=? AND version=2",
            (json.dumps(report), run_id),
        )
        reopened.conn.commit()
        tampered = json.loads(reopened.get_report(run_id).report_json)
        assert validate_report(tampered, evidence=records, plan_task_ids=["t1"]).status != "pass"


def test_measured_events_are_durable_and_idempotent_before_terminal_state(tmp_path: Path) -> None:
    with ResearchService(settings_for(tmp_path)) as service:
        service._run_repo.create(RunRow(id="run", thread_id="thread", query="Query"))
        event = {"event_type": "llm_call", "event_id": "usage-1", "total_tokens": 23}
        service._handle_worker_event("run", event)
        service._handle_worker_event("run", event)
        service._handle_worker_event(
            "run", {"event_type": "tool_started", "event_id": "call-1", "tool": "web_search"}
        )
        assert service.get_run("run").total_tokens == 23
        assert service.get_run("run").total_tool_calls == 1
        service._persist_results("run", {"token_usage": [], "tool_calls": []})
        assert service.get_run("run").total_tokens == 23


def test_legacy_duplicates_are_archived_without_losing_artifacts(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "legacy.sqlite")
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO runs (id,thread_id,query,created_at,updated_at) VALUES ('run','thread','query','2024','2024')"
    )
    for index in (1, 2):
        conn.execute(
            "INSERT INTO reports VALUES (?, 'run', 1, ?, '', ?)",
            (f"report{index}", json.dumps({"old": index}), f"2024-01-0{index}"),
        )
        conn.execute(
            "INSERT INTO validations VALUES (?, 'run', 1, 'combined', ?, ?)",
            (f"val{index}", json.dumps({"old": index}), f"2024-01-0{index}"),
        )
    conn.commit()
    init_db(conn)
    init_db(conn)
    assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM validations").fetchone()[0] == 1
    archives = conn.execute(
        "SELECT data_json FROM events WHERE event_type='artifact_migrated'"
    ).fetchall()
    assert len(archives) == 2
    assert {json.loads(row[0])["original"]["id"] for row in archives} == {"report1", "val1"}
    assert json.loads(conn.execute("SELECT report_json FROM reports").fetchone()[0]) == {"old": 2}
    conn.close()


def test_workers_own_separate_connections_and_honor_configured_path(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)

    def worker(index: int) -> str:
        with ResearchService(settings) as service:
            service._run_repo.create(RunRow(id=str(index), thread_id=str(index), query="Query"))
            return service.get_run(str(index)).id

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(worker, (1, 2))) == {"1", "2"}
    with ResearchService(settings) as service:
        assert len(service.list_runs()) == 2


def test_retrieval_identity_is_immutable_and_borrowed_connection_remains_owned_by_caller(
    tmp_path: Path,
) -> None:
    conn = get_connection(tmp_path / "borrowed.sqlite")
    settings = settings_for(tmp_path)
    evidence = Evidence(
        evidence_id="ev",
        research_task_id="t1",
        tool_name="web_search",
        url="https://example.com/x",
        title="Observed source",
        content="Original text",
        source_type="web",
    )
    with ResearchService(settings, db_conn=conn) as service:
        service._run_repo.create(RunRow(id="run", thread_id="thread", query="Query"))
        service._persist_results("run", {"evidence": [evidence]})
        service._persist_results("run", {"evidence": [evidence]})
        assert len(service.get_evidence("run")) == 1
        with pytest.raises(ValueError, match="immutable"):
            service._persist_results(
                "run", {"evidence": [evidence.model_copy(update={"content": "Replacement"})]}
            )
        assert service.get_evidence("run")[0].content == "Original text"
    assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 1
    conn.close()


@pytest.mark.parametrize("output", ["json", "markdown"])
def test_cli_runs_real_service_and_persists_exportable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output: str,
) -> None:
    settings = settings_for(tmp_path)
    install_local_models(monkeypatch, settings, _build_happy_path_llms())
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["What is X?", "--output", output]) == 0
    captured = capsys.readouterr()
    assert "Research status: completed" in captured.err
    if output == "json":
        assert json.loads(captured.out)["references"][0]["evidence_id"] == "ev1"
    else:
        assert "[ref1]" in captured.out and "https://example.com/x" in captured.out
    with ResearchService(settings) as service:
        assert len(service.list_runs()) == 1
        assert len(service.get_validations(service.list_runs()[0].id)) == 3


def test_cli_partial_status_and_approval_resume_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = settings_for(tmp_path, human_approval_enabled=True)
    install_local_models(monkeypatch, settings, _build_happy_path_llms())
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["What is X?"]) == 2
    assert "awaiting_approval" in capsys.readouterr().err
    with ResearchService(settings) as service:
        run_id = service.list_runs()[0].id
    assert cli.main(["--resume", run_id, "--approve", "--output", "json"]) == 0
    assert "completed" in capsys.readouterr().err
    with ResearchService(settings) as service:
        assert len(service.list_runs()) == 1


def test_cli_failure_closes_connection_and_sanitizes_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = settings_for(tmp_path)
    services = []

    def fail(service: ResearchService, *args: Any, **kwargs: Any) -> Any:
        services.append(service)
        raise RuntimeError("private-provider-body")

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(ResearchService, "_run_worker", fail)
    assert cli.main(["What is X?"]) == 1
    captured = capsys.readouterr()
    assert "private-provider-body" not in captured.out + captured.err
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        services[0].conn.execute("SELECT 1")
    with ResearchService(settings) as reopened:
        assert reopened.list_runs()[0].status == "failed"


def test_cli_best_effort_is_not_announced_as_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = settings_for(tmp_path, max_validation_revisions=0)
    models = _build_happy_path_llms()
    models["validator"] = FakeChatModel(
        responses=[AIMessage(content=_make_validation_result_json("needs_revision", 0.4))]
    )
    models["supervisor"].responses[-1] = AIMessage(
        content=_make_supervisor_decision_json("needs_revision", "report_generator", "final_review")
    )
    install_local_models(monkeypatch, settings, models)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["What is X?"]) == 2
    captured = capsys.readouterr()
    assert "Research status: best_effort" in captured.err
    assert "Research completed" not in captured.err


@pytest.mark.parametrize(
    "arguments", [[], ["x"], ["Query", "--approve"], ["Query", "--resume", "run"]]
)
def test_cli_rejects_invalid_requests_before_execution(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
