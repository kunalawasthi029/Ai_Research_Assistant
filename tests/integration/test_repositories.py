"""Integration tests for persistence repositories using a real in-memory SQLite DB."""

from __future__ import annotations

import sqlite3

import pytest

from research_agent.persistence.database import init_db
from research_agent.persistence.models import (
    EventRow,
    EvidenceRow,
    ReportRow,
    RunRow,
    ValidationRow,
)
from research_agent.persistence.repositories import (
    EventRepository,
    EvidenceRepository,
    ReportRepository,
    RunRepository,
    ValidationRepository,
)


@pytest.fixture
def db_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


# ──────────────────── RunRepository ────────────────────


class TestRunRepository:
    def test_create_and_get(self, db_conn: sqlite3.Connection) -> None:
        repo = RunRepository(db_conn)
        run = RunRow(id="run1", thread_id="t1", query="test query")
        repo.create(run)

        fetched = repo.get("run1")
        assert fetched is not None
        assert fetched.id == "run1"
        assert fetched.query == "test query"
        assert fetched.status == "pending"

    def test_get_nonexistent(self, db_conn: sqlite3.Connection) -> None:
        repo = RunRepository(db_conn)
        assert repo.get("no-such-id") is None

    def test_update_status(self, db_conn: sqlite3.Connection) -> None:
        repo = RunRepository(db_conn)
        repo.create(RunRow(id="run2", thread_id="t2", query="q"))

        repo.update_status(
            "run2",
            status="completed",
            completed_at="2024-01-01T00:00:00Z",
            total_tool_calls=5,
            elapsed_seconds=12.3,
        )

        fetched = repo.get("run2")
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.completed_at == "2024-01-01T00:00:00Z"
        assert fetched.total_tool_calls == 5
        assert fetched.elapsed_seconds == 12.3

    def test_list_all(self, db_conn: sqlite3.Connection) -> None:
        repo = RunRepository(db_conn)
        for i in range(3):
            repo.create(
                RunRow(
                    id=f"run{i}",
                    thread_id=f"t{i}",
                    query=f"q{i}",
                    created_at=f"2024-01-0{i + 1}T00:00:00Z",
                    updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                )
            )

        runs = repo.list_all()
        assert len(runs) == 3
        # Newest first
        assert runs[0].id == "run2"

    def test_list_all_with_limit(self, db_conn: sqlite3.Connection) -> None:
        repo = RunRepository(db_conn)
        for i in range(5):
            repo.create(RunRow(id=f"r{i}", thread_id=f"t{i}", query=f"q{i}"))
        assert len(repo.list_all(limit=2)) == 2


# ──────────────────── EventRepository ────────────────────


class TestEventRepository:
    def test_create_and_list(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = EventRepository(db_conn)

        repo.create(EventRow(id="e1", run_id="run1", event_type="status_change"))
        repo.create(EventRow(id="e2", run_id="run1", event_type="tool_call", agent="researcher"))

        events = repo.list_for_run("run1")
        assert len(events) == 2
        assert events[0].event_type == "status_change"

    def test_empty_run(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = EventRepository(db_conn)
        assert repo.list_for_run("run1") == []


# ──────────────────── EvidenceRepository ────────────────────


class TestEvidenceRepository:
    def test_create_and_list(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = EvidenceRepository(db_conn)

        repo.create(
            EvidenceRow(
                id="ev1",
                run_id="run1",
                task_id="t1",
                tool_name="web_search",
                url="https://example.com",
                title="Test",
                content="Content",
            )
        )

        evidence = repo.list_for_run("run1")
        assert len(evidence) == 1
        assert evidence[0].url == "https://example.com"

    def test_create_many(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = EvidenceRepository(db_conn)

        rows = [
            EvidenceRow(id=f"ev{i}", run_id="run1", task_id="t1", tool_name="web_search")
            for i in range(5)
        ]
        repo.create_many(rows)
        assert repo.count_for_run("run1") == 5

    def test_count_for_run(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = EvidenceRepository(db_conn)
        assert repo.count_for_run("run1") == 0


# ──────────────────── ReportRepository ────────────────────


class TestReportRepository:
    def test_create_and_get_latest(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = ReportRepository(db_conn)

        repo.create(ReportRow(id="rpt1", run_id="run1", version=1, report_json='{"v":1}'))
        repo.create(ReportRow(id="rpt2", run_id="run1", version=2, report_json='{"v":2}'))

        latest = repo.get_latest("run1")
        assert latest is not None
        assert latest.version == 2

    def test_list_for_run(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = ReportRepository(db_conn)
        repo.create(ReportRow(id="rpt1", run_id="run1", version=1, report_json="{}"))
        repo.create(ReportRow(id="rpt2", run_id="run1", version=2, report_json="{}"))

        reports = repo.list_for_run("run1")
        assert len(reports) == 2
        assert reports[0].version == 1  # ordered ascending

    def test_no_report(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = ReportRepository(db_conn)
        assert repo.get_latest("run1") is None


# ──────────────────── ValidationRepository ────────────────────


class TestValidationRepository:
    def test_create_and_list(self, db_conn: sqlite3.Connection) -> None:
        RunRepository(db_conn).create(RunRow(id="run1", thread_id="t1", query="q"))
        repo = ValidationRepository(db_conn)

        repo.create(
            ValidationRow(
                id="v1",
                run_id="run1",
                report_version=1,
                layer="deterministic",
                result_json='{"status":"pass"}',
            )
        )
        repo.create(
            ValidationRow(
                id="v2",
                run_id="run1",
                report_version=1,
                layer="semantic",
                result_json='{"status":"pass"}',
            )
        )

        vals = repo.list_for_run("run1")
        assert len(vals) == 2


# ──────────────────── Database Init ────────────────────


class TestDatabaseInit:
    def test_init_db_creates_tables(self, db_conn: sqlite3.Connection) -> None:
        """All 5 tables should exist after init."""
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "runs" in table_names
        assert "events" in table_names
        assert "evidence" in table_names
        assert "reports" in table_names
        assert "validations" in table_names

    def test_init_db_idempotent(self, db_conn: sqlite3.Connection) -> None:
        """Calling init_db twice should not error."""
        init_db(db_conn)  # Second call
        # If it didn't raise, it's idempotent
