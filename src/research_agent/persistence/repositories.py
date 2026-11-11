"""CRUD repositories for each database table."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from research_agent.persistence.models import (
    EventRow,
    EvidenceRow,
    ReportRow,
    RunRow,
    ValidationRow,
)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# ──────────────────── Runs ────────────────────


class RunRepository:
    """CRUD for the runs table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, run: RunRow) -> None:
        """Insert a new run."""
        self.conn.execute(
            """INSERT INTO runs
               (id, thread_id, query, objective, status, config_json,
                created_at, updated_at, completed_at, total_tool_calls,
                total_tokens, elapsed_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.id,
                run.thread_id,
                run.query,
                run.objective,
                run.status,
                run.config_json,
                run.created_at,
                run.updated_at,
                run.completed_at,
                run.total_tool_calls,
                run.total_tokens,
                run.elapsed_seconds,
            ),
        )
        self.conn.commit()

    def get(self, run_id: str) -> RunRow | None:
        """Fetch a run by ID."""
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunRow(**dict(row))

    def update_status(
        self,
        run_id: str,
        status: str,
        completed_at: str | None = None,
        total_tool_calls: int | None = None,
        total_tokens: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Update a run's status and optional metrics."""
        updates = ["status = ?", "updated_at = ?"]
        params: list[object] = [status, _utcnow_iso()]

        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)
        if total_tool_calls is not None:
            updates.append("total_tool_calls = ?")
            params.append(total_tool_calls)
        if total_tokens is not None:
            updates.append("total_tokens = ?")
            params.append(total_tokens)
        if elapsed_seconds is not None:
            updates.append("elapsed_seconds = ?")
            params.append(elapsed_seconds)

        params.append(run_id)
        self.conn.execute(f"UPDATE runs SET {', '.join(updates)} WHERE id = ?", params)
        self.conn.commit()

    def list_all(self, limit: int = 50) -> list[RunRow]:
        """List runs ordered by creation date (newest first)."""
        rows = self.conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [RunRow(**dict(r)) for r in rows]

    def claim(self, run_id: str, expected: tuple[str, ...]) -> bool:
        """Atomically reserve an execution slice, including across separate connections."""
        placeholders = ",".join("?" for _ in expected)
        cursor = self.conn.execute(
            f"UPDATE runs SET status='running', completed_at=NULL, updated_at=? WHERE id=? AND status IN ({placeholders})",
            (_utcnow_iso(), run_id, *expected),
        )
        self.conn.commit()
        return cursor.rowcount == 1

    def active(self) -> list[RunRow]:
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE status IN ('queued', 'running')"
        ).fetchall()
        return [RunRow(**dict(row)) for row in rows]

    def update_config(self, run_id: str, config_json: str) -> None:
        self.conn.execute(
            "UPDATE runs SET config_json=?, updated_at=? WHERE id=?",
            (config_json, _utcnow_iso(), run_id),
        )
        self.conn.commit()


# ──────────────────── Events ────────────────────


class EventRepository:
    """CRUD for the events table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, event: EventRow) -> None:
        self.conn.execute(
            """INSERT INTO events
               (id, run_id, event_type, agent, node, data_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING""",
            (
                event.id,
                event.run_id,
                event.event_type,
                event.agent,
                event.node,
                event.data_json,
                event.created_at,
            ),
        )
        self.conn.commit()

    def list_for_run(self, run_id: str) -> list[EventRow]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [EventRow(**dict(r)) for r in rows]

    def page(self, run_id: str, after: int = 0, limit: int = 200) -> list[tuple[int, EventRow]]:
        """Use append-only SQLite row IDs as stable SSE cursors, independent of timestamps."""
        rows = self.conn.execute(
            "SELECT rowid AS cursor, * FROM events WHERE run_id=? AND rowid>? ORDER BY rowid LIMIT ?",
            (run_id, after, limit),
        ).fetchall()
        return [
            (row["cursor"], EventRow(**{key: row[key] for key in dict(row) if key != "cursor"}))
            for row in rows
        ]

    def latest(self, run_id: str, event_type: str) -> EventRow | None:
        row = self.conn.execute(
            "SELECT * FROM events WHERE run_id=? AND event_type=? ORDER BY rowid DESC LIMIT 1",
            (run_id, event_type),
        ).fetchone()
        return EventRow(**dict(row)) if row else None


# ──────────────────── Evidence ────────────────────


class EvidenceRepository:
    """CRUD for the evidence table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, ev: EvidenceRow) -> None:
        """Persist an immutable retrieval record and its latest analysis."""
        self.create_many([ev])

    def create_many(self, evidence: list[EvidenceRow]) -> None:
        """Batch insert evidence rows."""
        immutable = (
            "run_id",
            "task_id",
            "tool_name",
            "url",
            "title",
            "content",
            "source_type",
            "metadata_json",
            "retrieved_at",
        )
        seen: dict[str, tuple[object, ...]] = {}
        for ev in evidence:
            original = self.conn.execute("SELECT * FROM evidence WHERE id = ?", (ev.id,)).fetchone()
            current = tuple(getattr(ev, name) for name in immutable)
            previous = tuple(original[name] for name in immutable) if original else seen.get(ev.id)
            if previous is not None and previous != current:
                raise ValueError("Evidence identity conflicts with its immutable retrieval record")
            seen[ev.id] = current
        self.conn.executemany(
            """INSERT INTO evidence
               (id, run_id, task_id, tool_name, url, title, content,
                source_type, metadata_json, retrieved_at, findings_json, source_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 findings_json=excluded.findings_json, source_json=excluded.source_json
               WHERE evidence.run_id=excluded.run_id""",
            [
                (
                    ev.id,
                    ev.run_id,
                    ev.task_id,
                    ev.tool_name,
                    ev.url,
                    ev.title,
                    ev.content,
                    ev.source_type,
                    ev.metadata_json,
                    ev.retrieved_at,
                    ev.findings_json,
                    ev.source_json,
                )
                for ev in evidence
            ],
        )
        self.conn.commit()

    def list_for_run(self, run_id: str) -> list[EvidenceRow]:
        rows = self.conn.execute(
            "SELECT * FROM evidence WHERE run_id = ? ORDER BY retrieved_at",
            (run_id,),
        ).fetchall()
        return [EvidenceRow(**dict(r)) for r in rows]

    def count_for_run(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM evidence WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row["cnt"] if row else 0


# ──────────────────── Reports ────────────────────


class ReportRepository:
    """CRUD for the reports table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, report: ReportRow) -> None:
        self.conn.execute(
            """INSERT INTO reports
               (id, run_id, version, report_json, report_markdown, created_at)
               VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id, version) DO UPDATE SET
                 report_json=excluded.report_json, report_markdown=excluded.report_markdown""",
            (
                report.id,
                report.run_id,
                report.version,
                report.report_json,
                report.report_markdown,
                report.created_at,
            ),
        )
        self.conn.commit()

    def get_latest(self, run_id: str) -> ReportRow | None:
        row = self.conn.execute(
            "SELECT * FROM reports WHERE run_id = ? ORDER BY version DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return ReportRow(**dict(row))

    def list_for_run(self, run_id: str) -> list[ReportRow]:
        rows = self.conn.execute(
            "SELECT * FROM reports WHERE run_id = ? ORDER BY version",
            (run_id,),
        ).fetchall()
        return [ReportRow(**dict(r)) for r in rows]


# ──────────────────── Validations ────────────────────


class ValidationRepository:
    """CRUD for the validations table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, val: ValidationRow) -> None:
        self.conn.execute(
            """INSERT INTO validations
               (id, run_id, report_version, layer, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id, report_version, layer) DO UPDATE SET
                 result_json=excluded.result_json""",
            (
                val.id,
                val.run_id,
                val.report_version,
                val.layer,
                val.result_json,
                val.created_at,
            ),
        )
        self.conn.commit()

    def list_for_run(self, run_id: str) -> list[ValidationRow]:
        rows = self.conn.execute(
            "SELECT * FROM validations WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [ValidationRow(**dict(r)) for r in rows]
