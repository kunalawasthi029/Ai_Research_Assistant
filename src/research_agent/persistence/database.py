"""SQLite database initialization and connection management."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data") / "research.sqlite"

# ──────────────────── Schema ────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    thread_id       TEXT NOT NULL,
    query           TEXT NOT NULL,
    objective       TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    config_json     TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    completed_at    TEXT,
    total_tool_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    elapsed_seconds REAL
);

CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    event_type      TEXT NOT NULL,
    agent           TEXT NOT NULL DEFAULT '',
    node            TEXT NOT NULL DEFAULT '',
    data_json       TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    task_id         TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    url             TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL DEFAULT '',
    source_type     TEXT NOT NULL DEFAULT 'web',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    retrieved_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    version         INTEGER NOT NULL DEFAULT 1,
    report_json     TEXT NOT NULL,
    report_markdown TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validations (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    report_version  INTEGER NOT NULL DEFAULT 1,
    layer           TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_evidence_run_id ON evidence(run_id);
CREATE INDEX IF NOT EXISTS idx_reports_run_id ON reports(run_id);
CREATE INDEX IF NOT EXISTS idx_validations_run_id ON validations(run_id);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the research SQLite database.

    Args:
        db_path: Path to the SQLite file. Defaults to data/research.sqlite.

    Returns:
        An open sqlite3 Connection with WAL mode and foreign keys enabled.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    deadline = time.monotonic() + 5
    try:
        while True:
            conn.execute(f"PRAGMA busy_timeout={max(1, int((deadline - time.monotonic()) * 1000))}")
            try:
                current_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                if current_mode not in {"wal", "memory"}:
                    conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", -1) & 255
                remaining = deadline - time.monotonic()
                if code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or remaining <= 0:
                    raise
                time.sleep(min(0.05, remaining))
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        conn.close()
        raise

    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    conn.executescript(_SCHEMA_SQL)
    if conn.execute("PRAGMA user_version").fetchone()[0] >= 1:
        return
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(evidence)")}
        for column, default in (("findings_json", "[]"), ("source_json", "{}")):
            if column not in columns:
                conn.execute(
                    f"ALTER TABLE evidence ADD COLUMN {column} TEXT NOT NULL DEFAULT '{default}'"
                )
        # Old releases could duplicate a report version or validation layer on resume.
        # Archive every displaced row in events before imposing unique version keys.
        for table, keys in (
            ("reports", ("run_id", "version")),
            ("validations", ("run_id", "report_version", "layer")),
        ):
            cursor = conn.execute(
                f"SELECT rowid AS migration_rowid, * FROM {table} ORDER BY created_at DESC, rowid DESC"
            )
            names = [column[0] for column in cursor.description]
            seen = set()
            for raw in cursor.fetchall():
                row = dict(zip(names, raw, strict=True))
                key = tuple(row[name] for name in keys)
                if key not in seen:
                    seen.add(key)
                    continue
                rowid = row.pop("migration_rowid")
                archive_id = str(
                    uuid5(NAMESPACE_URL, f"research-agent:migration:{table}:{row['id']}")
                )
                conn.execute(
                    "INSERT OR IGNORE INTO events (id, run_id, event_type, data_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        archive_id,
                        row["run_id"],
                        "artifact_migrated",
                        json.dumps({"table": table, "original": row}),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_version_key ON {table} ({', '.join(keys)})"
            )
        conn.execute("PRAGMA user_version=1")
    logger.info("Database schema initialized")
