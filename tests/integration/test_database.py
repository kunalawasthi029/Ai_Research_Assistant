"""Tests for database.py: connection management, WAL mode, init_db."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from research_agent.persistence.database import get_connection, init_db


class TestGetConnection:
    """Test get_connection creates a properly configured SQLite connection."""

    def test_creates_db_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "test.sqlite"
            conn = get_connection(db_path)

            assert db_path.exists()
            conn.close()

    def test_wal_mode_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            conn = get_connection(db_path)

            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert journal_mode == "wal"
            conn.close()

    def test_foreign_keys_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            conn = get_connection(db_path)

            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk == 1
            conn.close()

    def test_row_factory_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            conn = get_connection(db_path)

            assert conn.row_factory is sqlite3.Row
            conn.close()

    def test_string_path(self) -> None:
        """get_connection should accept a string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "str_test.sqlite")
            conn = get_connection(db_path)
            assert Path(db_path).exists()
            conn.close()


class TestInitDb:
    """Test init_db creates all expected tables."""

    def test_creates_all_tables(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}

        assert "runs" in table_names
        assert "events" in table_names
        assert "evidence" in table_names
        assert "reports" in table_names
        assert "validations" in table_names

    def test_creates_indexes(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)

        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        index_names = {r["name"] for r in indexes}

        assert "idx_events_run_id" in index_names
        assert "idx_evidence_run_id" in index_names
        assert "idx_reports_run_id" in index_names
        assert "idx_validations_run_id" in index_names

    def test_idempotent(self) -> None:
        """Running init_db twice should not raise."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        init_db(conn)  # Should not raise
