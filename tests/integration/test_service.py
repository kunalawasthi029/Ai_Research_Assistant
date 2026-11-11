"""Integration tests for ResearchService and CLI."""

from __future__ import annotations

import sqlite3

import pytest

from research_agent.config.settings import Settings
from research_agent.service.research_service import ResearchService


def _make_test_settings() -> Settings:
    """Create a Settings object with a dummy API key for DB-only tests."""
    return Settings(groq_api_key="test-key-not-used")


class TestResearchServiceInit:
    """Test service initialization without running a full research."""

    def test_service_creates_tables(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _ = ResearchService(settings=_make_test_settings(), db_conn=conn)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}

        assert "runs" in table_names
        assert "evidence" in table_names
        assert "reports" in table_names

    def test_list_runs_empty(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        service = ResearchService(settings=_make_test_settings(), db_conn=conn)
        assert service.list_runs() == []

    def test_get_run_nonexistent(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        service = ResearchService(settings=_make_test_settings(), db_conn=conn)
        assert service.get_run("no-such-id") is None


class TestCLIParsing:
    """Test CLI argument parsing (no actual research execution)."""

    def test_cli_module_imports(self) -> None:
        """The CLI module should import without errors."""
        from research_agent.cli import main

        assert callable(main)

    def test_cli_no_query_exits(self) -> None:
        """CLI should error when no query is provided."""
        from research_agent.cli import main

        with pytest.raises(SystemExit):
            main([])
