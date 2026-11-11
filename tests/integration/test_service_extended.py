"""Tests for research_service: persist_results, render_markdown, and query methods."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from unittest.mock import MagicMock

from research_agent.config.settings import Settings
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
from research_agent.schemas.report import Reference
from research_agent.service.research_service import ResearchService


def _make_test_settings() -> Settings:
    return Settings(groq_api_key="test-key-not-used")


def _make_service() -> ResearchService:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return ResearchService(settings=_make_test_settings(), db_conn=conn)


# ──────────────────── Query Methods ────────────────────


class TestServiceQueryMethods:
    """Test get_run, list_runs, get_evidence, get_report, get_events, get_validations."""

    def test_get_run_returns_row(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-1", thread_id="t-1", query="test query"))

        result = service.get_run("run-1")
        assert result is not None
        assert result.id == "run-1"
        assert result.query == "test query"

    def test_list_runs_returns_rows(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-1", thread_id="t-1", query="q1"))
        run_repo.create(RunRow(id="run-2", thread_id="t-2", query="q2"))

        runs = service.list_runs()
        assert len(runs) == 2

    def test_list_runs_respects_limit(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        for i in range(5):
            run_repo.create(RunRow(id=f"run-{i}", thread_id=f"t-{i}", query=f"q{i}"))

        runs = service.list_runs(limit=3)
        assert len(runs) == 3

    def test_get_evidence_returns_list(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-ev", thread_id="t-ev", query="q"))

        ev_repo = EvidenceRepository(service.conn)
        ev_repo.create(
            EvidenceRow(
                id="ev-1",
                run_id="run-ev",
                task_id="task-1",
                tool_name="web_search",
                url="https://example.com",
                title="Example",
                source_type="web",
            )
        )

        evidence = service.get_evidence("run-ev")
        assert len(evidence) == 1
        assert evidence[0].id == "ev-1"

    def test_get_evidence_empty(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-no-ev", thread_id="t", query="q"))

        evidence = service.get_evidence("run-no-ev")
        assert evidence == []

    def test_get_report_returns_latest(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-rep", thread_id="t", query="q"))

        report_repo = ReportRepository(service.conn)
        report_repo.create(
            ReportRow(
                id="rep-1",
                run_id="run-rep",
                version=1,
                report_json='{"title": "Test Report"}',
                report_markdown="# Test Report",
            )
        )

        report = service.get_report("run-rep")
        assert report is not None
        assert report.version == 1

    def test_get_report_nonexistent(self) -> None:
        service = _make_service()
        assert service.get_report("no-such-run") is None

    def test_get_events_returns_list(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-evt", thread_id="t", query="q"))

        event_repo = EventRepository(service.conn)
        event_repo.create(
            EventRow(
                id="evt-1",
                run_id="run-evt",
                event_type="status_change",
                data_json='{"status": "running"}',
            )
        )

        events = service.get_events("run-evt")
        assert len(events) == 1
        assert events[0].event_type == "status_change"

    def test_get_validations_returns_list(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-val", thread_id="t", query="q"))

        val_repo = ValidationRepository(service.conn)
        val_repo.create(
            ValidationRow(
                id="val-1",
                run_id="run-val",
                report_version=1,
                layer="deterministic",
                result_json='{"status": "pass"}',
            )
        )

        validations = service.get_validations("run-val")
        assert len(validations) == 1
        assert validations[0].layer == "deterministic"


# ──────────────────── _log_event ────────────────────


class TestServiceLogEvent:
    """Test the internal _log_event method."""

    def test_log_event_creates_record(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-log", thread_id="t", query="q"))

        service._log_event(
            run_id="run-log",
            event_type="test_event",
            agent="planner",
            node="plan_node",
            data={"key": "value"},
        )

        events = service.get_events("run-log")
        assert len(events) == 1
        assert events[0].event_type == "test_event"
        assert events[0].agent == "planner"
        data = json.loads(events[0].data_json)
        assert data["key"] == "value"

    def test_log_event_default_empty_data(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-log2", thread_id="t", query="q"))

        service._log_event(run_id="run-log2", event_type="simple")

        events = service.get_events("run-log2")
        assert len(events) == 1
        data = json.loads(events[0].data_json)
        assert data == {}


# ──────────────────── _render_markdown ────────────────────


class TestServiceRenderMarkdown:
    """Test markdown rendering from a Report schema."""

    def _make_mock_report(
        self,
        *,
        comparative_insights: list[str] | None = None,
        recommendations: list[str] | None = None,
    ) -> MagicMock:
        report = MagicMock()
        report.title = "Test Research Report"
        report.executive_summary = "This is the executive summary."
        report.research_objective = "Investigate testing patterns."
        report.methodology = "Literature review and testing."
        report.key_findings = ["Finding A", "Finding B"]

        section = MagicMock()
        section.heading = "Analysis Section"
        section.content = "Detailed analysis content."
        report.detailed_analysis = [section]

        report.comparative_insights = comparative_insights or []
        report.recommendations = recommendations or []
        report.limitations = ["Small sample size"]
        report.conclusion = "Testing is important."

        ref = Reference(
            ref_id="ref1",
            title="Source 1",
            url="https://example.com",
            evidence_id="ev-001",
            supporting_evidence_ids=["ev-002"],
            source_type="web",
            accessed_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        report.references = [ref]

        return report

    def test_renders_all_sections(self) -> None:
        service = _make_service()
        report = self._make_mock_report()

        md = service._render_markdown(report)

        assert "# Test Research Report" in md
        assert "## Executive Summary" in md
        assert "## Research Objective" in md
        assert "## Methodology" in md
        assert "## Key Findings" in md
        assert "1. Finding A" in md
        assert "2. Finding B" in md
        assert "## Detailed Analysis" in md
        assert "### Analysis Section" in md
        assert "## Limitations" in md
        assert "## Conclusion" in md
        assert "## References" in md
        assert "[Source 1](https://example.com)" in md
        assert "ev-001" in md
        assert "ev-002" in md

    def test_renders_comparative_insights(self) -> None:
        service = _make_service()
        report = self._make_mock_report(comparative_insights=["Insight 1", "Insight 2"])

        md = service._render_markdown(report)

        assert "## Comparative Insights" in md
        assert "- Insight 1" in md
        assert "- Insight 2" in md

    def test_renders_recommendations(self) -> None:
        service = _make_service()
        report = self._make_mock_report(recommendations=["Rec A", "Rec B"])

        md = service._render_markdown(report)

        assert "## Recommendations" in md
        assert "- Rec A" in md

    def test_omits_empty_optional_sections(self) -> None:
        service = _make_service()
        report = self._make_mock_report()

        md = service._render_markdown(report)

        assert "## Comparative Insights" not in md
        assert "## Recommendations" not in md


# ──────────────────── _persist_results ────────────────────


class TestServicePersistResults:
    """Test the _persist_results method."""

    def test_persist_evidence(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-persist", thread_id="t", query="q"))

        mock_ev = MagicMock()
        mock_ev.evidence_id = "ev-persist-1"
        mock_ev.research_task_id = "task-1"
        mock_ev.tool_name = "web_search"
        mock_ev.url = "https://example.com"
        mock_ev.title = "Example Page"
        mock_ev.content = "Page content"
        mock_ev.source_type = "web"
        mock_ev.metadata = {"key": "val"}
        mock_ev.retrieved_at = datetime(2024, 1, 1, tzinfo=UTC)

        state = {"evidence": [mock_ev], "tool_calls": []}

        service._persist_results("run-persist", state)

        evidence = service.get_evidence("run-persist")
        assert len(evidence) == 1
        assert evidence[0].id == "ev-persist-1"
        assert evidence[0].tool_name == "web_search"

    def test_persist_report(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-rep", thread_id="t", query="q"))

        mock_report = MagicMock()
        mock_report.version = 1
        mock_report.model_dump_json.return_value = '{"title": "Test"}'
        mock_report.title = "Test Report"
        mock_report.executive_summary = "Summary"
        mock_report.research_objective = "Objective"
        mock_report.methodology = "Method"
        mock_report.key_findings = ["Finding"]
        mock_report.detailed_analysis = []
        mock_report.comparative_insights = []
        mock_report.recommendations = []
        mock_report.limitations = ["Limit"]
        mock_report.conclusion = "Done"
        mock_report.references = []

        state = {"report": mock_report, "evidence": [], "tool_calls": []}

        service._persist_results("run-rep", state)

        report = service.get_report("run-rep")
        assert report is not None
        assert report.version == 1

    def test_persist_validation(self) -> None:
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-val", thread_id="t", query="q"))

        mock_report = MagicMock()
        mock_report.version = 2
        mock_report.model_dump_json.return_value = '{"title": "T"}'
        mock_report.title = "T"
        mock_report.executive_summary = "S"
        mock_report.research_objective = "O"
        mock_report.methodology = "M"
        mock_report.key_findings = []
        mock_report.detailed_analysis = []
        mock_report.comparative_insights = []
        mock_report.recommendations = []
        mock_report.limitations = []
        mock_report.conclusion = "C"
        mock_report.references = []

        mock_vr = MagicMock()
        mock_vr.layer = "deterministic"
        mock_vr.model_dump_json.return_value = '{"status": "pass"}'

        state = {
            "report": mock_report,
            "validation_result": mock_vr,
            "evidence": [],
            "tool_calls": [],
        }

        service._persist_results("run-val", state)

        validations = service.get_validations("run-val")
        assert len(validations) == 1
        assert validations[0].layer == "deterministic"

    def test_persist_empty_state(self) -> None:
        """Persisting an empty state should not crash."""
        service = _make_service()
        run_repo = RunRepository(service.conn)
        run_repo.create(RunRow(id="run-empty", thread_id="t", query="q"))

        service._persist_results("run-empty", {})

        assert service.get_evidence("run-empty") == []
        assert service.get_report("run-empty") is None
