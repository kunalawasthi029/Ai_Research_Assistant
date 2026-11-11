"""Tests for evaluation metrics."""

from __future__ import annotations

from evals.metrics import (
    citation_validity,
    compute_all_metrics,
    report_completeness,
    source_diversity,
    topic_coverage,
)


class TestTopicCoverage:
    def test_all_topics_found(self) -> None:
        text = "This report covers surface codes and logical qubits."
        assert topic_coverage(text, ["surface codes", "logical qubits"]) == 1.0

    def test_no_topics_found(self) -> None:
        text = "This is about cooking."
        assert topic_coverage(text, ["quantum", "physics"]) == 0.0

    def test_partial_coverage(self) -> None:
        text = "Quantum computing uses logical qubits."
        score = topic_coverage(text, ["logical qubits", "surface codes", "error rates"])
        assert abs(score - 1 / 3) < 0.01

    def test_empty_expected(self) -> None:
        assert topic_coverage("any text", []) == 1.0

    def test_case_insensitive(self) -> None:
        text = "SURFACE CODES are important."
        assert topic_coverage(text, ["surface codes"]) == 1.0


class TestSourceDiversity:
    def test_single_type(self) -> None:
        evidence = [{"source_type": "web"}, {"source_type": "web"}]
        assert source_diversity(evidence) == 1

    def test_multiple_types(self) -> None:
        evidence = [
            {"source_type": "web"},
            {"source_type": "arxiv"},
            {"source_type": "wikipedia"},
        ]
        assert source_diversity(evidence) == 3

    def test_empty(self) -> None:
        assert source_diversity([]) == 0


class TestCitationValidity:
    def test_all_valid(self) -> None:
        report = {
            "detailed_analysis": [{"evidence_ids": ["ev1", "ev2"]}],
            "references": [{"evidence_id": "ev1"}],
        }
        assert citation_validity(report, {"ev1", "ev2"}) == 1.0

    def test_some_broken(self) -> None:
        report = {
            "detailed_analysis": [{"evidence_ids": ["ev1", "ev99"]}],
            "references": [],
        }
        assert citation_validity(report, {"ev1"}) == 0.5

    def test_no_citations(self) -> None:
        report = {"detailed_analysis": [], "references": []}
        assert citation_validity(report, {"ev1"}) == 0.0


class TestReportCompleteness:
    def test_complete_report(self) -> None:
        report = {
            "title": "Test",
            "executive_summary": "Summary",
            "research_objective": "Objective",
            "methodology": "Methods",
            "key_findings": ["finding"],
            "detailed_analysis": [{"heading": "h", "content": "c"}],
            "limitations": ["lim"],
            "conclusion": "Conclusion",
            "references": [{"ref_id": "r1"}],
        }
        assert report_completeness(report) == 1.0

    def test_missing_sections(self) -> None:
        report = {"title": "Test", "executive_summary": "Summary"}
        score = report_completeness(report)
        assert score < 1.0

    def test_empty_report(self) -> None:
        assert report_completeness({}) == 0.0


class TestComputeAllMetrics:
    def test_returns_all_metrics(self) -> None:
        report = {
            "title": "Test",
            "executive_summary": "Summary about quantum",
            "research_objective": "Objective",
            "methodology": "Methods",
            "key_findings": ["finding"],
            "detailed_analysis": [{"heading": "h", "content": "c", "evidence_ids": ["ev1"]}],
            "limitations": ["lim"],
            "conclusion": "Conclusion",
            "references": [{"evidence_id": "ev1"}],
        }
        metrics = compute_all_metrics(
            report=report,
            report_text="This report covers quantum computing and surface codes.",
            expected_topics=["quantum", "surface codes"],
            evidence=[{"source_type": "web"}],
            evidence_ids={"ev1"},
        )
        assert "topic_coverage" in metrics
        assert "evidence_count" in metrics
        assert "source_diversity" in metrics
        assert "citation_validity" in metrics
        assert "report_completeness" in metrics
        assert metrics["topic_coverage"] == 1.0
        assert metrics["evidence_count"] == 1.0
        assert metrics["citation_id_resolution"] == 1.0
        assert metrics["citation_validity"] is None  # No network observation was made here.
