"""Unit tests for the Quality Validator agent (Layers A, B, and combined)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from langchain_core.messages import AIMessage

from research_agent.agents.validator import (
    _report_to_text,
    combine_validation_results,
    run_deterministic_validator,
    run_semantic_validator,
)
from research_agent.schemas.evidence import Evidence
from research_agent.schemas.report import Reference, Report, ReportSection
from research_agent.schemas.validation import (
    ValidationResult,
    ValidationScore,
)
from tests.fake_llm import FakeChatModel

# ──────────────────────── Fixtures ────────────────────────


def _make_evidence(evidence_id: str = "ev1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        research_task_id="t1",
        tool_name="web_search",
        url=f"https://example.com/{evidence_id}",
        title=f"Source {evidence_id}",
        content="Evidence content for testing.",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
        source_type="web",
    )


def _make_report(
    evidence_ids: list[str] | None = None,
    content_length: int = 300,
    ref_evidence_ids: list[str] | None = None,
    missing_executive_summary: bool = False,
) -> Report:
    """Build a Report with controllable properties for testing."""
    if evidence_ids is None:
        evidence_ids = ["ev1"]
    if ref_evidence_ids is None:
        ref_evidence_ids = evidence_ids

    now = datetime(2024, 1, 1, tzinfo=UTC)
    return Report(
        title="Test Report",
        executive_summary="" if missing_executive_summary else "Summary of key findings.",
        research_objective="Test research objective",
        methodology="Systematic analysis of sources.",
        key_findings=["Finding 1 about the topic", "Finding 2 about the topic"],
        detailed_analysis=[
            ReportSection(
                heading="Analysis Section",
                content="x" * content_length
                + " "
                + " ".join(f"[ref_{eid}]" for eid in evidence_ids),
                evidence_ids=evidence_ids,
            ),
        ],
        limitations=["Limited scope"],
        conclusion="The research shows promising results.",
        references=[
            Reference(
                ref_id=f"ref_{eid}",
                evidence_id=eid,
                url=f"https://example.com/{eid}",
                title=f"Source {eid}",
                source_type="web",
                accessed_at=now,
            )
            for eid in ref_evidence_ids
        ],
    )


def _make_semantic_result_json(
    status: str = "pass",
    overall_score: float = 0.85,
) -> str:
    """Build a valid ValidationResult JSON for the semantic validator."""
    return json.dumps(
        {
            "status": status,
            "layer": "semantic",
            "scores": [
                {
                    "criterion": criterion,
                    "score": overall_score,
                    "passed": overall_score >= 0.6,
                    "details": "Evidence assessment",
                }
                for criterion in (
                    "relevance",
                    "evidence_alignment",
                    "completeness",
                    "logical_consistency",
                    "objectivity",
                )
            ],
            "issues": [],
            "feedback": "Report is well-structured." if status == "pass" else "Needs improvement.",
            "overall_score": overall_score,
        }
    )


# ──────────────── Layer A: Deterministic ────────────────


class TestDeterministicValidator:
    """Tests for run_deterministic_validator (pure Python, no LLM)."""

    def test_clean_report_passes(self) -> None:
        report = _make_report(evidence_ids=["ev1"], content_length=300)
        result = run_deterministic_validator(
            report=report,
            evidence_ids=["ev1"],
            plan_task_ids=["t1"],
            evidence=[_make_evidence()],
        )
        assert isinstance(result, ValidationResult)
        assert result.status == "pass"
        assert result.layer == "deterministic"
        assert len(result.broken_citations) == 0
        assert len(result.missing_sections) == 0

    def test_missing_section_flagged(self) -> None:
        report = _make_report(missing_executive_summary=True)
        result = run_deterministic_validator(
            report=report,
            evidence_ids=["ev1"],
            plan_task_ids=["t1"],
            evidence=[_make_evidence()],
        )
        assert result.status == "needs_revision"
        assert "executive_summary" in result.missing_sections
        assert any(i.issue_type == "missing_section" for i in result.issues)

    def test_broken_citation_detected(self) -> None:
        """When a section cites ev99 but evidence store only has ev1."""
        report = _make_report(evidence_ids=["ev99"], ref_evidence_ids=["ev99"])
        result = run_deterministic_validator(
            report=report,
            evidence_ids=["ev1"],  # ev99 not in evidence store
            plan_task_ids=["t1"],
            evidence=[_make_evidence()],
        )
        assert "ev99" in result.broken_citations
        assert any(i.issue_type == "broken_citation" for i in result.issues)

    def test_short_content_flagged(self) -> None:
        report = _make_report(content_length=50)  # Below 200 minimum
        result = run_deterministic_validator(
            report=report,
            evidence_ids=["ev1"],
            plan_task_ids=["t1"],
            evidence=[_make_evidence()],
        )
        assert any(s.criterion == "minimum_content" and not s.passed for s in result.scores)
        assert any(i.issue_type == "empty_content" for i in result.issues)

    def test_duplicate_references_detected(self) -> None:
        """Two references with the same URL are flagged."""
        report = _make_report(ref_evidence_ids=["ev1", "ev1"])
        result = run_deterministic_validator(
            report=report,
            evidence_ids=["ev1"],
            plan_task_ids=["t1"],
            evidence=[_make_evidence()],
        )
        dup_score = next(s for s in result.scores if s.criterion == "no_duplicate_refs")
        assert not dup_score.passed

    def test_all_criteria_scored(self) -> None:
        report = _make_report()
        result = run_deterministic_validator(
            report=report,
            evidence_ids=["ev1"],
            plan_task_ids=["t1"],
            evidence=[_make_evidence()],
        )
        criteria = {s.criterion for s in result.scores}
        assert "section_completeness" in criteria
        assert "citation_traceability" in criteria
        assert "no_duplicate_refs" in criteria
        assert "minimum_content" in criteria

    def test_needs_research_when_many_broken(self) -> None:
        """When >30% citations are broken, status should be needs_research."""
        report = _make_report(
            evidence_ids=["ev1", "ev2", "ev3", "ev4"],
            ref_evidence_ids=["ev1", "ev2", "ev3", "ev4"],
        )
        result = run_deterministic_validator(
            report=report,
            evidence_ids=["ev1"],  # only ev1 exists, 3 out of 4 broken
            plan_task_ids=["t1"],
            evidence=[_make_evidence()],
        )
        assert result.status == "needs_research"


# ──────────────── Layer B: Semantic ────────────────


class TestSemanticValidator:
    """Tests for run_semantic_validator (uses FakeChatModel)."""

    def test_returns_valid_result(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_semantic_result_json())])
        report = _make_report()
        result = run_semantic_validator(
            report=report,
            evidence=[_make_evidence()],
            research_objective="Test objective",
            llm=llm,
        )
        assert isinstance(result, ValidationResult)
        assert result.layer == "semantic"
        assert result.overall_score == 0.85

    def test_prompt_includes_report_and_evidence(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_semantic_result_json())])
        report = _make_report()
        run_semantic_validator(
            report=report,
            evidence=[_make_evidence("ev1")],
            research_objective="Impact of AI",
            llm=llm,
        )
        user_msg = llm.invocations[0][-1].content
        assert "Impact of AI" in user_msg
        assert "ev1" in user_msg
        assert "Test Report" in user_msg  # report title

    def test_low_score_result(self) -> None:
        llm = FakeChatModel(
            responses=[
                AIMessage(
                    content=_make_semantic_result_json(status="needs_revision", overall_score=0.3)
                )
            ]
        )
        result = run_semantic_validator(
            report=_make_report(),
            evidence=[_make_evidence()],
            research_objective="Test",
            llm=llm,
        )
        assert result.status == "needs_revision"
        assert result.overall_score == 0.3


# ──────────────── Combine ────────────────


class TestCombineValidationResults:
    """Tests for combine_validation_results."""

    def _make_det_result(self, status: str = "pass", score: float = 1.0) -> ValidationResult:
        return ValidationResult(
            status=status,
            layer="deterministic",
            scores=[ValidationScore(criterion="test", score=score, passed=status == "pass")],
            issues=[],
            feedback="All good." if status == "pass" else "Issues found.",
            overall_score=score,
        )

    def _make_sem_result(self, status: str = "pass", score: float = 0.8) -> ValidationResult:
        return ValidationResult(
            status=status,
            layer="semantic",
            scores=[
                ValidationScore(criterion=criterion, score=score, passed=status == "pass")
                for criterion in (
                    "relevance",
                    "evidence_alignment",
                    "completeness",
                    "logical_consistency",
                    "objectivity",
                )
            ],
            issues=[],
            feedback="Accurate." if status == "pass" else "Inaccurate claims.",
            overall_score=score,
        )

    def test_both_pass(self) -> None:
        result = combine_validation_results(
            self._make_det_result("pass", 1.0),
            self._make_sem_result("pass", 0.8),
        )
        assert result.status == "pass"
        assert result.layer == "combined"
        assert result.overall_score == 0.9  # (1.0 + 0.8) / 2

    def test_det_fails_means_revision(self) -> None:
        result = combine_validation_results(
            self._make_det_result("needs_revision", 0.5),
            self._make_sem_result("pass", 0.8),
        )
        assert result.status == "needs_revision"

    def test_sem_below_threshold_fails(self) -> None:
        result = combine_validation_results(
            self._make_det_result("pass", 1.0),
            self._make_sem_result("pass", 0.4),
            semantic_threshold=0.6,
        )
        assert result.status == "needs_revision"

    def test_needs_research_propagated(self) -> None:
        result = combine_validation_results(
            self._make_det_result("needs_research", 0.3),
            self._make_sem_result("pass", 0.8),
        )
        assert result.status == "needs_research"

    def test_scores_merged(self) -> None:
        det = self._make_det_result()
        sem = self._make_sem_result()
        result = combine_validation_results(det, sem)
        assert len(result.scores) == 6  # one deterministic plus five semantic criteria
        criteria = {s.criterion for s in result.scores}
        assert "test" in criteria
        assert "evidence_alignment" in criteria

    def test_feedback_combined(self) -> None:
        result = combine_validation_results(
            self._make_det_result("needs_revision", 0.5),
            self._make_sem_result("needs_revision", 0.4),
            semantic_threshold=0.6,
        )
        assert "Deterministic" in result.feedback
        assert "Semantic" in result.feedback


# ──────────────── Helpers ────────────────


class TestReportToText:
    """Tests for the _report_to_text helper."""

    def test_includes_all_fields(self) -> None:
        report = _make_report()
        text = _report_to_text(report)
        assert "Test Report" in text
        assert "Summary of key findings" in text
        assert "Systematic analysis" in text
        assert "Limited scope" in text
        assert "promising results" in text
