"""Unit tests for all Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from research_agent.schemas.decisions import (
    SupervisorDecision,
    TokenUsage,
    ToolCallRecord,
)
from research_agent.schemas.evidence import (
    AnalysisResult,
    Evidence,
    Finding,
    GapReport,
)
from research_agent.schemas.plan import ResearchPlan, ResearchTask, SubQuestion
from research_agent.schemas.report import Reference, Report, ReportSection
from research_agent.schemas.validation import (
    ValidationIssue,
    ValidationResult,
    ValidationScore,
)

# ──────────────────────── Plan Schemas ────────────────────────


class TestSubQuestion:
    def test_valid_subquestion(self) -> None:
        sq = SubQuestion(
            question_id="q1",
            question="What is quantum computing?",
            priority=1,
            expected_source_types=["web", "wikipedia"],
        )
        assert sq.question_id == "q1"
        assert sq.priority == 1

    def test_priority_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SubQuestion(question_id="q1", question="test", priority=0)
        with pytest.raises(ValidationError):
            SubQuestion(question_id="q1", question="test", priority=6)


class TestResearchTask:
    def test_valid_task(self) -> None:
        task = ResearchTask(
            task_id="t1",
            question_id="q1",
            description="Search for quantum computing advances",
            search_queries=["quantum computing 2024 breakthroughs"],
            required_evidence="Recent papers or articles",
            expected_source_types=["arxiv", "web"],
            completion_criteria="At least 3 sources found",
            priority=1,
        )
        assert task.status == "pending"
        assert len(task.search_queries) == 1

    def test_empty_search_queries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchTask(
                task_id="t1",
                question_id="q1",
                description="test",
                search_queries=[],
                required_evidence="test",
                expected_source_types=[],
                completion_criteria="test",
                priority=1,
            )


class TestResearchPlan:
    def test_valid_plan(self) -> None:
        plan = ResearchPlan(
            research_objective="Understand quantum computing",
            sub_questions=[SubQuestion(question_id="q1", question="What is QC?", priority=1)],
            research_tasks=[
                ResearchTask(
                    task_id="t1",
                    question_id="q1",
                    description="Search QC",
                    search_queries=["quantum computing"],
                    required_evidence="articles",
                    completion_criteria="3 sources",
                    priority=1,
                )
            ],
        )
        assert plan.scope == "balanced"
        assert plan.estimated_complexity == "moderate"

    def test_empty_subquestions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchPlan(
                research_objective="test",
                sub_questions=[],
                research_tasks=[
                    ResearchTask(
                        task_id="t1",
                        question_id="q1",
                        description="test",
                        search_queries=["test"],
                        required_evidence="test",
                        completion_criteria="test",
                        priority=1,
                    )
                ],
            )


# ──────────────────────── Evidence Schemas ────────────────────────


class TestEvidence:
    def test_valid_evidence(self) -> None:
        e = Evidence(
            evidence_id="ev1",
            research_task_id="t1",
            tool_name="web_search",
            url="https://example.com/article",
            title="Test Article",
            content="Some content about quantum computing",
            source_type="web",
        )
        assert e.evidence_id == "ev1"
        assert isinstance(e.retrieved_at, datetime)
        assert e.metadata == {}

    def test_metadata_preserved(self) -> None:
        e = Evidence(
            evidence_id="ev2",
            research_task_id="t1",
            tool_name="arxiv_search",
            url="https://arxiv.org/abs/1234",
            title="Paper",
            content="Abstract",
            source_type="arxiv",
            metadata={"authors": ["Alice", "Bob"], "year": 2024},
        )
        assert e.metadata["authors"] == ["Alice", "Bob"]


class TestFinding:
    def test_valid_finding(self) -> None:
        f = Finding(
            finding_id="f1",
            research_task_id="t1",
            claim="Quantum computers can break RSA",
            supporting_evidence=["ev1"],
            source_url="https://example.com",
            source_title="Example",
            source_type="web",
            relevance_score=0.9,
            confidence=0.7,
        )
        assert f.verification_status == "unverified"

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Finding(
                finding_id="f1",
                research_task_id="t1",
                claim="test",
                supporting_evidence=["ev1"],
                source_url="https://example.com",
                source_title="Example",
                source_type="web",
                relevance_score=1.5,
                confidence=0.7,
            )

    def test_empty_evidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Finding(
                finding_id="f1",
                research_task_id="t1",
                claim="test",
                supporting_evidence=[],
                source_url="https://example.com",
                source_title="Example",
                source_type="web",
                relevance_score=0.5,
                confidence=0.5,
            )


class TestGapReport:
    def test_valid_gap_report(self) -> None:
        gr = GapReport(
            has_gaps=True,
            missing_topics=["error correction"],
            contradictions=[],
            coverage_score=0.6,
        )
        assert gr.has_gaps is True
        assert len(gr.missing_topics) == 1


class TestAnalysisResult:
    def test_valid_analysis(self) -> None:
        ar = AnalysisResult(
            summary="Key findings about quantum computing",
            findings=[],
            gap_report=GapReport(has_gaps=False, coverage_score=0.95),
            total_evidence_count=5,
            unique_source_count=4,
        )
        assert ar.total_evidence_count == 5


# ──────────────────────── Report Schemas ────────────────────────


class TestReport:
    def test_valid_report(self) -> None:
        now = datetime.now(UTC)
        report = Report(
            title="Quantum Computing Research Report",
            executive_summary="Summary of findings",
            research_objective="Understand QC advances",
            methodology="Multi-source research with validation",
            key_findings=["QC is advancing rapidly"],
            detailed_analysis=[
                ReportSection(
                    heading="Overview",
                    content="Quantum computing is...",
                    evidence_ids=["ev1"],
                )
            ],
            limitations=["Limited to English sources"],
            conclusion="QC shows promise",
            references=[
                Reference(
                    ref_id="ref1",
                    evidence_id="ev1",
                    url="https://example.com",
                    title="Source",
                    source_type="web",
                    accessed_at=now,
                )
            ],
        )
        assert report.version == 1
        assert report.title == "Quantum Computing Research Report"

    def test_empty_key_findings_rejected(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            Report(
                title="Test",
                executive_summary="Test",
                research_objective="Test",
                methodology="Test",
                key_findings=[],
                detailed_analysis=[ReportSection(heading="A", content="B")],
                limitations=["None"],
                conclusion="Test",
                references=[
                    Reference(
                        ref_id="r1",
                        evidence_id="e1",
                        url="https://x.com",
                        title="T",
                        source_type="web",
                        accessed_at=now,
                    )
                ],
            )


# ──────────────────────── Validation Schemas ────────────────────────


class TestValidationResult:
    def test_valid_result(self) -> None:
        vr = ValidationResult(
            status="pass",
            layer="combined",
            scores=[
                ValidationScore(
                    criterion="completeness",
                    score=0.95,
                    passed=True,
                    details="All sections present",
                )
            ],
            overall_score=0.95,
        )
        assert vr.status == "pass"

    def test_with_issues(self) -> None:
        vr = ValidationResult(
            status="needs_revision",
            layer="deterministic",
            issues=[
                ValidationIssue(
                    issue_type="missing_section",
                    severity="critical",
                    description="Missing methodology section",
                    suggestion="Add a methodology section",
                )
            ],
            overall_score=0.4,
        )
        assert len(vr.issues) == 1
        assert vr.issues[0].severity == "critical"


# ──────────────────────── Decision Schemas ────────────────────────


class TestSupervisorDecision:
    def test_valid_decision(self) -> None:
        d = SupervisorDecision(
            decision_type="plan_review",
            selected_action="approve",
            reasoning="Plan covers all aspects",
            target_node="researcher",
        )
        assert d.target_node == "researcher"


class TestToolCallRecord:
    def test_valid_record(self) -> None:
        r = ToolCallRecord(
            call_id="tc1",
            tool_name="web_search",
            arguments={"query": "quantum computing"},
            task_id="t1",
            success=True,
            duration_seconds=1.5,
        )
        assert r.success is True
        assert r.error is None


class TestTokenUsage:
    def test_valid_usage(self) -> None:
        u = TokenUsage(
            agent="planner",
            model="llama-3.3-70b-versatile",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            latency_seconds=1.2,
        )
        assert u.total_tokens == 700

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(
                agent="planner",
                model="test",
                prompt_tokens=-1,
                completion_tokens=0,
                total_tokens=0,
            )
