"""Unit tests for the Report Generator agent."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage

from research_agent.agents.report_generator import (
    _analysis_projection,
    _build_evidence_reference_block,
    run_report_generator,
)
from research_agent.schemas.evidence import (
    AnalysisResult,
    Evidence,
    Finding,
    GapReport,
)
from research_agent.schemas.plan import ResearchPlan
from research_agent.schemas.report import Report, ReportSection
from tests.fake_llm import FakeChatModel


def test_analysis_projection_omits_whole_claims_without_mutating_evidence_facts() -> None:
    from research_agent.llm.context import untrusted_block

    analysis = _make_analysis()
    first = analysis.findings[0].model_copy(update={"claim": "Fact " * 700})
    second = first.model_copy(update={"finding_id": "f2"})
    analysis = analysis.model_copy(update={"findings": [first, second]})
    projected = _analysis_projection(analysis, {"ev1"})
    assert projected["findings"][0]["claim"] == first.claim
    assert projected["omitted_finding_count"] == 1
    assert len(untrusted_block("ANALYSIS", projected)) <= 6000
    assert len(analysis.findings) == 2
    assert _analysis_projection(analysis, set())["findings"] == []


def test_reference_repair_receives_exact_stored_metadata_without_weakening_contract() -> None:
    bad = json.loads(_make_report_json())
    bad["references"][0]["url"] = "https://wrong.example/source"
    model = FakeChatModel(
        responses=[AIMessage(content=json.dumps(bad)), AIMessage(content=_make_report_json())]
    )
    result = run_report_generator(
        "Objective", _make_plan(), _make_analysis(), [_make_evidence()], model
    )
    assert result.references[0].url == "https://example.com/ev1"
    correction = model.invocations[1][-1].content
    assert "EXPECTED_REFERENCE" in correction
    assert "https://example.com/ev1" in correction
    assert "Copy the following stored metadata as data" in correction


def test_report_execution_timestamp_cannot_be_backdated_by_model_output() -> None:
    data = json.loads(_make_report_json())
    data["generated_at"] = "2000-01-01T00:00:00Z"
    before = datetime.now(UTC)
    result = run_report_generator(
        "Objective",
        _make_plan(),
        _make_analysis(),
        [_make_evidence()],
        FakeChatModel(responses=[AIMessage(content=json.dumps(data))]),
    )
    assert before <= result.generated_at <= datetime.now(UTC)


@pytest.mark.parametrize("defect", ["missing_inline", "unused_reference"])
def test_citation_construction_errors_are_repaired_before_report_is_returned(defect: str) -> None:
    from research_agent.llm.structured import AgentOutputError
    from research_agent.reporting.deterministic import validate_report

    bad = json.loads(_make_report_json())
    evidence = [_make_evidence(), _make_evidence("ev2")]
    if defect == "missing_inline":
        bad["detailed_analysis"][0]["content"] = "Observed source fact. " * 30
    else:
        bad["references"].append(
            {
                **bad["references"][0],
                "ref_id": "ref2",
                "evidence_id": "ev2",
                "url": evidence[1].url,
                "title": evidence[1].title,
            }
        )
    model = FakeChatModel(
        responses=[AIMessage(content=json.dumps(bad)), AIMessage(content=_make_report_json())]
    )
    report = run_report_generator("Objective", _make_plan(), _make_analysis(), evidence, model)
    assert validate_report(report, evidence=evidence, plan_task_ids=["t1"]).status == "pass"
    assert "Inline citations must" in model.invocations[1][-1].content
    assert len(model.invocations) == 2
    invalid = FakeChatModel(responses=[AIMessage(content=json.dumps(bad))] * 2)
    with pytest.raises(AgentOutputError):
        run_report_generator("Objective", _make_plan(), _make_analysis(), evidence, invalid)
    assert len(invalid.invocations) == 2


# ──────────────────────── Test fixtures ────────────────────────


def _make_evidence(evidence_id: str = "ev1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        research_task_id="t1",
        tool_name="web_search",
        url=f"https://example.com/{evidence_id}",
        title=f"Source {evidence_id}",
        content="Quantum computing breakthroughs in error correction.",
        source_type="web",
    )


def _make_plan() -> ResearchPlan:
    return ResearchPlan(
        research_objective="Quantum computing advances",
        sub_questions=[
            {
                "question_id": "q1",
                "question": "What are the latest breakthroughs?",
                "priority": 1,
            }
        ],
        research_tasks=[
            {
                "task_id": "t1",
                "question_id": "q1",
                "description": "Search for quantum computing breakthroughs",
                "search_queries": ["quantum computing 2024"],
                "required_evidence": "Papers or articles",
                "completion_criteria": "At least 3 sources",
                "priority": 1,
            }
        ],
    )


def _make_analysis() -> AnalysisResult:
    return AnalysisResult(
        summary="Quantum computing is progressing rapidly.",
        findings=[
            Finding(
                finding_id="f1",
                research_task_id="t1",
                claim="Error correction has improved",
                supporting_evidence=["ev1"],
                source_url="https://example.com/ev1",
                source_title="Source ev1",
                source_type="web",
                relevance_score=0.9,
                confidence=0.85,
                verification_status="verified",
            ),
        ],
        gap_report=GapReport(
            has_gaps=False,
            coverage_score=0.9,
        ),
        total_evidence_count=1,
        unique_source_count=1,
    )


def _make_report_json(evidence_id: str = "ev1") -> str:
    """Build a valid Report JSON for FakeChatModel."""
    now = datetime.now(UTC).isoformat()
    return json.dumps(
        {
            "title": "Quantum Computing Advances Report",
            "executive_summary": "This report covers recent quantum computing advances.",
            "research_objective": "Understand quantum computing progress",
            "methodology": "Systematic web search and analysis of recent sources.",
            "key_findings": [
                "Error correction has significantly improved",
                "New qubit architectures are emerging",
            ],
            "detailed_analysis": [
                {
                    "heading": "Error Correction Progress",
                    "content": (
                        "Recent advances in quantum error correction have been significant. "
                        "Researchers have demonstrated surface codes with lower error rates. " * 5
                    )
                    + " [ref1]",
                    "evidence_ids": [evidence_id],
                },
            ],
            "limitations": [
                "Limited to English-language sources",
                "Focus on 2024 publications only",
            ],
            "conclusion": "Quantum computing is making steady progress toward practical applications.",
            "references": [
                {
                    "ref_id": "ref1",
                    "evidence_id": evidence_id,
                    "url": f"https://example.com/{evidence_id}",
                    "title": f"Source {evidence_id}",
                    "source_type": "web",
                    "accessed_at": now,
                },
            ],
        }
    )


# ──────────────────────── Tests ────────────────────────


class TestBuildEvidenceReferenceBlock:
    """Tests for the evidence reference block builder."""

    def test_empty_evidence(self) -> None:
        result = _build_evidence_reference_block([])
        assert result == "No evidence available."

    def test_includes_evidence_metadata(self) -> None:
        ev = _make_evidence("ev1")
        result = _build_evidence_reference_block([ev])
        assert "ev1" in result
        assert "https://example.com/ev1" in result
        assert "Source ev1" in result
        assert "web_search" in result

    def test_multiple_evidence(self) -> None:
        evidence = [_make_evidence("ev1"), _make_evidence("ev2")]
        result = _build_evidence_reference_block(evidence)
        assert "ev1" in result
        assert "ev2" in result


class TestRunReportGenerator:
    """Tests for the report generator agent."""

    def test_returns_valid_report(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_report_json())])
        report = run_report_generator(
            objective="Quantum computing advances",
            plan=_make_plan(),
            analysis=_make_analysis(),
            evidence=[_make_evidence()],
            llm=llm,
        )
        assert isinstance(report, Report)
        assert report.title
        assert report.executive_summary
        assert len(report.key_findings) >= 1
        assert len(report.detailed_analysis) >= 1
        assert len(report.references) >= 1

    def test_report_version_set(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_report_json())])
        report = run_report_generator(
            objective="test",
            plan=_make_plan(),
            analysis=_make_analysis(),
            evidence=[_make_evidence()],
            llm=llm,
            version=3,
        )
        assert report.version == 3

    def test_revision_includes_feedback(self) -> None:
        """When validation_feedback is provided, the prompt includes it."""
        llm = FakeChatModel(responses=[AIMessage(content=_make_report_json())])
        run_report_generator(
            objective="test",
            plan=_make_plan(),
            analysis=_make_analysis(),
            evidence=[_make_evidence()],
            llm=llm,
            validation_feedback="Fix broken citations and expand analysis",
            version=2,
        )
        user_msg = llm.invocations[0][-1].content
        assert "Revision Required" in user_msg
        assert "Fix broken citations" in user_msg

    def test_report_sections_have_evidence_ids(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_report_json())])
        report = run_report_generator(
            objective="test",
            plan=_make_plan(),
            analysis=_make_analysis(),
            evidence=[_make_evidence()],
            llm=llm,
        )
        for section in report.detailed_analysis:
            assert isinstance(section, ReportSection)
            assert isinstance(section.evidence_ids, list)

    def test_prompt_includes_plan_and_analysis(self) -> None:
        """The prompt should contain plan and analysis summaries."""
        llm = FakeChatModel(responses=[AIMessage(content=_make_report_json())])
        run_report_generator(
            objective="Quantum advances",
            plan=_make_plan(),
            analysis=_make_analysis(),
            evidence=[_make_evidence()],
            llm=llm,
        )
        user_msg = llm.invocations[0][-1].content
        assert "Quantum advances" in user_msg
        assert "Quantum computing is progressing" in user_msg  # analysis summary
        assert "1 findings" in user_msg or "1 finding" in user_msg
