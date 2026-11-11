"""Unit tests for the Analyst agent."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from research_agent.agents.analyst import (
    _format_evidence_for_analysis,
    run_analyst,
)
from research_agent.schemas.evidence import (
    AnalysisResult,
    Evidence,
    Finding,
)
from tests.fake_llm import FakeChatModel


def _make_evidence(
    evidence_id: str = "ev1",
    task_id: str = "t1",
    url: str = "https://example.com/article",
    title: str = "Example Article",
    content: str = "Quantum computing has made strides.",
    tool_name: str = "web_search",
    source_type: str = "web",
) -> Evidence:
    """Create a minimal Evidence record for testing."""
    return Evidence(
        evidence_id=evidence_id,
        research_task_id=task_id,
        tool_name=tool_name,
        url=url,
        title=title,
        content=content,
        source_type=source_type,
    )


def _make_analysis_result_json(
    n_findings: int = 2,
    has_gaps: bool = False,
    coverage: float = 0.9,
) -> str:
    """Build a valid AnalysisResult JSON string for FakeChatModel."""
    findings = []
    for i in range(1, n_findings + 1):
        findings.append(
            {
                "finding_id": f"f{i}",
                "research_task_id": "t1",
                "claim": f"Finding {i} about quantum computing",
                "supporting_evidence": ["ev1"],
                "source_url": "https://example.com/article",
                "source_title": "Example Article",
                "source_type": "web",
                "relevance_score": 0.8,
                "confidence": 0.85,
                "verification_status": "verified",
            }
        )
    return json.dumps(
        {
            "summary": "Quantum computing is advancing rapidly.",
            "findings": findings,
            "gap_report": {
                "has_gaps": has_gaps,
                "missing_topics": ["quantum error correction"] if has_gaps else [],
                "contradictions": [],
                "low_confidence_areas": [],
                "suggested_tasks": ["Research error correction"] if has_gaps else [],
                "coverage_score": coverage,
            },
            "total_evidence_count": 3,
            "unique_source_count": 2,
        }
    )


class TestFormatEvidence:
    """Tests for the evidence formatting helper."""

    def test_empty_evidence(self) -> None:
        result = _format_evidence_for_analysis([])
        assert result == "No evidence collected."

    def test_single_evidence(self) -> None:
        ev = _make_evidence()
        result = _format_evidence_for_analysis([ev])
        assert "Total evidence pieces: 1" in result
        assert ev.evidence_id in result
        assert ev.url in result
        assert ev.title in result

    def test_dedup_tracking(self) -> None:
        """Evidence from the same URL still gets listed, unique count is correct."""
        ev1 = _make_evidence(evidence_id="ev1", url="https://example.com/a")
        ev2 = _make_evidence(evidence_id="ev2", url="https://example.com/a")
        result = _format_evidence_for_analysis([ev1, ev2])
        assert "Total evidence pieces: 2" in result
        assert "Unique sources: 1" in result

    def test_content_truncated(self) -> None:
        """Evidence content longer than 2000 chars is truncated."""
        long_content = "x" * 5000
        ev = _make_evidence(content=long_content)
        result = _format_evidence_for_analysis([ev])
        # The formatted text should not contain the full 5000 chars
        assert len(result) < 5000 + 500  # some overhead but well under full


class TestRunAnalyst:
    """Tests for the analyst agent."""

    def test_returns_valid_analysis_result(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_analysis_result_json())])
        evidence = [_make_evidence()]

        result = run_analyst(
            evidence=evidence,
            research_objective="Quantum computing advances",
            task_ids=["t1"],
            llm=llm,
        )

        assert isinstance(result, AnalysisResult)
        assert len(result.findings) == 2
        assert result.gap_report.coverage_score == 0.9
        assert result.gap_report.has_gaps is False

    def test_findings_have_required_fields(self) -> None:
        llm = FakeChatModel(responses=[AIMessage(content=_make_analysis_result_json(n_findings=1))])
        result = run_analyst(
            evidence=[_make_evidence()],
            research_objective="Test",
            task_ids=["t1"],
            llm=llm,
        )
        finding = result.findings[0]
        assert isinstance(finding, Finding)
        assert finding.finding_id == "f1"
        assert finding.claim
        assert len(finding.supporting_evidence) >= 1
        assert 0 <= finding.relevance_score <= 1
        assert 0 <= finding.confidence <= 1

    def test_gap_report_with_gaps(self) -> None:
        llm = FakeChatModel(
            responses=[AIMessage(content=_make_analysis_result_json(has_gaps=True, coverage=0.4))]
        )
        result = run_analyst(
            evidence=[_make_evidence()],
            research_objective="Test",
            task_ids=["t1"],
            llm=llm,
        )
        assert result.gap_report.has_gaps is True
        assert result.gap_report.coverage_score == 0.4
        assert len(result.gap_report.missing_topics) > 0
        assert len(result.gap_report.suggested_tasks) > 0

    def test_prompt_includes_task_ids(self) -> None:
        """The analyst prompt should include all research task IDs."""
        llm = FakeChatModel(responses=[AIMessage(content=_make_analysis_result_json())])
        run_analyst(
            evidence=[_make_evidence()],
            research_objective="Test",
            task_ids=["t1", "t2", "t3"],
            llm=llm,
        )
        # Check the messages sent to the LLM
        assert len(llm.invocations) == 1
        user_msg = llm.invocations[0][-1].content
        assert "t1" in user_msg
        assert "t2" in user_msg
        assert "t3" in user_msg

    def test_prompt_includes_objective(self) -> None:
        """The analyst prompt should include the research objective."""
        llm = FakeChatModel(responses=[AIMessage(content=_make_analysis_result_json())])
        run_analyst(
            evidence=[_make_evidence()],
            research_objective="Impact of AI on healthcare",
            task_ids=["t1"],
            llm=llm,
        )
        user_msg = llm.invocations[0][-1].content
        assert "Impact of AI on healthcare" in user_msg
