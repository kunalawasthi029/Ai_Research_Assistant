"""One bibliography entry can retain distinct, validated task passages of one URL."""

from __future__ import annotations

import pytest

from research_agent.agents.report_generator import validate_report_sources
from research_agent.llm.context import evidence_context
from research_agent.llm.structured import ContractError
from research_agent.reporting.deterministic import validate_report
from research_agent.schemas.report import ReportSection
from tests.unit.test_validator import _make_evidence, _make_report


def citation_case():
    first = _make_evidence()
    second = first.model_copy(
        update={
            "evidence_id": "ev2",
            "research_task_id": "t2",
            "content": "A different passage from the same source.",
        }
    )
    report = _make_report()
    report.references[0].supporting_evidence_ids = ["ev2"]
    report.detailed_analysis.append(
        ReportSection(
            heading="Another aspect",
            content="The second passage supports this separate aspect. [ref_ev1]",
            evidence_ids=["ev2"],
        )
    )
    return report, [first, second]


def test_one_source_reference_covers_two_explicit_task_excerpts() -> None:
    report, records = citation_case()
    assert len(report.references) == 1
    assert validate_report_sources(report, records).references[0].all_evidence_ids == ["ev1", "ev2"]
    assert validate_report(report, evidence=records, plan_task_ids=["t1", "t2"]).status == "pass"


@pytest.mark.parametrize("bad_id", ["unknown", "ev1"])
def test_missing_or_repeated_additional_excerpt_is_rejected(bad_id: str) -> None:
    report, records = citation_case()
    report.references[0].supporting_evidence_ids = [bad_id]
    with pytest.raises(ContractError):
        validate_report_sources(report, records)
    assert validate_report(report, evidence=records).status != "pass"


@pytest.mark.parametrize(
    "change", [{"url": "https://other.example/doc"}, {"source_type": "arxiv"}, {"content": ""}]
)
def test_additional_excerpt_cannot_smuggle_other_source_or_empty_support(change: dict) -> None:
    report, records = citation_case()
    records[1] = records[1].model_copy(update=change)
    result = validate_report(report, evidence=records)
    assert not next(s for s in result.scores if s.criterion == "reference_provenance").passed
    with pytest.raises(ContractError):
        validate_report_sources(report, records)


def test_reference_membership_does_not_automatically_cover_uncited_task() -> None:
    report, records = citation_case()
    report.detailed_analysis.pop()
    result = validate_report(report, evidence=records, plan_task_ids=["t1", "t2"])
    assert not next(s for s in result.scores if s.criterion == "plan_task_coverage").passed


def test_small_context_retains_two_documents_across_two_tasks() -> None:
    records = []
    for task in ["t1", "t2"]:
        for url in ["https://example.org/one", "https://example.org/two"]:
            records.append(
                _make_evidence(f"{task}-{len(records)}").model_copy(
                    update={
                        "research_task_id": task,
                        "url": url,
                        "tool_name": "webpage_extractor",
                        "content": "Relevant source passage. " * 80,
                    }
                )
            )
    original = [r.model_dump() for r in records]
    context = evidence_context(records, max_chars=6000)
    assert len(context.text) <= 6000
    assert len(context.records) == 4 and not context.omitted_ids
    assert len({r.url for r in context.records[:2]}) == 2
    assert [r.model_dump() for r in records] == original
