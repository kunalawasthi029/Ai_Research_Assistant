"""Regression checks for false passes using arbitrary and persisted report artifacts."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from research_agent.agents.validator import combine_validation_results, run_semantic_validator
from research_agent.reporting.deterministic import validate_report
from research_agent.schemas.report import Report
from research_agent.schemas.validation import ValidationIssue, ValidationScore
from tests.fake_llm import FakeChatModel
from tests.unit.test_validator import TestCombineValidationResults as ValidationFactory
from tests.unit.test_validator import _make_evidence, _make_report, _make_semantic_result_json


def test_json_roundtrip_is_independently_validated_against_stored_records() -> None:
    report = Report.model_validate_json(_make_report().model_dump_json())
    result = validate_report(report, evidence=[_make_evidence()], plan_task_ids=["t1"])
    assert result.status == "pass"
    assert all(score.passed for score in result.scores)
    assert validate_report(report, evidence_ids=["ev1"], plan_task_ids=["t1"]).status != "pass"


def test_visible_unicode_citations_resolve_the_same_stored_provenance_and_task() -> None:
    report = _make_report()
    report.detailed_analysis[0].content = report.detailed_analysis[0].content.replace(
        "[ref_ev1]", "【ref_ev1】"
    )
    result = validate_report(report, evidence=[_make_evidence()], plan_task_ids=["t1"])
    assert result.status == "pass"
    assert all(score.passed for score in result.scores)


@pytest.mark.parametrize(
    "citation",
    [
        "`【ref_ev1】`",
        r"\【ref_ev1】",
        "<!-- 【ref_ev1】 -->",
        "[ref_ev1】",
        "【ref_ev1]",
        "【invented】",
    ],
)
def test_hidden_escaped_mismatched_and_unknown_unicode_citations_do_not_pass(citation: str) -> None:
    report = _make_report()
    report.detailed_analysis[0].content = "Analysis " * 40 + citation
    result = validate_report(report, evidence=[_make_evidence()], plan_task_ids=["t1"])
    assert result.status != "pass"
    assert not next(s.passed for s in result.scores if s.criterion == "citation_traceability")


@pytest.mark.parametrize(
    "defect",
    [
        "url",
        "title",
        "source_type",
        "accessed_at",
        "duplicate_url",
        "duplicate_ref",
        "unknown_inline",
        "foreign_link",
        "uncited_reference",
        "no_section_ids",
        "code_only",
        "escaped_only",
        "hidden_only",
        "open_fence",
        "wrong_task",
        "missing_task",
        "blank_section",
        "blank_finding",
        "missing_reference",
    ],
)
def test_deterministic_defects_never_pass(defect: str) -> None:
    report = _make_report()
    evidence = [_make_evidence()]
    tasks = ["t1"]
    if defect in {"url", "title", "source_type"}:
        setattr(report.references[0], defect, "invented")
    elif defect == "accessed_at":
        report.references[0].accessed_at = report.references[0].accessed_at.replace(year=2025)
    elif defect in {"duplicate_url", "duplicate_ref"}:
        duplicate = report.references[0].model_copy()
        if defect == "duplicate_url":
            duplicate.ref_id = "ref2"
        report.references.append(duplicate)
    elif defect == "unknown_inline":
        report.executive_summary += " [invented]"
    elif defect == "foreign_link":
        report.detailed_analysis[0].content += " [ref_ev1](https://invented.example/fake)"
    elif defect in {"uncited_reference", "code_only", "escaped_only", "hidden_only", "open_fence"}:
        report.detailed_analysis[0].content = (
            "x" * 300
            + {
                "uncited_reference": "",
                "code_only": " `[ref_ev1]`",
                "escaped_only": r" \[ref_ev1]",
                "hidden_only": " <!-- [ref_ev1] -->",
                "open_fence": "\n```\n[ref_ev1]",
            }[defect]
        )
    elif defect == "no_section_ids":
        report.detailed_analysis[0].evidence_ids = []
    elif defect == "wrong_task":
        evidence[0].research_task_id = "unplanned"
    elif defect == "missing_task":
        tasks.append("t2")
    elif defect == "blank_section":
        report.detailed_analysis[0].content = " "
    elif defect == "blank_finding":
        report.key_findings = [" "]
    elif defect == "missing_reference":
        report.references = []
    result = validate_report(report, evidence=evidence, plan_task_ids=tasks)
    assert result.status != "pass", defect
    assert result.issues and any(not score.passed for score in result.scores)


def test_invalid_schema_returns_sanitized_failure() -> None:
    result = validate_report({"secret": "private-body"}, evidence=[], plan_task_ids=[])
    assert result.status == "fail"
    assert "private-body" not in result.model_dump_json()


@pytest.mark.parametrize(
    "defect",
    [
        "fail",
        "needs_revision",
        "unsupported",
        "major",
        "failed_criterion",
        "low_criterion",
        "empty",
        "missing_rubric",
    ],
)
def test_high_aggregate_never_overrides_semantic_failures(defect: str) -> None:
    factory = ValidationFactory()
    det, sem = factory._make_det_result(), factory._make_sem_result(score=0.99)
    if defect in {"fail", "needs_revision"}:
        sem.status = defect
    elif defect == "unsupported":
        sem.unsupported_claims = ["The evidence does not establish this claim"]
    elif defect == "major":
        sem.issues = [
            ValidationIssue(issue_type="low_relevance", severity="major", description="Off topic")
        ]
    elif defect == "failed_criterion":
        sem.scores[0].passed = False
    elif defect == "low_criterion":
        sem.scores[0].score = 0.1
    elif defect == "empty":
        sem.scores = []
    else:
        sem.scores.pop()
    result = combine_validation_results(det, sem)
    assert result.status != "pass"
    assert result.feedback != "Validation passed."


def test_failed_deterministic_criterion_blocks_pass_without_issues() -> None:
    factory = ValidationFactory()
    det = factory._make_det_result()
    det.scores.append(ValidationScore(criterion="duplicates", score=0.5, passed=False))
    assert combine_validation_results(det, factory._make_sem_result()).status != "pass"


def test_semantic_aggregate_is_derived_and_thresholds_are_configurable() -> None:
    value = json.loads(_make_semantic_result_json(overall_score=0.7))
    value["overall_score"] = 1.0
    model = FakeChatModel(responses=[AIMessage(content=json.dumps(value))])
    result = run_semantic_validator(
        _make_report(), [_make_evidence()], "Objective", model, semantic_threshold=0.8
    )
    assert result.status == "needs_revision"
    assert result.overall_score == 0.7
