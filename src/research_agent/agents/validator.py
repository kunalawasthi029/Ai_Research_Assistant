"""Quality Validator agent: two-layer validation of research reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.llm.context import context_for_model, research_time_context, untrusted_block
from research_agent.llm.structured import invoke_structured
from research_agent.reporting.deterministic import validate_report as run_deterministic_validator
from research_agent.schemas.evidence import Evidence
from research_agent.schemas.plan import ResearchPlan
from research_agent.schemas.report import Report
from research_agent.schemas.validation import (
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

__all__ = ["run_deterministic_validator", "run_semantic_validator", "combine_validation_results"]

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "validator.md"

SEMANTIC_CRITERIA = {
    "relevance",
    "evidence_alignment",
    "completeness",
    "logical_consistency",
    "objectivity",
}


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def run_semantic_validator(
    report: Report,
    evidence: list[Evidence],
    research_objective: str,
    llm: Any,
    *,
    semantic_threshold: float = 0.6,
    criterion_threshold: float = 0.6,
    research_plan: ResearchPlan | None = None,
) -> ValidationResult:
    """Layer B: semantic validation using an LLM judge.

    The judge receives BOTH the report AND the evidence to compare them.
    Uses the separately configured validator role; model diversity is not enforced.
    """
    if not 0 <= semantic_threshold <= 1 or not 0 <= criterion_threshold <= 1:
        raise ValueError("Validation thresholds must be between zero and one")
    system_prompt = _load_prompt() + "\n\n" + research_time_context()
    if not evidence:
        return ValidationResult(
            status="needs_research",
            layer="semantic",
            feedback="No retrieved evidence is available to assess factual support.",
            overall_score=0.0,
        )

    # Build evidence summary for the judge
    context = context_for_model(llm, evidence)
    evidence_summary = context.text

    # Build report text for evaluation
    report_text = untrusted_block("REPORT", report.model_dump(mode="json"))

    user_message = (
        f"Research Objective: {research_objective}\n\n"
        f"## Report to Evaluate:\n{report_text}\n\n"
        f"## Actual Evidence Collected:\n{evidence_summary}\n\n"
        f"Evaluate this report against the evidence and research objective. "
        f"Score each criterion and identify any unsupported claims."
    )
    if research_plan is not None:
        user_message += "\n" + untrusted_block("PLAN", research_plan.model_dump(mode="json"))

    result = invoke_structured(
        llm,
        ValidationResult,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
    )

    if context.omitted_ids and any(
        eid in context.omitted_ids for ref in report.references for eid in ref.all_evidence_ids
    ):
        result = result.model_copy(
            update={
                "status": "needs_research",
                "overall_score": 0.0,
                "feedback": "Cited evidence was omitted by the context budget; support is unassessed.",
            }
        )

    names = [score.criterion for score in result.scores]
    rubric_ok = set(names) == SEMANTIC_CRITERIA and len(names) == len(SEMANTIC_CRITERIA)
    score = sum(item.score for item in result.scores) / len(result.scores) if result.scores else 0.0
    blocked = bool(result.unsupported_claims) or any(
        issue.severity in {"major", "critical"} or issue.issue_type == "unsupported_claim"
        for issue in result.issues
    )
    if result.status == "pass" and (
        not rubric_ok
        or blocked
        or score < semantic_threshold
        or any(not item.passed or item.score < criterion_threshold for item in result.scores)
    ):
        result = result.model_copy(
            update={
                "status": "needs_research" if result.unsupported_claims else "needs_revision",
                "feedback": "Resolve unsupported claims, failed criteria or incomplete rubric. "
                + result.feedback,
            }
        )
    # Context omissions remain unassessed even if the model assigned high scores.
    if context.omitted_ids and any(
        eid in context.omitted_ids for ref in report.references for eid in ref.all_evidence_ids
    ):
        score = 0.0

    # Ensure layer is set correctly and aggregate is derived from criterion scores.
    result = ValidationResult(
        status=result.status,
        layer="semantic",
        scores=result.scores,
        issues=result.issues,
        missing_sections=result.missing_sections,
        unsupported_claims=result.unsupported_claims,
        broken_citations=result.broken_citations,
        feedback=result.feedback,
        recommendations=result.recommendations,
        overall_score=score,
    )

    logger.info(
        "Semantic validation: status=%s, score=%.2f, unsupported_claims=%d",
        result.status,
        result.overall_score,
        len(result.unsupported_claims),
    )

    return result


def combine_validation_results(
    det_result: ValidationResult,
    sem_result: ValidationResult,
    semantic_threshold: float = 0.6,
    criterion_threshold: float = 0.6,
) -> ValidationResult:
    """Combine deterministic and semantic results into a final verdict.

    Pass requires: Layer A fully green AND Layer B above threshold.
    """
    all_scores = det_result.scores + sem_result.scores
    all_issues = det_result.issues + sem_result.issues
    all_missing = list(set(det_result.missing_sections + sem_result.missing_sections))
    all_unsupported = list(set(det_result.unsupported_claims + sem_result.unsupported_claims))
    all_broken = list(set(det_result.broken_citations + sem_result.broken_citations))

    if not 0 <= semantic_threshold <= 1 or not 0 <= criterion_threshold <= 1:
        raise ValueError("Validation thresholds must be between zero and one")

    def green(result: ValidationResult) -> bool:
        return (
            result.status == "pass"
            and bool(result.scores)
            and all(score.passed for score in result.scores)
            and not result.missing_sections
            and not result.broken_citations
            and not result.unsupported_claims
            and not any(
                issue.severity in {"major", "critical"} or issue.issue_type == "unsupported_claim"
                for issue in result.issues
            )
        )

    det_passed = green(det_result)
    sem_passed = (
        green(sem_result)
        and {score.criterion for score in sem_result.scores} == SEMANTIC_CRITERIA
        and len(sem_result.scores) == len(SEMANTIC_CRITERIA)
        and sem_result.overall_score >= semantic_threshold
        and all(score.score >= criterion_threshold for score in sem_result.scores)
    )

    status: ValidationStatus
    if det_passed and sem_passed:
        status = "pass"
    elif det_result.status == "fail" or sem_result.status == "fail":
        status = "fail"
    elif (
        det_result.status == "needs_research"
        or sem_result.status == "needs_research"
        or all_unsupported
    ):
        status = "needs_research"
    else:
        status = "needs_revision"

    overall = (det_result.overall_score + sem_result.overall_score) / 2

    feedback_parts = []
    if not det_passed:
        feedback_parts.append(
            f"Deterministic: {det_result.feedback or 'Resolve failed criteria and issues.'}"
        )
    if not sem_passed:
        feedback_parts.append(
            f"Semantic: {sem_result.feedback or 'Resolve failed verdict, unsupported claims or score thresholds.'}"
        )

    return ValidationResult(
        status=status,
        layer="combined",
        scores=all_scores,
        issues=all_issues,
        missing_sections=all_missing,
        unsupported_claims=all_unsupported,
        broken_citations=all_broken,
        feedback="\n".join(feedback_parts) if feedback_parts else "Validation passed.",
        recommendations=sem_result.recommendations,
        overall_score=overall,
    )


def _report_to_text(report: Report) -> str:
    """Convert a Report to a readable text summary for the judge."""
    parts = [
        f"Title: {report.title}",
        f"Executive Summary: {report.executive_summary}",
        f"Objective: {report.research_objective}",
        f"Methodology: {report.methodology}",
        f"Key Findings: {'; '.join(report.key_findings)}",
    ]
    for section in report.detailed_analysis:
        parts.append(f"\n## {section.heading}\n{section.content}")
    if report.limitations:
        parts.append(f"\nLimitations: {'; '.join(report.limitations)}")
    parts.append(f"\nConclusion: {report.conclusion}")
    parts.append(f"\nReferences: {len(report.references)} sources cited")
    return "\n".join(parts)
