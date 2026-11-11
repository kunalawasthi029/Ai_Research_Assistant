"""Analyst agent: synthesizes evidence into structured findings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.llm.context import context_for_model, evidence_context
from research_agent.llm.structured import ContractError, invoke_structured
from research_agent.schemas.evidence import AnalysisResult, Evidence, GapReport

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.md"


def _load_prompt() -> str:
    """Load the analyst prompt from file."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _format_evidence_for_analysis(evidence_list: list[Evidence]) -> str:
    """Format evidence records into a structured text block for the analyst."""
    if not evidence_list:
        return "No evidence collected."

    return (
        f"Total evidence pieces: {len(evidence_list)}\n"
        + evidence_context(evidence_list).text
        + f"\nUnique sources: {len({item.url for item in evidence_list})}"
    )


def validate_findings(
    result: AnalysisResult, evidence: list[Evidence], task_ids: list[str]
) -> AnalysisResult:
    """Enforce provenance membership and derive coverage without claiming factual verification."""
    by_id = {item.evidence_id: item for item in evidence}
    if len(by_id) != len(evidence):
        raise ContractError("Evidence IDs must be unique.")
    finding_ids = set()
    claims = set()
    findings = []
    for finding in result.findings:
        if finding.finding_id in finding_ids:
            raise ContractError("Finding IDs must be unique.")
        finding_ids.add(finding.finding_id)
        if finding.research_task_id not in task_ids or any(
            ref not in by_id for ref in finding.supporting_evidence
        ):
            raise ContractError("Findings must reference known tasks and supplied evidence IDs.")
        support = [by_id[ref] for ref in finding.supporting_evidence]
        if not any(item.research_task_id == finding.research_task_id for item in support):
            raise ContractError("A finding needs supporting evidence from its own research task.")
        primary = next(
            (
                item
                for item in support
                if item.url == finding.source_url
                and item.title == finding.source_title
                and item.source_type == finding.source_type
            ),
            None,
        )
        if primary is None:
            raise ContractError("Finding source URL/title/type must match its supporting evidence.")
        key = (
            finding.research_task_id,
            finding.claim.strip().casefold(),
            tuple(sorted(set(finding.supporting_evidence))),
        )
        if key in claims:
            continue
        claims.add(key)
        findings.append(
            finding.model_copy(
                update={
                    "supporting_evidence": list(dict.fromkeys(finding.supporting_evidence)),
                    "collected_at": primary.retrieved_at,
                    "verification_status": "unverified"
                    if finding.verification_status == "verified"
                    else finding.verification_status,
                }
            )
        )
    missing = sorted(set(task_ids) - {finding.research_task_id for finding in findings})
    gaps = result.gap_report.model_copy(deep=True)
    if missing:
        gaps.has_gaps = True
        gaps.missing_topics = list(
            dict.fromkeys(
                [
                    *gaps.missing_topics,
                    *(f"Task without supported findings: {task_id}" for task_id in missing),
                ]
            )
        )
        gaps.suggested_tasks = list(
            dict.fromkeys(
                [
                    *gaps.suggested_tasks,
                    *(f"Collect evidence for task {task_id}" for task_id in missing),
                ]
            )
        )
    gaps.coverage_score = min(
        gaps.coverage_score, (len(set(task_ids)) - len(missing)) / max(len(set(task_ids)), 1)
    )
    return result.model_copy(
        update={
            "findings": findings,
            "gap_report": gaps,
            "total_evidence_count": len(evidence),
            "unique_source_count": len({item.url for item in evidence}),
        }
    )


def run_analyst(
    evidence: list[Evidence],
    research_objective: str,
    task_ids: list[str],
    llm: Any,
) -> AnalysisResult:
    """Execute the analyst agent to synthesize evidence into findings.

    Args:
        evidence: All collected evidence records
        research_objective: The research objective
        task_ids: IDs of all research tasks for coverage analysis
        llm: LLM instance

    Returns:
        A validated AnalysisResult with findings and gap report.
    """
    system_prompt = _load_prompt()
    if not evidence:
        return AnalysisResult(
            summary="No retrieved evidence is available for analysis.",
            findings=[],
            gap_report=GapReport(
                has_gaps=True,
                missing_topics=list(task_ids),
                suggested_tasks=[f"Collect evidence for task {task}" for task in task_ids],
                coverage_score=0.0,
            ),
            total_evidence_count=0,
            unique_source_count=0,
        )

    context = context_for_model(llm, evidence)
    evidence_text = context.text

    user_message = (
        f"Research Objective: {research_objective}\n\n"
        f"Research Task IDs to cover: {', '.join(task_ids)}\n\n"
        f"Collected Evidence:\n{evidence_text}\n\n"
        f"Analyze this evidence and produce a structured AnalysisResult."
    )

    visible = list(context.records)
    result = invoke_structured(
        llm,
        AnalysisResult,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
        validate=lambda result: validate_findings(result, visible, task_ids),
    )

    logger.info(
        "Analyst produced %d findings, coverage=%.2f, has_gaps=%s",
        len(result.findings),
        result.gap_report.coverage_score,
        result.gap_report.has_gaps,
    )

    return result
