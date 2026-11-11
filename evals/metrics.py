"""Evaluation metrics for comparing research systems.

Metrics:
- topic_coverage: How many expected topics appear in the report
- evidence_count: Number of evidence pieces collected
- source_diversity: Number of unique source types used
- citation_id_resolution: Fraction of citation IDs that resolve to evidence
- citation_validity: Unassessed until the runner measures network accessibility
- report_completeness: Whether all required sections are present
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from research_agent.schemas.report import Report


def topic_coverage(report_text: str, expected_topics: list[str]) -> float:
    """Score how many expected topics are mentioned in the report.

    Returns: float in [0, 1]
    """
    if not expected_topics:
        return 1.0
    text_lower = report_text.lower()
    found = sum(1 for topic in expected_topics if topic.lower() in text_lower)
    return found / len(expected_topics)


def source_diversity(evidence: list[dict[str, Any]]) -> int:
    """Count unique source types in the evidence."""
    return len({ev.get("source_type", "web") for ev in evidence})


def citation_id_resolution(report: dict[str, Any], evidence_ids: set[str]) -> float:
    """Check what fraction of cited evidence IDs exist.

    Returns: float in [0, 1]
    """
    cited: set[str] = set()
    for section in report.get("detailed_analysis", []):
        cited.update(section.get("evidence_ids", []))
    for ref in report.get("references", []):
        cited.add(ref.get("evidence_id", ""))

    cited.discard("")
    if not cited:
        return 0.0

    valid = sum(1 for cid in cited if cid in evidence_ids)
    return valid / len(cited)


# Compatibility for old Python callers. The protocol-v2 JSON metric with this
# name is URL accessibility, populated only after actual network observations.
citation_validity = citation_id_resolution


def report_completeness(report: dict[str, Any]) -> float:
    """Score how many required sections are present and non-empty.

    Returns: float in [0, 1]
    """
    required = [
        "title",
        "executive_summary",
        "research_objective",
        "methodology",
        "key_findings",
        "detailed_analysis",
        "limitations",
        "conclusion",
        "references",
    ]
    present = 0
    for field in required:
        value = report.get(field)
        if value is not None:
            if (
                isinstance(value, str)
                and value.strip()
                or isinstance(value, list)
                and len(value) > 0
            ):
                present += 1
    return present / len(required)


def compute_all_metrics(
    report: dict[str, Any],
    report_text: str,
    expected_topics: list[str],
    evidence: list[dict[str, Any]],
    evidence_ids: set[str],
) -> dict[str, float | None]:
    """Compute all evaluation metrics for a single report.

    Returns: dict with metric names and values
    """
    return {
        "topic_coverage": topic_coverage(report_text, expected_topics),
        "evidence_count": float(len(evidence)),
        "source_diversity": float(source_diversity(evidence)),
        "citation_validity": None,
        "citation_id_resolution": citation_id_resolution(report, evidence_ids),
        "report_completeness": report_completeness(report),
        "schema_valid": schema_valid(report),
        "duplicate_section_rate": duplicate_section_rate(report),
        "evidence_coverage": evidence_coverage(report, evidence_ids),
    }


def schema_valid(report: dict[str, Any]) -> float:
    """Construction metric, not evidence of factual reliability."""
    try:
        Report.model_validate(report)
        return 1.0
    except ValidationError:
        return 0.0


def evidence_coverage(report: dict[str, Any], available: set[str]) -> float:
    """Fraction of available evidence cited in analysis, not claim-support accuracy."""
    cited = {
        eid
        for section in report.get("detailed_analysis", [])
        for eid in section.get("evidence_ids", [])
    }
    return len(cited & available) / len(available) if available else 0.0


def duplicate_section_rate(report: dict[str, Any]) -> float:
    """Exact normalized duplicate content only; not a semantic plagiarism detector."""
    sections = [
        " ".join(s.get("content", "").lower().split()) for s in report.get("detailed_analysis", [])
    ]
    return (len(sections) - len(set(sections))) / len(sections) if sections else 0.0
