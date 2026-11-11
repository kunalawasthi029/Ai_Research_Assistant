"""Independent checks against stored evidence, without model calls or network access."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from research_agent.schemas.evidence import Evidence
from research_agent.schemas.report import Report
from research_agent.schemas.validation import (
    ValidationIssue,
    ValidationResult,
    ValidationScore,
    ValidationStatus,
)

REQUIRED_SECTIONS = (
    "title",
    "executive_summary",
    "research_objective",
    "methodology",
    "key_findings",
    "detailed_analysis",
    "limitations",
    "conclusion",
    "references",
)
_CITATION = re.compile(
    r"(?<![!\\])(?:\[(?P<ascii>[^\[\]【】\n]+)\]|【(?P<unicode>[^\[\]【】\n]+)】)"
    r"(?:\((?P<url>[^\s)]+)\))?"
)
_CODE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]*`|<!--[\s\S]*?(?:-->|\Z)")
_HTML = re.compile(r"<!--|</?[A-Za-z][A-Za-z0-9]*(?:\s[^>]*|/?)>")
_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")


def _url_key(url: str) -> str:
    """Ignore fragments/host case/default ports, retaining path/query distinctions."""
    try:
        if any(char.isspace() or ord(char) < 32 or char in '<>"' for char in url):
            return ""
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ""
        port = parsed.port
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        if port and port != {"http": 80, "https": 443}[parsed.scheme]:
            host += f":{port}"
        return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ""))
    except ValueError:
        return ""


def validate_report(
    report: Report | dict[str, Any],
    evidence_ids: list[str] | None = None,
    plan_task_ids: list[str] | None = None,
    *,
    evidence: list[Evidence] | None = None,
) -> ValidationResult:
    """Validate full provenance. Legacy ID-only input cannot establish a passing result."""
    issues: list[ValidationIssue] = []
    scores: list[ValidationScore] = []

    def criterion(
        name: str, failures: list[str], issue_type: str, *, research: bool = False
    ) -> None:
        scores.append(
            ValidationScore(
                criterion=name,
                score=0.0 if failures else 1.0,
                passed=not failures,
                details="; ".join(failures) or "Passed",
            )
        )
        for detail in failures:
            issues.append(
                ValidationIssue(
                    issue_type=issue_type,
                    severity="major",
                    description=detail,
                    suggestion="Collect and cite valid task evidence"
                    if research
                    else "Correct the report and revalidate",
                )
            )

    try:
        report = Report.model_validate(
            report.model_dump() if isinstance(report, Report) else report
        )
    except (ValidationError, TypeError, ValueError):
        return ValidationResult(
            status="fail",
            layer="deterministic",
            overall_score=0,
            scores=[ValidationScore(criterion="schema_validity", score=0, passed=False)],
            issues=[
                ValidationIssue(
                    issue_type="schema_error",
                    severity="critical",
                    description="Report does not conform to the report schema",
                )
            ],
            feedback="Repair the report schema before validation.",
        )
    criterion("schema_validity", [], "schema_error")

    def empty(value: Any) -> bool:
        return (
            value is None
            or (isinstance(value, str) and not value.strip())
            or (isinstance(value, list) and (not value or any(empty(item) for item in value)))
        )

    missing = [name for name in REQUIRED_SECTIONS if empty(getattr(report, name))]
    nested = [
        f"Analysis section {i} is blank"
        for i, section in enumerate(report.detailed_analysis, 1)
        if not section.heading.strip() or not section.content.strip()
    ]
    criterion(
        "section_completeness",
        [f"Required section '{name}' is empty" for name in missing] + nested,
        "missing_section",
    )

    records = evidence or []
    by_id = {item.evidence_id: item for item in records}
    known = set(evidence_ids or []) | set(by_id)
    provenance = []
    if evidence is None:
        provenance.append(
            "Full evidence records are required to verify provenance and task coverage"
        )
    if len(by_id) != len(records):
        provenance.append("Stored evidence IDs are not unique")
    refs = {ref.ref_id: ref for ref in report.references}
    ref_evidence = {eid for ref in report.references for eid in ref.all_evidence_ids}
    valid_refs = set()
    for ref in report.references:
        item = by_id.get(ref.evidence_id)
        if item is None:
            provenance.append(f"Reference '{ref.ref_id}' has no stored evidence record")
        elif (
            (ref.url, ref.title, ref.source_type, ref.accessed_at)
            != (item.url, item.title, item.source_type, item.retrieved_at)
            or not _url_key(ref.url)
            or not item.content.strip()
        ):
            provenance.append(
                f"Reference '{ref.ref_id}' differs from stored provenance or has empty support"
            )
        elif any(
            eid not in by_id
            or by_id[eid].url != ref.url
            or by_id[eid].source_type != ref.source_type
            or not by_id[eid].content.strip()
            for eid in ref.supporting_evidence_ids
        ):
            provenance.append(
                f"Reference '{ref.ref_id}' has additional excerpts with missing or mismatched source provenance"
            )
        else:
            valid_refs.add(ref.ref_id)
        if not _LABEL.fullmatch(ref.ref_id):
            provenance.append(
                "Reference labels must contain 1-80 letters, digits, underscores, dots, colons or hyphens"
            )
    criterion("reference_provenance", provenance, "provenance_mismatch", research=True)

    duplicates: list[str] = []
    for label, values in (
        ("URL", [_url_key(ref.url) for ref in report.references]),
        ("reference ID", [ref.ref_id for ref in report.references]),
        ("evidence ID", [eid for ref in report.references for eid in ref.all_evidence_ids]),
    ):
        duplicates.extend(
            f"Duplicate {label}: {value}" for value, count in Counter(values).items() if count > 1
        )
    criterion("no_duplicate_refs", duplicates, "duplicate_reference")

    broken = set()
    citation_errors = []
    used_refs = set()
    covered_tasks = set()

    def inspect(text: str, location: str, declared: set[str] | None = None) -> set[str]:
        cited = set()
        if _HTML.search(_CODE.sub("", text)) or "<!--" in text:
            citation_errors.append(
                f"Raw HTML is unsupported in {location}; use visible Markdown citations"
            )
        for match in _CITATION.finditer(_CODE.sub("", text)):
            label = match.group("ascii") or match.group("unicode")
            url = match.group("url")
            if url:
                matches = [ref for ref in report.references if ref.url == url]
                ref = next(
                    (item for item in matches if item.ref_id == label),
                    matches[0] if matches else None,
                )
            else:
                ref = refs.get(label)
            if ref is None:
                broken.add(label)
                citation_errors.append(f"Unresolved inline citation '{label}' in {location}")
                continue
            used_refs.add(ref.ref_id)
            supported = set(ref.all_evidence_ids)
            cited.update(supported if declared is None else supported & declared)
            if ref.ref_id not in valid_refs:
                citation_errors.append(
                    f"Inline citation '{label}' lacks verified provenance in {location}"
                )
            if declared is not None and not supported.intersection(declared):
                citation_errors.append(
                    f"Inline citation '{label}' is absent from {location}.evidence_ids"
                )
        return cited

    for index, section in enumerate(report.detailed_analysis, 1):
        declared = set(section.evidence_ids)
        unresolved = declared - known | declared - ref_evidence
        broken.update(unresolved)
        if unresolved:
            citation_errors.append(
                f"Section {index} contains unresolved evidence IDs: {sorted(unresolved)}"
            )
        cited = inspect(section.content, f"section {index}", declared)
        inspect(section.heading, f"section {index} heading", declared)
        if not declared or not cited:
            citation_errors.append(f"Section {index} requires inline citations and evidence_ids")
        if declared - cited:
            citation_errors.append(f"Section {index} declares evidence without an inline citation")
        for eid in cited & declared:
            if eid in by_id and any(
                eid in ref.all_evidence_ids and ref.ref_id in valid_refs
                for ref in report.references
            ):
                covered_tasks.add(by_id[eid].research_task_id)
    for name in (
        "title",
        "executive_summary",
        "research_objective",
        "methodology",
        "key_findings",
        "comparative_insights",
        "recommendations",
        "limitations",
        "conclusion",
    ):
        value = getattr(report, name)
        for text in value if isinstance(value, list) else [value]:
            if text:
                inspect(text, name)
    for ref in report.references:
        broken.update(set(ref.all_evidence_ids) - known)
        if ref.ref_id not in used_refs:
            citation_errors.append(f"Reference '{ref.ref_id}' is never cited inline")
    criterion("citation_traceability", citation_errors, "broken_citation")

    uncovered = sorted(set(plan_task_ids or []) - covered_tasks)
    criterion(
        "plan_task_coverage",
        [f"Task '{task}' has no cited section evidence" for task in uncovered],
        "incomplete_coverage",
        research=True,
    )
    length = sum(len(section.content.strip()) for section in report.detailed_analysis)
    criterion(
        "minimum_content",
        []
        if length >= 200
        else [f"Detailed analysis has {length} characters; at least 200 required"],
        "empty_content",
    )
    passed = all(score.passed for score in scores)
    status: ValidationStatus = "pass" if passed else "needs_revision"
    if not passed and not missing and (provenance or uncovered or broken):
        status = "needs_research"
    return ValidationResult(
        status=status,
        layer="deterministic",
        scores=scores,
        issues=issues,
        missing_sections=missing,
        broken_citations=sorted(broken),
        overall_score=sum(score.score for score in scores) / len(scores),
        feedback="All deterministic checks passed."
        if passed
        else "; ".join(dict.fromkeys(issue.description for issue in issues)),
    )
