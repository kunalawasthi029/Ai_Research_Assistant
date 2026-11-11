"""Serialize untrusted evidence and model-derived data into bounded escaped envelopes."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from research_agent.schemas.evidence import Evidence

logger = logging.getLogger(__name__)


def research_time_context() -> str:
    """Ground relative dates in the runtime clock without rewriting historical requests."""
    return (
        f"Research as-of date (UTC): {datetime.now(UTC).date().isoformat()}. "
        "Interpret latest/current/recent relative to this date unless the user specifies "
        "a historical period. Preserve explicit historical periods. Do not describe old "
        "results as current advances; state when the available evidence is stale."
    )


def untrusted_block(label: str, value: Any) -> str:
    """Prevent source text from closing its data envelope; this is not semantic sandboxing."""
    encoded = (
        json.dumps(value, ensure_ascii=False, default=str)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return f"<UNTRUSTED_{label}>\n{encoded}\n</UNTRUSTED_{label}>"


@dataclass(frozen=True)
class EvidenceContext:
    """Exactly the source records shown to a model and any context omissions."""

    text: str
    records: tuple[Evidence, ...]
    omitted_ids: tuple[str, ...]
    truncated_ids: tuple[str, ...]


def evidence_context(
    evidence: list[Evidence], *, max_chars: int = 30_000, content_chars: int = 2000
) -> EvidenceContext:
    """Budget serialized source text while retaining exact IDs, provenance and timestamps."""
    payload: list[dict[str, Any]] = []
    included: list[Evidence] = []
    omitted: list[str] = []
    truncated: list[str] = []
    if max_chars < 256 or content_chars < 1:
        raise ValueError("Evidence context budget must allow a bounded data envelope")
    # Stable round-robin prevents one large task from consuming all source space.
    groups: dict[str, deque[Evidence]] = defaultdict(deque)
    # Full-text extractions can arrive after many search snippets. Represent them
    # first within each task so a small context does not hide documents we fetched.
    # This is a retrieval-depth preference, not a claim of factual authority.
    for item in sorted(evidence, key=lambda item: item.tool_name != "webpage_extractor"):
        groups[item.research_task_id].append(item)
    if groups:
        document_slots = max(
            len(groups), sum(item.tool_name == "webpage_extractor" for item in evidence)
        )
        content_chars = min(content_chars, max(128, max_chars // document_slots - 700))
    ordered: list[Evidence] = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.popleft())
    # Prioritize distinct URLs without changing any original task/ID/provenance.
    seen_urls: set[str] = set()
    distinct: list[Evidence] = []
    repeated: list[Evidence] = []
    for item in ordered:
        (repeated if item.url in seen_urls else distinct).append(item)
        seen_urls.add(item.url)
    ordered = sorted(distinct + repeated, key=lambda item: item.tool_name != "webpage_extractor")
    for item in ordered:
        row = {
            "evidence_id": item.evidence_id,
            "task_id": item.research_task_id,
            "tool": item.tool_name,
            "url": item.url,
            "title": item.title,
            "source_type": item.source_type,
            "retrieved_at": item.retrieved_at.isoformat(),
            "content": item.content[:content_chars],
            "content_truncated": len(item.content) > content_chars
            or bool(item.metadata.get("content_truncated")),
        }
        next_truncated = [*truncated, item.evidence_id] if row["content_truncated"] else truncated
        preview = untrusted_block(
            "EVIDENCE",
            {
                "items": [*payload, row],
                "omitted_count": len(evidence),
                "truncated_ids": next_truncated,
            },
        )
        if len(preview) > max_chars:
            omitted.append(item.evidence_id)
            continue
        payload.append(row)
        included.append(item)
        truncated = next_truncated
    if omitted or truncated:
        logger.info(
            "evidence_context included=%d omitted=%d truncated=%d",
            len(included),
            len(omitted),
            len(truncated),
        )
    text = untrusted_block(
        "EVIDENCE", {"items": payload, "omitted_count": len(omitted), "truncated_ids": truncated}
    )
    return EvidenceContext(text, tuple(included), tuple(omitted), tuple(truncated))


def context_for_model(llm: Any, evidence: list[Evidence]) -> EvidenceContext:
    """Apply configured source-space budgets without pretending omitted sources were seen."""
    budget = getattr(llm, "evidence_context_max_chars", 30_000)
    if not isinstance(budget, int):
        budget = 30_000
    return evidence_context(evidence, max_chars=budget)
