"""Row-level dataclasses for the persistence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunRow:
    """A research run record."""

    id: str
    thread_id: str
    query: str
    objective: str = ""
    status: str = "pending"
    config_json: str = "{}"
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    completed_at: str | None = None
    total_tool_calls: int = 0
    total_tokens: int = 0
    elapsed_seconds: float | None = None


@dataclass
class EventRow:
    """A research event record (decision, tool call, status change, etc.)."""

    id: str
    run_id: str
    event_type: str
    agent: str = ""
    node: str = ""
    data_json: str = "{}"
    created_at: str = field(default_factory=_utcnow_iso)


@dataclass
class EvidenceRow:
    """A persisted evidence record."""

    id: str
    run_id: str
    task_id: str
    tool_name: str
    url: str = ""
    title: str = ""
    content: str = ""
    source_type: str = "web"
    metadata_json: str = "{}"
    retrieved_at: str = field(default_factory=_utcnow_iso)
    findings_json: str = "[]"
    source_json: str = "{}"


@dataclass
class ReportRow:
    """A persisted report record."""

    id: str
    run_id: str
    version: int = 1
    report_json: str = "{}"
    report_markdown: str = ""
    created_at: str = field(default_factory=_utcnow_iso)


@dataclass
class ValidationRow:
    """A persisted validation record."""

    id: str
    run_id: str
    report_version: int = 1
    layer: str = "combined"
    result_json: str = "{}"
    created_at: str = field(default_factory=_utcnow_iso)
