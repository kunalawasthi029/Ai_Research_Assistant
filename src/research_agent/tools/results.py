"""Normalize tool outcomes and preserve retrieved-source provenance as evidence."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from research_agent.schemas.evidence import Evidence


class ToolArguments(BaseModel):
    """Reject unknown arguments and blank strings consistently across all tools."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)


class RetrievedItem(BaseModel):
    """One actual retrieved source, separate from errors and computed values."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    url: str
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=100_000)
    source_type: Literal["web", "wikipedia", "arxiv"] = "web"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def require_source_url(cls, value: str) -> str:
        """Reject pseudo-URLs and credentials without claiming that the URL is reachable."""
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Source requires an HTTP(S) URL without credentials")
        if any(char.isspace() for char in value) or len(value) > 4096:
            raise ValueError("Invalid source URL")
        return value


class ToolOutput(BaseModel):
    """A shared JSON contract: retrieved items, auxiliary data, and explicit failures."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    status: Literal["success", "partial", "empty", "error"] = "success"
    items: list[RetrievedItem] = Field(default_factory=list, max_length=20)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ToolExecutionError(Exception):
    """An explicit safe tool-outcome failure, separate from a provider exception."""


def safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep observable arguments useful while masking common credential fields/patterns."""

    def redact(value: Any, depth: int = 0) -> Any:
        if depth > 4:
            return "[omitted]"
        if isinstance(value, dict):
            return {
                str(key): "[REDACTED]"
                if any(
                    term in str(key).lower()
                    for term in ("key", "token", "password", "authorization")
                )
                else redact(item, depth + 1)
                for key, item in list(value.items())[:20]
            }
        if isinstance(value, list):
            return [redact(item, depth + 1) for item in value[:20]]
        if isinstance(value, str):
            value = re.sub(r"(?:gsk_|tvly-)[A-Za-z0-9_-]+", "[REDACTED]", value)
            return re.sub(
                r"(?i)((?:api_key|token|password|authorization)=)[^&\s]+", r"\1[REDACTED]", value
            )[:1000]
        return value

    return {
        key: redact(value)
        if not any(term in key.lower() for term in ("key", "token", "password", "authorization"))
        else "[REDACTED]"
        for key, value in arguments.items()
    }


def failure(code: str) -> str:
    """Return a fixed, secret-safe failure message without treating it as evidence."""
    return ToolOutput(status="error", error=code).model_dump_json()


def evidence_from_output(output: ToolOutput, task_id: str, tool_name: str) -> list[Evidence]:
    """Create one record per retrieved item; computation/verification is auxiliary data."""
    if output.status not in {"success", "partial"}:
        return []
    return [
        Evidence(
            evidence_id=str(uuid.uuid4()),
            research_task_id=task_id,
            tool_name=tool_name,
            url=item.url,
            title=item.title,
            content=item.content[:10_000],
            source_type=item.source_type,
            retrieved_at=item.retrieved_at,
            metadata={
                **item.metadata,
                "snippet": item.content[:500],
                "content_truncated": bool(item.metadata.get("content_truncated"))
                or len(item.content) > 10_000,
            },
        )
        for item in output.items
    ]
