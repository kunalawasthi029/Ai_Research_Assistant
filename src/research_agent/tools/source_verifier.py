"""Verify public URL accessibility and retain original outcomes for duplicate URLs."""

from __future__ import annotations

import time
from urllib.parse import urlsplit, urlunsplit

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from research_agent.tools.results import ToolArguments, ToolOutput
from research_agent.tools.safe_http import FetchError, fetch_public
from research_agent.tools.ssrf_guard import SSRFError, validate_url_scheme

VERIFY_TIMEOUT = 10


class SourceVerifierInput(ToolArguments):
    """Bounded URL batch arguments."""

    urls: list[str] = Field(min_length=1, max_length=20, description="Public HTTP(S) URLs")


class VerificationResult(BaseModel):
    """An accessibility observation; not retrieved factual evidence."""

    url: str
    canonical_url: str
    final_url: str | None = None
    is_accessible: bool = False
    status_code: int | None = None
    is_duplicate: bool = False
    duplicate_of: str | None = None
    error: str | None = None


def _canonicalize_url(url: str) -> str:
    """Normalize host, default port and fragment, preserving meaningful path/query differences."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port and (parsed.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    return urlunsplit((parsed.scheme.lower(), host, parsed.path or "/", parsed.query, ""))


def verify_sources(urls: list[str], *, timeout_seconds: float = VERIFY_TIMEOUT) -> str:
    """Use one deadline for the batch and never turn an unsuccessful duplicate into success."""
    results: list[VerificationResult] = []
    seen: dict[str, VerificationResult] = {}
    deadline = time.monotonic() + timeout_seconds
    for url in urls:
        try:
            validate_url_scheme(url)
            canonical = _canonicalize_url(url)
        except (SSRFError, ValueError):
            results.append(
                VerificationResult(
                    url="[invalid URL]", canonical_url="", error="BLOCKED: invalid URL"
                )
            )
            continue
        if canonical in seen:
            original = seen[canonical]
            results.append(
                original.model_copy(
                    update={"url": url, "is_duplicate": True, "duplicate_of": original.url}
                )
            )
            continue
        result = VerificationResult(url=url, canonical_url=canonical)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FetchError("Timeout: batch deadline exceeded")
            page = fetch_public(url, method="HEAD", timeout_seconds=remaining)
            result.is_accessible = True
            result.status_code = page.status_code
            result.final_url = page.url
        except SSRFError:
            result.error = "BLOCKED: unsafe URL or redirect"
        except FetchError as error:
            result.error = str(error)
        except Exception:
            result.error = "Source verification error"
        seen[canonical] = result
        results.append(result)
    accessible = sum(result.is_accessible for result in results)
    return ToolOutput(
        status="success" if accessible == len(results) else "partial" if accessible else "error",
        data={"verifications": [result.model_dump() for result in results]},
        error="One or more sources inaccessible" if accessible < len(results) else None,
    ).model_dump_json()


source_verifier_tool = StructuredTool.from_function(
    func=verify_sources,
    name="source_verifier",
    description="Check public source URLs using HEAD and detect exact canonical duplicates. Returns accessibility observations, not article evidence.",
    args_schema=SourceVerifierInput,
)
