"""Extract bounded readable source content through the shared public-only HTTP client."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from pydantic import Field

from research_agent.tools.results import RetrievedItem, ToolArguments, ToolOutput, failure
from research_agent.tools.safe_http import FetchError, fetch_public
from research_agent.tools.ssrf_guard import SSRFError

MAX_CONTENT_LENGTH = 100_000
REQUEST_TIMEOUT = 15


class WebpageExtractorInput(ToolArguments):
    """Validated extraction arguments."""

    url: str = Field(min_length=1, max_length=4096, description="HTTP(S) page URL")
    focus: str = Field(
        default="",
        max_length=500,
        description="Task terms for selecting relevant verbatim passages from a long document",
    )


def focused_excerpt(text: str, focus: str) -> tuple[str, list[list[int]]]:
    """Select bounded, nonoverlapping verbatim windows; never summarize or invent text."""
    words = {word[:5] for word in re.findall(r"[a-zA-Z]{3,}", focus.lower())}
    words -= {
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "using",
        "what",
        "which",
        "about",
    }
    if "setup" in words:
        words.update({"estab", "initi", "hands"})
    if not words or len(text) <= 1800:
        return text, []
    candidates: list[tuple[set[str], int, int]] = []
    for start in range(0, len(text), 600):
        end = min(start + 600, len(text))
        window_words = {word[:5] for word in re.findall(r"[a-zA-Z]{3,}", text[start:end].lower())}
        candidates.append((words & window_words, start, end))
    chosen: list[tuple[set[str], int, int]] = []
    covered: set[str] = set()
    for _ in range(min(3, len(candidates))):
        best = max(candidates, key=lambda row: (len(row[0] - covered), len(row[0]), -row[1]))
        if not best[0]:
            break
        chosen.append(best)
        covered.update(best[0])
        candidates.remove(best)
    chosen.sort(key=lambda row: row[1])
    if not chosen:
        return text, []
    ranges = [[start, end] for _, start, end in chosen]
    return "\n...\n".join(text[start:end] for start, end in ranges), ranges


def extract_webpage(url: str, *, timeout_seconds: float = REQUEST_TIMEOUT, focus: str = "") -> str:
    """Retrieve one page and retain its final URL, title, text and collection time."""
    try:
        page = fetch_public(url, timeout_seconds=timeout_seconds)
        content_type = page.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
            return failure("Non-text content type")
        soup = BeautifulSoup(page.content, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else "Untitled source"
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if not text:
            return ToolOutput(
                status="empty", warnings=["Page contains no readable text"]
            ).model_dump_json()
        original_length = len(text)
        text, ranges = focused_excerpt(text, focus)
        truncated = bool(ranges) or len(text) > MAX_CONTENT_LENGTH
        return ToolOutput(
            items=[
                RetrievedItem(
                    url=page.url,
                    title=title[:500] or "Untitled source",
                    content=text[:MAX_CONTENT_LENGTH],
                    metadata={
                        "requested_url": url,
                        "content_truncated": truncated,
                        "source_text_chars": original_length,
                        "excerpt_ranges": ranges,
                    },
                )
            ],
            warnings=["Content truncated"] if truncated else [],
        ).model_dump_json()
    except SSRFError:
        return failure("BLOCKED: unsafe URL or redirect")
    except FetchError as error:
        return failure(str(error))
    except Exception:
        return failure("Error extracting content")


webpage_extractor_tool = StructuredTool.from_function(
    func=extract_webpage,
    name="webpage_extractor",
    description="Extract readable text from a public webpage. Returns normalized source items or an explicit error.",
    args_schema=WebpageExtractorInput,
)
