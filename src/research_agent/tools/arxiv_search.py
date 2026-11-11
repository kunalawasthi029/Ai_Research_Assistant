"""Retrieve bounded arXiv Atom results with paper metadata and real source URLs."""

from __future__ import annotations

from xml.etree import ElementTree

import httpx
from langchain_core.tools import StructuredTool
from pydantic import Field

from research_agent.tools.results import RetrievedItem, ToolArguments, ToolOutput, failure
from research_agent.tools.safe_http import FetchError, fetch_public
from research_agent.tools.ssrf_guard import SSRFError

MAX_ABSTRACT_LENGTH = 2000
ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivSearchInput(ToolArguments):
    """Validated academic search arguments."""

    query: str = Field(min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=10)


def search_arxiv(query: str, max_results: int = 5, *, timeout_seconds: float = 15) -> str:
    """Parse a bounded Atom feed, rejecting entity declarations before XML parsing."""
    url = str(
        httpx.URL(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            },
        )
    )
    try:
        page = fetch_public(url, timeout_seconds=timeout_seconds)
        if b"<!DOCTYPE" in page.content.upper() or b"<!ENTITY" in page.content.upper():
            return failure("Unsafe XML declaration")
        root = ElementTree.fromstring(page.content)
        items = []
        skipped = 0
        for entry in root.findall(f"{ATOM}entry")[:max_results]:
            try:
                paper_url = entry.findtext(f"{ATOM}id", "")
                if "arxiv.org/api/errors" in paper_url:
                    return failure("arXiv API query error")
                title = " ".join(entry.findtext(f"{ATOM}title", "").split())
                abstract = " ".join(entry.findtext(f"{ATOM}summary", "").split())
                authors = [
                    author.findtext(f"{ATOM}name", "") for author in entry.findall(f"{ATOM}author")
                ]
                items.append(
                    RetrievedItem(
                        url=paper_url,
                        title=title[:500],
                        content=abstract[:MAX_ABSTRACT_LENGTH],
                        source_type="arxiv",
                        metadata={
                            "authors": authors,
                            "published": entry.findtext(f"{ATOM}published", ""),
                            "content_truncated": len(abstract) > MAX_ABSTRACT_LENGTH,
                        },
                    )
                )
            except (ValueError, TypeError):
                skipped += 1
        return ToolOutput(
            status="partial" if skipped and items else "success" if items else "empty",
            items=items,
            warnings=["Skipped invalid arXiv entries"]
            if skipped
            else ["No arXiv papers found"]
            if not items
            else [],
        ).model_dump_json()
    except (FetchError, SSRFError):
        return failure("arXiv retrieval error")
    except Exception:
        return failure("arXiv response parsing error")


arxiv_search_tool = StructuredTool.from_function(
    func=search_arxiv,
    name="arxiv_search",
    description="Search arXiv academic preprints. Returns real paper URLs, titles, abstracts, authors and publication metadata.",
    args_schema=ArxivSearchInput,
)
