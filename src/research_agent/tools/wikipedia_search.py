"""Search Wikipedia through its bounded HTTP API and preserve article provenance."""

from __future__ import annotations

import json

import httpx
from langchain_core.tools import StructuredTool
from pydantic import Field

from research_agent.tools.results import RetrievedItem, ToolArguments, ToolOutput, failure
from research_agent.tools.safe_http import FetchError, fetch_public
from research_agent.tools.ssrf_guard import SSRFError

MAX_CONTENT_LENGTH = 10_000


class WikipediaSearchInput(ToolArguments):
    """Validated Wikipedia search arguments."""

    query: str = Field(min_length=1, max_length=1000)
    max_results: int = Field(default=3, ge=1, le=5)


def search_wikipedia(query: str, max_results: int = 3, *, timeout_seconds: float = 15) -> str:
    """Retrieve plain introductory extracts and canonical article URLs in one request."""
    url = str(
        httpx.URL(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": max_results,
                "prop": "extracts|info",
                "inprop": "url",
                "exintro": "1",
                "explaintext": "1",
            },
        )
    )
    try:
        page = fetch_public(url, timeout_seconds=timeout_seconds)
        body = json.loads(page.content)
        if "error" in body:
            return failure("Wikipedia API error")
        items = []
        skipped = 0
        for article in body.get("query", {}).get("pages", [])[:max_results]:
            try:
                text = article["extract"]
                items.append(
                    RetrievedItem(
                        url=article["fullurl"],
                        title=article["title"][:500],
                        content=text[:MAX_CONTENT_LENGTH],
                        source_type="wikipedia",
                        metadata={"content_truncated": len(text) > MAX_CONTENT_LENGTH},
                    )
                )
            except (ValueError, KeyError, TypeError):
                skipped += 1
        return ToolOutput(
            status="partial" if skipped and items else "success" if items else "empty",
            items=items,
            warnings=["Skipped missing or ambiguous Wikipedia pages"]
            if skipped
            else ["No Wikipedia articles found"]
            if not items
            else [],
        ).model_dump_json()
    except (FetchError, SSRFError):
        return failure("Wikipedia retrieval error")
    except Exception:
        return failure("Wikipedia response parsing error")


wikipedia_search_tool = StructuredTool.from_function(
    func=search_wikipedia,
    name="wikipedia_search",
    description="Search Wikipedia for background knowledge; returns normalized article titles, URLs and plain-text extracts.",
    args_schema=WikipediaSearchInput,
)
