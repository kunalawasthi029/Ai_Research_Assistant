"""Web search tool with configurable provider.

Uses Tavily when TAVILY_API_KEY is set, otherwise falls back to
DuckDuckGo via the ddgs package. Returns titles, URLs, snippets.
"""

from __future__ import annotations

import math
import os
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import Field, SecretStr

from research_agent.tools.results import RetrievedItem, ToolArguments, ToolOutput, failure
from research_agent.tools.search_cache import SearchCache, search_key


class WebSearchInput(ToolArguments):
    """Input schema for web search."""

    query: str = Field(min_length=1, max_length=1000, description="The search query string")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum results to return")


def _search_duckduckgo(
    query: str, max_results: int, timeout_seconds: float = 15
) -> list[dict[str, str]]:
    """Search using DuckDuckGo (no API key required)."""
    from ddgs import DDGS

    results = []
    try:
        with DDGS(timeout=max(1, math.ceil(timeout_seconds))) as ddgs:
            for r in ddgs.text(query, max_results=max_results, backend="duckduckgo"):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", "")),
                        "provider": "DuckDuckGo",
                    }
                )
    except Exception:
        raise RuntimeError("Search provider unavailable") from None
    return results


def _search_tavily(
    query: str,
    max_results: int,
    api_key: str,
    timeout_seconds: float = 15,
    *,
    search_depth: Literal["basic", "advanced"] = "basic",
) -> list[dict[str, str]]:
    """Search using Tavily API."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            timeout=timeout_seconds,
            search_depth=search_depth,
            auto_parameters=False,
            include_answer=False,
            include_raw_content=False,
        )
        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "provider": "Tavily",
                }
            )
        return results
    except Exception:
        # Do not repeat potentially billable requests on quota or ambiguous failures.
        raise RuntimeError("Tavily unavailable or usage limit reached") from None


def web_search(
    query: str,
    max_results: int = 5,
    *,
    timeout_seconds: float = 15,
    api_key: str | SecretStr | None = None,
    search_depth: Literal["basic", "advanced"] = "basic",
    result_limit: int = 5,
    fallback_enabled: bool = True,
    cache_path: str = "",
    cache_ttl_seconds: int = 900,
) -> str:
    """Execute a web search and return formatted results.

    Uses Tavily if TAVILY_API_KEY is set, otherwise DuckDuckGo.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY") if api_key is None else api_key
    if isinstance(tavily_key, SecretStr):
        tavily_key = tavily_key.get_secret_value()

    try:
        arguments = WebSearchInput(query=query, max_results=max_results)
        query, max_results = arguments.query, min(arguments.max_results, result_limit)
        if search_depth not in {"basic", "advanced"} or not 1 <= result_limit <= 10:
            return failure("Invalid search configuration")
        provider = "Tavily" if tavily_key else "DuckDuckGo"
        cache = SearchCache(cache_path, cache_ttl_seconds)
        key = search_key(query, provider, search_depth, max_results)
        cached = cache.get(key)
        if cached is not None:
            return cached.model_dump_json()
        warnings: list[str] = []
        if tavily_key:
            try:
                results = _search_tavily(
                    query, max_results, tavily_key, timeout_seconds, search_depth=search_depth
                )
            except RuntimeError:
                if not fallback_enabled:
                    return failure("Tavily unavailable or usage limit reached; fallback disabled")
                warnings.append("Tavily unavailable or usage limit reached; using DuckDuckGo")
                results = _search_duckduckgo(query, max_results, timeout_seconds)
        else:
            results = _search_duckduckgo(query, max_results, timeout_seconds)
        items = []
        skipped = 0
        for result in results[:max_results]:
            try:
                items.append(
                    RetrievedItem(
                        url=result["url"],
                        title=result["title"][:500] or "Untitled source",
                        content=result["snippet"][:10_000],
                        metadata={
                            "provider": result.get(
                                "provider", "Tavily" if tavily_key else "DuckDuckGo"
                            ),
                            "search_cache_hit": False,
                        },
                    )
                )
            except (ValueError, KeyError):
                skipped += 1
        if skipped:
            warnings.append("Skipped invalid search items")
        if not items:
            warnings.append("No results found")
        output = ToolOutput(
            status="partial" if warnings and items else "success" if items else "empty",
            items=items,
            warnings=warnings,
            data={
                "cache_hit": False,
                "search_depth": search_depth,
                "max_results": max_results,
                # Estimate only: failed requests may not be billed. Never a balance.
                "tavily_credits_upper_bound": (2 if search_depth == "advanced" else 1)
                if tavily_key
                else 0,
            },
        )
        cache.put(key, output)
        return output.model_dump_json()
    except Exception:
        return failure("Search provider unavailable")


def create_web_search_tool() -> StructuredTool:
    """Create the web_search tool instance."""
    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description=(
            "Search the web for current information on any topic. "
            "Returns titles, URLs, and snippets from search results. "
            "Use this for general research, news, and current events."
        ),
        args_schema=WebSearchInput,
    )
