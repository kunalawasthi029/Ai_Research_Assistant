"""Cost controls and cache freshness under real SQLite, with provider calls mocked."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from research_agent.config.settings import Settings
from research_agent.tools.registry import create_default_registry
from research_agent.tools.results import ToolOutput, evidence_from_output
from research_agent.tools.search_cache import SearchCache, search_key
from research_agent.tools.web_search import web_search

ROWS = [{"title": "Source", "url": "https://example.org/a", "content": "Observed snippet"}]


def test_server_caps_depth_and_results_even_when_model_requests_more() -> None:
    settings = Settings(
        groq_api_key="dummy",
        tavily_api_key="dummy",
        search_depth="advanced",
        search_max_results=2,
    )
    with patch("tavily.TavilyClient") as provider:
        provider.return_value.search.return_value = {"results": ROWS * 8}
        output = ToolOutput.model_validate_json(
            create_default_registry(settings=settings).execute(
                "web_search", {"query": "query", "max_results": 10}
            )
        )
    call = provider.return_value.search.call_args.kwargs
    assert call["search_depth"] == "advanced"
    assert call["max_results"] == 2
    assert call["auto_parameters"] is False
    assert call["include_answer"] is False
    assert len(output.items) == 2
    assert output.data["tavily_credits_upper_bound"] == 2


def test_repeated_search_uses_persistent_cache_and_original_retrieval_date(tmp_path: Path) -> None:
    path = str(tmp_path / "search.db")
    with patch("tavily.TavilyClient") as provider:
        provider.return_value.search.return_value = {"results": ROWS}
        first = ToolOutput.model_validate_json(
            web_search("query", api_key="dummy", cache_path=path)
        )
        second = ToolOutput.model_validate_json(
            web_search("query", api_key="dummy", cache_path=path)
        )
    assert provider.return_value.search.call_count == 1
    assert not first.data["cache_hit"]
    assert second.data["cache_hit"]
    assert second.data["tavily_credits_upper_bound"] == 0
    assert second.items[0].retrieved_at == first.items[0].retrieved_at
    evidence = evidence_from_output(second, "task", "web_search")[0]
    assert evidence.retrieved_at == first.items[0].retrieved_at
    assert evidence.metadata["search_cache_hit"] is True
    assert b"dummy" not in Path(path).read_bytes()


@pytest.mark.parametrize("variant", ["query", "depth", "limit", "expired", "disabled"])
def test_cache_does_not_hide_changed_request_or_stale_evidence(
    tmp_path: Path, variant: str
) -> None:
    options = {"api_key": "dummy", "cache_path": str(tmp_path / "search.db")}
    with patch("tavily.TavilyClient") as provider:
        provider.return_value.search.return_value = {"results": ROWS}
        web_search("first query", **options)
        if variant == "expired":
            with sqlite3.connect(options["cache_path"]) as connection:
                connection.execute("UPDATE search_results SET saved_at = 0")
        changed = {"search_depth": "advanced"} if variant == "depth" else {}
        if variant == "limit":
            changed["max_results"] = 1
        if variant == "disabled":
            changed["cache_ttl_seconds"] = 0
        output = ToolOutput.model_validate_json(
            web_search("other query" if variant == "query" else "first query", **options, **changed)
        )
    assert provider.return_value.search.call_count == 2
    assert not output.data["cache_hit"]


def test_quota_failure_is_not_retried_or_cached_and_fallback_is_explicit(tmp_path: Path) -> None:
    from tavily.errors import UsageLimitExceededError

    with patch("tavily.TavilyClient") as provider, patch("ddgs.DDGS") as fallback:
        provider.return_value.search.side_effect = UsageLimitExceededError("private-response")
        fallback.return_value.__enter__.return_value.text.return_value = [
            {"title": "Fallback", "href": "https://example.org/b", "body": "Snippet"}
        ]
        for _ in range(2):
            output = ToolOutput.model_validate_json(
                web_search("query", api_key="dummy", cache_path=str(tmp_path / "search.db"))
            )
            assert output.status == "partial"
            assert output.items[0].metadata["provider"] == "DuckDuckGo"
            assert "usage limit" in output.warnings[0]
            assert "private-response" not in output.model_dump_json()
        assert provider.return_value.search.call_count == 2
    assert not (tmp_path / "search.db").exists()


def test_disabled_fallback_preserves_honest_error_without_extra_provider_call() -> None:
    with patch("tavily.TavilyClient") as provider, patch("ddgs.DDGS") as fallback:
        provider.return_value.search.side_effect = RuntimeError("secret")
        output = ToolOutput.model_validate_json(
            web_search("query", api_key="dummy", fallback_enabled=False)
        )
    assert output.status == "error" and not output.items
    assert "fallback disabled" in output.error
    assert provider.return_value.search.call_count == 1
    fallback.assert_not_called()


def test_cache_corruption_and_unwritable_path_do_not_break_successful_search(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.db"
    path.write_text("not a SQLite file", encoding="utf-8")
    with patch("tavily.TavilyClient") as provider:
        provider.return_value.search.return_value = {"results": ROWS}
        for target in (str(path), str(path / "child.db")):
            output = ToolOutput.model_validate_json(
                web_search("q", api_key="dummy", cache_path=target)
            )
            assert output.status == "success" and not output.data["cache_hit"]


def test_cache_bounds_entries_and_rejects_malformed_payload(tmp_path: Path) -> None:
    path = str(tmp_path / "cache.db")
    cache = SearchCache(path, 900)
    with patch("tavily.TavilyClient") as provider:
        provider.return_value.search.return_value = {"results": ROWS}
        output = ToolOutput.model_validate_json(web_search("q", api_key="dummy"))
    for i in range(514):
        cache.put(str(i), output)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM search_results").fetchone()[0] == 512
        connection.execute("UPDATE search_results SET payload = '{}' WHERE cache_key = '513'")
    assert cache.get("513") is None
    assert search_key("q", "Tavily", "basic", 5) != search_key("q", "DuckDuckGo", "basic", 5)


@pytest.mark.parametrize(
    "options",
    [
        {"search_depth": "unlimited"},
        {"search_max_results": 0},
        {"search_max_results": 11},
        {"search_cache_ttl_seconds": -1},
        {"search_cache_ttl_seconds": 86401},
    ],
)
def test_invalid_resource_settings_fail_at_startup(options: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(groq_api_key="dummy", **options)
