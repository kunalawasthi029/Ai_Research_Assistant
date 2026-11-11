"""Verify all six tool contracts, provider failures and retrieved-source metadata."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from research_agent.config.settings import Settings
from research_agent.tools.arxiv_search import search_arxiv
from research_agent.tools.calculator import CalculatorError, calculate, safe_calculate
from research_agent.tools.registry import ToolRegistry, create_default_registry
from research_agent.tools.results import (
    ToolOutput,
    evidence_from_output,
    safe_arguments,
)
from research_agent.tools.source_verifier import _canonicalize_url, verify_sources
from research_agent.tools.web_search import web_search
from research_agent.tools.webpage_extractor import extract_webpage, focused_excerpt
from research_agent.tools.wikipedia_search import search_wikipedia
from tests.tool_http import install_http, reply


def test_focused_excerpt_selects_later_verbatim_passages_without_inventing_text() -> None:
    text = "Document metadata and table of contents. " * 1500 + (
        "Connection setup establishes state. Reliable delivery uses retransmission. "
        "Sequence numbers preserve ordering. " * 30
    )
    excerpt, ranges = focused_excerpt(text, "connection setup reliable delivery ordering")
    assert len(excerpt) <= 1810 and ranges
    assert any(start > 10_000 for start, _ in ranges)
    assert excerpt == "\n...\n".join(text[start:end] for start, end in ranges)
    assert "Reliable delivery" in excerpt and "ordering" in excerpt


def test_unfocused_or_unmatched_document_keeps_existing_text_contract() -> None:
    text = "Unrelated document. " * 1000
    assert focused_excerpt(text, "") == (text, [])
    assert focused_excerpt(text, "retransmission sequencing") == (text, [])
    assert focused_excerpt("Short text", "short") == ("Short text", [])


def test_excerpt_diversity_keeps_a_later_setup_mechanism_among_repeated_matches() -> None:
    text = ("Reliable ordering connection details. " * 100) + ("Unrelated text. " * 1000)
    text += "Handshake establishes the protocol connection. " * 10
    excerpt, ranges = focused_excerpt(text, "reliability ordering connection setup")
    assert "Handshake" in excerpt and "Reliable" in excerpt
    assert excerpt == "\n...\n".join(text[start:end] for start, end in ranges)


@pytest.mark.parametrize(
    "arguments", [{"url": "https://example.org"}, {"url": "https://example.org", "focus": ""}]
)
def test_registry_supplies_task_focus_when_extraction_focus_is_absent_or_empty(
    arguments: dict,
) -> None:
    registry = create_default_registry()
    tool = registry.get_tool("webpage_extractor")
    mock = MagicMock(return_value=ToolOutput(status="empty").model_dump_json())
    registry._tools["webpage_extractor"] = tool.model_copy(update={"func": mock})
    registry.extraction_focus = "Connection setup and reliable ordering"
    registry.execute("webpage_extractor", arguments)
    mock.assert_called_once_with(url="https://example.org", focus=registry.extraction_focus)


@pytest.mark.parametrize(
    "name",
    [
        "web_search",
        "webpage_extractor",
        "wikipedia_search",
        "arxiv_search",
        "calculator",
        "source_verifier",
    ],
)
def test_six_tools_have_validated_argument_contracts(name: str) -> None:
    registry = create_default_registry()
    assert len(registry) == 6
    assert registry.get_tool(name).args_schema is not None
    with pytest.raises(ValidationError):
        registry.execute(name, {})
    assert registry.get_call_count(name) == 1


@pytest.mark.parametrize(
    "name,args",
    [
        ("web_search", {"query": "test"}),
        ("wikipedia_search", {"query": "test"}),
        ("arxiv_search", {"query": "test"}),
        ("calculator", {"expression": "1"}),
        ("webpage_extractor", {"url": "https://example.org"}),
        ("source_verifier", {"urls": ["https://example.org"]}),
    ],
)
def test_hallucinated_extra_arguments_rejected_before_execution(name: str, args: dict) -> None:
    registry = create_default_registry()
    with pytest.raises(ValidationError):
        registry.execute(name, {**args, "unexpected": "ignored-before-now"})
    assert registry.get_call_count(name) == 1


def test_web_search_keeps_separate_real_urls() -> None:
    rows = [
        {"title": "First", "href": "https://example.org/one", "body": "One"},
        {"title": "Second", "href": "https://example.org/two", "body": "Two"},
    ]
    with patch("ddgs.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.return_value = rows
        result = ToolOutput.model_validate_json(web_search("query", timeout_seconds=7))
        ddgs.assert_called_once_with(timeout=7)
    evidence = evidence_from_output(result, "task-1", "web_search")
    assert [item.url for item in evidence] == [row["href"] for row in rows]
    assert [item.title for item in evidence] == ["First", "Second"]
    assert evidence[0].metadata["provider"] == "DuckDuckGo"
    assert evidence[0].retrieved_at.tzinfo is not None
    assert evidence[0].metadata["snippet"] == "One"


def test_ddg_failure_does_not_become_evidence() -> None:
    with patch("ddgs.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.side_effect = RuntimeError(
            "dummy-secret-body"
        )
        result = ToolOutput.model_validate_json(web_search("query"))
    assert result.status == "error"
    assert evidence_from_output(result, "task", "web_search") == []
    assert "dummy-secret" not in result.model_dump_json()


def test_tavily_and_fallback_report_actual_provider() -> None:
    with patch("tavily.TavilyClient") as tavily:
        tavily.return_value.search.return_value = {
            "results": [{"title": "Tavily", "url": "https://example.org/a", "content": "Snippet"}]
        }
        result = ToolOutput.model_validate_json(
            web_search("query", api_key="dummy", timeout_seconds=6)
        )
        assert result.items[0].metadata["provider"] == "Tavily"
        tavily.return_value.search.assert_called_once_with(
            query="query",
            max_results=5,
            timeout=6,
            search_depth="basic",
            auto_parameters=False,
            include_answer=False,
            include_raw_content=False,
        )
    with (
        patch("tavily.TavilyClient", side_effect=RuntimeError("private")),
        patch("ddgs.DDGS") as ddgs,
    ):
        ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Fallback", "href": "https://example.org/b", "body": "Fallback text"}
        ]
        result = ToolOutput.model_validate_json(web_search("query", api_key="dummy"))
        assert result.items[0].metadata["provider"] == "DuckDuckGo"
        assert result.status == "partial"
        assert "Tavily" in result.warnings[0]


def test_missing_tavily_module_falls_back() -> None:
    with patch.dict("sys.modules", {"tavily": None}), patch("ddgs.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.return_value = []
        assert (
            ToolOutput.model_validate_json(web_search("query", api_key="dummy")).status == "empty"
        )


def test_empty_and_malformed_search_results() -> None:
    with patch("research_agent.tools.web_search._search_duckduckgo", return_value=[]):
        assert ToolOutput.model_validate_json(web_search("query")).status == "empty"
    with patch(
        "research_agent.tools.web_search._search_duckduckgo",
        return_value=[
            {"title": "Invalid", "url": "search://fake", "snippet": "Do not cite"},
            {"title": "Valid", "url": "https://example.org/", "snippet": "Actual snippet"},
        ],
    ):
        result = ToolOutput.model_validate_json(web_search("query"))
        assert result.status == "partial"
        assert len(result.items) == 1


def test_page_extraction_preserves_final_url_and_removes_scripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return reply(302, headers={"location": "/article"})
        return reply(
            body="<title>Article</title><script>malicious()</script><p>Retrieved fact</p>",
            headers={"content-type": "text/html"},
        )

    install_http(monkeypatch, handler)
    result = ToolOutput.model_validate_json(extract_webpage("https://example.org/old"))
    assert result.items[0].url == "https://example.org/article"
    assert result.items[0].metadata["requested_url"] == "https://example.org/old"
    assert "Retrieved fact" in result.items[0].content
    assert "malicious" not in result.items[0].content


@pytest.mark.parametrize(
    "status,body,headers,error",
    [
        (200, b"binary", {"content-type": "image/png"}, "Non-text"),
        (403, b"private provider body", {}, "403"),
        (200, b"", {"content-type": "text/plain"}, None),
    ],
)
def test_page_error_and_empty_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: bytes,
    headers: dict[str, str],
    error: str | None,
) -> None:
    install_http(monkeypatch, lambda request: reply(status, body, headers=headers))
    result = ToolOutput.model_validate_json(extract_webpage("https://example.org/"))
    assert result.status == ("error" if error else "empty")
    assert not result.items
    assert "private provider body" not in result.model_dump_json()
    if error:
        assert error in result.error


def test_page_text_cap_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    install_http(
        monkeypatch,
        lambda request: reply(body="X" * 150_000, headers={"content-type": "text/plain"}),
    )
    result = ToolOutput.model_validate_json(extract_webpage("https://example.org/"))
    assert len(result.items[0].content) == 100_000
    assert result.items[0].metadata["content_truncated"]


def test_wikipedia_extracts_have_real_article_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["generator"] == "search"
        assert request.url.params["explaintext"] == "1"
        return reply(
            data={
                "query": {
                    "pages": [
                        {
                            "title": "Quantum computing",
                            "fullurl": "https://en.wikipedia.org/wiki/Quantum_computing",
                            "extract": "A" * 15_000,
                        }
                    ]
                }
            }
        )

    install_http(monkeypatch, handler)
    result = ToolOutput.model_validate_json(search_wikipedia("quantum"))
    assert result.items[0].source_type == "wikipedia"
    assert len(result.items[0].content) == 10_000
    assert result.items[0].metadata["content_truncated"]


@pytest.mark.parametrize(
    "body,status",
    [
        ({}, "empty"),
        ({"error": {"info": "provider-body"}}, "error"),
        ({"query": {"pages": [{"missing": True}]}}, "empty"),
    ],
)
def test_wikipedia_empty_missing_and_error(
    monkeypatch: pytest.MonkeyPatch, body: dict, status: str
) -> None:
    install_http(monkeypatch, lambda request: reply(data=body))
    result = ToolOutput.model_validate_json(search_wikipedia("query"))
    assert result.status == status
    assert not result.items


def test_arxiv_metadata_and_normalized_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1234.5678</id><title> Paper title </title><summary>Evidence\n abstract</summary><published>2026-01-01T00:00:00Z</published><author><name>Author One</name></author></entry></feed>'
    install_http(monkeypatch, lambda request: reply(body=xml))
    result = ToolOutput.model_validate_json(search_arxiv("test"))
    assert result.items[0].url == "https://arxiv.org/abs/1234.5678"
    assert result.items[0].metadata["authors"] == ["Author One"]
    assert result.items[0].content == "Evidence abstract"


@pytest.mark.parametrize(
    "xml,status",
    [
        ('<!DOCTYPE feed [<!ENTITY x "bad">]><feed/>', "error"),
        ("<not-valid", "error"),
        ('<feed xmlns="http://www.w3.org/2005/Atom"/>', "empty"),
        (
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/api/errors#bad</id></entry></feed>',
            "error",
        ),
    ],
)
def test_arxiv_unsafe_invalid_empty_or_provider_error(
    monkeypatch: pytest.MonkeyPatch, xml: str, status: str
) -> None:
    install_http(monkeypatch, lambda request: reply(body=xml))
    assert ToolOutput.model_validate_json(search_arxiv("test")).status == status


def test_verifier_duplicate_retains_original_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "HEAD"
        return reply(404)

    install_http(monkeypatch, handler)
    result = ToolOutput.model_validate_json(
        verify_sources(["https://example.org/a", "https://example.org/a#section"])
    )
    first, duplicate = result.data["verifications"]
    assert not first["is_accessible"]
    assert duplicate["is_duplicate"] and not duplicate["is_accessible"]
    assert duplicate["error"] == first["error"]
    assert len(requests) == 1
    assert not result.items


def test_verifier_success_and_canonicalization(monkeypatch: pytest.MonkeyPatch) -> None:
    install_http(monkeypatch, lambda request: reply())
    result = ToolOutput.model_validate_json(verify_sources(["https://example.org/a"]))
    assert result.data["verifications"][0]["is_accessible"]
    assert _canonicalize_url("HTTPS://EXAMPLE.ORG:443/a#b") == "https://example.org/a"
    assert _canonicalize_url("https://example.org/a/") != _canonicalize_url("https://example.org/a")
    assert _canonicalize_url("https://example.org/?a=1") != _canonicalize_url(
        "https://example.org/?a=2"
    )


@pytest.mark.parametrize(
    "expression", ["((2**1000)**1000)", "1e309", "True", "1j", "round(2, ndigits=3)", "[1, 2]"]
)
def test_calculator_bounds_intermediates_and_result_types(expression: str) -> None:
    with pytest.raises(CalculatorError):
        safe_calculate(expression)


def test_calculator_is_auxiliary_data_not_a_web_source() -> None:
    result = ToolOutput.model_validate_json(calculate("sqrt(144) + 2**3"))
    assert result.data["value"] == 20
    assert not evidence_from_output(result, "task", "calculator")


def test_failed_invocations_consume_per_tool_and_global_budgets() -> None:
    registry = ToolRegistry(default_call_limit=3, max_total_calls=1)
    tool = MagicMock(name="tool")
    tool.name = "failing"
    tool.invoke.side_effect = RuntimeError("private")
    registry.register(tool)
    with pytest.raises(RuntimeError):
        registry.execute("failing", {})
    assert registry.get_call_count("failing") == 1
    assert not registry.can_call("failing")
    with pytest.raises(RuntimeError, match="limit"):
        registry.execute("failing", {})
    assert tool.invoke.call_count == 1


def test_settings_configure_timeout_credentials_and_run_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        groq_api_key="test",
        tavily_api_key="dummy-private",
        per_tool_timeout_seconds=7,
        max_tool_calls_per_run=1,
        max_tool_calls_per_task=2,
    )
    registry = create_default_registry(settings=settings)
    assert registry.max_task_calls == 2
    with patch("research_agent.tools.web_search._search_tavily", return_value=[]) as search:
        registry.execute("web_search", {"query": "test"})
        search.assert_called_once_with("test", 5, "dummy-private", 7, search_depth="basic")
    assert not registry.can_call("calculator")


def test_argument_trace_redacts_secrets_without_changing_tool_inputs() -> None:
    original = {
        "api_key": "private",
        "url": "https://example.org/?token=private",
        "nested": {"password": "private"},
        "count": 3,
    }
    safe = safe_arguments(original)
    assert "private" not in str(safe)
    assert original["api_key"] == "private"
    assert safe["count"] == 3
