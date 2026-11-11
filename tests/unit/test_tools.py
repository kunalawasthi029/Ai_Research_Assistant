"""Unit tests for all tools (web_search, webpage_extractor, wikipedia, arxiv, source_verifier) and the registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from research_agent.tools.registry import ToolRegistry

# ──────────────────────── Tool Registry ────────────────────────


class TestToolRegistry:
    """Tests for the ToolRegistry."""

    def _make_mock_tool(self, name: str, description: str = "test tool") -> MagicMock:
        tool = MagicMock()
        tool.name = name
        tool.description = description
        return tool

    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = self._make_mock_tool("test_tool")
        registry.register(tool)
        assert registry.get_tool("test_tool") is tool

    def test_get_nonexistent_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get_tool("nonexistent")

    def test_get_all_tools(self) -> None:
        registry = ToolRegistry()
        t1 = self._make_mock_tool("tool1")
        t2 = self._make_mock_tool("tool2")
        registry.register(t1)
        registry.register(t2)
        assert len(registry.get_all_tools()) == 2

    def test_get_tool_names(self) -> None:
        registry = ToolRegistry()
        registry.register(self._make_mock_tool("alpha"))
        registry.register(self._make_mock_tool("beta"))
        assert set(registry.get_tool_names()) == {"alpha", "beta"}

    def test_call_limit_enforcement(self) -> None:
        registry = ToolRegistry()
        tool = self._make_mock_tool("limited")
        registry.register(tool, call_limit=2)

        assert registry.can_call("limited")
        registry.record_call("limited")
        registry.record_call("limited")
        assert not registry.can_call("limited")

        with pytest.raises(RuntimeError, match="exceeded"):
            registry.record_call("limited")

    def test_reset_counts(self) -> None:
        registry = ToolRegistry()
        tool = self._make_mock_tool("resettable")
        registry.register(tool, call_limit=1)
        registry.record_call("resettable")
        assert not registry.can_call("resettable")

        registry.reset_counts()
        assert registry.can_call("resettable")

    def test_get_call_count(self) -> None:
        registry = ToolRegistry()
        registry.register(self._make_mock_tool("counted"))
        registry.record_call("counted")
        registry.record_call("counted")
        assert registry.get_call_count("counted") == 2

    def test_contains(self) -> None:
        registry = ToolRegistry()
        registry.register(self._make_mock_tool("exists"))
        assert "exists" in registry
        assert "missing" not in registry

    def test_len(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(self._make_mock_tool("t1"))
        assert len(registry) == 1

    def test_get_tool_descriptions(self) -> None:
        registry = ToolRegistry()
        registry.register(self._make_mock_tool("t1", "desc1"))
        descriptions = registry.get_tool_descriptions()
        assert descriptions[0]["name"] == "t1"
        assert descriptions[0]["description"] == "desc1"


# ──────────────────────── Web Search Tool ────────────────────────
