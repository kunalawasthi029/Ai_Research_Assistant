"""Shared pytest fixtures for the research agent test suite."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from tests.fake_llm import FakeChatModel


@pytest.fixture(autouse=True)
def isolate_offline_settings(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep offline tests independent of developer dotenv files and credentials."""
    if request.node.get_closest_marker("live"):
        return
    from research_agent.config.settings import Settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


@pytest.fixture
def fake_llm_factory():
    """Factory fixture to create FakeChatModel instances with custom responses.

    Usage:
        def test_something(fake_llm_factory):
            llm = fake_llm_factory([
                AIMessage(content="response 1"),
                AIMessage(content="response 2"),
            ])
    """

    def _create(responses: list[Any]) -> FakeChatModel:
        messages = []
        for r in responses:
            if isinstance(r, str):
                messages.append(AIMessage(content=r))
            else:
                messages.append(r)
        return FakeChatModel(responses=messages)

    return _create


@pytest.fixture
def sample_settings():
    """Create a Settings instance for testing (no real API key needed)."""
    from research_agent.config.settings import Settings

    return Settings(
        groq_api_key="test-key-not-real",
        database_path=":memory:",
        checkpoint_db_path=":memory:",
        log_level="DEBUG",
    )


@pytest.fixture
def sample_research_state():
    """Create a minimal initial research state for testing."""
    from research_agent.graph.state import create_initial_state

    return create_initial_state(
        research_id="test-run-001",
        thread_id="test-thread-001",
        query="What are the latest advances in quantum computing?",
        objective="Research recent breakthroughs in quantum computing technology",
    )


def has_groq_key() -> bool:
    """Check if a real Groq API key is available."""
    return bool(os.environ.get("GROQ_API_KEY"))
