"""Opt-in live role capability checks using configured Groq models and safe errors."""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from research_agent.config.settings import Settings
from research_agent.llm.groq_client import create_llm

pytestmark = pytest.mark.live


def live_settings() -> Settings:
    """Require an explicitly exported key; never load developer dotenv implicitly."""
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("Export GROQ_API_KEY explicitly for live smoke tests")
    return Settings(_env_file=None, llm_timeout_seconds=20, llm_max_retries=0)


class Answer(BaseModel):
    """Minimal schema to test configured structured-output roles."""

    answer: int


@pytest.mark.parametrize(
    "role", ["supervisor", "planner", "analyst", "report_generator", "validator", "judge"]
)
def test_configured_role_structured_output(role: str) -> None:
    llm = create_llm(role, live_settings(), max_tokens=256)
    result = llm.with_structured_output(Answer).invoke(
        "What is 2 + 2? Return answer as an integer."
    )
    assert isinstance(result, Answer)
    assert result.answer == 4


def test_researcher_executes_tool_and_sends_result_back() -> None:
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two integers exactly."""
        return a + b

    llm = create_llm("researcher", live_settings(), max_tokens=256)
    bound = llm.bind_tools([add_numbers])
    messages = [HumanMessage(content="Use add_numbers to calculate 15 + 27.")]
    response = bound.invoke(messages)
    assert response.tool_calls
    results = []
    for call in response.tool_calls:
        assert call["name"] == "add_numbers"
        value = add_numbers.invoke(call["args"])
        assert value == 42
        results.append(ToolMessage(content=str(value), tool_call_id=call["id"]))
    final = llm.invoke([*messages, response, *results])
    assert "42" in str(final.content)
