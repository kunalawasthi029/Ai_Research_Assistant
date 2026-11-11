"""Check settings isolation, secret serialization and bounded model configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from research_agent.config.settings import Settings
from research_agent.llm.groq_client import create_llm
from research_agent.llm.model_config import get_model_config


def test_secret_repr_and_serialization() -> None:
    settings = Settings(groq_api_key="dummy-private-groq", tavily_api_key="dummy-private-tavily")
    assert "dummy-private" not in repr(settings)
    assert "groq_api_key" not in settings.model_dump()
    assert "tavily_api_key" not in settings.model_dump()
    assert "dummy-private" not in settings.model_dump_json()


def test_dotenv_is_ignored_by_offline_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "TAVILY_API_KEY=deliberate-dummy\nSUPERVISOR_MODEL=invalid-model\n"
    )
    monkeypatch.chdir(tmp_path)
    settings = Settings(groq_api_key="test")
    assert settings.tavily_api_key is None
    assert settings.supervisor_model == "qwen/qwen3.8-27b"


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tool_calls_per_task", 0),
        ("max_research_iterations", -1),
        ("llm_max_retries", 100),
        ("llm_timeout_seconds", 0),
        ("api_port", 70000),
        ("default_temperature", 3),
        ("cors_origins", '["*"]'),
    ],
)
def test_invalid_settings_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(groq_api_key="test", **{field: value})


def test_unknown_model_capabilities_fail_closed() -> None:
    config = get_model_config("researcher", "unverified-model")
    assert not config.supports_tool_calling
    with pytest.raises(ValueError, match="documented"):
        create_llm("researcher", Settings(groq_api_key="test", researcher_model="unverified-model"))


def test_report_completion_budget_is_separate_from_evidence_input_space() -> None:
    settings = Settings(groq_api_key="test")
    report = create_llm("report_generator", settings)
    supervisor = create_llm("supervisor", settings)
    judge = create_llm("judge", settings)
    assert report.max_tokens == 4096
    assert report.evidence_context_max_chars == 8000
    assert supervisor.max_tokens == 2048
    assert judge.max_tokens == 1024


def test_parallel_capability_is_model_specific() -> None:
    assert not get_model_config("researcher", "openai/gpt-oss-120b").supports_parallel_tools
    assert get_model_config("researcher", "llama-3.3-70b-versatile").supports_parallel_tools


def test_json_logging_redacts_credentials_and_exception_bodies() -> None:
    import json
    import logging

    from research_agent.config.logging import SafeJsonFormatter

    record = logging.LogRecord(
        "test", logging.ERROR, __file__, 1, "gsk_dummy_value api_key=private", (), None
    )
    record.research_id = "run-1"
    data = json.loads(SafeJsonFormatter().format(record))
    assert "private" not in data["message"]
    assert "gsk_dummy_value" not in data["message"]
    assert data["research_id"] == "run-1"
