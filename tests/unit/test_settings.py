"""Unit tests for configuration settings."""

from __future__ import annotations

import pytest

from research_agent.config.settings import Settings, get_settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_creates_with_required_key(self) -> None:
        s = Settings(groq_api_key="test-key")
        assert s.groq_api_key.get_secret_value() == "test-key"

    def test_default_model_assignments(self) -> None:
        s = Settings(groq_api_key="test-key")
        assert s.supervisor_model == "qwen/qwen3.8-27b"
        assert s.planner_model == "qwen/qwen3.8-27b"
        assert s.researcher_model == "openai/gpt-oss-20b"
        assert s.report_generator_model == "openai/gpt-oss-120b"
        assert s.judge_model == "openai/gpt-oss-20b"
        assert s.validator_model == "qwen/qwen3.8-27b"
        assert s.validator_model != s.report_generator_model
        assert s.judge_model not in {s.validator_model, s.report_generator_model}

    def test_get_model_for_role(self) -> None:
        s = Settings(groq_api_key="test-key")
        assert s.get_model_for_role("supervisor") == "qwen/qwen3.8-27b"
        assert s.get_model_for_role("planner") == "qwen/qwen3.8-27b"

    def test_get_model_for_unknown_role_raises(self) -> None:
        s = Settings(groq_api_key="test-key")
        with pytest.raises(ValueError, match="Unknown agent role"):
            s.get_model_for_role("nonexistent")

    def test_default_limits(self) -> None:
        s = Settings(groq_api_key="test-key")
        assert s.max_research_iterations == 3
        assert s.max_tool_calls_per_task == 5
        assert s.max_validation_revisions == 2
        assert s.max_plan_revisions == 2
        assert s.graph_recursion_limit == 50

    def test_cors_origins_parsing(self) -> None:
        s = Settings(groq_api_key="test-key")
        origins = s.get_cors_origins()
        assert origins == ["http://localhost:8501"]

    def test_human_approval_disabled_by_default(self) -> None:
        s = Settings(groq_api_key="test-key")
        assert s.human_approval_enabled is False

    def test_get_settings_with_overrides(self) -> None:
        s = get_settings(groq_api_key="override-key", max_research_iterations=5)
        assert s.groq_api_key.get_secret_value() == "override-key"
        assert s.max_research_iterations == 5

    def test_tavily_key_optional(self) -> None:
        s = Settings(groq_api_key="test-key")
        assert s.tavily_api_key is None
