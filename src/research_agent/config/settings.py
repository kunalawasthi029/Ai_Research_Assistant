"""Application configuration loaded from environment variables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_source_root = Path(__file__).resolve().parents[3]
ENV_FILE = (_source_root if (_source_root / "pyproject.toml").is_file() else Path.cwd()) / ".env"


class Settings(BaseSettings):
    """Central configuration for the research agent system.

    All values come from environment variables or .env file.
    No secrets have default values — they must be explicitly set.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    # --- API Keys (no defaults — must be set) ---
    groq_api_key: SecretStr = Field(description="Groq API key for LLM access", exclude=True)
    tavily_api_key: SecretStr | None = Field(
        default=None, description="Tavily API key for web search (optional)", exclude=True
    )
    app_api_key: SecretStr | None = Field(default=None, exclude=True)
    require_api_auth: bool = False

    @model_validator(mode="after")
    def require_deployment_key(self) -> Settings:
        if self.require_api_auth and (
            not self.app_api_key or not self.app_api_key.get_secret_value().strip()
        ):
            raise ValueError("APP_API_KEY is required when REQUIRE_API_AUTH is enabled")
        return self

    # --- Per-Role Model Configuration ---
    supervisor_model: str = "qwen/qwen3.8-27b"
    planner_model: str = "qwen/qwen3.8-27b"
    researcher_model: str = "openai/gpt-oss-20b"
    analyst_model: str = "qwen/qwen3.8-27b"
    report_generator_model: str = "openai/gpt-oss-120b"
    validator_model: str = "qwen/qwen3.8-27b"
    judge_model: str = "openai/gpt-oss-20b"

    # --- LLM Tuning ---
    default_temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    evidence_context_max_chars: int = Field(default=8_000, ge=1000, le=60_000)
    llm_timeout_seconds: float = Field(default=30, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_max_concurrency: int = Field(default=2, ge=1, le=16)
    llm_min_interval_seconds: float = Field(default=0, ge=0, le=60)
    structured_repair_attempts: int = Field(default=1, ge=0, le=2)
    semantic_validation_threshold: float = Field(default=0.6, ge=0, le=1)
    semantic_criterion_threshold: float = Field(default=0.6, ge=0, le=1)

    # --- Search cost and freshness controls (owned by the server, not the model) ---
    search_depth: Literal["basic", "advanced"] = "basic"
    search_max_results: int = Field(default=5, ge=1, le=10)
    search_fallback_enabled: bool = True
    search_cache_path: str = ""  # Opt in; cached public snippets may contain sensitive queries.
    search_cache_ttl_seconds: int = Field(default=900, ge=0, le=86400)

    # --- Workflow Limits ---
    max_research_iterations: int = Field(default=3, ge=1, le=10)
    max_tool_calls_per_task: int = Field(default=5, ge=1, le=20)
    max_tool_calls_per_run: int = Field(default=100, ge=1, le=1000)
    max_validation_revisions: int = Field(default=2, ge=0, le=10)
    max_plan_revisions: int = Field(default=2, ge=0, le=10)
    graph_recursion_limit: int = Field(default=50, ge=1, le=500)
    per_tool_timeout_seconds: int = Field(default=30, ge=1, le=300)
    overall_run_timeout_seconds: int = Field(default=600, ge=1, le=3600)

    # --- Human-in-the-Loop ---
    human_approval_enabled: bool = False
    research_depth: Literal["brief", "standard", "deep"] = "standard"
    research_scope: Literal["narrow", "balanced", "broad"] = "balanced"
    source_preferences: list[Literal["web", "academic", "wikipedia"]] = Field(
        default_factory=list, max_length=3
    )

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_max_workers: int = Field(default=2, ge=1, le=8)
    api_max_queued_jobs: int = Field(default=4, ge=0, le=64)
    api_shutdown_timeout_seconds: float = Field(default=15, ge=1, le=30)
    api_event_stream_timeout_seconds: float = Field(default=600, ge=1, le=3600)
    cors_origins: str = '["http://localhost:8501"]'

    # --- Database ---
    database_path: str = "data/runs.db"
    checkpoint_db_path: str = "data/checkpoints.db"

    # --- Logging ---
    log_level: str = "INFO"

    @field_validator("tavily_api_key", "app_api_key", mode="before")
    @classmethod
    def empty_optional_key(cls, value: object) -> object:
        """Treat the blank example key as an unconfigured optional provider."""
        return None if value == "" else value

    @field_validator("groq_api_key")
    @classmethod
    def require_nonempty_key(cls, value: SecretStr) -> SecretStr:
        """Reject empty required credentials without displaying their contents."""
        if not value.get_secret_value().strip():
            raise ValueError("Groq API key is required")
        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_origins(cls, value: str) -> str:
        """Reject malformed origins and wildcard credentialed CORS configuration."""
        origins = json.loads(value)
        if not isinstance(origins, list) or not all(
            isinstance(origin, str) and origin.startswith(("http://", "https://"))
            for origin in origins
        ):
            raise ValueError("CORS origins must be a JSON list of HTTP(S) origins")
        return value

    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins from JSON string."""
        origins: list[str] = json.loads(self.cors_origins)
        return origins

    def get_model_for_role(self, role: str) -> str:
        """Return the configured model ID for a given agent role."""
        role_map = {
            "supervisor": self.supervisor_model,
            "planner": self.planner_model,
            "researcher": self.researcher_model,
            "analyst": self.analyst_model,
            "report_generator": self.report_generator_model,
            "validator": self.validator_model,
            "judge": self.judge_model,
        }
        model = role_map.get(role)
        if model is None:
            raise ValueError(f"Unknown agent role: {role!r}. Valid roles: {list(role_map.keys())}")
        return model


def get_settings(**overrides: object) -> Settings:
    """Create a Settings instance, allowing overrides for testing."""
    return Settings(**overrides)  # type: ignore[arg-type]
