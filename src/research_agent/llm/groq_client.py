"""Central Groq LLM factory using ChatGroq."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from langchain_groq import ChatGroq
from pydantic import SecretStr

from research_agent.config.settings import Settings
from research_agent.llm.model_config import ModelConfig, get_model_config
from research_agent.llm.reliable import ReliableChatGroq

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def create_llm(
    role: str,
    settings: Settings,
    **overrides: object,
) -> BaseChatModel:
    """Create a ChatGroq instance configured for the given agent role.

    This is the single entry point for LLM creation in the entire system.
    The model ID comes from Settings (environment variables), not hard-coded.

    Args:
        role: Agent role name (supervisor, planner, researcher, etc.)
        settings: Application settings with API key and model configuration.
        **overrides: Override specific ModelConfig fields (temperature, max_tokens).

    Returns:
        A configured ChatGroq instance.
    """
    model_id = settings.get_model_for_role(role)
    config = get_model_config(role, model_id, **overrides)
    if role == "researcher" and not config.supports_tool_calling:
        raise ValueError("Researcher model has no documented tool-calling capability")
    if role != "researcher" and not config.supports_structured_output:
        raise ValueError("Model has no documented structured-output capability")
    temperature = float(cast(float, overrides.get("temperature", settings.default_temperature)))
    max_tokens = min(config.max_tokens, settings.max_tokens)

    logger.info(
        "Creating LLM for role=%s model=%s temperature=%.2f max_tokens=%d",
        role,
        config.model_id,
        temperature,
        max_tokens,
    )

    llm = ReliableChatGroq(
        model=config.model_id,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        concurrency_limit=settings.llm_max_concurrency,
        min_interval_seconds=settings.llm_min_interval_seconds,
        slot_timeout=settings.llm_timeout_seconds,
        repair_attempts=settings.structured_repair_attempts,
        supports_parallel_tools=config.supports_parallel_tools,
        agent_role=role,
        structured_output_method=config.structured_output_method,
        evidence_context_max_chars=settings.evidence_context_max_chars,
        reasoning_effort=config.reasoning_effort,
    )

    return llm


def create_llm_with_config(
    config: ModelConfig,
    api_key: str,
) -> ChatGroq:
    """Create a ChatGroq instance from an explicit ModelConfig.

    Lower-level factory for cases where the caller already has a ModelConfig
    (e.g., evaluation judge with custom settings).
    """
    return ReliableChatGroq(
        model=config.model_id,
        api_key=SecretStr(api_key),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=30,
        max_retries=2,
        supports_parallel_tools=config.supports_parallel_tools,
        structured_output_method=config.structured_output_method,
        reasoning_effort=config.reasoning_effort,
    )
