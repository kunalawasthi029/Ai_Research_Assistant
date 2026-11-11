"""Per-role model configuration with capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model deployment."""

    model_id: str
    temperature: float = 0.1
    max_tokens: int = 4096
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supports_parallel_tools: bool = False
    context_window: int = 0
    structured_output_method: Literal["function_calling", "json_schema"] = "function_calling"
    reasoning_effort: str | None = None


# Groq documentation and live model listing, checked 2026-11-10. Legacy Llama
# entries describe capabilities only; their availability is no longer assumed.
# Structured describes the configured parsing protocol, not factual accuracy or
# a guarantee that the provider will successfully produce every requested schema.
MODEL_CAPABILITIES: dict[str, tuple[bool, bool, bool, int]] = {
    "llama-3.1-8b-instant": (True, True, True, 131072),
    "llama-3.3-70b-versatile": (True, True, True, 131072),
    "openai/gpt-oss-20b": (True, True, False, 131072),
    "openai/gpt-oss-120b": (True, True, False, 131072),
    "qwen/qwen3.8-27b": (True, True, True, 131072),
}


# Default configurations per role.
# Actual model IDs come from Settings; these provide capability metadata.
ROLE_DEFAULTS: dict[str, dict[str, object]] = {
    "supervisor": {
        "temperature": 0.0,
        "max_tokens": 2048,
        "description": "Routing decisions — needs structured output, low creativity",
    },
    "planner": {
        "temperature": 0.2,
        "max_tokens": 4096,
        "description": "Query decomposition — needs structured output, moderate creativity",
    },
    "researcher": {
        "temperature": 0.1,
        "max_tokens": 2048,
        "description": "Tool selection — needs tool calling",
    },
    "analyst": {
        "temperature": 0.1,
        "max_tokens": 4096,
        "description": "Evidence synthesis — needs structured output",
    },
    "report_generator": {
        "temperature": 0.3,
        "max_tokens": 8192,
        "description": "Report writing — needs structured output, higher creativity",
    },
    "validator": {
        "temperature": 0.0,
        "max_tokens": 4096,
        "description": "Quality judgment — needs structured output, zero creativity",
    },
    "judge": {
        "temperature": 0.0,
        "max_tokens": 1024,
        "description": "Evaluation judge — independent model, zero creativity",
    },
}


def get_model_config(role: str, model_id: str, **overrides: object) -> ModelConfig:
    """Build a ModelConfig for the given role and model ID.

    Uses role-specific defaults merged with any explicit overrides.
    """
    defaults = ROLE_DEFAULTS.get(role, {})
    tool, structured, parallel, context = MODEL_CAPABILITIES.get(model_id, (False, False, False, 0))
    params = {
        "model_id": model_id,
        "temperature": defaults.get("temperature", 0.1),
        "max_tokens": defaults.get("max_tokens", 4096),
        "supports_tool_calling": tool,
        "supports_structured_output": structured,
        "supports_parallel_tools": parallel,
        "context_window": context,
        "structured_output_method": "json_schema"
        if model_id in {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"}
        else "function_calling",
        "reasoning_effort": "low"
        if model_id in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
        else None,
    }
    # Apply overrides (only known ModelConfig fields)
    for key in (
        "temperature",
        "max_tokens",
        "supports_tool_calling",
        "supports_structured_output",
        "supports_parallel_tools",
        "context_window",
        "structured_output_method",
        "reasoning_effort",
    ):
        if key in overrides:
            params[key] = overrides[key]
    return ModelConfig(**params)  # type: ignore[arg-type]
