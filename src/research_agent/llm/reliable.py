"""Bound synchronous Groq calls, repair parsed outputs, and log content-free metrics."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextvars import ContextVar
from copy import deepcopy
from functools import lru_cache
from typing import Any, Literal

from groq import APIError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError

from research_agent.graph.events import emit_event
from research_agent.llm.retry import (
    GroqAPIError,
    MalformedStructuredOutputError,
    MalformedToolCallError,
)

logger = logging.getLogger(__name__)
_active_schema: ContextVar[type[BaseModel] | None] = ContextVar(
    "active_output_schema", default=None
)


def _closed_schema(schema: type[BaseModel]) -> dict[str, Any] | None:
    """Require declared fields without silently discarding arbitrary mapping entries."""
    document = deepcopy(schema.model_json_schema())
    compatible = True

    def visit(node: Any) -> None:
        nonlocal compatible
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            if node.get("type") == "object":
                if "properties" not in node or node.get("additionalProperties") not in (
                    None,
                    False,
                ):
                    compatible = False
                    return
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            node.pop("default", None)
            for value in node.values():
                visit(value)

    visit(document)
    return document if compatible else None


@lru_cache(maxsize=16)
def _call_slots(limit: int) -> threading.BoundedSemaphore:
    """Share a process-wide call budget between roles with the same configured limit."""
    return threading.BoundedSemaphore(limit)


class _ModelPacer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_start: float | None = None

    def wait(self, interval: float) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.last_start + interval - now) if self.last_start is not None else 0
            if delay:
                emit_event("llm_wait", reason="model_call_spacing", duration_seconds=delay)
                time.sleep(delay)
            self.last_start = time.monotonic()


@lru_cache(maxsize=64)
def _model_pacer(model: str) -> _ModelPacer:
    return _ModelPacer()


class ReliableChatGroq(ChatGroq):
    """Use SDK bounded retries/Retry-After with guarded synchronous agent calls."""

    concurrency_limit: int = Field(default=2, ge=1, le=16, exclude=True)
    min_interval_seconds: float = Field(default=0, ge=0, le=60, exclude=True)
    slot_timeout: float = Field(default=30, gt=0, exclude=True)
    repair_attempts: int = Field(default=1, ge=0, le=2, exclude=True)
    supports_parallel_tools: bool = False
    agent_role: str = ""
    evidence_context_max_chars: int = Field(default=8_000, ge=1000, le=60_000, exclude=True)
    structured_output_method: Literal["function_calling", "json_schema"] = "function_calling"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Limit concurrent requests without logging prompts, bodies or credentials."""
        slots = _call_slots(self.concurrency_limit)
        if not slots.acquire(timeout=self.slot_timeout):
            raise GroqAPIError("LLM concurrency wait timed out")
        start = time.monotonic()
        try:
            if self.min_interval_seconds:
                _model_pacer(self.model_name).wait(self.min_interval_seconds)
                start = time.monotonic()
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            # Providers may return private reasoning separately from the answer.
            # It must not enter persisted graph messages or observable tool traces.
            for generation in result.generations:
                for field in ("reasoning_content", "reasoning"):
                    generation.message.additional_kwargs.pop(field, None)
            usage = (result.llm_output or {}).get("token_usage", {})
            if usage:
                emit_event(
                    "llm_call",
                    agent=self.agent_role,
                    model=self.model_name,
                    duration_seconds=time.monotonic() - start,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
            logger.info(
                "llm_call model=%s latency_seconds=%.3f total_tokens=%s",
                self.model_name,
                time.monotonic() - start,
                usage.get("total_tokens", 0),
            )
            return result
        except APIError as error:
            # Observable diagnostic metadata only. Never emit response bodies, URLs,
            # request headers or exception strings which can contain supplied text.
            emit_event(
                "llm_error",
                agent=self.agent_role,
                model=self.model_name,
                error_type=type(error).__name__,
                http_status=getattr(error, "status_code", None),
            )
            body = getattr(error, "body", None)
            if isinstance(body, dict):
                detail = body.get("error", body)
                if isinstance(detail, dict) and detail.get("code") == "tool_use_failed":
                    raise MalformedToolCallError("Model generated invalid tool arguments") from None
                if (
                    isinstance(detail, dict)
                    and detail.get("code") == "json_validate_failed"
                    and getattr(error, "status_code", None) == 400
                ):
                    issues: list[dict[str, str]] = []
                    schema = _active_schema.get()
                    rejected = detail.get("failed_generation")
                    if (
                        schema is not None
                        and isinstance(rejected, str)
                        and len(rejected) <= 200_000
                    ):
                        # Inspect only to derive schema-owned paths/types. Never accept
                        # rejected output or attach its content to exceptions/events.
                        try:
                            schema.model_validate(json.loads(rejected))
                        except ValidationError as invalid:
                            from research_agent.llm.structured import _schema_issues

                            issues = _schema_issues(invalid, schema)
                        except (ValueError, TypeError, RecursionError):
                            pass
                    raise MalformedStructuredOutputError(
                        "Model generated invalid structured JSON", schema_issues=issues
                    ) from None
            # Never relay provider bodies: they can echo prompts and supplied secrets.
            logger.warning("llm_error model=%s kind=%s", self.model_name, type(error).__name__)
            raise GroqAPIError(f"Groq request failed ({type(error).__name__})") from None
        finally:
            slots.release()

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel] | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] | None = None,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Repair schema failures using a bounded, content-free correction prompt."""
        if include_raw:
            raise ValueError("Raw model output is disabled to avoid retaining private reasoning")
        repair_attempts = kwargs.pop("repair_attempts", self.repair_attempts)
        if not isinstance(repair_attempts, int) or not 0 <= repair_attempts <= 2:
            raise ValueError("Repair attempts must be between zero and two")
        selected_method = method or self.structured_output_method
        native_schema = (
            _closed_schema(schema)
            if selected_method == "json_schema"
            and self.model_name in {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"}
            and isinstance(schema, type)
            and issubclass(schema, BaseModel)
            else None
        )
        parsed: Runnable
        if native_schema is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
            # langchain-groq 0.3.8 silently drops strict=True. Bind the wire format
            # explicitly, then retain the same local Pydantic/provenance validation.
            parsed = self.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": native_schema,
                    },
                }
            ) | PydanticOutputParser(pydantic_object=schema)
        else:
            parsed = super().with_structured_output(
                schema, method=selected_method, include_raw=False, **kwargs
            )

        def invoke(value: Any, config: Any = None) -> Any:
            messages = self._convert_input(value).to_messages()
            for attempt in range(repair_attempts + 1):
                try:
                    try:
                        token = _active_schema.set(
                            schema
                            if isinstance(schema, type) and issubclass(schema, BaseModel)
                            else None
                        )
                        try:
                            return parsed.invoke(messages, config=config)
                        finally:
                            _active_schema.reset(token)
                    except KeyError as error:
                        # LangChain 0.3's PydanticToolsParser indexes the requested
                        # schema by the model's returned function name. An unexpected
                        # name is malformed output, not an application KeyError.
                        frame = error.__traceback__
                        while frame and frame.tb_next:
                            frame = frame.tb_next
                        parser_error = bool(
                            frame
                            and frame.tb_frame.f_globals.get("__name__")
                            == "langchain_core.output_parsers.openai_tools"
                            and frame.tb_frame.f_code.co_name == "parse_result"
                        )
                        if not parser_error:
                            raise
                        raise OutputParserException("Unexpected structured function name") from None
                except (OutputParserException, ValidationError, MalformedStructuredOutputError):
                    if attempt == repair_attempts:
                        if repair_attempts == 0:
                            raise
                        raise GroqAPIError(
                            "Structured output remained invalid after bounded repair"
                        ) from None
                    logger.info(
                        "llm_schema_repair model=%s attempt=%d", self.model_name, attempt + 1
                    )
                    messages.append(
                        HumanMessage(
                            content="Return an actual data object with all required fields and valid types. Do not return a JSON Schema, field descriptions, a template, or commentary."
                        )
                    )
            raise AssertionError("Unreachable repair branch")

        return RunnableLambda(invoke)
