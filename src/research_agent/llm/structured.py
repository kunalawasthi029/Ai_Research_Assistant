"""Validate agent output contracts with one bounded, secret-safe repair budget."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar, get_args

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorType

from research_agent.graph.events import emit_event
from research_agent.llm.retry import MalformedStructuredOutputError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class AgentOutputError(Exception):
    """An agent exhausted its output-repair budget; no invalid result may escape."""


class ContractError(ValueError):
    """A fixed application-authored contract issue safe to feed back to the model."""


def _schema_issues(error: Exception, schema: type[BaseModel]) -> list[dict[str, str]]:
    """Only schema-owned field names/type codes, never invalid values or error messages."""
    if isinstance(error, MalformedStructuredOutputError) and error.schema_issues:
        return error.schema_issues
    validation = error if isinstance(error, ValidationError) else error.__cause__
    if not isinstance(validation, ValidationError):
        return []
    document = schema.model_json_schema()
    names = set(document.get("properties", {}))
    for definition in document.get("$defs", {}).values():
        names.update(definition.get("properties", {}))
    codes = set(get_args(ErrorType))
    issues = []
    for item in validation.errors(include_url=False, include_context=False, include_input=False)[
        :8
    ]:
        path = ".".join(
            str(part) if isinstance(part, int) or part in names else "<field>"
            for part in item["loc"][:8]
        )
        issues.append(
            {
                "field": path or "$",
                "type": item["type"] if item["type"] in codes else "validation_error",
            }
        )
    return issues


def invoke_structured(
    llm: Any,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    validate: Callable[[T], T] | None = None,
) -> T:
    """Repair parsing and provenance together, without multiplying factory repair loops."""
    attempts = min(2, max(0, int(getattr(llm, "repair_attempts", 1))))
    # ReliableChatGroq honors this override; the shared agent boundary owns repairs.
    structured = llm.with_structured_output(schema, repair_attempts=0)
    conversation = list(messages)
    if (
        conversation
        and isinstance(conversation[0], SystemMessage)
        and isinstance(conversation[0].content, str)
    ):
        conversation[0] = conversation[0].model_copy(
            update={
                "content": conversation[0].content
                + (
                    f"\nOutput contract: return one actual {schema.__name__} JSON data object, "
                    "not a JSON Schema or field descriptions. Populate its declared fields with "
                    "the requested data and valid types; include null for nullable fields when unused. "
                    "Do not wrap the object in another object or include private reasoning."
                )
            }
        )
    for attempt in range(attempts + 1):
        try:
            raw = structured.invoke(conversation)
            result = schema.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
            return validate(result) if validate else result
        except (
            OutputParserException,
            ValidationError,
            ValueError,
            MalformedStructuredOutputError,
        ) as error:
            fields = _schema_issues(error, schema)
            issue_kind = "malformed_structured_output"
            if isinstance(error, ContractError):
                issue_kind = "provenance_contract"
                for prefix, kind in (
                    ("Generated research reports require", "missing_references"),
                    ("Every reference must identify", "unknown_reference_evidence"),
                    ("Reference URL/title/type", "reference_metadata_mismatch"),
                    ("Reference IDs and referenced evidence", "duplicate_references"),
                    ("Section evidence IDs", "section_reference_mismatch"),
                    ("Inline citations must", "inline_citation_mismatch"),
                ):
                    if str(error).startswith(prefix):
                        issue_kind = kind
                        break
            elif isinstance(error, MalformedStructuredOutputError):
                issue_kind = "provider_schema_rejection"
            elif isinstance(error, ValidationError) or isinstance(error.__cause__, ValidationError):
                issue_kind = "schema_validation"
            emit_event(
                "agent_output_invalid",
                schema=schema.__name__,
                issue_kind=issue_kind,
                attempt=attempt + 1,
                allowed_attempts=attempts + 1,
                schema_issues=fields,
            )
            if attempt == attempts:
                raise AgentOutputError(
                    f"{schema.__name__} failed validation after bounded repair"
                ) from None
            detail = (
                str(error)
                if isinstance(error, ContractError)
                else (
                    f"Return an actual {schema.__name__} data object with all required fields "
                    "and valid types. Do not return a JSON Schema, field descriptions, or a template."
                )
            )
            if getattr(llm, "structured_output_method", None) == "function_calling":
                detail += f" Use the exact function name {schema.__name__}."
            if fields:
                detail += " Fix these schema field/type errors: " + json.dumps(fields)
            logger.info("agent_output_repair schema=%s attempt=%d", schema.__name__, attempt + 1)
            conversation.append(
                HumanMessage(
                    content=f"Output repair required: {detail} Correct your output using only the supplied data."
                )
            )
    raise AssertionError("Unreachable repair branch")
