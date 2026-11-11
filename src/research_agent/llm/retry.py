"""Retry and rate-limit handling for Groq API calls."""

from __future__ import annotations

import contextlib
import logging

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


class GroqRateLimitError(Exception):
    """Raised when Groq returns HTTP 429."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class GroqAPIError(Exception):
    """Raised for non-retryable Groq API errors."""

    pass


class MalformedStructuredOutputError(GroqAPIError):
    """Provider rejected generated structured output; one bounded repair may help."""

    def __init__(self, message: str, *, schema_issues: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.schema_issues = schema_issues or []


class MalformedToolCallError(MalformedStructuredOutputError):
    """Groq rejected malformed model-generated tool arguments; safe bounded repair is possible."""


class GroqModelUnavailableError(GroqAPIError):
    """Raised when the requested model is not available."""

    pass


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts with context."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Groq API retry attempt %d/%d: %s",
        retry_state.attempt_number,
        getattr(retry_state.retry_object.stop, "max_attempt_number", "bounded"),
        type(exception).__name__ if exception else "unknown",
    )


# Retry decorator for Groq API calls.
# Retries on rate limit errors with exponential backoff + jitter.
# Max 5 attempts, initial wait 2s, max wait 60s.
groq_retry = retry(
    retry=retry_if_exception_type(GroqRateLimitError),
    wait=wait_exponential_jitter(initial=2, max=60, jitter=2),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
    reraise=True,
)


def handle_groq_error(error: Exception) -> None:
    """Classify a Groq/LangChain error and raise the appropriate typed exception.

    This inspects the error to determine if it's retryable (rate limit)
    or fatal (auth, model not found, etc).
    """
    error_str = str(error).lower()

    # Rate limit (429)
    if "429" in error_str or "rate_limit" in error_str or "rate limit" in error_str:
        retry_after: float | None = None
        # Try to extract retry-after value
        if hasattr(error, "response"):
            response = getattr(error, "response", None)
            if response is not None and hasattr(response, "headers"):
                retry_header = response.headers.get("retry-after")
                if retry_header:
                    with contextlib.suppress(ValueError, TypeError):
                        retry_after = float(retry_header)
        raise GroqRateLimitError("Groq rate limit (429)", retry_after=retry_after) from None

    # Authentication errors
    if "401" in error_str or "authentication" in error_str or "api_key" in error_str:
        raise GroqAPIError("Authentication failed") from None

    # Model not found
    if "404" in error_str or "model_not_found" in error_str or "not found" in error_str:
        raise GroqModelUnavailableError("Model not available") from None

    # All other errors — not retryable
    raise GroqAPIError("Groq API error: request failed") from None
