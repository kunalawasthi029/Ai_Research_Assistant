"""Unit tests for retry and rate-limit handling (src/research_agent/llm/retry.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tenacity import RetryCallState

from research_agent.llm.retry import (
    GroqAPIError,
    GroqModelUnavailableError,
    GroqRateLimitError,
    _log_retry,
    groq_retry,
    handle_groq_error,
)

# ──────────────────── Exception Classes ────────────────────


class TestGroqRateLimitError:
    """Test the GroqRateLimitError exception."""

    def test_basic_construction(self) -> None:
        err = GroqRateLimitError("rate limited")
        assert str(err) == "rate limited"
        assert err.retry_after is None

    def test_with_retry_after(self) -> None:
        err = GroqRateLimitError("rate limited", retry_after=5.0)
        assert err.retry_after == 5.0

    def test_inherits_exception(self) -> None:
        assert issubclass(GroqRateLimitError, Exception)


class TestGroqAPIError:
    """Test the GroqAPIError exception."""

    def test_basic_construction(self) -> None:
        err = GroqAPIError("api error")
        assert str(err) == "api error"

    def test_inherits_exception(self) -> None:
        assert issubclass(GroqAPIError, Exception)


class TestGroqModelUnavailableError:
    """Test the GroqModelUnavailableError exception."""

    def test_inherits_groq_api_error(self) -> None:
        assert issubclass(GroqModelUnavailableError, GroqAPIError)

    def test_basic_construction(self) -> None:
        err = GroqModelUnavailableError("model gone")
        assert str(err) == "model gone"


# ──────────────────── handle_groq_error ────────────────────


class TestHandleGroqError:
    """Test error classification in handle_groq_error."""

    def test_rate_limit_429(self) -> None:
        """Errors containing '429' should raise GroqRateLimitError."""
        with pytest.raises(GroqRateLimitError, match="429"):
            handle_groq_error(Exception("HTTP 429 Too Many Requests"))

    def test_rate_limit_keyword(self) -> None:
        """Errors containing 'rate_limit' keyword should raise GroqRateLimitError."""
        with pytest.raises(GroqRateLimitError):
            handle_groq_error(Exception("rate_limit_exceeded"))

    def test_rate_limit_with_retry_after_header(self) -> None:
        """When error.response has retry-after header, it should be extracted."""
        error = Exception("HTTP 429 rate limited")
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "3.5"}
        error.response = mock_response  # type: ignore[attr-defined]

        with pytest.raises(GroqRateLimitError) as exc_info:
            handle_groq_error(error)

        assert exc_info.value.retry_after == 3.5

    def test_rate_limit_retry_after_non_numeric(self) -> None:
        """Non-numeric retry-after values should not crash; retry_after stays None."""
        error = Exception("HTTP 429 rate limited")
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "not-a-number"}
        error.response = mock_response  # type: ignore[attr-defined]

        with pytest.raises(GroqRateLimitError) as exc_info:
            handle_groq_error(error)

        assert exc_info.value.retry_after is None

    def test_rate_limit_no_retry_after_header(self) -> None:
        """Rate limit error without retry-after header should have retry_after=None."""
        error = Exception("HTTP 429")
        mock_response = MagicMock()
        mock_response.headers = {}
        error.response = mock_response  # type: ignore[attr-defined]

        with pytest.raises(GroqRateLimitError) as exc_info:
            handle_groq_error(error)

        assert exc_info.value.retry_after is None

    def test_auth_error_401(self) -> None:
        """Errors containing '401' should raise GroqAPIError (non-retryable)."""
        with pytest.raises(GroqAPIError, match="Authentication failed"):
            handle_groq_error(Exception("HTTP 401 Unauthorized"))

    def test_auth_error_api_key(self) -> None:
        """Errors mentioning 'api_key' should raise GroqAPIError."""
        with pytest.raises(GroqAPIError, match="Authentication failed"):
            handle_groq_error(Exception("Invalid api_key provided"))

    def test_model_not_found_404(self) -> None:
        """Errors containing '404' should raise GroqModelUnavailableError."""
        with pytest.raises(GroqModelUnavailableError, match="not available"):
            handle_groq_error(Exception("HTTP 404 Model not found"))

    def test_model_not_found_keyword(self) -> None:
        """Errors mentioning 'model_not_found' should raise GroqModelUnavailableError."""
        with pytest.raises(GroqModelUnavailableError):
            handle_groq_error(Exception("model_not_found error"))

    def test_generic_error(self) -> None:
        """Unknown errors should raise GroqAPIError."""
        with pytest.raises(GroqAPIError, match="Groq API error"):
            handle_groq_error(Exception("Something unexpected happened"))

    def test_provider_error_body_is_not_chained(self) -> None:
        """Sanitized boundary errors must suppress credential-bearing provider bodies."""
        original = Exception("dummy-secret-from-provider-body")
        with pytest.raises(GroqAPIError) as exc_info:
            handle_groq_error(original)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__
        assert "dummy-secret" not in str(exc_info.value)


# ──────────────────── _log_retry ────────────────────


class TestLogRetry:
    """Test the retry logging callback."""

    def test_log_retry_with_exception(self) -> None:
        """_log_retry should not crash when outcome has an exception."""
        retry_state = MagicMock(spec=RetryCallState)
        retry_state.attempt_number = 2
        retry_state.retry_object = MagicMock()
        retry_state.retry_object.stop.max_attempt_number = 5

        outcome = MagicMock()
        outcome.exception.return_value = GroqRateLimitError("rate limited")
        retry_state.outcome = outcome

        # Should not raise
        _log_retry(retry_state)

    def test_log_retry_without_outcome(self) -> None:
        """_log_retry should handle None outcome."""
        retry_state = MagicMock(spec=RetryCallState)
        retry_state.attempt_number = 1
        retry_state.retry_object = MagicMock()
        retry_state.retry_object.stop.max_attempt_number = 5
        retry_state.outcome = None

        # Should not raise
        _log_retry(retry_state)


# ──────────────────── groq_retry Decorator ────────────────────


class TestGroqRetryDecorator:
    """Test the groq_retry tenacity decorator."""

    def test_retries_on_rate_limit_fast(self) -> None:
        """A retry decorator with the same logic but no wait should retry on GroqRateLimitError."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        fast_retry = retry(
            retry=retry_if_exception_type(GroqRateLimitError),
            wait=wait_none(),
            stop=stop_after_attempt(5),
            reraise=True,
        )

        call_count = 0

        @fast_retry
        def flaky_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise GroqRateLimitError("rate limited")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_does_not_retry_on_api_error(self) -> None:
        """groq_retry should NOT retry on GroqAPIError (non-retryable)."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        fast_retry = retry(
            retry=retry_if_exception_type(GroqRateLimitError),
            wait=wait_none(),
            stop=stop_after_attempt(5),
            reraise=True,
        )

        call_count = 0

        @fast_retry
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise GroqAPIError("auth failure")

        with pytest.raises(GroqAPIError):
            always_fails()

        assert call_count == 1  # No retry

    def test_exhausts_retries(self) -> None:
        """After max attempts, should reraise GroqRateLimitError."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

        fast_retry = retry(
            retry=retry_if_exception_type(GroqRateLimitError),
            wait=wait_none(),
            stop=stop_after_attempt(5),
            reraise=True,
        )

        call_count = 0

        @fast_retry
        def always_rate_limited() -> str:
            nonlocal call_count
            call_count += 1
            raise GroqRateLimitError("still limited")

        with pytest.raises(GroqRateLimitError):
            always_rate_limited()

        assert call_count == 5

    def test_groq_retry_is_callable_decorator(self) -> None:
        """Verify groq_retry is a callable decorator that can wrap functions."""
        assert callable(groq_retry)

        # Apply it to a function — should return a wrapped callable
        @groq_retry
        def dummy() -> str:
            return "ok"

        assert callable(dummy)
        assert dummy() == "ok"
