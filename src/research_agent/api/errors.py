"""Sanitized, uniform HTTP errors with request correlation."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from research_agent.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)


def error_response(
    request: Request, status: int, code: str, detail: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    payload = ErrorResponse(error=code, detail=detail, status_code=status, request_id=request_id)
    return JSONResponse(
        status_code=status,
        content=payload.model_dump(),
        headers={**(headers or {}), "X-Request-ID": request_id},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "api_error request_id=%s error_type=%s",
        getattr(request.state, "request_id", ""),
        type(exc).__name__,
    )
    return error_response(request, 500, "InternalError", "The request could not be completed.")


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return error_response(request, 400, "InvalidRequest", "The request could not be processed.")


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Never serialize errors(), input, context or body: Pydantic can echo supplied secrets.
    return error_response(
        request, 422, "ValidationError", "Request fields are invalid. Check the API schema."
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    codes = {
        400: "InvalidRequest",
        404: "NotFound",
        405: "MethodNotAllowed",
        409: "Conflict",
        503: "Unavailable",
        504: "Timeout",
    }
    details = {
        400: "The request or options are invalid.",
        404: "The requested resource was not found.",
        405: "The method is not allowed.",
        409: "The run is active or cannot resume with this request.",
        503: "Research capacity or storage is unavailable. Retry later.",
        504: "The request timed out.",
    }
    return error_response(
        request,
        exc.status_code,
        codes.get(exc.status_code, "RequestError"),
        details.get(exc.status_code, "The request could not be completed."),
        exc.headers,
    )
