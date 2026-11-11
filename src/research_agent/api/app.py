"""FastAPI factory with lifespan-owned bounded jobs and independent DB connections."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from research_agent.api.errors import (
    error_response,
    generic_exception_handler,
    http_exception_handler,
    validation_error_handler,
    value_error_handler,
)
from research_agent.api.jobs import JobManager
from research_agent.api.middleware import (
    ApiKeyMiddleware,
    ConfiguredCorsMiddleware,
    RequestIdMiddleware,
    SanitizedErrorMiddleware,
)
from research_agent.api.routes import router
from research_agent.api.schemas import ErrorResponse
from research_agent.config.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configured = settings or get_settings()
        app.state.settings = configured
        manager = await asyncio.to_thread(JobManager, configured)
        app.state.jobs = manager
        try:
            yield
        finally:
            await asyncio.to_thread(manager.close)
            app.state.jobs = None

    app = FastAPI(
        title="AI Research Agent",
        description="Research, reports and durable execution history",
        version="0.1.0",
        lifespan=lifespan,
        responses={
            status: {"model": ErrorResponse} for status in (400, 401, 404, 409, 422, 500, 503, 504)
        },
    )
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]

    async def storage_error(request: Request, error: Exception) -> JSONResponse:
        return error_response(
            request, 503, "Unavailable", "Research storage is unavailable. Retry later."
        )

    app.add_exception_handler(sqlite3.Error, storage_error)
    app.add_middleware(SanitizedErrorMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(ConfiguredCorsMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
