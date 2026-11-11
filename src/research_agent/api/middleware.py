"""Request correlation and CORS using validated lifespan configuration."""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from uuid import uuid4

from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from research_agent.api.errors import error_response, generic_exception_handler

logger = logging.getLogger(__name__)


class ApiKeyMiddleware:
    """Keep health probes public; protect data and job routes when configured."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope.get("method") != "OPTIONS"
            and scope.get("path") != "/api/v1/health"
        ):
            settings = getattr(scope["app"].state, "settings", None)
            key = getattr(settings, "app_api_key", None)
            if key and key.get_secret_value():
                supplied = Headers(scope=scope).get("x-api-key", "")
                if not secrets.compare_digest(supplied.encode(), key.get_secret_value().encode()):
                    response = error_response(
                        Request(scope), 401, "Unauthorized", "Backend authentication is required."
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = (
            dict(scope.get("headers", []))
            .get(b"x-request-id", b"")
            .decode("ascii", errors="ignore")
        )
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", supplied) else str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        started = False
        start = time.monotonic()

        async def correlated(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                message["headers"] = [*headers, (b"x-request-id", request_id.encode("ascii"))]
            await send(message)

        try:
            await self.app(scope, receive, correlated)
        except Exception as error:
            if started:
                raise
            response = await generic_exception_handler(Request(scope), error)
            await response(scope, receive, correlated)
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": scope["method"],
                        "route": getattr(scope.get("route"), "path", "<unmatched>"),
                        "status": status,
                        "duration_seconds": time.monotonic() - start,
                    }
                )
            )


class ConfiguredCorsMiddleware:
    """Load origins after startup without loading credentials at module import time."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings = getattr(scope["app"].state, "settings", None)
        origins = settings.get_cors_origins() if settings else []
        cors = CORSMiddleware(
            self.app,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID", "Last-Event-ID", "X-API-Key"],
            expose_headers=["X-Request-ID"],
        )
        headers = Headers(scope=scope)
        if (
            scope["method"] == "OPTIONS"
            and "origin" in headers
            and "access-control-request-method" in headers
        ):
            response = cors.preflight_response(headers)
            if response.status_code >= 400:
                response = error_response(
                    Request(scope),
                    400,
                    "CorsRejected",
                    "Cross-origin request is not allowed.",
                    {
                        key: value
                        for key, value in response.headers.items()
                        if key not in {"content-length", "content-type"}
                    },
                )
            await response(scope, receive, send)
            return
        await cors(scope, receive, send)


class SanitizedErrorMiddleware:
    """Convert unhandled failures inside CORS so browser clients receive readable errors."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = False

        async def observed(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, observed)
        except Exception as error:
            if scope["type"] != "http" or started:
                raise
            response = await generic_exception_handler(Request(scope), error)
            await response(scope, receive, send)
