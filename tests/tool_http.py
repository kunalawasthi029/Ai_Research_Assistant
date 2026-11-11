"""Mock public HTTP transport without bypassing URL and redirect validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from research_agent.tools import safe_http, ssrf_guard


def reply(
    status: int = 200,
    body: str | bytes = b"",
    *,
    data: Any = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Return an unread byte stream, just like a network transport response."""
    payload = (
        json.dumps(data).encode()
        if data is not None
        else body.encode()
        if isinstance(body, str)
        else body
    )
    return httpx.Response(status, headers=headers, stream=httpx.ByteStream(payload))


def install_http(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Replace sockets, retaining the real fetch loop and address checks."""
    monkeypatch.setattr(
        ssrf_guard.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    monkeypatch.setattr(safe_http, "PinnedTransport", lambda: httpx.MockTransport(handler))
