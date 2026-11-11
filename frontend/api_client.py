"""HTTP-only client with bounded requests, finite event snapshots and safe errors."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any, cast
from urllib.parse import quote, urlsplit

import httpx


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, request_id: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", request_id) else ""


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Preserve complete event frames, including multiline data and reconnect IDs."""
    event = "message"
    identifier: int | None = None
    data: list[str] = []

    def frame() -> dict[str, Any]:
        try:
            payload = json.loads("\n".join(data))
        except (ValueError, TypeError):
            raise APIError("The API returned an invalid event stream.") from None
        if not isinstance(payload, dict):
            raise APIError("The API returned an invalid event stream.")
        return {"event_type": event, "event_id": identifier, "data": payload}

    for line in lines:
        if not line:
            if data:
                yield frame()
            event, identifier, data = "message", None, []
        elif not line.startswith(":"):
            key, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if key == "event":
                event = value
            elif key == "id":
                if not value.isascii() or not value.isdecimal() or len(value) > 19:
                    raise APIError("The API returned an invalid event cursor.")
                identifier = int(value)
            elif key == "data":
                data.append(value)
    if data:
        yield frame()


class APIClient:
    def __init__(
        self, base_url: str = "http://localhost:8000/api/v1", *, api_key: str = ""
    ) -> None:
        try:
            parsed = urlsplit(base_url.strip())
            _ = parsed.port
        except (ValueError, TypeError):
            raise APIError("Enter a valid HTTP or HTTPS API URL.") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise APIError("Enter an HTTP or HTTPS API URL without credentials, query or fragment.")
        self.base_url = base_url.strip().rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url + "/",
            timeout=httpx.Timeout(10, connect=3),
            follow_redirects=False,
            headers={"X-API-Key": api_key} if api_key else {},
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except (httpx.HTTPError, RuntimeError):
            raise APIError("Cannot reach the API. Check the server and API URL.") from None
        if response.is_error or response.is_redirect:
            messages = {
                401: "Backend authentication failed. Check the server configuration.",
                400: "Check the request fields and server limits.",
                404: "The run or report is not available.",
                409: "This run cannot resume with the current request. Review its status and saved plan; a new run may be required.",
                422: "Check the research query and options.",
                503: "The server is busy or unavailable. Try again shortly.",
            }
            raise APIError(
                messages.get(response.status_code, "The API could not complete the request."),
                response.status_code,
                response.headers.get("X-Request-ID", ""),
            )
        return response

    def _json(self, method: str, path: str, *, array: bool = False, **kwargs: Any) -> Any:
        try:
            value = self._request(method, path, **kwargs).json()
            if array:
                valid = isinstance(value, list) and all(isinstance(row, dict) for row in value)
            else:
                valid = isinstance(value, dict)
            if not valid:
                raise APIError("The API returned an invalid response.")
            return value
        except (ValueError, TypeError):
            raise APIError("The API returned an invalid response.") from None

    @staticmethod
    def _run_path(run_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", run_id) or run_id in {".", ".."}:
            raise APIError("Enter a valid run ID.")
        return "research/" + quote(run_id, safe="")

    def health_check(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._json("GET", "health"))

    def get_tools(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._json("GET", "tools", array=True))

    def start_research(
        self, query: str, objective: str = "", *, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._json(
                "POST",
                "research",
                json={"query": query, "objective": objective, "options": options or {}},
            ),
        )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]], self._json("GET", "research", params={"limit": limit}, array=True)
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._json("GET", self._run_path(run_id)))

    def get_report(self, run_id: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._json("GET", self._run_path(run_id) + "/report", params={"format": "json"}),
        )

    def get_sources(self, run_id: str) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]], self._json("GET", self._run_path(run_id) + "/sources", array=True)
        )

    def get_events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        response = self._request(
            "GET", self._run_path(run_id) + "/events", params={"after": after, "follow": "false"}
        )
        frames = list(parse_sse(response.iter_lines()))
        if any(row["event_type"] == "error" for row in frames):
            raise APIError("Execution events are unavailable. Refresh to retry.")
        return frames

    def resume_research(self, run_id: str, approved: bool | None = None) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._json("POST", self._run_path(run_id) + "/resume", json={"approved": approved}),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
