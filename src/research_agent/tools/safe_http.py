"""Fetch public HTTP resources with checked redirects, pinned DNS and bounded bodies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from research_agent.tools.ssrf_guard import SSRFError, validate_url

REDIRECT_CODES = {301, 302, 303, 307, 308}


class FetchError(Exception):
    """A safe retrieval error code, never a raw provider body or credentialed URL."""


@dataclass(frozen=True)
class FetchedPage:
    """The final URL, response metadata and bounded uncompressed body bytes."""

    url: str
    status_code: int
    headers: httpx.Headers
    content: bytes


class PinnedTransport(httpx.BaseTransport):
    """Connect to a validated IP while retaining the original HTTP host and TLS name."""

    def __init__(self) -> None:
        # Pool origins become IP literals after pinning. Disable keep-alive so
        # redirects between different TLS names sharing an IP cannot reuse a
        # connection authenticated only for the first name.
        self._transport = httpx.HTTPTransport(
            retries=0, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Resolve once at the socket boundary; never let the connection re-resolve DNS."""
        ip = validate_url(str(request.url))
        headers = request.headers.copy()
        headers["host"] = request.url.netloc.decode("ascii")
        pinned = httpx.Request(
            method=request.method,
            url=request.url.copy_with(host=ip),
            headers=headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": request.url.host},
        )
        return self._transport.handle_request(pinned)

    def close(self) -> None:
        """Close pooled sockets."""
        self._transport.close()


def fetch_public(
    url: str,
    *,
    method: str = "GET",
    timeout_seconds: float = 15,
    max_bytes: int = 1_000_000,
    max_redirects: int = 3,
    retries: int = 1,
) -> FetchedPage:
    """Bound retries/redirects within one deadline and check every target before sending."""
    method = method.upper()
    deadline = time.monotonic() + timeout_seconds
    current = url
    redirects = 0
    attempts = 0
    with httpx.Client(
        transport=PinnedTransport(), trust_env=False, follow_redirects=False
    ) as client:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FetchError("Timeout: retrieval deadline exceeded")
            try:
                validate_url(current)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FetchError("Timeout: retrieval deadline exceeded")
                with client.stream(
                    method,
                    current,
                    timeout=remaining,
                    headers={
                        "User-Agent": "ResearchAgent/0.1 (research source retrieval)",
                        "Accept-Encoding": "identity",
                    },
                ) as response:
                    if response.status_code in REDIRECT_CODES:
                        if redirects >= max_redirects or not response.headers.get("location"):
                            raise FetchError("Redirect limit or missing destination")
                        target = str(response.url.join(response.headers["location"]))
                        if response.url.scheme == "https" and httpx.URL(target).scheme != "https":
                            raise SSRFError("BLOCKED: insecure redirect downgrade")
                        current = target
                        redirects += 1
                        continue
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempts < retries:
                            attempts += 1
                            # Never disregard a provider wait that exceeds this tool's deadline.
                            raw_wait = response.headers.get("retry-after", "0")
                            try:
                                wait = max(0.0, float(raw_wait))
                            except ValueError:
                                try:
                                    when = parsedate_to_datetime(raw_wait)
                                    wait = max(0.0, (when - datetime.now(UTC)).total_seconds())
                                except (ValueError, TypeError, OverflowError):
                                    wait = 0.0
                            if wait >= deadline - time.monotonic():
                                raise FetchError("Retry-After exceeds retrieval deadline")
                            if wait:
                                time.sleep(wait)
                            continue
                    response.raise_for_status()
                    if (
                        method != "HEAD"
                        and response.headers.get("content-encoding", "identity").lower()
                        != "identity"
                    ):
                        raise FetchError("Unsupported compressed response")
                    content = bytearray()
                    if method != "HEAD":
                        for chunk in response.iter_raw():
                            if time.monotonic() >= deadline:
                                raise FetchError("Timeout: retrieval deadline exceeded")
                            if len(content) + len(chunk) > max_bytes:
                                raise FetchError("Response body exceeds byte limit")
                            content.extend(chunk)
                    if time.monotonic() >= deadline:
                        raise FetchError("Timeout: retrieval deadline exceeded")
                    return FetchedPage(
                        str(response.url), response.status_code, response.headers, bytes(content)
                    )
            except SSRFError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempts >= retries:
                    raise FetchError("Timeout or network request error") from None
                attempts += 1
            except httpx.HTTPStatusError as error:
                raise FetchError(f"HTTP error {error.response.status_code}") from None
            except httpx.HTTPError:
                raise FetchError("HTTP request error") from None
