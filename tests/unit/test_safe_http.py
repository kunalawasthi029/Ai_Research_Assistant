"""Probe redirect safety, public-only connections, byte limits and request retry caps."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from research_agent.tools.safe_http import FetchError, PinnedTransport, fetch_public
from research_agent.tools.ssrf_guard import SSRFError, validate_url
from tests.tool_http import install_http, reply


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::1]/",
        "http://[2002:7f00:1::]/",
        "http://224.0.0.1/",
        "http://100.64.0.1/",
        "http://[ff02::1]/",
        "http://0.0.0.0/",
        "file:///secret",
        "http://user:password@example.org/",
        "http://example.org:8080/",
        "http://example.org:bad/",
    ],
)
def test_nonpublic_unsafe_and_credentialed_urls_rejected(url: str) -> None:
    with pytest.raises(SSRFError):
        validate_url(url)


@pytest.mark.parametrize("private", ["127.0.0.1", "169.254.169.254", "[::ffff:127.0.0.1]"])
def test_unsafe_redirect_is_blocked_before_second_request(
    monkeypatch: pytest.MonkeyPatch, private: str
) -> None:
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return reply(302, headers={"location": f"https://{private}/secret"})

    install_http(monkeypatch, handler)
    with pytest.raises(SSRFError):
        fetch_public("https://example.org/start")
    assert sent == ["https://example.org/start"]


def test_relative_redirect_and_loop_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if len(paths) == 1:
            return reply(302, headers={"location": "../article"})
        return reply(body="Actual text")

    install_http(monkeypatch, handler)
    assert fetch_public("https://example.org/a/start").url == "https://example.org/article"
    assert paths == ["/a/start", "/article"]
    paths.clear()
    install_http(
        monkeypatch,
        lambda request: paths.append(request.url.path) or reply(302, headers={"location": "/loop"}),
    )
    with pytest.raises(FetchError, match="Redirect limit"):
        fetch_public("https://example.org/start", max_redirects=2)
    assert len(paths) == 3


def test_https_downgrade_is_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []
    install_http(
        monkeypatch,
        lambda request: (
            sent.append(request) or reply(302, headers={"location": "http://example.org/insecure"})
        ),
    )
    with pytest.raises(SSRFError):
        fetch_public("https://example.org/")
    assert len(sent) == 1


def test_download_limit_stops_stream_early(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stream(httpx.SyncByteStream):
        chunks_read = 0

        def __iter__(self):
            for _ in range(10):
                self.chunks_read += 1
                yield b"x" * 16_384

    stream = Stream()
    install_http(monkeypatch, lambda request: httpx.Response(200, stream=stream))
    with pytest.raises(FetchError, match="byte limit"):
        fetch_public("https://example.org/", max_bytes=100)
    assert stream.chunks_read == 1


def test_compressed_response_is_rejected_without_decompression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_http(monkeypatch, lambda request: reply(headers={"content-encoding": "gzip"}))
    with pytest.raises(FetchError, match="compressed"):
        fetch_public("https://example.org/")


@pytest.mark.parametrize("method", ["HEAD", "head"])
def test_head_checks_status_without_reading_or_decompressing_a_body(monkeypatch, method):
    class NeverRead(httpx.SyncByteStream):
        def __iter__(self):
            raise AssertionError("HEAD must not read a response body")
            yield b""

    install_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-encoding": "gzip"}, stream=NeverRead()
        ),
    )
    page = fetch_public("https://example.org/", method=method)
    assert page.status_code == 200
    assert page.content == b""


def test_retry_budget_and_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return reply(429 if len(sent) == 1 else 200, headers={"retry-after": "0.01"})

    install_http(monkeypatch, handler)
    with patch("research_agent.tools.safe_http.time.sleep") as sleep:
        fetch_public("https://example.org/", retries=1)
    sleep.assert_called_once_with(0.01)
    assert len(sent) == 2
    install_http(monkeypatch, lambda request: reply(429, headers={"retry-after": "1000"}))
    with pytest.raises(FetchError, match="Retry-After"):
        fetch_public("https://example.org/", timeout_seconds=1)


def test_network_failure_retries_once_and_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        raise httpx.ConnectError("private response body")

    install_http(monkeypatch, handler)
    with pytest.raises(FetchError) as error:
        fetch_public("https://example.org/")
    assert len(sent) == 2
    assert "private" not in str(error.value)


def test_http_date_retry_after_respects_tool_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    install_http(
        monkeypatch,
        lambda request: reply(429, headers={"retry-after": "Wed, 01 Jan 2099 00:00:00 GMT"}),
    )
    with pytest.raises(FetchError, match="Retry-After"):
        fetch_public("https://example.org/", timeout_seconds=1)


def test_mixed_public_private_dns_answers_block_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_agent.tools import ssrf_guard

    monkeypatch.setattr(
        ssrf_guard.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ],
    )
    with pytest.raises(SSRFError):
        validate_url("https://example.org/")


def test_transport_pins_ip_and_retains_http_host_and_tls_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_agent.tools import safe_http

    resolutions = []

    def resolve(url: str) -> str:
        resolutions.append(url)
        return "93.184.216.34"

    monkeypatch.setattr(safe_http, "validate_url", resolve)
    observed = []
    transport = PinnedTransport()
    transport._transport = httpx.MockTransport(lambda request: observed.append(request) or reply())
    try:
        response = transport.handle_request(
            httpx.Request("GET", "https://example.org/article", headers={"host": "attacker"})
        )
        response.close()
    finally:
        transport.close()
    assert observed[0].url.host == "93.184.216.34"
    assert observed[0].headers["host"] == "example.org"
    assert observed[0].extensions["sni_hostname"] == "example.org"
    assert resolutions == ["https://example.org/article"]
