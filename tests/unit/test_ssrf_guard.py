"""Unit tests for the SSRF guard."""

from __future__ import annotations

import pytest

from research_agent.tools.ssrf_guard import (
    SSRFError,
    validate_url,
    validate_url_host,
    validate_url_scheme,
)


class TestValidateUrlScheme:
    """Tests for scheme validation."""

    def test_http_allowed(self) -> None:
        validate_url_scheme("http://example.com")

    def test_https_allowed(self) -> None:
        validate_url_scheme("https://example.com")

    def test_ftp_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url_scheme("ftp://example.com/file")

    def test_file_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url_scheme("file:///etc/passwd")

    def test_javascript_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url_scheme("javascript:alert(1)")

    def test_data_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url_scheme("data:text/html,<h1>test</h1>")

    def test_empty_scheme_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url_scheme("://example.com")


class TestValidateUrlHost:
    """Tests for host/IP validation."""

    def test_loopback_ipv4_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://127.0.0.1/admin")

    def test_loopback_127_x_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://127.0.0.2/")

    def test_private_10_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://10.0.0.1/internal")

    def test_private_172_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://172.16.0.1/")

    def test_private_192_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://192.168.1.1/")

    def test_link_local_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://169.254.169.254/metadata")

    def test_ipv6_loopback_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://[::1]/")

    def test_no_hostname_blocked(self) -> None:
        with pytest.raises(SSRFError, match="No hostname"):
            validate_url_host("http:///path")

    def test_zero_ip_blocked(self) -> None:
        with pytest.raises(SSRFError, match="Blocked private"):
            validate_url_host("http://0.0.0.0/")


class TestValidateUrl:
    """Integration tests for full URL validation."""

    def test_valid_public_url(self) -> None:
        # This should not raise (example.com resolves to public IP)
        # Note: requires DNS resolution, so may fail without network
        try:
            result = validate_url("https://example.com")
            assert result  # Returns resolved IP
        except SSRFError:
            pytest.skip("DNS resolution failed — likely no network in sandbox")

    def test_private_ip_with_https(self) -> None:
        """SSRF check should block private IPs even with HTTPS."""
        with pytest.raises(SSRFError):
            validate_url("https://192.168.1.1/admin")

    def test_ftp_with_public_host(self) -> None:
        """Non-http scheme blocked before host check."""
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url("ftp://example.com/file")
