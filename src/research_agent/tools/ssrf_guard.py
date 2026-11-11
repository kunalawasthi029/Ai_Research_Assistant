"""SSRF protection for URL fetching tools.

Blocks requests to private, loopback, and link-local IP addresses,
non-http(s) schemes, and unsafe redirects. Used by webpage_extractor
and source_verifier.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Private and reserved IP ranges that must never be accessed
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 private
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

_ALLOWED_SCHEMES = {"http", "https"}


class SSRFError(Exception):
    """Raised when a URL fails SSRF validation."""

    pass


def validate_url_scheme(url: str) -> None:
    """Reject non-http(s) schemes."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        raise SSRFError("BLOCKED: malformed URL") from None
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError("Blocked scheme: only HTTP(S) is allowed")
    if parsed.username is not None or parsed.password is not None:
        raise SSRFError("BLOCKED: URL credentials are not allowed")
    if port not in (None, 80, 443):
        raise SSRFError("BLOCKED: only ports 80 and 443 are allowed")
    if any(char.isspace() or ord(char) < 32 for char in url) or len(url) > 4096:
        raise SSRFError("BLOCKED: malformed URL")


def validate_url_host(url: str) -> str:
    """Resolve the URL's hostname and block private/reserved IPs.

    Returns the resolved IP address as a string for logging.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise SSRFError("No hostname in URL")

    # Check if hostname is a raw IP address
    try:
        ip = ipaddress.ip_address(hostname)
        _check_ip_blocked(ip, url)
        return str(ip)
    except ValueError:
        pass  # Not a raw IP — it's a hostname, resolve it

    # DNS resolution
    try:
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        raise SSRFError("DNS resolution failed") from None

    if not addr_infos:
        raise SSRFError("No DNS results")

    # Check ALL resolved IPs (not just the first)
    for addr_info in addr_infos:
        ip_str = addr_info[4][0]
        ip = ipaddress.ip_address(ip_str)
        _check_ip_blocked(ip, url)

    return str(addr_infos[0][4][0])


def _check_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, url: str) -> None:
    """Raise SSRFError if the IP falls in a blocked range."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            _check_ip_blocked(ip.ipv4_mapped, url)
        if ip.sixtofour is not None or ip.teredo is not None:
            raise SSRFError("BLOCKED: IPv6 transition address")
    if not ip.is_global or ip.is_reserved or ip.is_multicast:
        raise SSRFError("Blocked private/reserved IP")
    for network in _BLOCKED_NETWORKS:
        if ip in network:
            raise SSRFError("Blocked private/reserved IP")


def validate_url(url: str) -> str:
    """Full SSRF validation: scheme + host resolution.

    Returns the resolved IP for logging.
    """
    validate_url_scheme(url)
    resolved_ip = validate_url_host(url)
    logger.debug("SSRF check passed")
    return resolved_ip
