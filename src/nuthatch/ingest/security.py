# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Security validation for URL-based ingest.

Purpose: prevent SSRF and accidental fetches of local / private
    addresses when the corpus accepts URL-based ingest in addition
    to file-system drops. Validates a URL is safe to fetch *before*
    the fetch happens; rejects URLs pointing at loopback, link-local,
    private RFC 1918 ranges, metadata-service IPs, and non-allowed
    schemes.

Inputs: a candidate URL string. Optional allowlist of additional
    schemes (default: `http`, `https` only). Optional allowlist of
    hostnames the caller pre-approves.

Outputs: `SecurityResult(allowed, reason, resolved_ip)`.

Pattern reused from `kestrel`'s `security.py` (the URL / SSRF guard
in the external pipeline library nuthatch borrows from). nuthatch's
implementation is a fresh write shaped by that pattern: scheme
allowlist, host resolution, address-block rejection, plus an
explicit hostname allowlist for corpora that whitelist a few
trusted sources.

Assumptions: SSRF rejection is conservative — when in doubt, reject.
    A caller that needs to fetch from an internal source can pass
    `allowed_hosts` explicitly.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


_BLOCKED_HOST_LITERAL: frozenset[str] = frozenset(
    {"localhost", "ip6-localhost", "ip6-loopback", "metadata", "metadata.google.internal"}
)


@dataclass(frozen=True)
class SecurityResult:
    allowed: bool
    reason: str | None
    resolved_ip: str | None = None


def _is_blocked_address(addr: str) -> bool:
    """True if `addr` is loopback, link-local, private, or a known metadata IP."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_link_local or ip.is_private:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    # AWS / GCP / Azure metadata service IPs.
    if str(ip) in {"169.254.169.254", "fd00:ec2::254"}:
        return True
    return False


def validate_url(
    url: str,
    *,
    allowed_schemes: frozenset[str] | None = None,
    allowed_hosts: frozenset[str] | None = None,
) -> SecurityResult:
    """Decide whether `url` is safe for the ingest pipeline to fetch.

    `allowed_hosts` lets a caller whitelist a small set of trusted
    hostnames (e.g. an internal preprint server). When the host is
    in `allowed_hosts`, the IP-address block check is skipped (the
    caller is asserting they trust the address).
    """
    if not url or not isinstance(url, str):
        return SecurityResult(allowed=False, reason="empty_url")

    schemes = allowed_schemes or frozenset({"http", "https"})

    parsed = urlparse(url)
    if not parsed.scheme:
        return SecurityResult(allowed=False, reason="missing_scheme")
    if parsed.scheme.lower() not in schemes:
        return SecurityResult(
            allowed=False, reason=f"disallowed_scheme:{parsed.scheme}"
        )
    if not parsed.hostname:
        return SecurityResult(allowed=False, reason="missing_host")

    host = parsed.hostname.lower()

    if host in _BLOCKED_HOST_LITERAL:
        return SecurityResult(allowed=False, reason=f"blocked_host_literal:{host}")

    if allowed_hosts and host in {h.lower() for h in allowed_hosts}:
        return SecurityResult(allowed=True, reason=None)

    # Resolve and check.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return SecurityResult(allowed=False, reason=f"dns_failure:{exc}")
    addresses = {str(info[4][0]) for info in infos}
    for addr in addresses:
        if _is_blocked_address(addr):
            return SecurityResult(
                allowed=False,
                reason=f"blocked_ip:{addr}",
                resolved_ip=addr,
            )
    first_addr = next(iter(addresses), None)
    return SecurityResult(allowed=True, reason=None, resolved_ip=first_addr)
