# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Tests for `nuthatch.ingest.security.validate_url` (SSRF / URL guard)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nuthatch.ingest.security import validate_url


class TestSchemeRejection:
    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://example.com/x", "gopher://x.example.com"],
    )
    def test_disallowed_scheme(self, url: str) -> None:
        r = validate_url(url)
        assert not r.allowed
        assert r.reason is not None
        assert "disallowed_scheme" in r.reason

    def test_empty(self) -> None:
        assert not validate_url("").allowed

    def test_missing_scheme(self) -> None:
        assert not validate_url("example.com/x").allowed


class TestHostLiteralRejection:
    @pytest.mark.parametrize("host", ["localhost", "ip6-localhost", "metadata"])
    def test_blocked_host_literal(self, host: str) -> None:
        r = validate_url(f"http://{host}/x")
        assert not r.allowed
        assert r.reason is not None
        assert "blocked_host_literal" in r.reason


class TestPrivateAddressRejection:
    @pytest.mark.parametrize(
        "addr",
        ["127.0.0.1", "192.168.1.10", "10.0.0.5", "169.254.169.254"],
    )
    def test_private_address_blocked(self, addr: str) -> None:
        with patch(
            "nuthatch.ingest.security.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", (addr, 0))],
        ):
            r = validate_url("http://target.example.com/x")
            assert not r.allowed
            assert r.reason is not None
            assert "blocked_ip" in r.reason
            assert r.resolved_ip == addr

    def test_public_address_passes(self) -> None:
        with patch(
            "nuthatch.ingest.security.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("8.8.8.8", 0))],
        ):
            r = validate_url("https://example.com/x")
            assert r.allowed
            assert r.resolved_ip == "8.8.8.8"


class TestAllowedHostsBypass:
    def test_allowed_host_skips_ip_check(self) -> None:
        # Even if it resolves to a private address, allowed_hosts wins.
        with patch(
            "nuthatch.ingest.security.socket.getaddrinfo",
            return_value=[(2, 1, 0, "", ("192.168.1.10", 0))],
        ):
            r = validate_url(
                "http://internal-server.example/x",
                allowed_hosts=frozenset({"internal-server.example"}),
            )
            assert r.allowed
