# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-only

"""Credential-store boundary for hosted LLM providers.

The default reader delegates to the host's Python ``keyring`` backend.  Whether
that backend is usable depends on the host configuration; failures are reported
as closed error kinds rather than falling back to environment variables.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

_SERVICE = "nuthatch"
_ACCOUNTS = {"openai": "openai-api-key", "anthropic": "anthropic-api-key"}


class CredentialErrorKind(StrEnum):
    """Stable, non-secret reasons why a hosted credential cannot be used."""

    ABSENT = "absent_store_entry"
    STORE_UNAVAILABLE = "store_unavailable"
    ACCESS_DENIED = "access_denied"
    INVALID = "invalid_credential"


class CredentialError(RuntimeError):
    """Credential lookup failure that never includes the credential value."""

    _SETUP_HINT = (
        "store the key with: keyring set nuthatch <provider>-api-key "
        "(accounts: openai-api-key, anthropic-api-key)"
    )

    def __init__(self, provider: str, kind: CredentialErrorKind) -> None:
        self.provider = provider
        self.kind = kind
        message = f"{provider} credential error: {kind.value}"
        if kind is CredentialErrorKind.ABSENT:
            message = f"{message} — {self._SETUP_HINT}"
        super().__init__(message)


class CredentialReader(Protocol):
    """Minimal injectable contract for a credential-store implementation."""

    def read(self, provider: str) -> str: ...


class KeyringCredentialReader:
    """Read Nuthatch credentials through the configured Python keyring backend."""

    def read(self, provider: str) -> str:
        account = _ACCOUNTS.get(provider)
        if account is None:
            raise ValueError(f"unsupported hosted provider: {provider!r}")

        try:
            import keyring
            from keyring import errors
        except ImportError as exc:
            raise CredentialError(provider, CredentialErrorKind.STORE_UNAVAILABLE) from exc

        try:
            credential = keyring.get_password(_SERVICE, account)
        except PermissionError as exc:
            raise CredentialError(provider, CredentialErrorKind.ACCESS_DENIED) from exc
        except errors.KeyringLocked as exc:
            raise CredentialError(provider, CredentialErrorKind.ACCESS_DENIED) from exc
        except errors.KeyringError as exc:
            raise CredentialError(provider, CredentialErrorKind.STORE_UNAVAILABLE) from exc

        if credential is None:
            raise CredentialError(provider, CredentialErrorKind.ABSENT)
        if not credential.strip():
            raise CredentialError(provider, CredentialErrorKind.INVALID)
        return credential
