# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from nuthatch.util.credentials import (
    CredentialError,
    CredentialErrorKind,
    KeyringCredentialReader,
)
from nuthatch.util.llm_backends import build_llm


class FakeReader:
    def __init__(self, *, failure: CredentialErrorKind | None = None) -> None:
        self.failure = failure
        self.providers: list[str] = []

    def read(self, provider: str) -> str:
        self.providers.append(provider)
        if self.failure is not None:
            raise CredentialError(provider, self.failure)
        return "test-only-credential"


class FakeChatModel:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _fake_module(name: str, class_name: str) -> ModuleType:
    module = ModuleType(name)
    setattr(module, class_name, FakeChatModel)
    return module


@pytest.mark.parametrize(
    ("backend", "module_name", "class_name"),
    [
        ("openai", "langchain_openai", "ChatOpenAI"),
        ("anthropic", "langchain_anthropic", "ChatAnthropic"),
    ],
)
def test_hosted_backends_use_injected_reader(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    module_name: str,
    class_name: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ignored-environment-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ignored-environment-value")
    monkeypatch.setitem(sys.modules, module_name, _fake_module(module_name, class_name))
    reader = FakeReader()

    model = build_llm(backend, credential_reader=reader)

    assert reader.providers == [backend]
    assert model.kwargs["api_key"] == "test-only-credential"


@pytest.mark.parametrize("kind", list(CredentialErrorKind))
def test_hosted_credential_failures_are_closed(
    monkeypatch: pytest.MonkeyPatch,
    kind: CredentialErrorKind,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        _fake_module("langchain_openai", "ChatOpenAI"),
    )
    reader = FakeReader(failure=kind)

    with pytest.raises(CredentialError) as caught:
        build_llm("openai", credential_reader=reader)

    assert caught.value.kind is kind
    assert str(caught.value) == f"openai credential error: {kind.value}"
    assert "test-only-credential" not in str(caught.value)


class FakeKeyringError(Exception):
    pass


class FakeKeyringLocked(FakeKeyringError):
    pass


def _install_fake_keyring(
    monkeypatch: pytest.MonkeyPatch,
    result: str | None | BaseException,
) -> None:
    keyring = ModuleType("keyring")
    errors = ModuleType("keyring.errors")
    errors.KeyringError = FakeKeyringError  # type: ignore[attr-defined]
    errors.KeyringLocked = FakeKeyringLocked  # type: ignore[attr-defined]

    def get_password(service: str, account: str) -> str | None:
        assert service == "nuthatch"
        assert account == "openai-api-key"
        if isinstance(result, BaseException):
            raise result
        return result

    keyring.get_password = get_password  # type: ignore[attr-defined]
    keyring.errors = errors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring)
    monkeypatch.setitem(sys.modules, "keyring.errors", errors)


@pytest.mark.parametrize(
    ("result", "kind"),
    [
        (None, CredentialErrorKind.ABSENT),
        ("   ", CredentialErrorKind.INVALID),
        (PermissionError(), CredentialErrorKind.ACCESS_DENIED),
        (FakeKeyringLocked(), CredentialErrorKind.ACCESS_DENIED),
        (FakeKeyringError(), CredentialErrorKind.STORE_UNAVAILABLE),
    ],
)
def test_keyring_reader_maps_closed_failures(
    monkeypatch: pytest.MonkeyPatch,
    result: str | None | BaseException,
    kind: CredentialErrorKind,
) -> None:
    _install_fake_keyring(monkeypatch, result)

    with pytest.raises(CredentialError) as caught:
        KeyringCredentialReader().read("openai")

    assert caught.value.kind is kind
