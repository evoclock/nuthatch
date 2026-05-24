# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.corpus.registry`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nuthatch.corpus.registry import Registry, default_registry_path


class TestDefaultRegistryPath:
    def test_uses_xdg_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        path = default_registry_path()
        assert path == tmp_path / "nuthatch" / "registry.toml"

    def test_falls_back_to_home_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        path = default_registry_path()
        assert path == Path.home() / ".config" / "nuthatch" / "registry.toml"


class TestRoundTrip:
    def test_save_then_load_preserves_corpora(self, tmp_path: Path) -> None:
        registry = Registry()
        registry.add("ml", tmp_path / "ml-papers")
        registry.add("bio", tmp_path / "bio-genomics")
        registry.set_default("ml")

        path = tmp_path / "registry.toml"
        registry.save(path)

        loaded = Registry.load(path)
        assert loaded.names() == ["bio", "ml"]
        assert loaded.default_corpus == "ml"
        assert loaded.resolve("ml") == (tmp_path / "ml-papers").resolve()
        assert loaded.resolve("bio") == (tmp_path / "bio-genomics").resolve()


class TestLoadMissing:
    def test_returns_empty_registry(self, tmp_path: Path) -> None:
        loaded = Registry.load(tmp_path / "does-not-exist.toml")
        assert loaded.corpora == {}
        assert loaded.default_corpus is None


class TestResolve:
    def test_resolves_named_corpus(self, tmp_path: Path) -> None:
        r = Registry()
        r.add("ml", tmp_path / "ml")
        assert r.resolve("ml") == (tmp_path / "ml").resolve()

    def test_resolves_default_when_name_is_none(self, tmp_path: Path) -> None:
        r = Registry()
        r.add("ml", tmp_path / "ml")
        r.set_default("ml")
        assert r.resolve(None) == (tmp_path / "ml").resolve()

    def test_returns_none_for_unknown(self) -> None:
        r = Registry()
        assert r.resolve("nope") is None

    def test_returns_none_when_no_default_and_no_name(self) -> None:
        r = Registry()
        assert r.resolve(None) is None


class TestMutations:
    def test_add_overwrites(self, tmp_path: Path) -> None:
        r = Registry()
        r.add("ml", tmp_path / "old")
        r.add("ml", tmp_path / "new")
        assert r.resolve("ml") == (tmp_path / "new").resolve()

    def test_remove(self, tmp_path: Path) -> None:
        r = Registry()
        r.add("ml", tmp_path / "ml")
        assert r.remove("ml") is True
        assert r.resolve("ml") is None
        assert r.remove("ml") is False

    def test_remove_clears_default(self, tmp_path: Path) -> None:
        r = Registry()
        r.add("ml", tmp_path / "ml")
        r.set_default("ml")
        r.remove("ml")
        assert r.default_corpus is None

    def test_set_default_unknown_raises(self) -> None:
        r = Registry()
        with pytest.raises(KeyError):
            r.set_default("ghost")


class TestSaveCreatesDirectory:
    def test_save_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        r = Registry()
        r.add("ml", tmp_path / "ml")
        target = tmp_path / "deep" / "nested" / "registry.toml"
        r.save(target)
        assert target.is_file()
        # No leftover .tmp file.
        assert not target.with_suffix(target.suffix + ".tmp").exists()
        # Cleanup env hygiene check
        assert os.environ.get("XDG_CONFIG_HOME") != str(target.parent)
