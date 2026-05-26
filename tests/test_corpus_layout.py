# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.corpus.layout`."""

from __future__ import annotations

from pathlib import Path

from nuthatch.corpus.layout import (
    CORPUS_MARKER,
    CorpusLayout,
    discover_corpus_root,
    init_corpus,
)


class TestInitCorpus:
    def test_creates_marker_and_standard_subdirs(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "ml-papers")
        assert layout.kg.is_dir()
        assert layout.kg.name == CORPUS_MARKER
        for sub in (
            "inbox",
            "processed",
            "quarantine",
            "rejected",
            "cards",
            "html",
            "notes",
            "graph",
            "exports",
        ):
            assert (layout.root / sub).is_dir(), f"{sub} not created"

    def test_idempotent(self, tmp_path: Path) -> None:
        first = init_corpus(tmp_path / "c")
        # Drop a file into inbox so we can prove re-init doesn't nuke it.
        (first.inbox / "preexisting.pdf").write_bytes(b"x")
        second = init_corpus(tmp_path / "c")
        assert second.root == first.root
        assert (first.inbox / "preexisting.pdf").exists()

    def test_creates_audit_dir(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        assert layout.audit_dir.is_dir()

    def test_resolved_paths_are_absolute(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        for attr in ("kg", "inbox", "processed", "rejected", "manifest_path", "audit_dir"):
            assert getattr(layout, attr).is_absolute()


class TestCorpusLayout:
    def test_canonical_subpath_names(self, tmp_path: Path) -> None:
        layout = CorpusLayout(root=tmp_path / "c")
        assert layout.kg == tmp_path / "c" / ".kg"
        assert layout.manifest_path == tmp_path / "c" / ".kg" / "manifest.jsonl"
        assert layout.inbox == tmp_path / "c" / "inbox"
        assert layout.processed == tmp_path / "c" / "processed"
        assert layout.rejected == tmp_path / "c" / "rejected"
        assert layout.quarantine == tmp_path / "c" / "quarantine"


class TestDiscoverCorpusRoot:
    def test_finds_root_from_inside_corpus(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        found = discover_corpus_root(layout.inbox / "deep" / "nested")
        # The deep dir doesn't have to exist; discover walks up from the path.
        assert found == layout.root

    def test_finds_root_from_subdir(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        (layout.processed / "sub").mkdir()
        found = discover_corpus_root(layout.processed / "sub")
        assert found == layout.root

    def test_returns_none_when_no_marker(self, tmp_path: Path) -> None:
        # A plain directory with no .kg/ anywhere up to home.
        plain = tmp_path / "plain"
        plain.mkdir()
        assert discover_corpus_root(plain) is None
