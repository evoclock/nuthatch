# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.state_machine.IngestOrchestrator`."""

from __future__ import annotations

from pathlib import Path

from nuthatch.corpus.layout import CorpusLayout, init_corpus
from nuthatch.ingest.manifest import IngestStatus
from nuthatch.ingest.state_machine import IngestOrchestrator


def _seed_inbox(layout: CorpusLayout, filename: str, content: bytes) -> Path:
    p = layout.inbox / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


class TestIngestInboxBasics:
    def test_empty_inbox_returns_no_results(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        results = IngestOrchestrator(layout).ingest_inbox()
        assert results == []

    def test_single_pdf_moves_to_papers(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        source = _seed_inbox(layout, "attention.pdf", b"fake-pdf-bytes-but-non-zero")

        results = IngestOrchestrator(layout).ingest_inbox()

        assert len(results) == 1
        r = results[0]
        assert r.status is IngestStatus.INGESTED
        assert r.destination is not None
        assert r.destination.parent == layout.papers
        assert r.destination.exists()
        assert not source.exists()  # moved out of inbox

    def test_unsupported_format_goes_to_quarantine(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        source = _seed_inbox(layout, "weird.xyz", b"x")

        results = IngestOrchestrator(layout).ingest_inbox()

        assert results[0].status is IngestStatus.QUARANTINED
        assert results[0].reason is not None
        assert "unsupported_format" in results[0].reason
        assert (layout.quarantine / "weird.xyz").exists()
        assert not source.exists()

    def test_empty_file_goes_to_quarantine(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "empty.pdf", b"")
        results = IngestOrchestrator(layout).ingest_inbox()
        assert results[0].status is IngestStatus.QUARANTINED
        assert results[0].reason == "empty_file"


class TestDedup:
    def test_second_run_with_same_file_is_a_noop(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "p.pdf", b"abc")
        orchestrator = IngestOrchestrator(layout)
        first = orchestrator.ingest_inbox()
        assert first[0].status is IngestStatus.INGESTED

        # Drop a fresh copy of the same content (different filename even):
        _seed_inbox(layout, "p-copy.pdf", b"abc")
        second = orchestrator.ingest_inbox()
        assert len(second) == 1
        assert second[0].status is IngestStatus.DUPLICATE

    def test_dedup_uses_content_not_filename(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "first.pdf", b"unique-bytes-1")
        IngestOrchestrator(layout).ingest_inbox()
        # Same name, different content; should be a brand-new ingest.
        _seed_inbox(layout, "first.pdf", b"unique-bytes-2")
        results = IngestOrchestrator(layout).ingest_inbox()
        assert results[0].status is IngestStatus.INGESTED


class TestRecursiveWalk:
    def test_subdirectories_are_walked(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        (layout.inbox / "arxiv").mkdir()
        (layout.inbox / "bioarxiv").mkdir()
        (layout.inbox / "arxiv" / "a.pdf").write_bytes(b"a")
        (layout.inbox / "bioarxiv" / "b.pdf").write_bytes(b"b")
        results = IngestOrchestrator(layout).ingest_inbox()
        assert len(results) == 2
        assert all(r.status is IngestStatus.INGESTED for r in results)


class TestHiddenFilesSkipped:
    def test_dotfiles_in_inbox_are_ignored(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, ".gitkeep", b"")
        _seed_inbox(layout, ".DS_Store", b"x")
        results = IngestOrchestrator(layout).ingest_inbox()
        assert results == []


class TestManifestSideEffects:
    def test_every_result_appends_a_manifest_entry(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "good.pdf", b"abc")
        _seed_inbox(layout, "bad.xyz", b"abc")
        orch = IngestOrchestrator(layout)
        orch.ingest_inbox()
        entries = list(orch.manifest.iter_entries())
        statuses = {e.status for e in entries}
        assert IngestStatus.INGESTED in statuses
        assert IngestStatus.QUARANTINED in statuses


class TestNameCollisionUnderPapers:
    def test_collision_appends_counter(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        # A file with the same target name already exists (not from ingest).
        (layout.papers / "p.pdf").write_bytes(b"existing-content")
        _seed_inbox(layout, "p.pdf", b"new-content")
        results = IngestOrchestrator(layout).ingest_inbox()
        assert results[0].status is IngestStatus.INGESTED
        assert results[0].destination is not None
        assert results[0].destination.name == "p-1.pdf"
        # Original file untouched
        assert (layout.papers / "p.pdf").read_bytes() == b"existing-content"
