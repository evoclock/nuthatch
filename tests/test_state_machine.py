# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Tests for `nuthatch.ingest.state_machine.IngestOrchestrator` (Sprint 2 pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

from nuthatch.corpus.layout import CorpusLayout, init_corpus
from nuthatch.ingest.manifest import IngestStatus
from nuthatch.ingest.state_machine import IngestOrchestrator
from nuthatch.schema.profile import FieldSpec, SchemaProfile


class _AcceptAnyProfile(SchemaProfile):
    """Test profile: passes any non-empty document."""

    profile_name = "test_accept_any"
    fields = (FieldSpec("title", required=True, expected_type=str),)


_DEFAULT_TEST_MARKDOWN = (
    "# Sample title\n\n" + "Body paragraph with enough text to clear the qc.check_extract_yield "
    "floor of 200 chars. Lorem ipsum dolor sit amet consectetur adipiscing "
    "elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n"
)


def _make_extractor(text: str = _DEFAULT_TEST_MARKDOWN):
    """Return a callable that yields fixed markdown regardless of input path."""

    def _extract(path: Path) -> str:
        return text

    return _extract


def _seed_inbox(layout: CorpusLayout, filename: str, content: bytes) -> Path:
    p = layout.inbox / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _make_orchestrator(
    layout: CorpusLayout,
    *,
    extractor=None,
    text: str = _DEFAULT_TEST_MARKDOWN,
) -> IngestOrchestrator:
    return IngestOrchestrator(
        layout,
        extractor=extractor or _make_extractor(text),
        schema_profile=_AcceptAnyProfile,
    )


class TestIngestInboxBasics:
    def test_empty_inbox_returns_no_results(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        results = _make_orchestrator(layout).ingest_inbox()
        assert results == []

    def test_single_pdf_moves_to_processed_mirroring_source_subdir(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        source = _seed_inbox(layout, "attention.pdf", b"fake-pdf-bytes")

        results = _make_orchestrator(layout).ingest_inbox()

        assert len(results) == 1
        r = results[0]
        assert r.status is IngestStatus.INGESTED
        assert r.destination is not None
        # Source provenance preserved: inbox/foo.pdf -> processed/inbox/foo.pdf.
        assert r.destination == layout.processed / "inbox" / "attention.pdf"
        assert r.destination.exists()
        assert not source.exists()  # moved out of inbox

    def test_unsupported_format_goes_to_quarantine(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "weird.xyz", b"x")

        results = _make_orchestrator(layout).ingest_inbox()

        assert results[0].status is IngestStatus.QUARANTINED
        assert results[0].reason is not None
        assert "unsupported_format" in results[0].reason
        # New quarantine layout: <quarantine>/<reason-slug>/<file>
        assert any(p.name == "weird.xyz" for p in layout.quarantine.rglob("weird.xyz"))

    def test_empty_file_goes_to_quarantine(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "empty.pdf", b"")
        results = _make_orchestrator(layout).ingest_inbox()
        assert results[0].status is IngestStatus.QUARANTINED
        assert results[0].reason == "empty_file"


class TestQuarantineSidecar:
    def test_quarantine_writes_reason_sidecar(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "bad.xyz", b"content")
        _make_orchestrator(layout).ingest_inbox()
        sidecars = list(layout.quarantine.rglob("*.reason.json"))
        assert len(sidecars) == 1
        payload = json.loads(sidecars[0].read_text())
        assert payload["original_filename"] == "bad.xyz"
        assert "unsupported_format" in payload["reason"]
        assert "stage" in payload["details"]


class TestSchemaGate:
    def test_extractor_yields_too_little_text_quarantines(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "thin.pdf", b"non-zero")
        # Extractor returns near-empty markdown → qc fails.
        results = _make_orchestrator(layout, text="x").ingest_inbox()
        assert results[0].status is IngestStatus.QUARANTINED
        assert results[0].reason is not None
        assert "extract_yield_too_low" in results[0].reason

    def test_schema_failure_quarantines(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "headless.pdf", b"non-zero")
        # Extractor returns markdown with no `# Title` heading → schema fails.
        body = "Lorem ipsum " * 50
        results = _make_orchestrator(layout, text=body).ingest_inbox()
        assert results[0].status is IngestStatus.QUARANTINED
        assert results[0].reason is not None
        assert "missing" in results[0].reason

    def test_extractor_crash_marks_failed_not_quarantine(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "p.pdf", b"non-zero")

        def boom(_path: Path) -> str:
            raise RuntimeError("OCR died")

        results = IngestOrchestrator(
            layout, extractor=boom, schema_profile=_AcceptAnyProfile
        ).ingest_inbox()
        assert results[0].status is IngestStatus.FAILED
        assert results[0].reason is not None
        assert "extract_error" in results[0].reason
        # File NOT moved; user should retry, not lose it.
        assert (layout.inbox / "p.pdf").exists()


class TestDedup:
    def test_second_run_with_same_file_is_a_noop(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "p.pdf", b"abc")
        orchestrator = _make_orchestrator(layout)
        first = orchestrator.ingest_inbox()
        assert first[0].status is IngestStatus.INGESTED

        _seed_inbox(layout, "p-copy.pdf", b"abc")
        second = orchestrator.ingest_inbox()
        assert len(second) == 1
        assert second[0].status is IngestStatus.DUPLICATE

    def test_dedup_uses_content_not_filename(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "first.pdf", b"unique-bytes-1")
        _make_orchestrator(layout).ingest_inbox()
        _seed_inbox(layout, "first.pdf", b"unique-bytes-2")
        results = _make_orchestrator(layout).ingest_inbox()
        assert results[0].status is IngestStatus.INGESTED


class TestRecursiveWalk:
    def test_subdirectories_are_walked(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        (layout.inbox / "arxiv").mkdir()
        (layout.inbox / "bioarxiv").mkdir()
        (layout.inbox / "arxiv" / "a.pdf").write_bytes(b"a")
        (layout.inbox / "bioarxiv" / "b.pdf").write_bytes(b"b")
        results = _make_orchestrator(layout).ingest_inbox()
        assert len(results) == 2
        assert all(r.status is IngestStatus.INGESTED for r in results)


class TestHiddenFilesSkipped:
    def test_dotfiles_in_inbox_are_ignored(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, ".gitkeep", b"")
        _seed_inbox(layout, ".DS_Store", b"x")
        results = _make_orchestrator(layout).ingest_inbox()
        assert results == []


class TestManifestSideEffects:
    def test_every_result_appends_a_manifest_entry(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "good.pdf", b"abc")
        _seed_inbox(layout, "bad.xyz", b"abc")
        orch = _make_orchestrator(layout)
        orch.ingest_inbox()
        entries = list(orch.manifest.iter_entries())
        statuses = {e.status for e in entries}
        assert IngestStatus.INGESTED in statuses
        assert IngestStatus.QUARANTINED in statuses


class TestNameCollisionUnderProcessed:
    def test_collision_appends_counter(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        # Pre-seed a processed file at the path the inbox file will
        # map to (processed/inbox/p.pdf, mirroring the source subdir).
        target = layout.processed / "inbox" / "p.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing-content")
        _seed_inbox(layout, "p.pdf", b"new-content")
        results = _make_orchestrator(layout).ingest_inbox()
        assert results[0].status is IngestStatus.INGESTED
        assert results[0].destination is not None
        assert results[0].destination.name == "p-1.pdf"
        assert target.read_bytes() == b"existing-content"


class TestCorpusRootRecursion:
    """Verify the wider-scope scan: any user subdir is picked up, reserved
    dirs are skipped, root-level files work too. This is the PhD-KB rule:
    users organise their corpora however they like; we walk the tree."""

    def _seed_at(self, layout: CorpusLayout, rel: str, content: bytes) -> Path:
        p = layout.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def test_root_level_pdf_is_picked_up(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        self._seed_at(layout, "loose.pdf", b"abc")
        results = _make_orchestrator(layout).ingest_corpus()
        assert len(results) == 1
        assert results[0].source_filename == "loose.pdf"
        assert results[0].status is IngestStatus.INGESTED

    def test_user_subdirs_are_picked_up(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        self._seed_at(layout, "arxiv/2026.05/a.pdf", b"a")
        self._seed_at(layout, "bioarxiv/b.pdf", b"b")
        self._seed_at(layout, "topics/genetics/c.pdf", b"c")
        results = _make_orchestrator(layout).ingest_corpus()
        names = {r.source_filename for r in results}
        assert names == {"a.pdf", "b.pdf", "c.pdf"}
        assert all(r.status is IngestStatus.INGESTED for r in results)

    def test_reserved_dirs_are_skipped(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        # Seed pre-existing files in every reserved dir.
        (layout.processed / "already_in_papers.pdf").write_bytes(b"old")
        (layout.cards / "old_card.md").write_text("frontmatter")
        (layout.quarantine / "quar.pdf").write_bytes(b"q")
        (layout.graph / "g.json").write_text("{}")
        (layout.html / "h.html").write_text("<html/>")
        (layout.exports / "e.txt").write_text("e")
        # Seed one real input.
        self._seed_at(layout, "arxiv/real.pdf", b"abc")
        results = _make_orchestrator(layout).ingest_corpus()
        # ONLY the real input gets processed.
        assert len(results) == 1
        assert results[0].source_filename == "real.pdf"
        # Reserved-dir files still in place, untouched.
        assert (layout.processed / "already_in_papers.pdf").exists()
        assert (layout.cards / "old_card.md").exists()
        assert (layout.quarantine / "quar.pdf").exists()

    def test_communities_and_reports_dirs_also_reserved(self, tmp_path: Path) -> None:
        # These aren't in _STANDARD_SUBDIRS so init doesn't create them;
        # the user (or render step) might. Confirm they're still skipped.
        layout = init_corpus(tmp_path / "c")
        (layout.root / "communities").mkdir()
        (layout.root / "communities" / "0.md").write_text("---\n---\n")
        (layout.root / "reports").mkdir()
        (layout.root / "reports" / "old.md").write_text("old report")
        self._seed_at(layout, "inbox/real.pdf", b"abc")
        results = _make_orchestrator(layout).ingest_corpus()
        assert len(results) == 1
        assert results[0].source_filename == "real.pdf"

    def test_hidden_files_skipped(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        self._seed_at(layout, "arxiv/.DS_Store", b"junk")
        self._seed_at(layout, ".hidden.pdf", b"junk")
        self._seed_at(layout, "arxiv/real.pdf", b"abc")
        results = _make_orchestrator(layout).ingest_corpus()
        assert len(results) == 1
        assert results[0].source_filename == "real.pdf"

    def test_ingest_inbox_alias_still_works(self, tmp_path: Path) -> None:
        # Back-compat: old callers using ingest_inbox() should not break.
        layout = init_corpus(tmp_path / "c")
        _seed_inbox(layout, "via_inbox.pdf", b"abc")
        results = _make_orchestrator(layout).ingest_inbox()
        assert len(results) == 1
        assert results[0].source_filename == "via_inbox.pdf"

    def test_duplicate_hash_across_subdirs_dedupes(self, tmp_path: Path) -> None:
        """Two copies of the same file in different subdirs: one wins,
        the other is flagged DUPLICATE without re-extraction."""
        layout = init_corpus(tmp_path / "c")
        # Same bytes in both locations.
        self._seed_at(layout, "arxiv/dup.pdf", b"identical-bytes")
        self._seed_at(layout, "bioarxiv/dup.pdf", b"identical-bytes")
        results = _make_orchestrator(layout).ingest_corpus()
        statuses = sorted(r.status.value for r in results)
        # Exactly one INGESTED, one DUPLICATE.
        assert statuses == sorted(["ingested", "duplicate"])
