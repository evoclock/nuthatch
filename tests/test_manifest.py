# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.manifest`."""

from __future__ import annotations

import json
from pathlib import Path

from nuthatch.ingest.manifest import IngestStatus, ManifestEntry, ManifestStore


def _entry(
    *,
    hash_: str = "deadbeef",
    source: str = "p.pdf",
    status: IngestStatus = IngestStatus.INGESTED,
    reason: str | None = None,
    dest: str | None = "papers/p.pdf",
) -> ManifestEntry:
    return ManifestEntry.now(
        hash_=hash_,
        source_filename=source,
        destination=dest,
        status=status,
        reason=reason,
        extractor_version="placeholder-v0",
    )


class TestEntryRoundTrip:
    def test_to_jsonl_and_from_jsonl_are_inverses(self) -> None:
        entry = _entry()
        round_tripped = ManifestEntry.from_jsonl(entry.to_jsonl())
        assert round_tripped.hash == entry.hash
        assert round_tripped.source_filename == entry.source_filename
        assert round_tripped.destination == entry.destination
        assert round_tripped.status is entry.status
        assert round_tripped.reason == entry.reason
        assert round_tripped.extractor_version == entry.extractor_version
        assert round_tripped.timestamp_utc == entry.timestamp_utc

    def test_jsonl_payload_is_valid_json(self) -> None:
        entry = _entry()
        payload = json.loads(entry.to_jsonl())
        assert payload["status"] == "ingested"
        assert payload["hash"] == "deadbeef"


class TestManifestStore:
    def test_append_creates_file_and_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / ".kg" / "manifest.jsonl"
        store = ManifestStore(path)
        assert not path.exists()
        store.append(_entry())
        assert path.exists()
        assert len(path.read_text().splitlines()) == 1

    def test_appends_in_order(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path / "manifest.jsonl")
        store.append(_entry(hash_="a"))
        store.append(_entry(hash_="b"))
        store.append(_entry(hash_="c"))
        entries = list(store.iter_entries())
        assert [e.hash for e in entries] == ["a", "b", "c"]

    def test_known_hashes_only_returns_ingested(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path / "manifest.jsonl")
        store.append(_entry(hash_="a", status=IngestStatus.INGESTED))
        store.append(_entry(hash_="b", status=IngestStatus.DUPLICATE))
        store.append(_entry(hash_="c", status=IngestStatus.QUARANTINED, reason="bad"))
        store.append(_entry(hash_="d", status=IngestStatus.FAILED, reason="bad"))
        # Only the ingested hash counts as "this file is already in the corpus".
        # A previously quarantined file should be re-tried; a duplicate's hash
        # is already covered by the original ingested record under that hash.
        assert store.known_hashes() == {"a"}

    def test_iter_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        store = ManifestStore(path)
        store.append(_entry(hash_="a"))
        # Manually inject a blank line.
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n")
        store.append(_entry(hash_="b"))
        assert [e.hash for e in store.iter_entries()] == ["a", "b"]

    def test_count(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path / "manifest.jsonl")
        assert store.count() == 0
        store.append(_entry(hash_="a"))
        store.append(_entry(hash_="b"))
        assert store.count() == 2

    def test_iter_on_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path / "no-such-file.jsonl")
        assert list(store.iter_entries()) == []
        assert store.known_hashes() == set()
