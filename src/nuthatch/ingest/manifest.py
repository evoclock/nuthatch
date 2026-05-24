# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Append-only ingest manifest.

Every file the orchestrator touches gets one line in
`.kg/manifest.jsonl`. The manifest is the authoritative record of
what is in the corpus, when it landed, and what happened to it. It
backs the dedup lookup (the orchestrator queries the manifest before
hashing again) and the audit trail (`nuthatch status` reads it).

Format (one JSON object per line):

    {
      "hash": "<sha256-hex>",
      "source_filename": "1706.03762.pdf",
      "destination": "papers/1706.03762.pdf",
      "status": "ingested",
      "reason": null,
      "extractor_version": "placeholder-v0",
      "timestamp_utc": "2026-05-24T13:00:00+00:00"
    }

The shape is forward-compatible: future fields land additively; the
loader ignores unknown keys.

JSONL was chosen over a single-document JSON file so appends are
O(1) and parsing the manifest does not require holding it all in
memory (matters once a corpus has tens of thousands of papers).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class IngestStatus(StrEnum):
    """Terminal status of a single inbox file after orchestration."""

    INGESTED = "ingested"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"
    FAILED = "failed"


@dataclass(slots=True)
class ManifestEntry:
    """One row in the manifest."""

    hash: str
    source_filename: str
    destination: str | None
    status: IngestStatus
    reason: str | None
    extractor_version: str
    timestamp_utc: str

    @classmethod
    def now(
        cls,
        *,
        hash_: str,
        source_filename: str,
        destination: str | None,
        status: IngestStatus,
        reason: str | None,
        extractor_version: str,
    ) -> ManifestEntry:
        """Build an entry stamped with the current UTC time (ISO-8601)."""
        return cls(
            hash=hash_,
            source_filename=source_filename,
            destination=destination,
            status=status,
            reason=reason,
            extractor_version=extractor_version,
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
        )

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "hash": self.hash,
                "source_filename": self.source_filename,
                "destination": self.destination,
                "status": self.status.value,
                "reason": self.reason,
                "extractor_version": self.extractor_version,
                "timestamp_utc": self.timestamp_utc,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_jsonl(cls, line: str) -> ManifestEntry:
        raw = json.loads(line)
        return cls(
            hash=raw["hash"],
            source_filename=raw["source_filename"],
            destination=raw.get("destination"),
            status=IngestStatus(raw["status"]),
            reason=raw.get("reason"),
            extractor_version=raw["extractor_version"],
            timestamp_utc=raw["timestamp_utc"],
        )


class ManifestStore:
    """Append-only file-backed manifest.

    Constructed with a path; lazily creates the file on first
    `append`. Reads stream from disk, so a long-running session is
    not bottlenecked by manifest size.
    """

    __slots__ = ("_path",)

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, entry: ManifestEntry) -> None:
        """Append `entry` as a single JSONL line."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_jsonl() + "\n")

    def iter_entries(self) -> Iterator[ManifestEntry]:
        """Yield every entry in append order. Skips blank lines."""
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                yield ManifestEntry.from_jsonl(stripped)

    def known_hashes(self) -> set[str]:
        """Return the set of hashes the manifest has seen.

        Used by the orchestrator as the dedup lookup. Loads the
        whole manifest into a set; acceptable up to roughly
        100k-paper corpora before the set itself becomes a memory
        concern.
        """
        return {
            entry.hash for entry in self.iter_entries() if entry.status is IngestStatus.INGESTED
        }

    def count(self) -> int:
        return sum(1 for _ in self.iter_entries())
