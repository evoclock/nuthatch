# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Ingest orchestrator: walks files from `inbox/` to `papers/` via the spine.

The spine for Sprint 1:

    inbox file -> hash + dedup -> placeholder extract -> manifest log -> route

`route` moves the file to one of:

- `papers/`           on success (`IngestStatus.INGESTED`)
- (left in inbox)     on byte-exact dedup hit (`IngestStatus.DUPLICATE`)
- `quarantine/`       on placeholder-extract failure (`IngestStatus.QUARANTINED`)

The "placeholder extract" stage is a deliberate no-op for Sprint 1:
it returns `True` for any file with a non-zero size and one of the
known suffixes (`.pdf`, `.html`, `.txt`, `.md`). Real extraction
(Docling / Chandra-OCR / metadata schema validation) lands in
Sprint 2 by replacing `_placeholder_extract` with a proper pipeline
stage. The orchestrator shape stays the same.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from nuthatch.corpus.layout import CorpusLayout
from nuthatch.ingest.dedup import hash_file
from nuthatch.ingest.manifest import IngestStatus, ManifestEntry, ManifestStore

# Sentinel marking the Sprint 1 placeholder. Once Sprint 2 lands
# Docling + Chandra-OCR, bump to a real version string like
# `docling-2.5.0+chandra-2.1.0` so the manifest distinguishes
# files extracted under different toolchains.
_PLACEHOLDER_EXTRACTOR_VERSION: str = "placeholder-v0"

# File suffixes the placeholder extract accepts. Anything else is
# quarantined with reason `unsupported_format`.
_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".html", ".htm", ".txt", ".md"})

_LOG = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class IngestResult:
    """Per-file outcome of an `ingest_inbox` run."""

    source_filename: str
    status: IngestStatus
    reason: str | None
    destination: Path | None


class IngestOrchestrator:
    """Drive the Sprint 1 ingest spine against a `CorpusLayout`.

    `ingest_inbox()` processes every file currently under
    `<corpus>/inbox/` once and returns the per-file results. Files
    that fail dedup are LEFT in the inbox (so the user can decide
    to delete them); files that pass extract are MOVED to `papers/`.
    """

    __slots__ = ("_layout", "_manifest")

    def __init__(self, layout: CorpusLayout) -> None:
        self._layout = layout
        self._manifest = ManifestStore(layout.manifest_path)

    @property
    def layout(self) -> CorpusLayout:
        return self._layout

    @property
    def manifest(self) -> ManifestStore:
        return self._manifest

    def ingest_inbox(self) -> list[IngestResult]:
        """Process every file currently in `inbox/`. Returns per-file results.

        Subdirectories under `inbox/` are walked recursively (so the
        user can `mv arxiv/ inbox/` and have it work). Hidden files
        (`.gitkeep`, `.DS_Store`, dotfiles in general) are skipped.
        """

        if not self._layout.inbox.is_dir():
            return []

        known = self._manifest.known_hashes()
        results: list[IngestResult] = []

        for source in self._iter_inbox_files():
            try:
                file_hash = hash_file(source)
            except OSError as exc:
                _LOG.warning("Could not read %s: %s", source, exc)
                results.append(
                    IngestResult(
                        source_filename=source.name,
                        status=IngestStatus.FAILED,
                        reason=f"read_error: {exc}",
                        destination=None,
                    )
                )
                self._record(
                    file_hash="",
                    source=source,
                    destination=None,
                    status=IngestStatus.FAILED,
                    reason=f"read_error: {exc}",
                )
                continue

            if file_hash in known:
                results.append(
                    IngestResult(
                        source_filename=source.name,
                        status=IngestStatus.DUPLICATE,
                        reason=f"existing_hash:{file_hash[:12]}",
                        destination=None,
                    )
                )
                self._record(
                    file_hash=file_hash,
                    source=source,
                    destination=None,
                    status=IngestStatus.DUPLICATE,
                    reason=f"existing_hash:{file_hash[:12]}",
                )
                continue

            ok, reason = _placeholder_extract(source)
            if not ok:
                dest = self._move(source, self._layout.quarantine / source.name)
                results.append(
                    IngestResult(
                        source_filename=source.name,
                        status=IngestStatus.QUARANTINED,
                        reason=reason,
                        destination=dest,
                    )
                )
                self._record(
                    file_hash=file_hash,
                    source=source,
                    destination=dest,
                    status=IngestStatus.QUARANTINED,
                    reason=reason,
                )
                continue

            dest = self._move(source, self._layout.papers / source.name)
            results.append(
                IngestResult(
                    source_filename=source.name,
                    status=IngestStatus.INGESTED,
                    reason=None,
                    destination=dest,
                )
            )
            self._record(
                file_hash=file_hash,
                source=source,
                destination=dest,
                status=IngestStatus.INGESTED,
                reason=None,
            )
            known.add(file_hash)

        return results

    def _iter_inbox_files(self) -> list[Path]:
        """Sorted, deterministic walk over real files in `inbox/`."""
        files: list[Path] = []
        for path in self._layout.inbox.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            files.append(path)
        files.sort()
        return files

    def _move(self, source: Path, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        final = _unique_destination(dest)
        shutil.move(str(source), str(final))
        return final

    def _record(
        self,
        *,
        file_hash: str,
        source: Path,
        destination: Path | None,
        status: IngestStatus,
        reason: str | None,
    ) -> None:
        rel_dest: str | None = None
        if destination is not None:
            try:
                rel_dest = str(destination.resolve().relative_to(self._layout.root))
            except ValueError:
                rel_dest = str(destination)
        entry = ManifestEntry.now(
            hash_=file_hash,
            source_filename=source.name,
            destination=rel_dest,
            status=status,
            reason=reason,
            extractor_version=_PLACEHOLDER_EXTRACTOR_VERSION,
        )
        self._manifest.append(entry)


def _placeholder_extract(path: Path) -> tuple[bool, str | None]:
    """Sprint 1 placeholder. Returns (ok, reason_when_not_ok).

    Accepts any file with a known suffix and non-zero size. Real
    extraction lands in Sprint 2.
    """

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        return False, f"unsupported_format:{suffix or 'none'}"
    try:
        if path.stat().st_size == 0:
            return False, "empty_file"
    except OSError as exc:
        return False, f"stat_error:{exc}"
    return True, None


def _unique_destination(desired: Path) -> Path:
    """If `desired` already exists, append `-1`, `-2`, ... until it doesn't.

    Conflict resolution for the rare case where two source files
    share a basename but differ in content (different hashes). The
    duplicate-by-hash case is short-circuited earlier in the
    orchestrator; this is a last-line safety net.
    """

    if not desired.exists():
        return desired
    stem = desired.stem
    suffix = desired.suffix
    parent = desired.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
