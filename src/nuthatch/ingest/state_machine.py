# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Ingest orchestrator: walks files from `inbox/` to `papers/` via the spine.

The Sprint 2 spine:

    inbox file
      -> hash dedup
      -> preflight (suffix + non-zero size)
      -> extract (real OCR via `ingest.extract` for PDFs;
                  `read_text` for `.txt` / `.md` / `.html`)
      -> qc (`ingest.qc.check_extract_yield`)
      -> metadata + schema (`ingest.metadata.extract_and_validate`)
      -> route(pass/fail)
      -> manifest log

`route` moves the file to one of:

- `papers/`           on success (`IngestStatus.INGESTED`)
- (left in inbox)     on byte-exact dedup hit (`IngestStatus.DUPLICATE`)
- `quarantine/<reason>/` on any pipeline-stage failure
  (`IngestStatus.QUARANTINED`); a `.reason.json` sidecar lands next
  to the file (see `ingest.quarantine`)

The extractor is dependency-injected so tests can substitute a
fast no-op while the real path runs Docling / Chandra-OCR / EasyOCR
under the routing decision pinned in `docs/DECISIONS.md`.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nuthatch.corpus.layout import CorpusLayout
from nuthatch.ingest.dedup import hash_file
from nuthatch.ingest.manifest import IngestStatus, ManifestEntry, ManifestStore
from nuthatch.ingest.metadata import extract_and_validate
from nuthatch.ingest.qc import check_extract_yield
from nuthatch.ingest.quarantine import quarantine_file
from nuthatch.schema.profile import SchemaProfile
from nuthatch.schema.profiles import ArxivPaperProfile

# Sentinel marking the Sprint 2 default extractor version. Bumped
# when the routing logic or backend versions change so the manifest
# distinguishes files extracted under different toolchains.
_EXTRACTOR_VERSION: str = "router-v1"

# File suffixes the preflight accepts. Anything else is quarantined
# with reason `unsupported_format`.
_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".html", ".htm", ".txt", ".md"})

# Callable signature for an extractor: `(path) -> markdown_string`.
Extractor = Callable[[Path], str]

_LOG = logging.getLogger(__name__)


def _default_extractor(path: Path) -> str:
    """Default extractor: routes PDFs through `ingest.extract`, reads text otherwise.

    Heavy-import the OCR machinery only when first invoked on a PDF
    so test runs that never touch PDFs do not pull torch into memory.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from nuthatch.ingest.extract import extract

        return extract(path).text
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass(slots=True, frozen=True)
class IngestResult:
    """Per-file outcome of an `ingest_inbox` run."""

    source_filename: str
    status: IngestStatus
    reason: str | None
    destination: Path | None


class IngestOrchestrator:
    """Drive the Sprint 2 ingest pipeline against a `CorpusLayout`.

    `ingest_inbox()` processes every file currently under
    `<corpus>/inbox/` once and returns the per-file results. Files
    that fail dedup are LEFT in the inbox (so the user can decide
    to delete them); files that pass the full pipeline are MOVED to
    `papers/`; files that fail extract / qc / schema land in
    `quarantine/<reason>/` with a `.reason.json` sidecar.

    Dependency-injection:
    - `extractor`: callable `(Path) -> str` returning markdown.
      Defaults to the real PDF router + plain-text reader.
    - `schema_profile`: subclass of `SchemaProfile` to validate
      extracted metadata against. Defaults to `ArxivPaperProfile`.
    """

    __slots__ = ("_extractor", "_layout", "_manifest", "_profile")

    def __init__(
        self,
        layout: CorpusLayout,
        *,
        extractor: Extractor | None = None,
        schema_profile: type[SchemaProfile] | None = None,
    ) -> None:
        self._layout = layout
        self._manifest = ManifestStore(layout.manifest_path)
        self._extractor = extractor or _default_extractor
        self._profile = schema_profile or ArxivPaperProfile

    @property
    def layout(self) -> CorpusLayout:
        return self._layout

    @property
    def manifest(self) -> ManifestStore:
        return self._manifest

    @property
    def schema_profile(self) -> type[SchemaProfile]:
        return self._profile

    def ingest_corpus(self) -> list[IngestResult]:
        """Process every source file anywhere in the corpus tree.

        Walks `<corpus>/` recursively, skipping the
        `CORPUS_RESERVED_DIRS` set (`.kg/`, `papers/`, `quarantine/`,
        `cards/`, `html/`, `communities/`, `graph/`, `exports/`,
        `reports/`) and hidden files. Users organise inputs however
        they like (`inbox/`, `arxiv/`, `bioarxiv/`, root-level PDFs);
        the scan finds them.

        Lesson from PhD KB: corpora are not flat. Forcing
        `<corpus>/inbox/*.pdf` would lose the user's organisation.
        """

        if not self._layout.root.is_dir():
            return []

        known = self._manifest.known_hashes()
        results: list[IngestResult] = []

        for source in self._iter_source_files():
            try:
                file_hash = hash_file(source)
            except OSError as exc:
                _LOG.warning("Could not read %s: %s", source, exc)
                results.append(
                    self._fail(
                        source,
                        file_hash="",
                        status=IngestStatus.FAILED,
                        reason=f"read_error: {exc}",
                        destination=None,
                    )
                )
                continue

            if file_hash in known:
                results.append(
                    self._record_and_result(
                        source=source,
                        file_hash=file_hash,
                        destination=None,
                        status=IngestStatus.DUPLICATE,
                        reason=f"existing_hash:{file_hash[:12]}",
                    )
                )
                continue

            # Preflight: suffix + size. Cheap reject for obvious
            # garbage before paying the OCR / extract cost.
            ok, preflight_reason = _preflight(source)
            if not ok:
                dest = quarantine_file(
                    source,
                    self._layout.quarantine,
                    reason=preflight_reason or "preflight_failed",
                    details={"stage": "preflight"},
                )
                results.append(
                    self._record_and_result(
                        source=source,
                        file_hash=file_hash,
                        destination=dest,
                        status=IngestStatus.QUARANTINED,
                        reason=preflight_reason,
                    )
                )
                continue

            # Real extraction. Failures here are infrastructure
            # problems (OCR crash, model unavailable). Fail-not-
            # quarantine: the user should retry, not lose the file.
            try:
                markdown = self._extractor(source)
            except Exception as exc:
                _LOG.exception("Extractor crashed on %s", source)
                results.append(
                    self._fail(
                        source,
                        file_hash=file_hash,
                        status=IngestStatus.FAILED,
                        reason=f"extract_error: {exc!s}",
                        destination=None,
                    )
                )
                continue

            qc = check_extract_yield(markdown)
            if not qc.passed:
                dest = quarantine_file(
                    source,
                    self._layout.quarantine,
                    reason=qc.reason or "extract_yield_failed",
                    details={"stage": "qc", "qc": qc.details},
                )
                results.append(
                    self._record_and_result(
                        source=source,
                        file_hash=file_hash,
                        destination=dest,
                        status=IngestStatus.QUARANTINED,
                        reason=qc.reason,
                    )
                )
                continue

            extracted, validation = extract_and_validate(markdown, self._profile)
            if not validation.passed:
                dest = quarantine_file(
                    source,
                    self._layout.quarantine,
                    reason=validation.reason or "schema_invalid",
                    details={
                        "stage": "schema",
                        "profile": self._profile.profile_name,
                        "extracted": extracted,
                        "missing_required": validation.missing_required,
                        "type_mismatches": validation.type_mismatches,
                    },
                )
                results.append(
                    self._record_and_result(
                        source=source,
                        file_hash=file_hash,
                        destination=dest,
                        status=IngestStatus.QUARANTINED,
                        reason=validation.reason,
                    )
                )
                continue

            dest = self._move(source, self._layout.papers / source.name)
            # Persist extracted markdown + meta so `nuthatch embed` can
            # consume without re-running OCR / Docling. doc_id is the
            # FINAL papers/ filename stem (handles name collisions
            # automatically since _move() applied `-1`, `-2` suffixes).
            doc_id = dest.stem
            self._persist_extracted(
                doc_id=doc_id,
                source_filename=source.name,
                file_hash=file_hash,
                markdown=markdown,
                metadata=extracted,
            )
            results.append(
                self._record_and_result(
                    source=source,
                    file_hash=file_hash,
                    destination=dest,
                    status=IngestStatus.INGESTED,
                    reason=None,
                )
            )
            known.add(file_hash)

        return results

    def _fail(
        self,
        source: Path,
        *,
        file_hash: str,
        status: IngestStatus,
        reason: str | None,
        destination: Path | None,
    ) -> IngestResult:
        """Record a manifest failure entry and return the matching result."""
        return self._record_and_result(
            source=source,
            file_hash=file_hash,
            destination=destination,
            status=status,
            reason=reason,
        )

    def _record_and_result(
        self,
        *,
        source: Path,
        file_hash: str,
        destination: Path | None,
        status: IngestStatus,
        reason: str | None,
    ) -> IngestResult:
        self._record(
            file_hash=file_hash,
            source=source,
            destination=destination,
            status=status,
            reason=reason,
        )
        return IngestResult(
            source_filename=source.name,
            status=status,
            reason=reason,
            destination=destination,
        )

    def _iter_source_files(self) -> list[Path]:
        """Sorted, deterministic walk over all corpus source files.

        Delegates to `CorpusLayout.iter_source_files()`, which skips
        reserved nuthatch-managed subdirs.
        """
        return self._layout.iter_source_files()

    def ingest_inbox(self) -> list[IngestResult]:
        """Backwards-compat alias for `ingest_corpus`.

        Old call sites (CLI, tests written before the recursion
        widening) used `ingest_inbox()`; the rename to
        `ingest_corpus()` honours the wider scan. Both methods do the
        same thing now.
        """
        return self.ingest_corpus()

    def _persist_extracted(
        self,
        *,
        doc_id: str,
        source_filename: str,
        file_hash: str,
        markdown: str,
        metadata: dict[str, Any],
    ) -> None:
        """Write the extracted markdown + sidecar meta json under `.kg/extracted/`.

        Idempotent: overwriting is fine (re-ingest of the same source
        produces byte-identical output if the extractor is deterministic).
        """
        extracted_dir = self._layout.extracted_dir
        extracted_dir.mkdir(parents=True, exist_ok=True)
        (extracted_dir / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
        meta = {
            "doc_id": doc_id,
            "source_filename": source_filename,
            "file_hash": file_hash,
            "extractor_version": _EXTRACTOR_VERSION,
            "ingested_at": datetime.now(UTC).isoformat(),
            "metadata": metadata,
        }
        (extracted_dir / f"{doc_id}.meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

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
            extractor_version=_EXTRACTOR_VERSION,
        )
        self._manifest.append(entry)


def _preflight(path: Path) -> tuple[bool, str | None]:
    """Cheap pre-extraction check: suffix + non-zero size.

    Returns `(ok, reason_when_not_ok)`. Fail-fast for obvious garbage
    so the orchestrator doesn't pay OCR cost on 0-byte files or files
    with formats nuthatch isn't built to handle.
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
