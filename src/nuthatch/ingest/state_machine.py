# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Ingest orchestrator: walks files from a user subdir to `processed/<subdir>/`.

The Sprint 2 spine:

    source file (anywhere under <corpus>/, outside reserved dirs)
      -> hash dedup
      -> preflight (suffix + non-zero size)
      -> extract (real OCR via `ingest.extract` for PDFs;
                  `read_text` for `.txt` / `.md` / `.html`)
      -> qc (`ingest.qc.check_extract_yield`)
      -> metadata + schema (`ingest.metadata.extract_and_validate`)
      -> route(pass/fail)
      -> manifest log

`route` moves the file to one of:

- `processed/<source_subdir>/`  on success (`IngestStatus.INGESTED`).
  Source provenance is preserved: a file ingested from
  `<corpus>/bioarxiv/foo.pdf` lands at
  `<corpus>/processed/bioarxiv/foo.pdf`. Watcher/scan skips this
  subtree so it's never re-fed.
- (left in place)               on byte-exact dedup hit (`IngestStatus.DUPLICATE`)
- `quarantine/<reason>/` on any pipeline-stage failure
  (`IngestStatus.QUARANTINED`); a `.reason.json` sidecar lands next
  to the file recording the original subdir so a fix-pass can route
  the file back to `processed/<original_subdir>/` after the issue
  is resolved (see `ingest.quarantine`).

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
from nuthatch.ingest.profile_routing import select_profile_for_filename
from nuthatch.ingest.qc import check_extract_yield
from nuthatch.ingest.quarantine import quarantine_file
from nuthatch.schema.profile import SchemaProfile
from nuthatch.schema.profiles import ArxivPaperProfile, InternalDocProfile

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


def _make_skip_chandra_extractor() -> Extractor:
    """Default extractor variant that forces non-Chandra routing.

    Used by `IngestOrchestrator(skip_chandra=True)`. Math-heavy
    papers will have broken `$...$` spans which the orchestrator
    captures into `<corpus>/.kg/math_retry.jsonl` for the
    post-Thursday batch-Chandra patching pass.
    """

    def _extract(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from nuthatch.ingest.extract import extract

            return extract(path, skip_chandra=True).text
        return path.read_text(encoding="utf-8", errors="replace")

    return _extract


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
    - `schema_profile`: subclass of `SchemaProfile` used as a HARD
      OVERRIDE. When set, every file in the corpus is validated
      against this profile regardless of filename pattern. When
      unset (the default), the orchestrator routes per-file via
      `select_profile_for_filename`: arxiv preprints validate
      against `ArxivPaperProfile`, bioRxiv preprints against
      `BiorxivPaperProfile`, everything else falls back to
      `InternalDocProfile`.
    """

    __slots__ = (
        "_extractor",
        "_layout",
        "_manifest",
        "_profile_override",
        "_skip_chandra",
    )

    def __init__(
        self,
        layout: CorpusLayout,
        *,
        extractor: Extractor | None = None,
        schema_profile: type[SchemaProfile] | None = None,
        skip_chandra: bool = False,
    ) -> None:
        self._layout = layout
        self._manifest = ManifestStore(layout.manifest_path)
        if extractor is not None:
            self._extractor = extractor
        elif skip_chandra:
            self._extractor = _make_skip_chandra_extractor()
        else:
            self._extractor = _default_extractor
        self._profile_override = schema_profile
        self._skip_chandra = skip_chandra

    @property
    def layout(self) -> CorpusLayout:
        return self._layout

    @property
    def manifest(self) -> ManifestStore:
        return self._manifest

    @property
    def schema_profile(self) -> type[SchemaProfile]:
        """Return the configured override, or `ArxivPaperProfile` as a stable default
        for back-compat with callers that introspect the profile at construction time.

        Per-file routing happens inside `_iter_processed`; this accessor is
        for tests and external introspection.
        """
        return self._profile_override or ArxivPaperProfile

    def _resolve_profile(self, filename: str) -> type[SchemaProfile]:
        """Pick the profile to validate `filename` against."""
        if self._profile_override is not None:
            return self._profile_override
        return select_profile_for_filename(filename, fallback=InternalDocProfile)

    def ingest_corpus(
        self,
        *,
        on_progress: Callable[[int, int, IngestResult], None] | None = None,
    ) -> list[IngestResult]:
        """Process every source file anywhere in the corpus tree.

        Walks `<corpus>/` recursively, skipping the
        `CORPUS_RESERVED_DIRS` set (`.kg/`, `papers/`, `quarantine/`,
        `cards/`, `html/`, `communities/`, `graph/`, `exports/`,
        `reports/`) and hidden files. Users organise inputs however
        they like (`inbox/`, `arxiv/`, `bioarxiv/`, root-level PDFs);
        the scan finds them.

        Lesson from PhD KB: corpora are not flat. Forcing
        `<corpus>/inbox/*.pdf` would lose the user's organisation.

        `on_progress(index, total, result)` is called after each file
        is processed (index is 1-based; total is the count of files
        the scan found). Used by the CLI to print per-file progress
        so the operator sees what's happening on a 100+-file run.
        """
        results: list[IngestResult] = []
        all_sources = self._gather_source_files()
        total = len(all_sources)
        for i, result in enumerate(self._iter_processed(all_sources), start=1):
            results.append(result)
            if on_progress is not None:
                on_progress(i, total, result)
        return results

    def _gather_source_files(self) -> list[Path]:
        """Return the sorted list of files the orchestrator will process."""
        if not self._layout.root.is_dir():
            return []
        return self._iter_source_files()

    def _iter_processed(self, all_sources: list[Path]):
        """Yield an `IngestResult` per source, in scan order."""
        known = self._manifest.known_hashes()

        for source in all_sources:
            try:
                file_hash = hash_file(source)
            except OSError as exc:
                _LOG.warning("Could not read %s: %s", source, exc)
                yield (
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
                yield (
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
                    corpus_root=self._layout.root,
                )
                yield (
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
                yield (
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
                    corpus_root=self._layout.root,
                )
                yield (
                    self._record_and_result(
                        source=source,
                        file_hash=file_hash,
                        destination=dest,
                        status=IngestStatus.QUARANTINED,
                        reason=qc.reason,
                    )
                )
                continue

            profile = self._resolve_profile(source.name)
            extracted, validation = extract_and_validate(
                markdown,
                profile,
                source_filename=source.name,
                metadata_cache_dir=self._layout.kg / "metadata_cache",
            )
            if not validation.passed:
                dest = quarantine_file(
                    source,
                    self._layout.quarantine,
                    reason=validation.reason or "schema_invalid",
                    details={
                        "stage": "schema",
                        "profile": profile.profile_name,
                        "extracted": extracted,
                        "missing_required": validation.missing_required,
                        "type_mismatches": validation.type_mismatches,
                    },
                    corpus_root=self._layout.root,
                )
                yield (
                    self._record_and_result(
                        source=source,
                        file_hash=file_hash,
                        destination=dest,
                        status=IngestStatus.QUARANTINED,
                        reason=validation.reason,
                    )
                )
                continue

            dest = self._move(source, self._processed_destination(source))
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
            # When --skip-chandra is on, Docling handles every PDF
            # (including math-heavy ones). Docling's math output for
            # those is typically broken LaTeX fragments. Validate and
            # flag for a later batch-Chandra patch pass.
            if self._skip_chandra:
                self._maybe_flag_math_retry(
                    doc_id=doc_id, source_filename=source.name, markdown=markdown
                )
            yield (
                self._record_and_result(
                    source=source,
                    file_hash=file_hash,
                    destination=dest,
                    status=IngestStatus.INGESTED,
                    reason=None,
                )
            )
            known.add(file_hash)

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

    def _maybe_flag_math_retry(
        self, *, doc_id: str, source_filename: str, markdown: str
    ) -> None:
        """Append a JSONL line to `.kg/math_retry.jsonl` if Docling's
        math output is broken enough to warrant a later Chandra retry.

        Called only when `skip_chandra=True`. The retry log is the
        input for the post-Thursday batch-Chandra patching pass (see
        `docs/DECISIONS.md` § Execution model and the strand doc
        post-Thursday section).
        """
        from nuthatch.ingest.math_validator import validate_math

        result = validate_math(markdown)
        if not result.needs_retry:
            return
        retry_path = self._layout.kg / "math_retry.jsonl"
        retry_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "doc_id": doc_id,
            "source_filename": source_filename,
            "broken_count": result.broken_count,
            "real_count": result.real_count,
            "broken_ratio": round(result.broken_ratio, 3),
            "broken_spans_sample": result.broken_spans,
            "extracted_md_path": str(
                (self._layout.extracted_dir / f"{doc_id}.md").relative_to(
                    self._layout.root
                )
            ),
            "deferred_at": datetime.now(UTC).isoformat(),
        }
        with retry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

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

    def _processed_destination(self, source: Path) -> Path:
        """Mirror the source subdir under `processed/`.

        A file ingested from `<corpus>/bioarxiv/foo.pdf` returns
        `<corpus>/processed/bioarxiv/foo.pdf`. Files at corpus root
        (no subdir) go straight to `<corpus>/processed/foo.pdf`.

        Files outside the corpus root fall back to a flat
        `processed/<filename>` placement; that shouldn't happen
        through the normal scan but the fallback keeps the move safe
        if a test passes an unrelated path.
        """
        try:
            rel = source.resolve().relative_to(self._layout.root)
        except ValueError:
            return self._layout.processed / source.name
        return self._layout.processed / rel

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
