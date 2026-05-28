# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Pre-flight triage for PDF source files.

Purpose: tell the operator BEFORE the expensive Docling extraction
    which PDFs will likely ingest cleanly, which look risky, and
    which should be deferred. Runs against the cheap pdftotext
    extraction (no GPU, no model load, ~50ms per PDF) so a 200-paper
    corpus can be classified in a few seconds.

The triage approximates what the real ingest extractor will do
without paying the Docling cost. Approximation, not guarantee:
Docling output differs from pdftotext (added markdown structure,
different layout decisions), so a PASS here is "likely to ingest",
not "certain to ingest". But it catches the worst offenders (no
detectable title, no detectable authors, no abstract paragraph)
without spinning up the GPU.

Decision classes:

- PASS: title detected, 2+ authors detected, abstract paragraph
  detected. Highest-confidence "this will ingest cleanly".
- FLAG: title + authors detected but abstract is questionable
  (no `Abstract` keyword in the first ~10kB). Likely to recover
  via the no-heading paragraph fallback but worth flagging.
- DEFER: missing title OR missing authors after the full
  heuristic cascade. Likely quarantine target.

Inputs: a `CorpusLayout` + the set of PDF candidates (defaults to
    every unprocessed PDF found by the recursive source-walk).

Outputs: per-PDF classification + reason; a summary table; an
    optional auto-defer action that moves DEFER PDFs to
    `<corpus>/defer/` (a reserved subdir the scan skips).

Assumptions: pdftotext is installed (ships with poppler-utils on
    every Linux distro; macOS via `brew install poppler`). Falls
    back to "UNKNOWN" classification when pdftotext fails on a PDF.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from nuthatch.corpus.layout import CorpusLayout


class TriageClass(StrEnum):
    PASS = "PASS"
    FLAG = "FLAG"
    DEFER = "DEFER"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TriageResult:
    """Per-PDF triage outcome."""

    path: Path
    classification: TriageClass
    reason: str
    has_title: bool
    has_authors: bool
    has_abstract: bool


# Approximate heuristics. Looser than the real extractor because we
# only have pdftotext output, not Docling's markdown structure.
_NAME_TOKEN = r"[A-Z][a-z][\w'.-]*"
_AUTHOR_CSV_RE = re.compile(
    rf"\b{_NAME_TOKEN}(?:\s+{_NAME_TOKEN})*\s*[,\d*]*\s*,\s*"
    rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN})*\b",
)
_AFFIL_DIGIT_NAME_RE = re.compile(
    rf"\b{_NAME_TOKEN}(?:\s+{_NAME_TOKEN})*\d+",
)
_SINGLE_AUTHOR_WITH_EMAIL_RE = re.compile(
    rf"\b{_NAME_TOKEN}\s+{_NAME_TOKEN}\b[^\n]*@",
)
_LONG_PARA_RE = re.compile(r"\n\s*\n([^\n]{250,})", re.DOTALL)
_ABSTRACT_WORD_RE = re.compile(r"\b[Aa]bstract\b|\bSUMMARY\b")
# Title-line detector. Accepts an optional leading line number
# (`1   Title text`), an optional `Title:` label (`Title: Real title`),
# or an `Article type: X` preamble (in which case the next non-blank
# line should be the title — handled by the broader search below).
_TITLE_LINE_RE = re.compile(
    r"^\s*(?:\d+\s+)?(?:Title:\s*)?[A-Z][^\n]{15,200}$",
    re.MULTILINE,
)
_ARTICLE_TYPE_PRECEDES_TITLE_RE = re.compile(
    r"^\s*(?:\d+\s+)?Article type:[^\n]+\n+\s*(?:\d+\s+)?([A-Z][^\n]{15,200})$",
    re.MULTILINE,
)


def triage_pdf(pdf: Path) -> TriageResult:
    """Classify one PDF using cheap pdftotext extraction."""
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-f", "1", "-l", "2", str(pdf), "-"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return TriageResult(
            path=pdf,
            classification=TriageClass.UNKNOWN,
            reason="pdftotext unavailable or timed out",
            has_title=False,
            has_authors=False,
            has_abstract=False,
        )

    # Skip the bioRxiv watermark paragraph (always lines 1-3 of page 1).
    body = "\n".join(out.split("\n")[5:])

    has_title = bool(_TITLE_LINE_RE.search(body) or _ARTICLE_TYPE_PRECEDES_TITLE_RE.search(body))
    has_authors = bool(
        _AUTHOR_CSV_RE.search(body)
        or _AFFIL_DIGIT_NAME_RE.search(body)
        or _SINGLE_AUTHOR_WITH_EMAIL_RE.search(body)
    )
    has_abstract = bool(_ABSTRACT_WORD_RE.search(body) or _LONG_PARA_RE.search(body))

    if has_title and has_authors and has_abstract:
        return TriageResult(
            pdf,
            TriageClass.PASS,
            "title + authors + abstract detected",
            has_title,
            has_authors,
            has_abstract,
        )
    if has_title and has_authors:
        return TriageResult(
            pdf,
            TriageClass.FLAG,
            "no Abstract keyword on p1; will rely on paragraph fallback",
            has_title,
            has_authors,
            has_abstract,
        )
    missing = []
    if not has_title:
        missing.append("title")
    if not has_authors:
        missing.append("authors")
    if not has_abstract:
        missing.append("abstract")
    return TriageResult(
        pdf,
        TriageClass.DEFER,
        f"missing: {', '.join(missing)}",
        has_title,
        has_authors,
        has_abstract,
    )


def triage_corpus(
    layout: CorpusLayout,
    *,
    paths: Iterable[Path] | None = None,
) -> list[TriageResult]:
    """Classify every PDF the recursive source-walk would feed to ingest.

    When `paths` is omitted, walks `layout.iter_source_files()` and
    filters for `.pdf` (the recursive walk already skips
    `processed/`, `quarantine/`, `defer/`, `benchmark_test/`, etc.).
    """
    if paths is None:
        paths = [p for p in layout.iter_source_files() if p.suffix.lower() == ".pdf"]
    return [triage_pdf(p) for p in paths]


def auto_defer(
    results: Iterable[TriageResult],
    layout: CorpusLayout,
) -> list[Path]:
    """Move DEFER-classified PDFs to `<corpus>/defer/<original_subdir>/`.

    The destination mirrors each file's source subdir so provenance
    survives (e.g. `bioarxiv/foo.pdf` -> `defer/bioarxiv/foo.pdf`).
    Files stay on disk for a future re-triage; the `defer/` dir is
    reserved by the layout, so the next `nuthatch ingest` run does
    not pick them up.
    """
    moved: list[Path] = []
    defer_root = layout.root / "defer"
    defer_root.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r.classification is not TriageClass.DEFER:
            continue
        try:
            rel = r.path.resolve().relative_to(layout.root.resolve())
            dest = defer_root / rel
        except ValueError:
            dest = defer_root / r.path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(r.path), str(dest))
        moved.append(dest)
    return moved
