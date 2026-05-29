# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Hybrid chunking with full-document coverage invariant.

Purpose: split a paper's extracted markdown into overlapping chunks
    suitable for embedding + retrieval, while enforcing the
    full-document-coverage rule from DECISIONS.md (every byte of the
    extracted text lands in at least one chunk).

Inputs: raw markdown string + chunk-size config (target chars,
    overlap chars).

Outputs: list of `Chunk(text, start_offset, end_offset, ordinal)`,
    plus `check_coverage(text, chunks)` for the invariant.

Assumptions: structure-aware chunking prefers paragraph and section
    boundaries when they fit within `max_chars`. The fallback is
    fixed-window slicing with overlap. The character offset is into
    the input string, so reconstruction by index is exact.

The full-document-coverage invariant is the load-bearing rule: a
knowledge graph that surfaces "the relevant section" to an LLM is
only as honest as its chunks. Partial coverage means a query can
miss the one paragraph where the paper's actual contribution lives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Default chunking parameters. Tunable per-corpus via config.
# Sized to match PhD KB's ~512-word/64-word overlap pattern at
# ~6 chars/word, biased larger to preserve more context per chunk
# for retrieval (per DECISIONS.md: "we go with more not less").
_DEFAULT_MAX_CHARS: int = 3000
_DEFAULT_OVERLAP_CHARS: int = 400

# Boundary regex for structure-aware chunking. Order matters: try
# strongest boundaries first (markdown headings, then paragraphs,
# then sentence ends).
_HEADING_RE = re.compile(r"\n#{1,6}\s+[^\n]+\n")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class Chunk:
    """A single chunk with its offsets into the source text."""

    text: str
    start_offset: int
    end_offset: int
    ordinal: int


@dataclass(frozen=True)
class CoverageResult:
    """Outcome of `check_coverage`: did the chunk union cover every byte?"""

    passed: bool
    total_chars: int
    covered_chars: int
    gaps: list[tuple[int, int]] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        if self.total_chars == 0:
            return 100.0
        return 100.0 * self.covered_chars / self.total_chars


def _find_break_points(text: str) -> list[int]:
    """Return preferred break-point offsets, strongest first."""
    points: set[int] = set()
    for rx in (_HEADING_RE, _PARAGRAPH_RE, _SENTENCE_RE):
        for m in rx.finditer(text):
            points.add(m.start())
    return sorted(points)


def _choose_split(_text: str, start: int, target_end: int, break_points: list[int]) -> int:
    """Pick the best split offset at or before `target_end`, but after `start`.

    Falls through to `target_end` if no break point fits within the
    last 20 percent of the chunk window (avoids over-shrinking chunks
    on prose without natural boundaries).
    """
    min_acceptable = start + int(0.8 * (target_end - start))
    candidates = [p for p in break_points if min_acceptable <= p <= target_end]
    if candidates:
        return candidates[-1]
    return target_end


def chunk_text(
    text: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Hybrid chunker. Prefers structural boundaries; falls back to fixed window.

    Guarantees full-document coverage: the union of `(start, end)`
    intervals across the returned chunks equals `(0, len(text))`.
    Overlap between adjacent chunks is encouraged for retrieval recall.
    """
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")

    n = len(text)
    if n <= max_chars:
        return [Chunk(text=text, start_offset=0, end_offset=n, ordinal=0)]

    breaks = _find_break_points(text)
    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    while start < n:
        target_end = min(start + max_chars, n)
        end = _choose_split(text, start, target_end, breaks) if target_end < n else n
        # Defensive: ensure forward progress
        if end <= start:
            end = min(start + max_chars, n)
        chunks.append(
            Chunk(
                text=text[start:end],
                start_offset=start,
                end_offset=end,
                ordinal=ordinal,
            )
        )
        ordinal += 1
        if end >= n:
            break
        start = max(end - overlap_chars, end - max_chars + 1)
    return chunks


def check_coverage(text: str, chunks: list[Chunk]) -> CoverageResult:
    """Verify the union of chunk intervals covers every byte of `text`."""
    if not text:
        return CoverageResult(passed=True, total_chars=0, covered_chars=0)
    n = len(text)
    intervals = sorted((c.start_offset, c.end_offset) for c in chunks)
    covered = 0
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, start))
        # Advance cursor by the new coverage (allowing overlap).
        if end > cursor:
            covered += end - max(cursor, start)
            cursor = end
    if cursor < n:
        gaps.append((cursor, n))
    return CoverageResult(
        passed=not gaps,
        total_chars=n,
        covered_chars=covered,
        gaps=gaps,
    )
