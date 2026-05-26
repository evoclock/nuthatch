# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Metadata extraction + schema validation.

Purpose: lift the metadata fields a `SchemaProfile` requires out of
    an extractor's markdown, then run profile validation. Pluggable
    so a richer extractor (LLM-driven, like PhD KB's
    `01_extract_metadata.py` Claude-Code call) can replace the
    heuristic default without changing the orchestrator surface.

Inputs: extracted markdown + an active `SchemaProfile`. Optional
    `extractor` callable for plugin replacement.

Outputs: `(extracted_metadata: dict, validation: ValidationResult)`.

Pattern reused from `~/PhD-knowledge-base/scripts/01_extract_metadata.py`:
the JSON-schema shape (title, authors, year, doi, abstract,
key_claims, topics, methods, relevance), the controlled-vocabulary
discipline (topics drawn from a per-corpus list), and the
separation of extract-then-validate. nuthatch's implementation is
a fresh heuristic plus a plugin hook for richer extractors.

The default heuristic extractor pulls what regex + first-block
parsing can reliably get from a paper's markdown:

- title from the first `# heading` or a `Title:` line
- authors from a line below the title (heuristic; LLM extractor
  would do better)
- year from a 4-digit year token in the head
- doi from a `10.NNNN/...` pattern anywhere
- arxiv id from `arXiv:NNNN.NNNNN[vN]`
- abstract from the first paragraph after an `## Abstract` heading
  (or the first multi-sentence paragraph if no heading)
- topics, key_claims, methods, relevance: heuristic stubs — the
  default extractor leaves these empty for the schema gate to
  flag if required. A user opting into the LLM extractor (Sprint
  3+) fills them.

Assumptions: markdown is one paper's worth, not a multi-doc corpus.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from nuthatch.schema.profile import SchemaProfile, ValidationResult

# Patterns.
_HEADING_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_TITLE_LABEL_RE = re.compile(r"^\s*Title:\s*(.+?)\s*$", re.MULTILINE)
_ARXIV_ID_RE = re.compile(
    r"arXiv\s*[:=]?\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE
)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PATENT_NUMBER_RE = re.compile(
    r"\b(US|EP|WO)\d{6,10}[A-Z]?\d?\b", re.IGNORECASE
)
_AUTHORS_LINE_RE = re.compile(
    r"^\s*(?:Authors?:|By)\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
_ABSTRACT_HEADING_RE = re.compile(
    r"^#{1,3}\s*Abstract\s*\n+(.+?)(?:\n#{1,3}\s|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)

# Signature for a metadata extractor; default is the heuristic below.
# Sprint 3+ ships an LLM-driven extractor as an optional plugin.
MetadataExtractor = Callable[[str], dict[str, Any]]


def _first_match(rx: re.Pattern[str], text: str, group: int = 1) -> str | None:
    m = rx.search(text)
    if not m:
        return None
    return m.group(group).strip()


def _split_authors(raw: str) -> list[str]:
    """Split an author string by common separators."""
    if not raw:
        return []
    parts = re.split(r"\s*(?:,|;|\band\b|&)\s*", raw)
    return [p.strip() for p in parts if p.strip()]


def extract_metadata_heuristic(markdown: str) -> dict[str, Any]:
    """Heuristic metadata lift. Best-effort; conservative on false positives.

    Fields populated: title, authors, year, doi, arxiv_id, patent_number,
    abstract. Optional list fields (key_claims, methods, topics,
    relevance) are left empty; a plugin extractor fills them.
    """
    out: dict[str, Any] = {}

    title = _first_match(_TITLE_LABEL_RE, markdown) or _first_match(
        _HEADING_TITLE_RE, markdown
    )
    if title:
        out["title"] = title

    authors_raw = _first_match(_AUTHORS_LINE_RE, markdown)
    if authors_raw:
        out["authors"] = _split_authors(authors_raw)

    arxiv = _first_match(_ARXIV_ID_RE, markdown)
    if arxiv:
        out["arxiv_id"] = arxiv

    doi = _first_match(_DOI_RE, markdown)
    if doi:
        out["doi"] = doi

    patent = _first_match(_PATENT_NUMBER_RE, markdown, group=0)
    if patent:
        out["patent_number"] = patent.upper()

    year = _first_match(_YEAR_RE, markdown, group=0)
    if year:
        out["year"] = int(year)

    abstract_m = _ABSTRACT_HEADING_RE.search(markdown)
    if abstract_m:
        abstract = re.sub(r"\s+", " ", abstract_m.group(1)).strip()
        if abstract:
            out["abstract"] = abstract

    return out


def extract_and_validate(
    markdown: str,
    profile: type[SchemaProfile],
    *,
    extractor: MetadataExtractor | None = None,
    source_filename: str | None = None,
    metadata_cache_dir: Any = None,
) -> tuple[dict[str, Any], ValidationResult]:
    """Run an extractor + profile validation in one call.

    `extractor` defaults to `extract_metadata_heuristic`. A plugin
    extractor (LLM-driven, per the PhD KB pattern) can be passed
    when the corpus opts into richer metadata.

    When `source_filename` is provided and looks like an arxiv or
    bioRxiv preprint filename, the publisher's API is consulted FIRST
    for authoritative title / authors / abstract / year / doi /
    arxiv_id. Body-text heuristic extraction fills any gaps the
    publisher metadata didn't cover. This preserves PDF body text
    as the source of truth for the corpus content while using the
    publisher record (which IS the canonical source for paper
    metadata) for fields the PDF body would only give us via brittle
    heuristics.

    Source-metadata fetches are cached under `metadata_cache_dir`
    (typically `<corpus>/.kg/metadata_cache/`) so re-ingest is a
    free local file read.
    """
    extract = extractor or extract_metadata_heuristic
    body_extracted = extract(markdown)

    source_extracted: dict[str, Any] = {}
    if source_filename is not None:
        from nuthatch.ingest.source_metadata import enrich_from_source

        source_meta = enrich_from_source(
            source_filename, cache_dir=metadata_cache_dir
        )
        if source_meta is not None:
            source_extracted = source_meta.to_dict()

    # Source-metadata takes precedence on overlapping fields: the
    # publisher record is authoritative for title / authors / etc.,
    # the body is authoritative for everything else (key claims,
    # topics, methods narrative).
    merged: dict[str, Any] = {**body_extracted, **source_extracted}

    result = profile.validate(merged)
    return merged, result
