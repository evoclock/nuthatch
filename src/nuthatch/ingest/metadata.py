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
# Title: accept #, ##, or ### so we match Docling output (which uses
# `##` for paper titles, not `#`). Take the FIRST heading at any of
# those levels. Body of a paper rarely has multiple top headings
# before the first content section, so this is safe.
_HEADING_TITLE_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
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

# Author lift from Docling-style author lines: `## [Name](url)` or
# `[Name](url)` immediately after the title. Filter out obvious non-
# author URLs (CC licence badges, ORCID images, etc.) by requiring
# the link target be an orcid.org URL or a mailto:, since author
# names in scholarly PDFs typically link to one of those.
_LEADING_MD_LINK_AUTHOR_RE = re.compile(
    r"^\s*(?:#{1,3}\s+)?\[([^\]\n]+)\]\((?:https?://orcid\.org/|mailto:)[^)]+\)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Cap how far into the document we scan for the author block. After
# the Abstract or Introduction heading is reached we stop; author
# lines never live below that.
_AUTHOR_SCAN_HEADING_RE = re.compile(
    r"^#{1,3}\s+(?:Abstract|Introduction|Background|Keywords|Summary)\b",
    re.MULTILINE | re.IGNORECASE,
)

# Plain-text author extraction. Three patterns observed in
# Docling-converted preprints:
#   (a) `Name1, Name2, Name3, and Name4`  — one comma-separated line
#   (b) `Name <affiliation> email@inst.tld`  — one author per line, with email
#   (c) `[Name](orcid|mailto:url)` — handled by _LEADING_MD_LINK_AUTHOR_RE
#
# A "name token" must have lowercase characters following the leading
# capital (e.g. `Jueun`), or be a single capital + period (e.g. `J.`).
# This excludes all-caps acronyms (KAIST, MIT, ACM) so the multi-token
# match stops at the first affiliation word.
_NAME_TOKEN = r"[A-Z](?:\.|[a-z][\w'.-]*)"
_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}"

# Author-footnote markers Docling emits next to names. Built from
# chr() so the source file stays ASCII-only (avoids RUF001 on the
# literal glyphs): U+2217 ASTERISK OPERATOR, U+2020 DAGGER,
# U+2021 DOUBLE DAGGER.
_FOOTNOTE_MARKERS = chr(0x2217) + chr(0x2020) + chr(0x2021)

_EMAIL_BEARING_AUTHOR_LINE_RE = re.compile(
    rf"""
    ^\s*                                # line start
    (?P<name>{_NAME})                   # the name (1-4 tokens; no acronyms)
    (?:\s|[{_FOOTNOTE_MARKERS}]|\*)     # whitespace, footnote marker, or ASCII asterisk
    .*?                                 # affiliation text (any)
    [\w._%+-]+@[\w.-]+\.[A-Za-z]{{2,}}  # an email anywhere in the rest
    """,
    re.MULTILINE | re.VERBOSE | re.UNICODE,
)

# A line of comma/and-separated name-shaped tokens.
_PLAIN_CSV_AUTHORS_LINE_RE = re.compile(
    rf"""
    ^\s*
    (?:
        {_NAME}                                       # first name
        (?:                                           # then 1+ separators + name
            (?:\s*,\s*(?:and\s+)?|\s+and\s+)
            {_NAME}
        )+
    )
    \s*$
    """,
    re.MULTILINE | re.VERBOSE,
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


def _extract_leading_authors(markdown: str) -> list[str]:
    """Lift authors from the post-title, pre-abstract region.

    Cascades three patterns until one yields names:

    1. `[Name](orcid|mailto:url)` markdown links — papers that
       Docling renders with orcid hyperlinks (e.g. ACM venues).
    2. Per-line `Name <affiliation> email@inst.tld` — common
       Docling output for arxiv preprints where each author has
       their own line with affiliation + contact mashed in.
    3. A single line of comma/and-separated Title-Case names
       (e.g. `Clement Wang, Antoine Vialle, Robin Vaysse, and
       Thomas Bonald`) — minimal-formatting preprints.

    Scans only the post-title region, stopping at the first
    content section heading (Abstract / Introduction / ...) or
    after 5000 chars to keep false-positives down. Deduplicates
    while preserving order.
    """
    cutoff = _AUTHOR_SCAN_HEADING_RE.search(markdown)
    region = markdown[: cutoff.start()] if cutoff else markdown[:5000]

    def _seen_list(names):
        seen: dict[str, None] = {}
        for n in names:
            cleaned = re.sub(r"\s+", " ", n).strip()
            if cleaned and cleaned not in seen:
                seen[cleaned] = None
        return list(seen)

    # Pattern 1: markdown-link authors (orcid / mailto).
    linked = [m.group(1) for m in _LEADING_MD_LINK_AUTHOR_RE.finditer(region)]
    if linked:
        return _seen_list(linked)

    # Pattern 2: email-bearing one-per-line author lines.
    email_form = [m.group("name") for m in _EMAIL_BEARING_AUTHOR_LINE_RE.finditer(region)]
    if email_form:
        return _seen_list(email_form)

    # Pattern 3: a single line of comma/and-separated Title-Case names.
    csv_match = _PLAIN_CSV_AUTHORS_LINE_RE.search(region)
    if csv_match:
        # _split_authors handles commas / semicolons / "and" / &.
        return _seen_list(_split_authors(csv_match.group(0)))

    return []


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
    else:
        leading = _extract_leading_authors(markdown)
        if leading:
            out["authors"] = leading

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
