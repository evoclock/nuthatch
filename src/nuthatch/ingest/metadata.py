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
# those levels that ISN'T a section name (Abstract / Introduction /
# Methods / ...). Body of a paper sometimes has the Abstract heading
# appear before the actual title heading on the rendered page (or the
# title is plain text with no heading at all), so we have to filter
# out section names rather than just taking the first match.
_HEADING_TITLE_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
_TITLE_LABEL_RE = re.compile(r"^\s*Title:\s*(.+?)\s*$", re.MULTILINE)

# Headings that are section names, not paper titles. Anything starting
# with one of these (after stripping trailing digits / punctuation /
# colons) is rejected as a title candidate. Real titles are rarely a
# single word matching one of these.
_SECTION_HEADING_NAMES: frozenset[str] = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "summary",
        "overview",
        "methods",
        "method",
        "materials and methods",
        "materials",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "acknowledgements",
        "acknowledgments",
        "supplementary",
        "appendix",
        "data availability",
        "author contributions",
        "funding",
        "conflict of interest",
        "ethics statement",
        "ethics declaration",
        "keywords",
        "main",
        "main text",
        "results and discussion",
    }
)
_ARXIV_ID_RE = re.compile(r"arXiv\s*[:=]?\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PATENT_NUMBER_RE = re.compile(r"\b(US|EP|WO)\d{6,10}[A-Z]?\d?\b", re.IGNORECASE)
_AUTHORS_LINE_RE = re.compile(r"^\s*(?:Authors?:|By)\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_ABSTRACT_HEADING_RE = re.compile(
    r"^#{1,3}\s*Abstract\s*\n+(.+?)(?:\n#{1,3}\s|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)

# bioRxiv-style papers often render the abstract as the first long
# paragraph after authors+affiliations with NO heading at all. When
# `_ABSTRACT_HEADING_RE` misses, this fallback picks up such a block:
# the first paragraph >= 250 chars that does NOT look like an
# author/affiliation line (no email, no digit prefix, no all-caps run).
_LONG_PARAGRAPH_MIN_CHARS: int = 250

# Affiliation markers that get glued to author surnames in Docling
# output: digits, asterisks (regular + operator), daggers, double
# daggers, section, pilcrow, plus parenthesised affiliation groups.
# Stripped from candidate author lines before applying the CSV regex,
# so `Buralkin1,2,3` becomes `Buralkin` and the multi-author line can
# be parsed.
_AFFIL_MARKER_RE = re.compile(
    r"[\d"
    + chr(0x2217)
    + chr(0x2020)
    + chr(0x2021)
    + r"\*"
    + chr(0x00A7)
    + chr(0x00B6)
    + r"\#"
    + r"]+|\([^)]+\)"
)

# LaTeX-math superscripts that some bioRxiv preprints render
# explicitly as `$^{1}$`, `$^{2,3}$`, `$^{*}$`, `$^{3*}$` after
# Docling conversion. Treated as affiliation markers for stripping
# purposes; matches the whole LaTeX expression so the resulting
# author line is `Author, Author, Author` with nothing leftover.
_LATEX_SUPERSCRIPT_RE = re.compile(r"\$\^\{[^}]*\}\$|\^\{[^}]*\}")

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
# capital (e.g. `Jueun`), OR be one or more capital-letter initials
# joined by periods (e.g. `J.` or `S.A.` or `S.A.B.`). This excludes
# all-caps acronyms (KAIST, MIT, ACM) so the multi-token match stops
# at the first affiliation word.
_NAME_TOKEN = r"[A-Z](?:(?:\.[A-Z])*\.|[a-z][\w'.-]*)"
_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}"

# Author-footnote markers Docling emits next to names. Built from
# chr() so the source file stays ASCII-only (avoids RUF001 on the
# literal glyphs): U+2217 ASTERISK OPERATOR, U+2020 DAGGER,
# U+2021 DOUBLE DAGGER.
_FOOTNOTE_MARKERS = chr(0x2217) + chr(0x2020) + chr(0x2021)

# Pattern 2 requires 2+ name tokens so single capitalised words
# in boilerplate phrases like "Correspondence should be addressed
# to H.S. (Heewon.Seo@ucalgary.ca)" don't pollute the author list.
# Real single-author papers are handled by the pattern-4 fallback
# which requires email-surname confirmation.
_NAME_2PLUS = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{1,3}}"

_EMAIL_BEARING_AUTHOR_LINE_RE = re.compile(
    rf"""
    ^\s*                                # line start
    (?P<name>{_NAME_2PLUS})             # the name (2-4 tokens; no acronyms)
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


def _looks_like_section_heading(heading: str) -> bool:
    """True when a heading is a known section name, not a paper title.

    Strips trailing digits, punctuation, and colons so `Abstract 10:`,
    `Methods (a)`, and `Introduction.` all match the canonical names
    in `_SECTION_HEADING_NAMES`. A real paper title rarely reduces to
    one of those tokens.
    """
    if not heading:
        return False
    normalised = re.sub(r"[\W\d]+$", "", heading.strip()).strip().lower()
    if normalised in _SECTION_HEADING_NAMES:
        return True
    # `Abstract: ...` style: take the prefix before the colon.
    prefix = normalised.split(":", 1)[0].strip()
    return prefix in _SECTION_HEADING_NAMES


def _extract_title(markdown: str) -> str | None:
    """Pick the paper title from `Title:` label, heading, or table cell.

    Cascades three sources in order:

    1. `Title:` labelled line (`_TITLE_LABEL_RE`).
    2. First `#{1,3} heading` whose normalised stem is NOT a known
       section name (filters out `## Abstract` / `## Introduction`
       headings Docling sometimes emits before the actual title).
    3. First markdown-table cell in the document's leading region
       whose contents look like a title (long enough, not a section
       name) — bioRxiv preprints with line-numbered Word manuscripts
       can come through as `| 1 | Title... |` tables. Stops at the
       first match.

    Returns None when none of the three find a candidate.
    """
    labelled = _first_match(_TITLE_LABEL_RE, markdown)
    if labelled:
        return labelled
    for m in _HEADING_TITLE_RE.finditer(markdown):
        candidate = _LATEX_SUPERSCRIPT_RE.sub("", m.group(1)).strip()
        # Strip leading line numbers (`## 1 Title text`) and `Title:`
        # labels (`## Title: Real title`) introduced by Docling on
        # line-numbered Word manuscripts.
        candidate = re.sub(r"^\d+\s+", "", candidate).strip()
        candidate = re.sub(r"^Title:\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s+\d+\s*$", "", candidate).strip()  # trailing line number
        if not candidate or _looks_like_section_heading(candidate):
            continue
        return candidate
    # Fallback: walk the first ~30 non-blank table-cell lines.
    seen_lines = 0
    for line in markdown.split("\n"):
        if seen_lines >= 30:
            break
        cell = _maybe_table_cell(line)
        if not cell or cell == line:
            continue
        seen_lines += 1
        cleaned = _LATEX_SUPERSCRIPT_RE.sub("", cell).strip()
        cleaned = re.sub(r"\s+\d+\s*$", "", cleaned).strip()
        if not cleaned or _looks_like_section_heading(cleaned):
            continue
        # Skip cells that look like author CSV lines (have commas
        # and Title-Case run patterns); titles are usually free prose.
        if "," in cleaned and _PLAIN_CSV_AUTHORS_LINE_RE.fullmatch(_strip_affil_markers(cleaned)):
            continue
        if len(cleaned) >= 20:
            return cleaned
    return None


def _strip_line_prefix(line: str) -> str:
    """Strip leading markdown list / heading markers and line numbers.

    Docling renders the author block of a line-numbered Word manuscript
    (the journal-review default for biology submissions) as either:

        - Chenwei Zhou, 1 Chanjuan Dong, 1 Weiye Zhao, 1 and Fu-Sen Liang 1, *
        1 Xiaoqin Huang1, Ivan Ovcharenko1*

    Both forms break the CSV-author regex which expects `^name, name, ...`.
    Stripping `- ` / `* ` / `<digit>. ` list markers + a leading
    standalone line-number lets the same regex parse the cleaned line.
    """
    cleaned = re.sub(r"^[-*]\s+|^\d+[.)]\s+", "", line)
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    return cleaned


def _strip_affil_markers(line: str) -> str:
    """Remove affiliation digits / asterisks / daggers from an author line.

    Docling glues affiliation references directly onto surnames
    (`Buralkin1,2,3`, `Park2,3,*`, `Mattick† 1`). Some bioRxiv
    preprints encode the markers as explicit LaTeX-math superscripts
    (`Sun$^{1}$, Choi$^{2}$, Yin$^{3*}$`). Naive removal of marker
    characters alone leaves stray punctuation (`Buralkin,, ,`) that
    the multi-name regex cannot recover from; we additionally
    collapse runs of `,` and surrounding whitespace introduced by
    the strip.
    """
    cleaned = _LATEX_SUPERSCRIPT_RE.sub("", line)
    cleaned = _AFFIL_MARKER_RE.sub("", cleaned)
    # Collapse comma runs introduced when the marker was sandwiched
    # between commas: `Buralkin,, , Hu` -> `Buralkin , Hu`.
    cleaned = re.sub(r"(?:\s*,)+\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    # Trim commas glued to the start/end (no leading/trailing comma
    # is a valid CSV-author shape).
    cleaned = re.sub(r"^\s*,+\s*|\s*,+\s*$", "", cleaned)
    # Collapse whitespace runs.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _maybe_table_cell(line: str) -> str:
    """If `line` is a markdown table row, return its rightmost non-empty cell.

    Docling renders some bioRxiv title pages as multi-column tables
    (line-number column + content column). The data we care about
    lives in the last cell; the leading `|` and number columns are
    structure, not content. Lines that are not table rows pass through
    unchanged. Separator rows (`|---|---|`) collapse to empty so the
    caller skips them.
    """
    s = line.strip()
    if not (s.startswith("|") and s.count("|") >= 2):
        return line
    cells = [c.strip() for c in s.split("|") if c.strip()]
    if not cells:
        return ""
    # If every cell is just dashes (separator row), return empty.
    if all(set(c) <= set("-:") for c in cells):
        return ""
    return cells[-1]


def _extract_abstract(markdown: str) -> str | None:
    """Lift the abstract via heading match, falling back to first long paragraph.

    bioRxiv-style papers often render the abstract as a paragraph
    with NO `## Abstract` heading. The fallback picks the first
    paragraph >= `_LONG_PARAGRAPH_MIN_CHARS` chars in the post-
    title region that doesn't look like an author/affiliation
    line (no `@` email, no leading digits, no all-caps acronym
    line, not a heading).
    """
    m = _ABSTRACT_HEADING_RE.search(markdown)
    if m:
        cleaned = re.sub(r"\s+", " ", m.group(1)).strip()
        if cleaned:
            return cleaned

    # Fallback: walk paragraphs after the title, return the first
    # long, prose-shaped one.
    paragraphs = re.split(r"\n\s*\n", markdown[:20000])
    for para in paragraphs:
        text = re.sub(r"\s+", " ", para).strip()
        if len(text) < _LONG_PARAGRAPH_MIN_CHARS:
            continue
        if text.startswith("#"):
            continue
        if "@" in text and re.search(r"[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
            continue
        # bioRxiv watermark and license boilerplate to skip.
        if text.startswith("bioRxiv preprint") or "CC-BY" in text or "All rights reserved" in text:
            continue
        return text
    return None


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

    def _seen_list(names: list[str]) -> list[str]:
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
    # Sometimes the line wraps across 2-3 markdown lines (an ORCID
    # parenthesis crosses a soft break, etc.), so we walk a 3-line
    # sliding window and try each join. For each candidate string,
    # apply FOUR progressive cleaners and try the CSV regex on each:
    #   1. raw line(s)
    #   2. + extract rightmost table cell when it's a `|...|` row
    #   3. + strip leading markdown list marker / line number
    #   4. + strip affiliation markers (digits / `*` / daggers /
    #      LaTeX-math superscripts) glued to surnames
    # First match wins.
    lines = region.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        # Build join variants: this line alone + this+next + ... up to
        # 6 lines total. Highly-affiliated author lists (every author
        # with 7 superscripts) can wrap onto 5+ lines after Docling.
        variants: list[str] = []
        accumulated: list[str] = []
        j = i
        while len(accumulated) < 6 and j < len(lines):
            t = lines[j].strip()
            if t:
                accumulated.append(t)
                variants.append(" ".join(accumulated))
            j += 1

        # Try variants LONGEST -> SHORTEST so highly-affiliated author
        # lists that wrap across many lines return the full author
        # list, not just the first few names visible on the first
        # joined slice.
        for raw in reversed(variants):
            text = raw
            in_table = _maybe_table_cell(text)
            if in_table and in_table != text:
                text = in_table
            depref = _strip_line_prefix(text)
            for candidate in (text, depref, _strip_affil_markers(depref)):
                if not candidate:
                    continue
                csv_match = _PLAIN_CSV_AUTHORS_LINE_RE.fullmatch(candidate)
                if csv_match:
                    return _seen_list(_split_authors(csv_match.group(0)))

    # Pattern 4: SINGLE-AUTHOR papers. The CSV regex requires 2+ names;
    # papers with one author would never match. Look for a standalone
    # name line (1-4 capitalized tokens, no comma, no body text) in
    # the post-title region. Confirm by checking a nearby line carries
    # an email matching the surname.
    name_only_re = re.compile(rf"^\s*({_NAME})\s*$")
    for i, line in enumerate(lines):
        text = _strip_line_prefix(_strip_affil_markers(line.strip()))
        if not text:
            continue
        m = name_only_re.fullmatch(text)
        if not m:
            continue
        name = m.group(1)
        # Confirm: look in the next 8 lines for an email that mentions
        # the surname (case-insensitive). Avoids accepting random
        # capitalised words like "Introduction" as a name.
        surname = name.split()[-1].lower()
        confirmed = False
        for nxt in lines[i + 1 : i + 9]:
            if re.search(rf"[\w._%+-]*{re.escape(surname)}[\w._%+-]*@", nxt, re.IGNORECASE):
                confirmed = True
                break
        if confirmed:
            return [name]

    return []


def extract_metadata_heuristic(markdown: str) -> dict[str, Any]:
    """Heuristic metadata lift. Best-effort; conservative on false positives.

    Fields populated: title, authors, year, doi, arxiv_id, patent_number,
    abstract. Optional list fields (key_claims, methods, topics,
    relevance) are left empty; a plugin extractor fills them.
    """
    out: dict[str, Any] = {}

    title = _extract_title(markdown)
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

    abstract = _extract_abstract(markdown)
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

        source_meta = enrich_from_source(source_filename, cache_dir=metadata_cache_dir)
        if source_meta is not None:
            source_extracted = source_meta.to_dict()

    # Source-metadata takes precedence on overlapping fields: the
    # publisher record is authoritative for title / authors / etc.,
    # the body is authoritative for everything else (key claims,
    # topics, methods narrative).
    merged: dict[str, Any] = {**body_extracted, **source_extracted}

    result = profile.validate(merged)
    return merged, result
