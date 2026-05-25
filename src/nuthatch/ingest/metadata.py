# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Metadata extraction + schema validation.

Purpose: take the markdown an extractor produced and lift the fields
    a `SchemaProfile` requires out of it. The current implementation
    is heuristic (regex + first-block parsing); a real Sprint 4
    metadata layer would use the structured layout JSON Docling
    produces. This module gives the orchestrator something to call
    today while keeping the contract stable.

Inputs: extracted markdown (one paper's content) + an active
    `SchemaProfile` subclass.

Outputs: `(extracted_metadata: dict, validation: ValidationResult)`.

Assumptions: the markdown is one paper's worth, not a multi-doc
    corpus. Heuristics target academic paper conventions (title in
    `# heading`, `arXiv:` / `doi:` markers, year as a 4-digit number).
    Subclass-specific overrides land in the profile, not here.
"""

from __future__ import annotations

import re

from nuthatch.schema.profile import SchemaProfile, ValidationResult

# Headers, identifiers, years. The patterns are deliberately broad;
# false positives are preferable to missing real values, because the
# schema validation step downstream will catch genuinely missing
# fields with a clear reason.
_TITLE_LINE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_ARXIV_ID_RE = re.compile(r"arXiv\s*[:=]?\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PATENT_NUMBER_RE = re.compile(
    r"\b(US|EP|WO)\d{6,10}[A-Z]?\d?\b", re.IGNORECASE
)


def extract_metadata_heuristic(markdown: str) -> dict[str, object]:
    """Best-effort metadata lift from a paper's extracted markdown.

    The orchestrator combines this with a `SchemaProfile.validate()`
    call to decide whether the doc passes the ingest gate.
    """
    out: dict[str, object] = {}

    title_match = _TITLE_LINE_RE.search(markdown)
    if title_match:
        out["title"] = title_match.group(1).strip()

    arxiv_match = _ARXIV_ID_RE.search(markdown)
    if arxiv_match:
        out["arxiv_id"] = arxiv_match.group(1)

    doi_match = _DOI_RE.search(markdown)
    if doi_match:
        out["doi"] = doi_match.group(1)

    patent_match = _PATENT_NUMBER_RE.search(markdown)
    if patent_match:
        out["patent_number"] = patent_match.group(0).upper()

    year_match = _YEAR_RE.search(markdown)
    if year_match:
        out["year"] = int(year_match.group(0))

    return out


def extract_and_validate(
    markdown: str, profile: type[SchemaProfile]
) -> tuple[dict[str, object], ValidationResult]:
    """Run heuristic extraction + profile validation in one call.

    Returns the extracted metadata dict and the validation result.
    The caller decides what to do on failure (typically quarantine
    via `nuthatch.ingest.quarantine.quarantine_file`).
    """
    extracted = extract_metadata_heuristic(markdown)
    result = profile.validate(extracted)
    return extracted, result
