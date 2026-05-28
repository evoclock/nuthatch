# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""SchemaProfile: per-corpus metadata contract aligned with kb-reports.md.

Purpose: declare the metadata fields a document must yield to be
    admitted into a corpus's graph. Pluggable per corpus; the built-in
    profiles live in `nuthatch.schema.profiles` and the four
    `arxiv_paper`, `biorxiv_paper`, `patent`, `internal_doc` are
    shipped out of the box. Users add their own via YAML or by
    subclassing `SchemaProfile`.

The field vocabulary aligns with the **kb-reports.md frontmatter
contract** at `~/project-planning-agent/conventions/kb-reports.md`
so that nuthatch's per-paper cards index alongside PhD KB reports
and proposal notes in Obsidian Dataview queries with a single
shared schema. Paper-specific fields (authors, doi, arxiv_id, year)
extend the kb-reports vocabulary; operational fields (relevance,
half_life_days, status) match it directly.

Inputs at validation time: a dict of extracted metadata (typically
    from `nuthatch.ingest.metadata`).

Outputs: `ValidationResult` with the list of missing required fields
    and the populated optional fields.

Assumptions: field validation is by presence and (optionally) type.
    Cross-field constraints live in the profile subclass, not in the
    base contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class FieldSpec:
    """A single field's contract within a profile."""

    name: str
    required: bool = True
    expected_type: type | tuple[type, ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of running a `SchemaProfile` against extracted metadata."""

    passed: bool
    missing_required: list[str]
    type_mismatches: list[tuple[str, str]]
    extracted: dict[str, Any]

    @property
    def reason(self) -> str | None:
        """Quarantine reason, or None if the doc passed."""
        if self.passed:
            return None
        parts = []
        if self.missing_required:
            parts.append("missing:" + ",".join(self.missing_required))
        if self.type_mismatches:
            parts.append("bad_type:" + ",".join(f"{n}({t})" for n, t in self.type_mismatches))
        return ";".join(parts)


# Fields shared across all kb-reports-aligned profiles. Profiles
# extend with paper-specific fields (authors, doi, arxiv_id) on top.
_KB_REPORTS_SHARED_FIELDS: tuple[FieldSpec, ...] = (
    # Required by kb-reports.md
    FieldSpec("title", required=True, expected_type=str),
    FieldSpec("date", required=False, expected_type=str),  # ingest-time default
    FieldSpec("status", required=False, expected_type=str),  # default 'exploratory'
    # Tags + committee_member: flat arrays, controlled vocab.
    FieldSpec("tags", required=False, expected_type=(list, tuple)),
    FieldSpec("committee_member", required=False, expected_type=(list, tuple)),
    # Relevance + decay: numeric defaults set by the card renderer.
    FieldSpec("relevance", required=False, expected_type=(int, float)),
    FieldSpec("half_life_days", required=False, expected_type=int),
    # Operational pointers.
    FieldSpec("aim_ref", required=False, expected_type=str),
    FieldSpec("parents", required=False, expected_type=(list, tuple)),
    FieldSpec("supersedes", required=False, expected_type=(list, tuple)),
)


class SchemaProfile:
    """Base class for per-corpus schema profiles.

    Subclass and override `fields` (or `profile_name` for instrumentation
    in the manifest). The base class provides field-presence and
    type-check validation; subclasses may add cross-field checks by
    overriding `validate`.

    All profiles inherit the kb-reports.md shared fields automatically
    via `_KB_REPORTS_SHARED_FIELDS`; subclass `fields` is concatenated
    with the shared set in `all_fields()`.
    """

    profile_name: ClassVar[str] = "base"
    fields: ClassVar[tuple[FieldSpec, ...]] = ()

    @classmethod
    def all_fields(cls) -> tuple[FieldSpec, ...]:
        """Per-class `fields` PLUS the kb-reports shared field set."""
        # If a subclass declares 'title' in its own fields, prefer that
        # (allows overriding required-ness or type). Dedup by name.
        seen: dict[str, FieldSpec] = {}
        for spec in cls.fields:
            seen[spec.name] = spec
        for spec in _KB_REPORTS_SHARED_FIELDS:
            seen.setdefault(spec.name, spec)
        return tuple(seen.values())

    @classmethod
    def required_field_names(cls) -> list[str]:
        return [f.name for f in cls.all_fields() if f.required]

    @classmethod
    def validate(cls, extracted: dict[str, Any]) -> ValidationResult:
        missing: list[str] = []
        bad_types: list[tuple[str, str]] = []
        for spec in cls.all_fields():
            value = extracted.get(spec.name)
            if value is None or value == "":
                if spec.required:
                    missing.append(spec.name)
                continue
            if spec.expected_type is not None and not isinstance(value, spec.expected_type):
                bad_types.append((spec.name, type(value).__name__))
        passed = not missing and not bad_types
        return ValidationResult(
            passed=passed,
            missing_required=missing,
            type_mismatches=bad_types,
            extracted=dict(extracted),
        )
