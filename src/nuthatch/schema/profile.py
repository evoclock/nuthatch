# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""SchemaProfile: per-corpus metadata contract.

Purpose: declare the metadata fields a document must yield to be
    admitted into a corpus's graph. Pluggable per corpus; the built-in
    profiles live in `nuthatch.schema.profiles` and the four
    `arxiv_paper`, `biorxiv_paper`, `patent`, `internal_doc` are
    shipped out of the box. Users add their own via YAML or by
    subclassing `SchemaProfile`.

Inputs: at validation time, a dict of extracted metadata (typically
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
            parts.append(
                "bad_type:" + ",".join(f"{n}({t})" for n, t in self.type_mismatches)
            )
        return ";".join(parts)


class SchemaProfile:
    """Base class for per-corpus schema profiles.

    Subclass and override `fields` (or `profile_name` for instrumentation
    in the manifest). The base class provides field-presence and
    type-check validation; subclasses may add cross-field checks by
    overriding `validate`.
    """

    profile_name: ClassVar[str] = "base"
    fields: ClassVar[tuple[FieldSpec, ...]] = ()

    @classmethod
    def required_field_names(cls) -> list[str]:
        return [f.name for f in cls.fields if f.required]

    @classmethod
    def validate(cls, extracted: dict[str, Any]) -> ValidationResult:
        missing: list[str] = []
        bad_types: list[tuple[str, str]] = []
        for spec in cls.fields:
            value = extracted.get(spec.name)
            if value is None or value == "":
                if spec.required:
                    missing.append(spec.name)
                continue
            if spec.expected_type is not None and not isinstance(
                value, spec.expected_type
            ):
                bad_types.append((spec.name, type(value).__name__))
        passed = not missing and not bad_types
        return ValidationResult(
            passed=passed,
            missing_required=missing,
            type_mismatches=bad_types,
            extracted=dict(extracted),
        )
