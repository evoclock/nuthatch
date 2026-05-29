# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the SchemaProfile base + the four built-in profiles."""

from __future__ import annotations

import pytest

from nuthatch.schema.profile import FieldSpec, SchemaProfile, ValidationResult
from nuthatch.schema.profiles import (
    BUILTIN_PROFILES,
    ArxivPaperProfile,
    BiorxivPaperProfile,
    InternalDocProfile,
    PatentProfile,
)


class _MiniProfile(SchemaProfile):
    profile_name = "mini"
    fields = (
        FieldSpec("title", required=True, expected_type=str),
        FieldSpec("year", required=True, expected_type=int),
        FieldSpec("tags", required=False, expected_type=(list, tuple)),
    )


class TestBaseProfile:
    def test_fully_populated_passes(self) -> None:
        result = _MiniProfile.validate({"title": "Hello", "year": 2026, "tags": ["a", "b"]})
        assert result.passed
        assert result.reason is None
        assert result.missing_required == []
        assert result.type_mismatches == []

    def test_missing_required_fails(self) -> None:
        result = _MiniProfile.validate({"title": "Hello"})
        assert not result.passed
        assert "year" in result.missing_required
        assert result.reason is not None
        assert "year" in result.reason

    def test_empty_string_treated_as_missing(self) -> None:
        result = _MiniProfile.validate({"title": "", "year": 2026})
        assert not result.passed
        assert "title" in result.missing_required

    def test_bad_type_fails(self) -> None:
        result = _MiniProfile.validate({"title": "Hello", "year": "2026"})
        assert not result.passed
        assert ("year", "str") in result.type_mismatches
        assert result.reason is not None
        assert "bad_type" in result.reason

    def test_optional_field_absence_ok(self) -> None:
        result = _MiniProfile.validate({"title": "Hello", "year": 2026})
        assert result.passed

    def test_extra_fields_ignored(self) -> None:
        result = _MiniProfile.validate({"title": "Hello", "year": 2026, "rogue": "value"})
        assert result.passed
        assert result.extracted["rogue"] == "value"

    def test_required_field_names(self) -> None:
        assert _MiniProfile.required_field_names() == ["title", "year"]


class TestBuiltinProfilesRegistry:
    def test_all_four_registered(self) -> None:
        assert set(BUILTIN_PROFILES.keys()) == {
            "arxiv_paper",
            "biorxiv_paper",
            "patent",
            "internal_doc",
        }

    @pytest.mark.parametrize("name,cls", list(BUILTIN_PROFILES.items()))
    def test_profile_name_matches_registry_key(self, name: str, cls: type) -> None:
        assert cls.profile_name == name

    @pytest.mark.parametrize("cls", list(BUILTIN_PROFILES.values()))
    def test_profile_has_at_least_one_required_field(self, cls: type) -> None:
        assert len(cls.required_field_names()) > 0


class TestArxivProfile:
    def test_minimal_arxiv_passes(self) -> None:
        result = ArxivPaperProfile.validate(
            {
                "title": "Attention Is All You Need",
                "authors": ["Vaswani", "Shazeer"],
                "abstract": "We propose a new...",
                "arxiv_id": "1706.03762",
                "year": 2017,
            }
        )
        assert result.passed

    def test_arxiv_without_id_fails(self) -> None:
        result = ArxivPaperProfile.validate(
            {
                "title": "Untitled",
                "authors": ["Anon"],
                "abstract": "Body",
                "year": 2026,
            }
        )
        assert not result.passed
        assert "arxiv_id" in result.missing_required


class TestBiorxivProfile:
    def test_minimal_biorxiv_passes(self) -> None:
        result = BiorxivPaperProfile.validate(
            {
                "title": "A study",
                "authors": ("Smith", "Jones"),
                "abstract": "...",
                "doi": "10.1101/2026.05.25.999999",
                "year": 2026,
            }
        )
        assert result.passed

    def test_biorxiv_without_doi_fails(self) -> None:
        result = BiorxivPaperProfile.validate(
            {
                "title": "A study",
                "authors": ["Smith"],
                "abstract": "...",
                "year": 2026,
            }
        )
        assert not result.passed
        assert "doi" in result.missing_required


class TestPatentProfile:
    def test_minimal_patent_passes(self) -> None:
        result = PatentProfile.validate(
            {
                "title": "Method and apparatus for X",
                "patent_number": "US12345678B2",
                "inventors": ["Jane Doe"],
                "filing_date": "2024-01-15",
                "publication_date": "2026-03-12",
                "abstract": "Disclosed is a method...",
            }
        )
        assert result.passed


class TestInternalDocProfile:
    def test_minimal_internal_doc_passes(self) -> None:
        result = InternalDocProfile.validate(
            {
                "title": "Sprint 2 plan",
                "author": "Julen",
                "date": "2026-05-25",
            }
        )
        assert result.passed


class TestValidationResultReason:
    def test_reason_none_on_pass(self) -> None:
        r = ValidationResult(passed=True, missing_required=[], type_mismatches=[], extracted={})
        assert r.reason is None

    def test_reason_includes_missing_and_types(self) -> None:
        r = ValidationResult(
            passed=False,
            missing_required=["title"],
            type_mismatches=[("year", "str")],
            extracted={},
        )
        assert r.reason is not None
        assert "missing:title" in r.reason
        assert "bad_type:year(str)" in r.reason
