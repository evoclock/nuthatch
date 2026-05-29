# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for filename-based schema-profile routing."""

from __future__ import annotations

from nuthatch.ingest.profile_routing import select_profile_for_filename
from nuthatch.schema.profile import FieldSpec, SchemaProfile
from nuthatch.schema.profiles import (
    ArxivPaperProfile,
    BiorxivPaperProfile,
    InternalDocProfile,
)


class _FallbackProfile(SchemaProfile):
    profile_name = "test_fallback"
    fields = (FieldSpec("title", required=True, expected_type=str),)


class TestSelectProfileForFilename:
    def test_arxiv_filename_routes_to_arxiv(self) -> None:
        assert select_profile_for_filename("2605.15308v1.pdf") is ArxivPaperProfile
        assert select_profile_for_filename("2605.15308.pdf") is ArxivPaperProfile

    def test_biorxiv_filename_routes_to_biorxiv(self) -> None:
        assert select_profile_for_filename("2026.05.14.725010v1.full.pdf") is BiorxivPaperProfile
        assert select_profile_for_filename("2021.12.06.471493v2.full.pdf") is BiorxivPaperProfile

    def test_unknown_filename_uses_default_fallback(self) -> None:
        assert select_profile_for_filename("internal_memo.pdf") is InternalDocProfile
        assert (
            select_profile_for_filename("Mendel_1866_Bateson_translation.pdf") is InternalDocProfile
        )

    def test_fallback_is_overridable(self) -> None:
        assert (
            select_profile_for_filename("Mendel_1866.pdf", fallback=_FallbackProfile)
            is _FallbackProfile
        )

    def test_arxiv_takes_precedence_over_fallback(self) -> None:
        # An arxiv filename should always pick ArxivPaperProfile, even
        # when the caller supplied a custom fallback.
        assert (
            select_profile_for_filename("2605.15308v1.pdf", fallback=_FallbackProfile)
            is ArxivPaperProfile
        )
