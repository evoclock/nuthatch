# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.metadata`."""

from __future__ import annotations

from nuthatch.ingest.metadata import extract_and_validate, extract_metadata_heuristic
from nuthatch.schema.profile import FieldSpec, SchemaProfile


class _TitleAndYearProfile(SchemaProfile):
    profile_name = "test_title_year"
    fields = (
        FieldSpec("title", required=True, expected_type=str),
        FieldSpec("year", required=True, expected_type=int),
    )


class TestExtractMetadataHeuristic:
    def test_pulls_title_from_h1(self) -> None:
        md = "# A great paper\n\nBody text."
        assert extract_metadata_heuristic(md)["title"] == "A great paper"

    def test_pulls_year(self) -> None:
        md = "Published in 2017 in Nature."
        assert extract_metadata_heuristic(md)["year"] == 2017

    def test_pulls_doi(self) -> None:
        md = "DOI: 10.1038/nature12345 for this paper."
        assert extract_metadata_heuristic(md)["doi"] == "10.1038/nature12345"

    def test_pulls_arxiv_id(self) -> None:
        md = "See arXiv:1706.03762v5 for the original."
        assert extract_metadata_heuristic(md)["arxiv_id"] == "1706.03762v5"

    def test_pulls_patent_number(self) -> None:
        md = "Patent US12345678B2 was issued."
        assert extract_metadata_heuristic(md)["patent_number"] == "US12345678B2"

    def test_empty_input_yields_empty_dict(self) -> None:
        assert extract_metadata_heuristic("") == {}


class TestExtractAndValidate:
    def test_passing_doc(self) -> None:
        md = "# Hello world\n\nBody text from 2024 about things."
        extracted, validation = extract_and_validate(md, _TitleAndYearProfile)
        assert validation.passed
        assert extracted["title"] == "Hello world"
        assert extracted["year"] == 2024

    def test_missing_title_fails(self) -> None:
        md = "No heading here. Year 2024."
        extracted, validation = extract_and_validate(md, _TitleAndYearProfile)
        assert not validation.passed
        assert "title" in validation.missing_required

    def test_missing_year_fails(self) -> None:
        md = "# Headed but no year"
        extracted, validation = extract_and_validate(md, _TitleAndYearProfile)
        assert not validation.passed
        assert "year" in validation.missing_required
