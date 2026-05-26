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

    def test_pulls_title_from_h2_docling_style(self) -> None:
        # Docling emits ## for paper titles, not #. The heuristic must
        # cope with that or every Docling-converted PDF loses its title.
        md = "## FORGE: Self-Evolving Agent Memory\n\nBody text."
        assert (
            extract_metadata_heuristic(md)["title"]
            == "FORGE: Self-Evolving Agent Memory"
        )

    def test_pulls_authors_from_leading_orcid_links(self) -> None:
        md = (
            "## My Paper\n\n"
            "## [Alice Smith](https://orcid.org/0000-0001-2345-6789)\n\n"
            "alice@example.org University of Foo\n\n"
            "[Bob Jones](https://orcid.org/0000-0002-3456-7890)\n\n"
            "## Abstract\n\n"
            "We did stuff.\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Alice Smith", "Bob Jones"]

    def test_authors_label_still_takes_precedence(self) -> None:
        # Explicit `Authors:` line wins over the leading-link heuristic
        # when both are present.
        md = (
            "## My Paper\n\n"
            "Authors: Carol Lin, Dave Park\n\n"
            "[Bob Jones](https://orcid.org/0000-0002-3456-7890)\n\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Carol Lin", "Dave Park"]

    def test_leading_link_authors_stop_at_abstract(self) -> None:
        # Links inside the abstract / body must NOT be parsed as authors.
        md = (
            "## My Paper\n\n"
            "[Alice Smith](https://orcid.org/0000-0001-2345-6789)\n\n"
            "## Abstract\n\n"
            "See related work by [Eve Watson](https://orcid.org/9999).\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Alice Smith"]

    def test_pulls_authors_from_email_bearing_lines(self) -> None:
        # Docling's typical output for arxiv preprints: one author per
        # line with affiliation and email mashed together. No orcid links.
        md = (
            "## AMUSE: Anytime Muon with Stable Gradient Evaluation\n\n"
            "Jueun Kim KAIST jueunkim@kaist.ac.kr\n\n"
            "Jihun Yun KRAFTON jihuny@krafton.com\n\n"
            "Chulhee Yun KAIST chulhee.yun@kaist.ac.kr\n\n"
            "## Abstract\n\n"
            "Modern deep learning relies on AdamW...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Jueun Kim", "Jihun Yun", "Chulhee Yun"]

    def test_pulls_authors_from_plain_csv_line(self) -> None:
        # Minimal-formatting preprints render authors as a single line
        # of comma- and "and"-separated names.
        md = (
            "## Implicit Regularization of Mini-Batch Training in GNNs\n\n"
            "Clement Wang, Antoine Vialle, Robin Vaysse, and Thomas Bonald\n\n"
            "Institut Polytechnique de Paris\n\n"
            "## Abstract\n\n"
            "Mini-batch training of GNNs...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == [
            "Clement Wang",
            "Antoine Vialle",
            "Robin Vaysse",
            "Thomas Bonald",
        ]

    def test_no_false_positive_authors_from_body_text(self) -> None:
        # Body sentences that happen to contain capitalized words
        # must NOT be parsed as authors. The post-title region is
        # bounded by the first content-section heading.
        md = (
            "## A Paper Title\n\n"
            "## Abstract\n\n"
            "We compare Adam, SGD, and Muon. Authors of those "
            "optimizers include Kingma, Loshchilov, and others.\n"
        )
        out = extract_metadata_heuristic(md)
        # No authors found in the post-title pre-abstract region
        # (which is empty here).
        assert "authors" not in out

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
        _extracted, validation = extract_and_validate(md, _TitleAndYearProfile)
        assert not validation.passed
        assert "title" in validation.missing_required

    def test_missing_year_fails(self) -> None:
        md = "# Headed but no year"
        _extracted, validation = extract_and_validate(md, _TitleAndYearProfile)
        assert not validation.passed
        assert "year" in validation.missing_required
