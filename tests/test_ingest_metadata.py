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

    def test_title_skips_section_heading_when_picking_first_h2(self) -> None:
        # Docling sometimes emits the Abstract section heading BEFORE
        # the real title heading. A naive "first heading wins" rule
        # would pick "Abstract" as the title; the section-name filter
        # must reject it.
        md = (
            "## Abstract\n\n"
            "Random text that looks like an abstract block but isn't.\n\n"
            "## Robust Random Forests for Genomic Prediction\n\n"
            "Vanda M. Lourenco, Joseph O. Ogutu, Hans-Peter Piepho\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["title"] == "Robust Random Forests for Genomic Prediction"

    def test_title_skips_abstract_with_trailing_digits(self) -> None:
        # Real-world case: `## Abstract 10` from a numbered-line PDF
        # got picked as the title. The section-name filter strips
        # trailing digits / punctuation before matching.
        md = (
            "## Abstract 10\n\n"
            "boilerplate\n\n"
            "## Actual Paper Title Goes Here\n\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["title"] == "Actual Paper Title Goes Here"

    def test_csv_authors_with_embedded_affiliation_digits(self) -> None:
        # Docling output: `Ilia Buralkin1,2,3 , Hu Chen2,3 , ...`
        # The stripped form (digits + asterisks removed) should parse
        # as four authors. Without this fix, the line failed to match
        # because affiliation markers aren't part of name tokens.
        md = (
            "## scDeepVariant\n\n"
            "Ilia Buralkin1,2,3 , Hu Chen2,3 , Zhandong Liu2,3,* , and Junseok Park2,3,*\n\n"
            "## Abstract\n\n"
            "We introduce scDeepVariant...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == [
            "Ilia Buralkin",
            "Hu Chen",
            "Zhandong Liu",
            "Junseok Park",
        ]

    def test_csv_authors_with_daggers(self) -> None:
        # Dagger + space + digit pattern: `Name† 1 , Name† 1 , and Name1`
        md = (
            "## Inferring Gene Presence\n\n"
            "John S.A. Mattick" + chr(0x2020) + " 1 , Wesley C. DeMontigny"
            + chr(0x2020) + " 1 , and Charles F. Delwiche1\n\n"
            "## Abstract\n\n"
            "Increasing access...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == [
            "John S.A. Mattick",
            "Wesley C. DeMontigny",
            "Charles F. Delwiche",
        ]

    def test_abstract_fallback_first_long_paragraph_no_heading(self) -> None:
        # bioRxiv preprints often render the abstract as a paragraph
        # WITHOUT a `## Abstract` heading. The fallback picks the
        # first long prose paragraph in the post-title region.
        long_para = (
            "Increasing access to genomic data has revolutionized our "
            "understanding of biology. Organisms that were previously "
            "unculturable or otherwise difficult to study have been "
            "investigated using metagenomic sequencing and bioinformatic "
            "assemblies, illuminating biological diversity that was "
            "previously invisible. However, as the availability of "
            "genomic data has grown, so has the challenge posed by "
            "incomplete genomes."
        )
        md = f"## A Paper Title\n\nAuthors etc.\n\n{long_para}\n"
        out = extract_metadata_heuristic(md)
        assert out["abstract"].startswith("Increasing access to genomic data")

    def test_csv_authors_rendered_as_markdown_list_item(self) -> None:
        # Real-world: line-numbered Word manuscripts on bioRxiv get
        # rendered by Docling as `- Author, Author, ...` list items
        # with trailing line numbers. The strip-line-prefix step
        # must remove the `- ` marker so the CSV regex can match.
        md = (
            "## Hierarchical Interplay 1\n\n"
            "- Chenwei Zhou, 1 Chanjuan Dong, 1 Weiye Zhao, 1 and Fu-Sen Liang 1, * 2\n\n"
            "## SUMMARY\n\n"
            "H3K27ac and H3K4me3...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == [
            "Chenwei Zhou",
            "Chanjuan Dong",
            "Weiye Zhao",
            "Fu-Sen Liang",
        ]

    def test_csv_authors_with_leading_line_number(self) -> None:
        # Word manuscripts with margin line-numbers can also yield
        # plain `1 Author, Author` style after Docling.
        md = (
            "## A Paper 1\n\n"
            "2 Xiaoqin Huang1, Ivan Ovcharenko1*\n\n"
            "## ABSTRACT\n\n"
            "We present...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Xiaoqin Huang", "Ivan Ovcharenko"]

    def test_csv_authors_with_latex_superscript_markers(self) -> None:
        # Some bioRxiv preprints encode affiliation markers as
        # explicit LaTeX math: `Sun$^{1}$, Choi$^{2}$, Yin$^{3*}$`.
        # The strip must remove the whole `$^{...}$` expression.
        md = (
            "## AI predictions and the expansion of scientific frontiers\n\n"
            "Mengyi Sun$^{1}$, Sukwoong Choi$^{2}$, Yian Yin$^{3*}$\n\n"
            "## Abstract\n\n"
            "We show that...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Mengyi Sun", "Sukwoong Choi", "Yian Yin"]

    def test_title_strips_leading_line_number(self) -> None:
        # Line-numbered Word manuscripts get rendered by Docling as
        # `## 1 Real Title Text` — the leading line number must be
        # stripped from the title candidate.
        md = (
            "## 1 Loss of Adaptive Capacity Drives Climate Vulnerability\n\n"
            "Kristen Ruegg, Christen M. Bossu\n\n"
            "## Abstract\n\n"
            "We show...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["title"] == "Loss of Adaptive Capacity Drives Climate Vulnerability"

    def test_title_strips_label_prefix(self) -> None:
        # Some PDFs render `## Title: Real Title Here`; the label
        # is metadata, not part of the title.
        md = (
            "## Title: Combinatorial transcription factor interactions\n\n"
            "Author A, Author B\n\n"
            "## Abstract\n\n"
            "We show...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["title"] == "Combinatorial transcription factor interactions"

    def test_csv_authors_with_hash_co_first_author_marker(self) -> None:
        # bioRxiv co-first-author convention: `Author 1#, Author 1#`.
        # `#` must strip alongside digits + asterisks.
        md = (
            "## Evo 2 Predicts Cardiomyopathy-Associated Variants\n\n"
            "Atsumasa Kurozumi 1#, Naoto Otsuka 1#, Masamichi Ito 1, "
            "Toshinaru Kawakami 1, Takayuki Isagawa 2, Satoshi Kodera 1, "
            "Norihiko Takeda 1,2\n\n"
            "## Abstract\n\n"
            "We predict...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == [
            "Atsumasa Kurozumi",
            "Naoto Otsuka",
            "Masamichi Ito",
            "Toshinaru Kawakami",
            "Takayuki Isagawa",
            "Satoshi Kodera",
            "Norihiko Takeda",
        ]

    def test_csv_authors_wrapping_many_lines_via_sliding_window(self) -> None:
        # Highly-affiliated author lists (each author has 7 superscripts)
        # can wrap across 5+ markdown lines. Sliding window must
        # accumulate enough lines.
        md = (
            "## A Paper\n\n"
            "Tianchu Zeng1,2,3,4,5,6,7*, Hetu Li1,3,4,5,6,7*,\n"
            "Shaoshi Zhang1,2,3,4,5,6,7,8*,\n"
            "Yan Quan Tan1,2,3,4,5,6,7, Fang Tian1,2,3,4,5,6,7,\n"
            "Csaba Orban1,3,4,5,6,7\n\n"
            "## Abstract\n\n"
            "We show...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == [
            "Tianchu Zeng",
            "Hetu Li",
            "Shaoshi Zhang",
            "Yan Quan Tan",
            "Fang Tian",
            "Csaba Orban",
        ]

    def test_single_author_paper(self) -> None:
        # Single-author papers don't match the CSV regex (requires 2+
        # names). The pattern-4 fallback finds a standalone name line
        # and confirms via an email mentioning the surname.
        md = (
            "## Reproducible transcriptional modules\n\n"
            "Heewon Seo\n\n"
            "Snyder Institute for Chronic Diseases, University of Calgary\n\n"
            "Correspondence should be addressed to H.S. (Heewon.Seo@ucalgary.ca)\n\n"
            "## Abstract\n\n"
            "Glioblastoma comprises a complex ecosystem...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Heewon Seo"]

    def test_single_name_NOT_confirmed_by_email_is_rejected(self) -> None:
        # Defensive: a random capitalised heading-like line in the
        # post-title region must NOT be picked as a single author
        # unless an email confirms it. Prevents false positives like
        # treating "Introduction" or "Methods" as an author.
        md = (
            "## A Paper Title\n\n"
            "Some Stray Header\n\n"
            "## Abstract\n\n"
            "Body text...\n"
        )
        out = extract_metadata_heuristic(md)
        assert "authors" not in out

    def test_csv_authors_wrapped_across_two_lines_mid_paren(self) -> None:
        # bioRxiv: ORCID iDs in parens occasionally wrap mid-paren
        # across markdown lines. Sliding-line-join must reassemble.
        md = (
            "## Temperature alters specificity in a host-parasite interaction\n\n"
            "2 Abbey Ramirez1* (ORCID iD: 0009-0000-4698-6432) and Amanda Gibson1 (ORCID iD: 0000-\n"
            "0002-0867-4953)\n\n"
            "## ABSTRACT\n\n"
            "The Red Queen Hypothesis...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Abbey Ramirez", "Amanda Gibson"]

    def test_title_and_authors_from_markdown_table(self) -> None:
        # Docling renders some line-numbered bioRxiv Word manuscripts
        # as a multi-column markdown table; both title and author
        # extraction must walk table cells.
        md = (
            "| 1   | Using Deep Learning Models of Gene Regulation to Guide Drug Prioritization |\n"
            "|-----|--------------------------------------------------------------------------|\n"
            "| 2   | Xiaoqin Huang 1 , Ivan Ovcharenko 1 *                                    |\n"
            "| 3   |                                                                          |\n\n"
            "## Abstract\n\n"
            "We present...\n"
        )
        out = extract_metadata_heuristic(md)
        assert "Using Deep Learning Models" in out["title"]
        assert out["authors"] == ["Xiaoqin Huang", "Ivan Ovcharenko"]

    def test_csv_authors_with_parenthesised_orcid_ids(self) -> None:
        # bioRxiv: `Abbey Ramirez1* (ORCID iD: 0009-...) and Amanda Gibson1 (ORCID iD: 0000-...)`
        # Parenthesised groups + affiliation digits stripped together.
        md = (
            "## Temperature alters specificity 1\n\n"
            "2 Abbey Ramirez1* (ORCID iD: 0009-0000-4698-6432) "
            "and Amanda Gibson1 (ORCID iD: 0000-0002-0867-4953)\n\n"
            "## ABSTRACT\n\n"
            "The Red Queen Hypothesis proposes...\n"
        )
        out = extract_metadata_heuristic(md)
        assert out["authors"] == ["Abbey Ramirez", "Amanda Gibson"]

    def test_abstract_fallback_skips_bioarxiv_watermark(self) -> None:
        # The first paragraph in a Docling-converted bioRxiv PDF is
        # often the license watermark. The fallback must skip it and
        # pick the next long paragraph.
        watermark = (
            "bioRxiv preprint doi: https://doi.org/10.1101/2026.01.01.123456; "
            "this version posted Jan 2, 2026. The copyright holder for "
            "this preprint (which was not certified by peer review) is "
            "the author/funder. All rights reserved. No reuse allowed "
            "without permission."
        )
        real_abstract = (
            "We present a new method for solving the protein folding "
            "problem using a deep learning architecture trained on the "
            "Protein Data Bank. Our approach outperforms existing methods "
            "by 12% on the CASP benchmark, with particularly strong "
            "improvements on disordered regions and protein-protein "
            "interactions."
        )
        md = f"{watermark}\n\n## A Paper Title\n\nAlice, Bob, Carol\n\n{real_abstract}\n"
        out = extract_metadata_heuristic(md)
        assert out["abstract"].startswith("We present a new method")

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
