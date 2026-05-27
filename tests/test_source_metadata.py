# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.source_metadata` (arxiv + bioRxiv API fetchers)
with mocked HTTP so the suite runs offline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nuthatch.ingest.security import SecurityResult
from nuthatch.ingest.source_metadata import (
    SourceMetadata,
    _parse_arxiv_atom,
    _parse_biorxiv_json,
    enrich_from_source,
    extract_arxiv_id_from_filename,
    extract_biorxiv_doi_from_filename,
    fetch_arxiv_metadata,
    fetch_biorxiv_metadata,
)

_PERMISSIVE = lambda _url: SecurityResult(allowed=True, reason=None)  # noqa: E731

_FAKE_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2605.15308v1</id>
    <title>SMCEVOLVE: Principled Scientific Discovery</title>
    <summary>We introduce SMCEVOLVE, a Sequential Monte Carlo approach.</summary>
    <published>2026-05-14T00:00:00Z</published>
    <author><name>Jiachen Jiang</name></author>
    <author><name>Huminhao Zhu</name></author>
    <author><name>Zhihui Zhu</name></author>
    <arxiv:doi>10.48550/arXiv.2605.15308</arxiv:doi>
  </entry>
</feed>"""

_FAKE_BIORXIV_JSON = """{
  "collection": [
    {
      "title": "A study of regulatory networks in mouse",
      "abstract": "We show that ...",
      "authors": "Alice Smith; Bob Jones; Carol Lin",
      "date": "2026-05-14"
    }
  ]
}"""


# -- filename parsers -----------------------------------------------------


class TestFilenameParsers:
    def test_arxiv_modern_format(self) -> None:
        assert extract_arxiv_id_from_filename("2605.15308v1.pdf") == "2605.15308"
        assert extract_arxiv_id_from_filename("2605.15308.pdf") == "2605.15308"
        assert extract_arxiv_id_from_filename("2605.15308v22") == "2605.15308"

    def test_arxiv_negative_cases(self) -> None:
        assert extract_arxiv_id_from_filename("random_paper.pdf") is None
        assert extract_arxiv_id_from_filename("smith_2010.pdf") is None
        assert extract_arxiv_id_from_filename("cs0507001.pdf") is None  # old format

    def test_biorxiv_doi(self) -> None:
        # bioRxiv preprint naming: YYYY.MM.DD.NNNNNN[vN][.full].pdf
        assert (
            extract_biorxiv_doi_from_filename("2026.05.14.725010v1.full.pdf")
            == "10.1101/2026.05.14.725010"
        )
        assert (
            extract_biorxiv_doi_from_filename("2026.05.14.725010.pdf")
            == "10.1101/2026.05.14.725010"
        )

    def test_biorxiv_negative_cases(self) -> None:
        assert extract_biorxiv_doi_from_filename("foo.pdf") is None
        assert extract_biorxiv_doi_from_filename("2605.15308v1.pdf") is None


# -- atom + json parsers --------------------------------------------------


class TestArxivAtomParser:
    def test_full_record(self) -> None:
        meta = _parse_arxiv_atom(_FAKE_ARXIV_ATOM, arxiv_id="2605.15308")
        assert meta is not None
        assert meta.title == "SMCEVOLVE: Principled Scientific Discovery"
        assert meta.authors == ("Jiachen Jiang", "Huminhao Zhu", "Zhihui Zhu")
        assert "Sequential Monte Carlo" in (meta.abstract or "")
        assert meta.year == 2026
        assert meta.arxiv_id == "2605.15308"
        assert meta.source == "arxiv"
        assert meta.doi == "10.48550/arXiv.2605.15308"

    def test_malformed_xml_returns_none(self) -> None:
        assert _parse_arxiv_atom("not xml at all", arxiv_id="x") is None

    def test_empty_feed_returns_none(self) -> None:
        assert (
            _parse_arxiv_atom(
                '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>',
                arxiv_id="x",
            )
            is None
        )


class TestBiorxivJsonParser:
    def test_full_record(self) -> None:
        meta = _parse_biorxiv_json(_FAKE_BIORXIV_JSON, doi="10.1101/2026.05.14.725010")
        assert meta is not None
        assert meta.title == "A study of regulatory networks in mouse"
        assert meta.authors == ("Alice Smith", "Bob Jones", "Carol Lin")
        assert meta.year == 2026
        assert meta.doi == "10.1101/2026.05.14.725010"
        assert meta.source == "biorxiv"

    def test_malformed_json_returns_none(self) -> None:
        assert _parse_biorxiv_json("not json", doi="x") is None

    def test_empty_collection_returns_none(self) -> None:
        assert _parse_biorxiv_json('{"collection": []}', doi="x") is None


# -- HTTP fetchers (mocked) -----------------------------------------------


class TestFetchArxivMetadata:
    def test_hits_api_and_caches(self, tmp_path: Path) -> None:
        with patch(
            "nuthatch.ingest.source_metadata._fetch_url",
            return_value=_FAKE_ARXIV_ATOM,
        ) as mock_fetch:
            meta = fetch_arxiv_metadata("2605.15308", cache_dir=tmp_path, url_validator=_PERMISSIVE)
            assert meta is not None
            assert meta.title.startswith("SMCEVOLVE")
            mock_fetch.assert_called_once()

        # Re-fetch hits the cache, not the network.
        cache_file = tmp_path / "arxiv_2605.15308.json"
        assert cache_file.is_file()
        with patch(
            "nuthatch.ingest.source_metadata._fetch_url",
            side_effect=AssertionError("should not call network"),
        ):
            meta2 = fetch_arxiv_metadata(
                "2605.15308", cache_dir=tmp_path, url_validator=_PERMISSIVE
            )
            assert meta2 is not None
            assert meta2.title == meta.title

    def test_fetch_failure_returns_none(self, tmp_path: Path) -> None:
        with patch("nuthatch.ingest.source_metadata._fetch_url", return_value=None):
            assert (
                fetch_arxiv_metadata("2605.15308", cache_dir=tmp_path, url_validator=_PERMISSIVE)
                is None
            )


class TestFetchBiorxivMetadata:
    def test_hits_api_and_caches(self, tmp_path: Path) -> None:
        with patch(
            "nuthatch.ingest.source_metadata._fetch_url",
            return_value=_FAKE_BIORXIV_JSON,
        ):
            meta = fetch_biorxiv_metadata(
                "10.1101/2026.05.14.725010",
                cache_dir=tmp_path,
                url_validator=_PERMISSIVE,
            )
            assert meta is not None
            assert meta.authors[0] == "Alice Smith"

        cache_file = tmp_path / "biorxiv_10.1101_2026.05.14.725010.json"
        assert cache_file.is_file()


# -- the convenience entrypoint -------------------------------------------


class TestEnrichFromSource:
    def test_arxiv_filename_dispatches_to_arxiv_fetcher(self, tmp_path: Path) -> None:
        with patch(
            "nuthatch.ingest.source_metadata._fetch_url",
            return_value=_FAKE_ARXIV_ATOM,
        ):
            meta = enrich_from_source(
                "2605.15308v1.pdf",
                cache_dir=tmp_path,
                url_validator=_PERMISSIVE,
            )
            assert isinstance(meta, SourceMetadata)
            assert meta.source == "arxiv"
            assert meta.arxiv_id == "2605.15308"

    def test_biorxiv_filename_dispatches_to_biorxiv_fetcher(self, tmp_path: Path) -> None:
        with patch(
            "nuthatch.ingest.source_metadata._fetch_url",
            return_value=_FAKE_BIORXIV_JSON,
        ):
            meta = enrich_from_source(
                "2026.05.14.725010v1.full.pdf",
                cache_dir=tmp_path,
                url_validator=_PERMISSIVE,
            )
            assert isinstance(meta, SourceMetadata)
            assert meta.source == "biorxiv"

    def test_unrecognised_filename_returns_none(self, tmp_path: Path) -> None:
        assert (
            enrich_from_source(
                "internal_report_q3.pdf",
                cache_dir=tmp_path,
                url_validator=_PERMISSIVE,
            )
            is None
        )

    def test_arxiv_api_miss_returns_filename_fallback(self, tmp_path: Path) -> None:
        # Publisher API returned nothing (e.g. rate-limit, transient).
        # We should still emit a stub carrying the arxiv_id from the
        # filename so the schema gate's identifier requirement passes.
        with patch("nuthatch.ingest.source_metadata._fetch_url", return_value=None):
            meta = enrich_from_source(
                "2605.15308v1.pdf",
                cache_dir=tmp_path,
                url_validator=_PERMISSIVE,
            )
        assert meta is not None
        assert meta.arxiv_id == "2605.15308"
        assert meta.source == "arxiv_filename_fallback"
        # No title / authors / abstract from a fallback; body extraction
        # must fill those.
        assert meta.title is None
        assert meta.authors == ()

    def test_biorxiv_api_miss_returns_filename_fallback(self, tmp_path: Path) -> None:
        with patch("nuthatch.ingest.source_metadata._fetch_url", return_value=None):
            meta = enrich_from_source(
                "2026.05.20.726505v1.full.pdf",
                cache_dir=tmp_path,
                url_validator=_PERMISSIVE,
            )
        assert meta is not None
        assert meta.doi == "10.1101/2026.05.20.726505"
        assert meta.year == 2026
        assert meta.source == "biorxiv_filename_fallback"

    def test_blocked_url_falls_back_without_network(self, tmp_path: Path) -> None:
        # If validate_url disallows the fetch we must NOT touch the
        # network. The filename-fallback path is allowed because it
        # derives its data from the local filename only.
        blocker = lambda _u: SecurityResult(allowed=False, reason="test-block")  # noqa: E731
        with patch(
            "nuthatch.ingest.source_metadata.urllib.request.urlopen",
            side_effect=AssertionError("should not be reached"),
        ):
            meta = enrich_from_source(
                "2605.15308v1.pdf",
                cache_dir=tmp_path,
                url_validator=blocker,
            )
        # We get a fallback stub, NOT publisher metadata.
        assert meta is not None
        assert meta.source == "arxiv_filename_fallback"
        assert meta.arxiv_id == "2605.15308"
        assert meta.title is None and meta.authors == ()


# -- end-to-end: extract_and_validate prefers source metadata over body ---


class TestExtractAndValidateMerge:
    def test_source_metadata_overrides_body_for_authors(self, tmp_path: Path) -> None:
        from nuthatch.ingest.metadata import extract_and_validate
        from nuthatch.schema.profile import FieldSpec, SchemaProfile

        class _RequireAuthors(SchemaProfile):
            profile_name = "test_req_authors"
            fields = (
                FieldSpec("title", required=True, expected_type=str),
                FieldSpec("authors", required=True, expected_type=list),
            )

        # Body has no detectable authors; source metadata supplies them.
        body_md = "# A Paper Title\n\nBody text without an obvious author line."

        with patch(
            "nuthatch.ingest.source_metadata._fetch_url",
            return_value=_FAKE_ARXIV_ATOM,
        ):
            extracted, validation = extract_and_validate(
                body_md,
                _RequireAuthors,
                source_filename="2605.15308v1.pdf",
                metadata_cache_dir=tmp_path,
            )
            assert validation.passed, (
                f"schema should pass once arxiv supplies authors: "
                f"missing={validation.missing_required}"
            )
            assert extracted["authors"] == [
                "Jiachen Jiang",
                "Huminhao Zhu",
                "Zhihui Zhu",
            ]
            assert "SMCEVOLVE" in extracted["title"]
