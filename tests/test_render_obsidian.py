# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.render.obsidian.export_vault`."""

from __future__ import annotations

from pathlib import Path

from nuthatch.corpus.layout import init_corpus
from nuthatch.render.obsidian import export_vault


def _sample_metadata() -> dict[str, dict]:
    """Generic document fixtures. nuthatch is document-agnostic; tests
    must not assume paper-specific authorship or domain content."""
    return {
        "doc_a": {
            "title": "Document A",
            "year": 2024,
            "authors": ["Author One", "Author Two"],
            "topics": ["topic_x"],
        },
        "doc_b": {
            "title": "Document B",
            "year": 2025,
            "authors": ["Author Three"],
            "topics": ["topic_y"],
        },
    }


class TestExportVault:
    def test_writes_cards(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        rv = export_vault(layout, paper_metadata=_sample_metadata())
        assert rv.n_cards == 2
        assert (layout.root / "cards" / "doc_a.md").is_file()
        assert (layout.root / "cards" / "doc_b.md").is_file()

    def test_writes_dashboard_and_index(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        rv = export_vault(layout, paper_metadata=_sample_metadata())
        assert rv.dashboard_path.is_file()
        assert rv.index_path.is_file()
        dashboard = rv.dashboard_path.read_text()
        assert "dataview" in dashboard.lower()
        assert 'FROM "cards"' in dashboard

    def test_appends_log(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        export_vault(layout, paper_metadata=_sample_metadata())
        log = (layout.root / "log.md").read_text()
        assert "Cards written:** 2" in log
        # Run again; should prepend a second entry.
        export_vault(layout, paper_metadata=_sample_metadata())
        log = (layout.root / "log.md").read_text()
        assert log.count("Cards written:** 2") == 2

    def test_writes_community_pages(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        rv = export_vault(
            layout,
            paper_metadata=_sample_metadata(),
            community_membership={"0": ["doc_a", "doc_b"]},
        )
        assert rv.n_communities == 1
        page = (layout.root / "communities" / "0.md").read_text()
        assert "Community 0" in page
        assert "[[doc_a|" in page
        assert "[[doc_b|" in page

    def test_no_communities_when_membership_absent(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        rv = export_vault(layout, paper_metadata=_sample_metadata())
        assert rv.n_communities == 0
