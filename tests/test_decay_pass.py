# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sprint 8 tests: frontmatter round-trip, supersession scan + apply,
end-to-end decay pass orchestrator, pinned-status preservation,
dry-run no-write contract, idempotency."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import networkx as nx
import yaml

from nuthatch.corpus import init_corpus
from nuthatch.decay import (
    apply_supersession,
    find_supersession_pairs,
    parse_card,
    render_report,
    run_decay_pass,
    update_frontmatter,
    write_card,
)

# -- frontmatter round-trip -----------------------------------------------


class TestFrontmatter:
    def test_round_trip_preserves_body(self, tmp_path: Path) -> None:
        path = tmp_path / "card.md"
        path.write_text(
            "---\ntitle: Test\nrelevance: 0.5\n---\n\n# Test\n\nbody\n",
            encoding="utf-8",
        )
        front, body = parse_card(path)
        assert front["title"] == "Test"
        assert "body" in body
        # Round-trip with an update.
        updated = update_frontmatter(front, {"relevance": 0.3})
        write_card(path, updated, body)
        front2, body2 = parse_card(path)
        assert front2["relevance"] == 0.3
        assert body2 == body

    def test_no_frontmatter_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "noyaml.md"
        path.write_text("just a body, no fence\n", encoding="utf-8")
        front, body = parse_card(path)
        assert front == {}
        assert "just a body" in body

    def test_update_writes_through_none(self) -> None:
        merged = update_frontmatter({"a": 1, "b": 2}, {"b": None, "c": 3})
        assert merged == {"a": 1, "b": None, "c": 3}


# -- supersession ----------------------------------------------------------


class TestSupersessionScan:
    def test_finds_pairs(self) -> None:
        cards = {
            "p2": {"supersedes": ["p1"]},
            "p3": {"supersedes": "p2"},  # scalar form
            "p4": {},
        }
        pairs = find_supersession_pairs(cards)
        keys = {(p.new_doc_id, p.old_doc_id) for p in pairs}
        assert keys == {("p2", "p1"), ("p3", "p2")}


class TestSupersessionApply:
    def test_adds_edge_flips_status_and_downweights(self) -> None:
        g = nx.MultiDiGraph()
        g.add_node("doc::p1", node_type="document")
        g.add_node("doc::p2", node_type="document")
        cards = {
            "p1": {"status": "validated", "relevance": 1.0},
            "p2": {"status": "exploratory", "supersedes": ["p1"]},
        }
        pairs = find_supersession_pairs(cards)
        result = apply_supersession(g, cards, pairs)
        assert len(result.pairs_applied) == 1
        # Edge exists with relation supersedes.
        edges = [(u, v, d) for u, v, d in g.edges(data=True) if d.get("relation") == "supersedes"]
        assert edges == [
            ("doc::p2", "doc::p1", {"relation": "supersedes", "confidence": "EXTRACTED"})
        ]
        assert cards["p1"]["status"] == "superseded"
        assert cards["p1"]["relevance"] == 0.1  # downweighted by default 0.1

    def test_skips_unknown_old_doc(self) -> None:
        g = nx.MultiDiGraph()
        cards = {"p2": {"supersedes": ["external_paper_we_dont_have"]}}
        pairs = find_supersession_pairs(cards)
        result = apply_supersession(g, cards, pairs)
        assert result.pairs_applied == []
        assert len(result.pairs_skipped) == 1


# -- end-to-end decay pass -------------------------------------------------


def _write_card_file(
    cards_dir: Path,
    doc_id: str,
    *,
    title: str,
    relevance: float = 1.0,
    last_touched: str | None = "2026-05-01",
    half_life_days: int | None = 365,
    status: str = "exploratory",
    supersedes: list[str] | None = None,
) -> None:
    front: dict = {
        "title": title,
        "id": doc_id,
        "doc_id": doc_id,
        "type": "paper",
        "status": status,
        "relevance": relevance,
        "half_life_days": half_life_days,
        "last_touched": last_touched,
    }
    if supersedes is not None:
        front["supersedes"] = supersedes
    body = f"# {title}\n\nbody for {doc_id}\n"
    text = (
        "---\n"
        + yaml.dump(front, default_flow_style=False, sort_keys=False).rstrip()
        + "\n---\n"
        + body
    )
    (cards_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")


class TestRunDecayPass:
    def test_basic_decay_writes_back_card_relevance(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        # last_touched 400 days ago + half_life 365 -> attenuation ~0.47.
        old_date = (date(2026, 5, 26) - timedelta(days=400)).isoformat()
        _write_card_file(cards_dir, "p1", title="Paper 1", last_touched=old_date)

        result = run_decay_pass(layout, now=date(2026, 5, 26))
        assert result.n_cards_scanned == 1
        assert result.n_decayed == 1
        # Card on disk now has decayed relevance < 1.0.
        front, _ = parse_card(cards_dir / "p1.md")
        assert front["relevance"] < 1.0
        # last_touched is NOT bumped (it means "last human edit");
        # the decay pass updates relevance but leaves last_touched
        # alone so the next pass is idempotent.
        assert front["last_touched"] == old_date

    def test_pinned_status_not_decayed(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        old_date = (date(2026, 5, 26) - timedelta(days=2000)).isoformat()
        _write_card_file(
            cards_dir,
            "pinned1",
            title="Pinned classic",
            last_touched=old_date,
            status="pinned",
            relevance=1.0,
        )
        result = run_decay_pass(layout, now=date(2026, 5, 26))
        assert result.n_pinned == 1
        front, _ = parse_card(cards_dir / "pinned1.md")
        # Pinned card's last_touched is NOT bumped (stable means stable).
        assert front["last_touched"] == old_date

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        old_date = (date(2026, 5, 26) - timedelta(days=500)).isoformat()
        _write_card_file(cards_dir, "p1", title="Paper", last_touched=old_date)
        original = (cards_dir / "p1.md").read_text(encoding="utf-8")

        result = run_decay_pass(layout, now=date(2026, 5, 26), dry_run=True)
        assert result.n_decayed == 1
        assert result.cards_written == 0
        assert (cards_dir / "p1.md").read_text(encoding="utf-8") == original

    def test_supersession_flips_old_card(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        _write_card_file(cards_dir, "old", title="Old", relevance=1.0)
        _write_card_file(cards_dir, "new", title="New", supersedes=["old"], relevance=1.0)
        result = run_decay_pass(layout, now=date(2026, 5, 26))
        assert result.n_superseded_this_pass == 1
        front_old, _ = parse_card(cards_dir / "old.md")
        assert front_old["status"] == "superseded"
        assert front_old["relevance"] < 0.5  # downweighted by 0.1

    def test_archive_candidates_listed(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        very_old = (date(2026, 5, 26) - timedelta(days=4000)).isoformat()
        _write_card_file(
            cards_dir,
            "stale",
            title="Stale",
            last_touched=very_old,
            half_life_days=365,
        )
        result = run_decay_pass(layout, now=date(2026, 5, 26))
        ids = {c.doc_id for c in result.archive_candidates}
        assert "stale" in ids

    def test_idempotency_second_run_no_writes(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        old_date = (date(2026, 5, 26) - timedelta(days=400)).isoformat()
        _write_card_file(cards_dir, "p1", title="Paper", last_touched=old_date)
        first = run_decay_pass(layout, now=date(2026, 5, 26))
        assert first.cards_written == 1
        # Second pass on the SAME `now`: post-decay value already matches
        # what the kernel would compute, so no card is re-written.
        second = run_decay_pass(layout, now=date(2026, 5, 26))
        assert second.n_decayed == 0
        assert second.cards_written == 0

    def test_render_report_emits_summary_table(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        very_old = (date(2026, 5, 26) - timedelta(days=4000)).isoformat()
        _write_card_file(cards_dir, "stale", title="S", last_touched=very_old)
        result = run_decay_pass(layout, now=date(2026, 5, 26))
        md = render_report(result, corpus_name="corpus")
        assert "# Decay pass report" in md
        assert "Archive candidates" in md
        assert "stale" in md
