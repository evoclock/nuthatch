# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.clustering.relabel`.

Covers the pure helpers (label cleaning, frontmatter parsing, prompt
member-block construction) and the file-rewrite round-trip with a
stub LLM-invocation callable so the suite stays hermetic (no Ollama
required).
"""

from __future__ import annotations

import json
from pathlib import Path

from nuthatch.clustering.relabel import (
    _build_member_block,
    _clean_label,
    _read_card_meta,
    discover_index_paths,
    relabel_communities,
    relabel_index_file,
)


class TestCleanLabel:
    def test_strips_preamble(self) -> None:
        assert _clean_label("Topical name: Circadian Biology") == "Circadian Biology"
        assert _clean_label("Name:Foo Bar") == "Foo Bar"
        assert _clean_label("Cluster: Quantum Stuff") == "Quantum Stuff"

    def test_strips_quotes(self) -> None:
        assert _clean_label('"Hello World"') == "Hello World"
        assert _clean_label("'Alpha Beta'") == "Alpha Beta"
        assert _clean_label("`Backticked`") == "Backticked"

    def test_takes_only_first_line(self) -> None:
        assert _clean_label("Foo Bar\nthen an explanation") == "Foo Bar"

    def test_caps_at_six_words(self) -> None:
        long_in = "one two three four five six seven eight nine"
        assert _clean_label(long_in) == "one two three four five six"

    def test_passes_through_short(self) -> None:
        assert _clean_label("Graph Neural Networks") == "Graph Neural Networks"

    def test_handles_empty(self) -> None:
        assert _clean_label("") == ""
        assert _clean_label("   ") == ""


class TestReadCardMeta:
    def test_extracts_title_and_tags_block(self, tmp_path: Path) -> None:
        card = tmp_path / "doc.md"
        card.write_text(
            "---\n"
            "title: A Test Paper\n"
            "tags:\n"
            "- alpha\n"
            "- beta\n"
            "- gamma\n"
            "---\n"
            "body content here\n",
            encoding="utf-8",
        )
        meta = _read_card_meta(card)
        assert meta["title"] == "A Test Paper"
        assert meta["tags"] == ["alpha", "beta", "gamma"]

    def test_extracts_inline_tag_list(self, tmp_path: Path) -> None:
        card = tmp_path / "doc.md"
        card.write_text(
            "---\ntitle: Another\ntags: [x, y, z]\n---\n",
            encoding="utf-8",
        )
        meta = _read_card_meta(card)
        assert meta["title"] == "Another"
        assert meta["tags"] == ["x", "y", "z"]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        meta = _read_card_meta(tmp_path / "nope.md")
        assert meta == {"title": "", "tags": []}

    def test_no_frontmatter_returns_empty(self, tmp_path: Path) -> None:
        card = tmp_path / "doc.md"
        card.write_text("plain text only, no frontmatter\n", encoding="utf-8")
        meta = _read_card_meta(card)
        assert meta == {"title": "", "tags": []}


class TestBuildMemberBlock:
    def test_renders_titles_with_tags(self, tmp_path: Path) -> None:
        for stem, title in [("a", "Paper A"), ("b", "Paper B")]:
            (tmp_path / f"{stem}.md").write_text(
                f"---\ntitle: {title}\ntags:\n- t1\n- t2\n---\n",
                encoding="utf-8",
            )
        block = _build_member_block(["doc::a", "doc::b"], tmp_path)
        assert "- Paper A (t1, t2)" in block
        assert "- Paper B (t1, t2)" in block

    def test_strips_doc_prefix_for_lookup(self, tmp_path: Path) -> None:
        # Card filename has no `doc::` prefix; member id does.
        (tmp_path / "raw_id.md").write_text(
            "---\ntitle: Title One\n---\n", encoding="utf-8",
        )
        block = _build_member_block(["doc::raw_id"], tmp_path)
        assert "Title One" in block
        # Sanity: the bare-id form also works (some flat partitions
        # don't carry the doc:: prefix).
        block2 = _build_member_block(["raw_id"], tmp_path)
        assert "Title One" in block2

    def test_falls_back_to_id_when_card_missing(self, tmp_path: Path) -> None:
        block = _build_member_block(["doc::ghost"], tmp_path)
        assert "ghost" in block

    def test_truncates_with_overflow_indicator(self, tmp_path: Path) -> None:
        # 15 members, max 12 in prompt; overflow indicator line appended.
        for i in range(15):
            (tmp_path / f"d{i}.md").write_text(
                f"---\ntitle: T{i}\n---\n", encoding="utf-8",
            )
        members = [f"doc::d{i}" for i in range(15)]
        block = _build_member_block(members, tmp_path, max_members=12)
        assert "(and 3 more)" in block


class TestRelabelIndexFile:
    def test_round_trip_with_stub_llm(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        (cards_dir / "doc1.md").write_text(
            "---\ntitle: Doc One\n---\n", encoding="utf-8",
        )

        index_path = tmp_path / "communities.json"
        original = {
            "schema_version": 1,
            "backend": "test",
            "rigor": "principled",
            "runtime_seconds": 0.5,
            "n_levels": 1,
            "notes": "preserved-field",
            "flat": {"doc::doc1": 0},
            "hierarchy": {"doc::doc1": [0]},
            "members": {"0": ["doc::doc1"], "1": ["doc::ghost"]},
            "core_nodes": {"0": [], "1": []},
            "labels": {"0": "stale label zero", "1": "stale label one"},
        }
        index_path.write_text(json.dumps(original), encoding="utf-8")

        # Stub LLM: deterministic, returns "Cluster <N>" given any prompt.
        calls: list[str] = []

        def stub_invoke(prompt: str) -> str:
            calls.append(prompt)
            # The prompt embeds the member block; we don't inspect it,
            # we just confirm we were called per-community.
            return f"Topical name: Cluster {len(calls)}"

        diff = relabel_index_file(index_path, cards_dir, stub_invoke)
        assert len(calls) == 2, "one LLM call per community"
        # Old label preserved in the diff, new label applied.
        assert diff[0][0] == "stale label zero"
        assert diff[0][1] == "Cluster 1"
        assert diff[1][1] == "Cluster 2"

        # File rewritten in place; non-label fields preserved.
        rewritten = json.loads(index_path.read_text(encoding="utf-8"))
        assert rewritten["labels"] == {"0": "Cluster 1", "1": "Cluster 2"}
        assert rewritten["notes"] == "preserved-field"
        assert rewritten["flat"] == {"doc::doc1": 0}

    def test_keeps_old_label_when_llm_raises(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        index_path = tmp_path / "communities.json"
        original = {
            "members": {"0": ["doc::doc1"]},
            "labels": {"0": "fallback label"},
        }
        index_path.write_text(json.dumps(original), encoding="utf-8")

        def failing_invoke(prompt: str) -> str:
            raise RuntimeError("LLM unreachable")

        diff = relabel_index_file(index_path, cards_dir, failing_invoke)
        assert diff[0] == ("fallback label", "fallback label")
        rewritten = json.loads(index_path.read_text(encoding="utf-8"))
        assert rewritten["labels"] == {"0": "fallback label"}

    def test_missing_file_returns_empty_diff(self, tmp_path: Path) -> None:
        diff = relabel_index_file(
            tmp_path / "missing.json", tmp_path, lambda p: "x",
        )
        assert diff == {}


class TestRelabelCommunities:
    def test_applies_to_every_path(self, tmp_path: Path) -> None:
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        paths = []
        for name in ("communities.json", "communities_sbm.json"):
            p = tmp_path / name
            p.write_text(
                json.dumps({
                    "members": {"0": ["doc::a"]},
                    "labels": {"0": f"old-{name}"},
                }),
                encoding="utf-8",
            )
            paths.append(p)

        def stub_invoke(prompt: str) -> str:
            return "Fresh Label"

        diffs = relabel_communities(paths, cards_dir, llm_invoke=stub_invoke)
        assert set(diffs.keys()) == set(paths)
        for p in paths:
            rewritten = json.loads(p.read_text(encoding="utf-8"))
            assert rewritten["labels"] == {"0": "Fresh Label"}


class TestDiscoverIndexPaths:
    def test_canonical_first_then_suffixed_sorted(
        self, tmp_path: Path,
    ) -> None:
        kg = tmp_path / ".kg"
        kg.mkdir()
        (kg / "communities.json").write_text("{}", encoding="utf-8")
        (kg / "communities_sbm.json").write_text("{}", encoding="utf-8")
        (kg / "communities_leiden.json").write_text("{}", encoding="utf-8")
        (kg / "communities_embeddings.json").write_text("{}", encoding="utf-8")

        paths = discover_index_paths(kg)
        # Canonical first.
        assert paths[0].name == "communities.json"
        # Remaining are sorted lexicographically.
        rest = [p.name for p in paths[1:]]
        assert rest == sorted(rest)
        assert set(rest) == {
            "communities_embeddings.json",
            "communities_leiden.json",
            "communities_sbm.json",
        }

    def test_no_canonical_returns_only_suffixed(self, tmp_path: Path) -> None:
        kg = tmp_path / ".kg"
        kg.mkdir()
        (kg / "communities_sbm.json").write_text("{}", encoding="utf-8")
        paths = discover_index_paths(kg)
        assert [p.name for p in paths] == ["communities_sbm.json"]

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        kg = tmp_path / ".kg"
        kg.mkdir()
        assert discover_index_paths(kg) == []
