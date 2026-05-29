# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for `nuthatch.publish` canonical-promotion behaviour.

The MCP server expects `<kb>/.kg/communities.json` and
`<kb>/.kg/community_centroids.npy` (unsuffixed). When the source
corpus was clustered with `--output-suffix` only, the canonical names
never existed; publish must promote a preferred backend (sbm >
leiden > embeddings) to the canonical name at the dest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nuthatch.publish import publish_corpus


def _make_minimal_corpus(root: Path) -> None:
    """Create a corpus skeleton sufficient for publish_corpus to run."""
    (root / "cards").mkdir(parents=True, exist_ok=True)
    (root / "communities").mkdir(parents=True, exist_ok=True)
    (root / ".kg" / "graph").mkdir(parents=True, exist_ok=True)
    (root / ".kg" / "extracted").mkdir(parents=True, exist_ok=True)
    # A minimal graph so the merge step has something to copy/merge.
    graph_payload = {
        "schema_version": 1,
        "graph_type": "MultiDiGraph",
        "data": {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [],
            "edges": [],
        },
    }
    (root / ".kg" / "graph" / "graph.json").write_text(
        json.dumps(graph_payload),
        encoding="utf-8",
    )


def _write_index(path: Path, label: str) -> None:
    """Minimal community-index payload sufficient for promotion logic."""
    payload = {
        "schema_version": 1,
        "backend": label,
        "rigor": "principled",
        "runtime_seconds": 0.1,
        "n_levels": 1,
        "notes": "",
        "flat": {"doc::a": 0, "doc::b": 0, "doc::c": 1},
        "hierarchy": {"doc::a": [0], "doc::b": [0], "doc::c": [1]},
        "members": {"0": ["doc::a", "doc::b"], "1": ["doc::c"]},
        "core_nodes": {"0": [], "1": []},
        "labels": {"0": f"{label} cluster zero", "1": f"{label} cluster one"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_centroids(path: Path, n: int = 2, dim: int = 8) -> None:
    """Tiny float32 centroids matrix for promotion-target tests."""
    arr = np.zeros((n, dim), dtype=np.float32)
    np.save(path, arr)


class TestCanonicalPromotion:
    def test_promotes_sbm_when_canonical_missing(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        _make_minimal_corpus(corpus)
        # Only the suffixed index exists; no canonical communities.json.
        _write_index(corpus / ".kg" / "communities_sbm.json", "sbm")
        _write_centroids(corpus / ".kg" / "community_centroids_sbm.npy")

        dest = tmp_path / "kb"
        publish_corpus(
            corpus_root=corpus,
            dest=dest,
            kb_name="test-kb",
            include_chroma=False,
            include_eval=False,
            include_d3_html=False,
        )

        canonical_index = dest / ".kg" / "communities.json"
        canonical_centroids = dest / ".kg" / "community_centroids.npy"
        assert canonical_index.is_file(), (
            "publish must promote suffixed sbm index to canonical name"
        )
        assert canonical_centroids.is_file(), (
            "publish must promote suffixed sbm centroids to canonical name"
        )

        # And canonical content matches the promoted file.
        promoted = json.loads(canonical_index.read_text(encoding="utf-8"))
        original = json.loads((dest / ".kg" / "communities_sbm.json").read_text(encoding="utf-8"))
        assert promoted["labels"] == original["labels"]

    def test_prefers_sbm_over_leiden(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        _make_minimal_corpus(corpus)
        _write_index(corpus / ".kg" / "communities_sbm.json", "sbm-content")
        _write_index(corpus / ".kg" / "communities_leiden.json", "leiden-content")

        dest = tmp_path / "kb"
        publish_corpus(
            corpus_root=corpus,
            dest=dest,
            kb_name="test-kb",
            include_chroma=False,
            include_eval=False,
            include_d3_html=False,
        )

        promoted = json.loads((dest / ".kg" / "communities.json").read_text(encoding="utf-8"))
        # sbm wins over leiden; labels carry the sbm-content marker.
        assert any("sbm-content" in v for v in promoted["labels"].values()), (
            f"expected sbm promotion, got labels {promoted['labels']!r}"
        )

    def test_falls_back_to_leiden_when_sbm_absent(
        self,
        tmp_path: Path,
    ) -> None:
        corpus = tmp_path / "corpus"
        _make_minimal_corpus(corpus)
        # No sbm. Only leiden + embeddings.
        _write_index(corpus / ".kg" / "communities_leiden.json", "leiden-x")
        _write_index(corpus / ".kg" / "communities_embeddings.json", "emb-x")

        dest = tmp_path / "kb"
        publish_corpus(
            corpus_root=corpus,
            dest=dest,
            kb_name="test-kb",
            include_chroma=False,
            include_eval=False,
            include_d3_html=False,
        )

        promoted = json.loads((dest / ".kg" / "communities.json").read_text(encoding="utf-8"))
        assert any("leiden-x" in v for v in promoted["labels"].values()), (
            f"expected leiden promotion when sbm absent, got {promoted['labels']!r}"
        )

    def test_preserves_existing_canonical(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        _make_minimal_corpus(corpus)
        # Canonical already present at source; no promotion needed.
        _write_index(corpus / ".kg" / "communities.json", "canonical-already")
        _write_index(corpus / ".kg" / "communities_sbm.json", "sbm-different")

        dest = tmp_path / "kb"
        publish_corpus(
            corpus_root=corpus,
            dest=dest,
            kb_name="test-kb",
            include_chroma=False,
            include_eval=False,
            include_d3_html=False,
        )

        promoted = json.loads((dest / ".kg" / "communities.json").read_text(encoding="utf-8"))
        assert any("canonical-already" in v for v in promoted["labels"].values()), (
            "publish must NOT overwrite an existing canonical index"
        )


class TestSuffixedCentroidsGlobbed:
    def test_copies_suffixed_centroids(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        _make_minimal_corpus(corpus)
        _write_index(corpus / ".kg" / "communities_sbm.json", "sbm")
        _write_centroids(corpus / ".kg" / "community_centroids_sbm.npy")
        _write_centroids(corpus / ".kg" / "community_centroids_leiden.npy")

        dest = tmp_path / "kb"
        publish_corpus(
            corpus_root=corpus,
            dest=dest,
            kb_name="test-kb",
            include_chroma=False,
            include_eval=False,
            include_d3_html=False,
        )

        assert (dest / ".kg" / "community_centroids_sbm.npy").is_file()
        assert (dest / ".kg" / "community_centroids_leiden.npy").is_file()
