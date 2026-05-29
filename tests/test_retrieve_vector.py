# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for `nuthatch.retrieve.vector.VectorRetriever` with a fake VectorStore."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nuthatch.embed.store import Neighbour
from nuthatch.retrieve.vector import VectorRetriever


class _FakeEmbedder:
    """Returns a fixed vector regardless of input."""

    def encode(self, texts):
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]


class _FakeStore:
    """Records calls and returns a fixed neighbour list."""

    def __init__(self, neighbours: list[Neighbour]) -> None:
        self._neighbours = neighbours
        self.last_k: int | None = None
        self.last_where: dict | None = None

    def add(self, **_: Any) -> None:
        return None

    def query(
        self,
        *,
        embedding: Sequence[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Neighbour]:
        self.last_k = k
        self.last_where = where
        return self._neighbours[:k]

    def delete_by_doc(self, doc_id: str) -> int:
        return 0

    def count(self) -> int:
        return len(self._neighbours)


def _n(chunk_id: str, distance: float, title: str = "", doc_id: str = "doc") -> Neighbour:
    return Neighbour(
        chunk_id=chunk_id,
        distance=distance,
        metadata={"title": title, "doc_id": doc_id, "document": f"text-{chunk_id}"},
    )


class TestVectorRetrieverSearch:
    def test_empty_query_returns_empty(self) -> None:
        store = _FakeStore([_n("c1", 0.1)])
        rv = VectorRetriever(store, embedder=_FakeEmbedder())
        assert rv.search("") == []
        assert rv.search("   ") == []

    def test_basic_search(self) -> None:
        store = _FakeStore([_n("c1", 0.1, "Title One"), _n("c2", 0.3, "Title Two")])
        rv = VectorRetriever(store, embedder=_FakeEmbedder())
        hits = rv.search("anything", k=2)
        assert [h.chunk_id for h in hits] == ["c1", "c2"]
        # Scores monotonically decreasing.
        assert hits[0].score > hits[1].score
        assert store.last_k == 2

    def test_search_passes_where(self) -> None:
        store = _FakeStore([_n("c1", 0.1)])
        rv = VectorRetriever(store, embedder=_FakeEmbedder())
        rv.search("anything", where={"doc_id": "X"})
        assert store.last_where == {"doc_id": "X"}


class TestKeywordOverlay:
    def test_overlay_promotes_title_matches(self) -> None:
        # Dense order: c1 (worst title match), c2 (best title match).
        store = _FakeStore(
            [
                _n("c1", 0.05, title="Unrelated header"),
                _n("c2", 0.30, title="evolution in mendelian populations"),
            ]
        )
        rv = VectorRetriever(store, embedder=_FakeEmbedder())
        hits = rv.search_with_keyword_overlay("mendelian populations", k=2)
        # c2 promoted to first despite worse dense distance.
        assert hits[0].chunk_id == "c2"

    def test_no_keyword_match_keeps_dense_order(self) -> None:
        store = _FakeStore([_n("c1", 0.05, title="alpha"), _n("c2", 0.30, title="beta")])
        rv = VectorRetriever(store, embedder=_FakeEmbedder())
        hits = rv.search_with_keyword_overlay("no overlap here", k=2)
        assert [h.chunk_id for h in hits] == ["c1", "c2"]
