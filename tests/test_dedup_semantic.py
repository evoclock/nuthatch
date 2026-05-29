# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for `nuthatch.dedup.semantic`.

Scope: pure-logic tests against `classify` and the `check` flow with a
mock embedder. No real model loading, no GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from nuthatch.dedup.semantic import (
    DedupConfig,
    DedupOutcome,
    Neighbour,
    SemanticDeduper,
    _cosine,
    classify,
)


class TestCosine:
    def test_identical_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestClassify:
    @pytest.mark.parametrize(
        "cos,outcome",
        [
            (1.00, DedupOutcome.NEAR_DUPLICATE),
            (0.95, DedupOutcome.NEAR_DUPLICATE),
            (0.94, DedupOutcome.BORDERLINE),
            (0.80, DedupOutcome.BORDERLINE),
            (0.79, DedupOutcome.UNIQUE),
            (0.10, DedupOutcome.UNIQUE),
            (-0.5, DedupOutcome.UNIQUE),
        ],
    )
    def test_thresholds(self, cos: float, outcome: DedupOutcome) -> None:
        assert classify(cos) is outcome


class _MockEmbedder:
    """Stand-in for SentenceTransformer; returns a known vector per input."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def encode(self, text: str, normalize_embeddings: bool = True) -> Any:
        # Mimic numpy.ndarray.tolist; just return the list directly.
        return _ListAsArray(self._mapping[text])


class _ListAsArray(list):
    """Trivial wrapper so the test mock's `encode` output supports tolist()."""

    def tolist(self) -> list[float]:
        return list(self)


class _MockReranker:
    def __init__(self, score: float) -> None:
        self.score = score
        self.called = False

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.called = True
        return [self.score for _ in pairs]


class TestSemanticDeduperCheck:
    def _make_deduper(
        self,
        mapping: dict[str, list[float]],
        *,
        reranker_score: float | None = None,
    ) -> SemanticDeduper:
        cfg = DedupConfig(reranker_model=None if reranker_score is None else "mock")
        d = SemanticDeduper(cfg)
        d._embedder = _MockEmbedder(mapping)
        if reranker_score is not None:
            d._reranker = _MockReranker(reranker_score)
        return d

    def test_no_existing_returns_unique(self) -> None:
        d = self._make_deduper({"candidate": [1.0, 0.0]})
        result = d.check("candidate", existing=[], candidate_id="c1")
        assert result.outcome is DedupOutcome.UNIQUE
        assert result.duplicate_of is None
        assert result.neighbours == []
        assert result.candidate_id == "c1"

    def test_identical_doc_is_near_duplicate(self) -> None:
        d = self._make_deduper({"candidate": [1.0, 0.0]})
        result = d.check(
            "candidate",
            existing=[("a", [1.0, 0.0])],
        )
        assert result.outcome is DedupOutcome.NEAR_DUPLICATE
        assert result.duplicate_of == "a"

    def test_orthogonal_doc_is_unique(self) -> None:
        d = self._make_deduper({"candidate": [1.0, 0.0]})
        result = d.check("candidate", existing=[("a", [0.0, 1.0])])
        assert result.outcome is DedupOutcome.UNIQUE
        assert result.duplicate_of is None

    def test_borderline_no_reranker_stays_borderline(self) -> None:
        d = self._make_deduper({"candidate": [0.9, 0.4358]})
        # 0.9*1 + 0.4358*0 = 0.9; norms 1 and 1 → cosine ~0.9, borderline
        result = d.check("candidate", existing=[("a", [1.0, 0.0])])
        assert result.outcome is DedupOutcome.BORDERLINE
        assert result.duplicate_of is None

    def test_borderline_promoted_by_reranker(self) -> None:
        d = self._make_deduper({"candidate": [0.9, 0.4358]}, reranker_score=0.95)
        result = d.check(
            "candidate",
            existing=[("a", [1.0, 0.0])],
            candidate_lookup_text={"a": "existing text"},
        )
        assert result.outcome is DedupOutcome.NEAR_DUPLICATE
        assert result.duplicate_of == "a"
        # Reranker should have been invoked.
        assert d._reranker.called

    def test_borderline_not_promoted_when_score_below_threshold(self) -> None:
        d = self._make_deduper({"candidate": [0.9, 0.4358]}, reranker_score=0.6)
        result = d.check(
            "candidate",
            existing=[("a", [1.0, 0.0])],
            candidate_lookup_text={"a": "existing text"},
        )
        assert result.outcome is DedupOutcome.BORDERLINE
        assert result.duplicate_of is None

    def test_top_k_neighbours_returned_sorted(self) -> None:
        d = self._make_deduper({"candidate": [1.0, 0.0]})
        existing = [
            ("a", [0.5, 0.866]),  # cos ~0.5
            ("b", [1.0, 0.0]),  # cos 1.0
            ("c", [0.0, 1.0]),  # cos 0
            ("d", [-1.0, 0.0]),  # cos -1
        ]
        result = d.check("candidate", existing=existing)
        ids = [n.doc_id for n in result.neighbours]
        assert ids == ["b", "a", "c", "d"]
        # Scores should be monotonic non-increasing.
        scores = [n.cosine for n in result.neighbours]
        assert scores == sorted(scores, reverse=True)

    def test_neighbour_dataclass_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        n = Neighbour(doc_id="x", cosine=0.5)
        with pytest.raises(FrozenInstanceError):
            n.cosine = 0.6  # type: ignore[misc]


class TestDedupConfigDefaults:
    def test_default_models(self) -> None:
        cfg = DedupConfig()
        assert cfg.embedding_model == "BAAI/bge-m3"
        assert cfg.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_reranker_can_be_disabled(self) -> None:
        cfg = DedupConfig(reranker_model=None)
        assert cfg.reranker_model is None
