# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Semantic dedup via embedding similarity, with optional cross-encoder rerank.

Purpose: judge whether a candidate paper is a near-duplicate of any
    already-ingested paper. Catches preprint vs published versions,
    multiple PDF revisions, and OCR-rebuilt re-ingests that hash dedup
    misses.

Inputs: a corpus of `(doc_id, embedding)` pairs for previously
    ingested documents, plus the candidate's text or pre-computed
    embedding.

Outputs: `DedupResult` listing the top-k nearest neighbours, scores,
    and the dedup outcome (`UNIQUE`, `NEAR_DUPLICATE`, `BORDERLINE`).

Assumptions: default embedding model is `BAAI/bge-m3` (multilingual,
    ~2 GB). The optional reranker is `BAAI/bge-reranker-v2-m3`. Both
    are loaded lazily so the module imports without GPU. Configuration
    keys mirror the per-corpus `dedup.*` fields described in
    `docs/DECISIONS.md`.

The bi-encoder bands the candidate against existing docs cheaply. The
reranker rescores only the borderline band so total cost stays low.
Decision thresholds default to the values that fell out of the
McDonald-Kreitman benchmark; tune per corpus in config.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Defaults track DECISIONS.md / Ingest. Override per corpus via config.
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

# Threshold tiers for the bi-encoder cosine similarity. Pairs at or
# above `_NEAR_DUPLICATE_THRESHOLD` are duplicates without further
# checking. Pairs at or above `_BORDERLINE_THRESHOLD` go to the
# reranker. Everything below is unique.
_NEAR_DUPLICATE_THRESHOLD: float = 0.95
_BORDERLINE_THRESHOLD: float = 0.80

# Reranker decision threshold. The reranker outputs a score on (0, 1)
# tracking semantic equivalence. Above this, the borderline pair is
# promoted to duplicate; below, it stays unique.
_RERANKER_PROMOTION_THRESHOLD: float = 0.85


class DedupOutcome(StrEnum):
    UNIQUE = "unique"
    NEAR_DUPLICATE = "near_duplicate"
    BORDERLINE = "borderline"


@dataclass(frozen=True)
class DedupConfig:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str | None = DEFAULT_RERANKER_MODEL
    near_duplicate_threshold: float = _NEAR_DUPLICATE_THRESHOLD
    borderline_threshold: float = _BORDERLINE_THRESHOLD
    reranker_promotion_threshold: float = _RERANKER_PROMOTION_THRESHOLD
    top_k: int = 5


@dataclass(frozen=True)
class Neighbour:
    doc_id: str
    cosine: float
    reranker_score: float | None = None


@dataclass(frozen=True)
class DedupResult:
    outcome: DedupOutcome
    duplicate_of: str | None
    neighbours: list[Neighbour]
    candidate_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for two equal-length vectors of floats."""
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(num) / float(da * db)


def classify(
    cosine: float,
    *,
    near: float = _NEAR_DUPLICATE_THRESHOLD,
    border: float = _BORDERLINE_THRESHOLD,
) -> DedupOutcome:
    """Map a cosine score to a tier without invoking the reranker."""
    if cosine >= near:
        return DedupOutcome.NEAR_DUPLICATE
    if cosine >= border:
        return DedupOutcome.BORDERLINE
    return DedupOutcome.UNIQUE


class SemanticDeduper:
    """Stateful deduper holding an embedding model and optional reranker.

    Models load lazily on first call to keep import cheap. The corpus
    of existing `(doc_id, embedding)` pairs is supplied per-call rather
    than held internally; the actual storage layer is the Sprint 3
    vector store, not this module.
    """

    def __init__(self, config: DedupConfig | None = None) -> None:
        self.config = config or DedupConfig()
        self._embedder: Any = None
        self._reranker: Any = None

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def _ensure_reranker(self) -> Any | None:
        if self.config.reranker_model is None:
            return None
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.config.reranker_model)
        return self._reranker

    def embed(self, text: str) -> list[float]:
        """Return the bi-encoder embedding for `text`."""
        emb = self._ensure_embedder().encode(text, normalize_embeddings=True)
        return [float(x) for x in emb.tolist()]

    def check(
        self,
        candidate_text: str,
        existing: Iterable[tuple[str, Sequence[float]]],
        *,
        candidate_id: str | None = None,
        candidate_lookup_text: dict[str, str] | None = None,
    ) -> DedupResult:
        """Decide whether `candidate_text` is a duplicate of any existing doc.

        `existing` yields `(doc_id, embedding)` for previously ingested
        documents. Cosines are computed against the candidate; the
        top-k are kept. If the top hit lands in the borderline tier and
        a reranker is configured, the reranker re-scores using the
        existing-doc text from `candidate_lookup_text[doc_id]`.
        """
        cand_emb = self.embed(candidate_text)
        cosines = [(doc_id, _cosine(cand_emb, list(emb))) for doc_id, emb in existing]
        cosines.sort(key=lambda kv: kv[1], reverse=True)
        top = cosines[: self.config.top_k]
        if not top:
            return DedupResult(
                outcome=DedupOutcome.UNIQUE,
                duplicate_of=None,
                neighbours=[],
                candidate_id=candidate_id,
            )

        best_id, best_cos = top[0]
        outcome = classify(
            best_cos,
            near=self.config.near_duplicate_threshold,
            border=self.config.borderline_threshold,
        )

        # Promote borderline pairs to duplicate via reranker, when
        # available and when lookup text is supplied.
        reranker_score: float | None = None
        if outcome is DedupOutcome.BORDERLINE and candidate_lookup_text:
            reranker = self._ensure_reranker()
            if reranker is not None:
                other_text = candidate_lookup_text.get(best_id, "")
                if other_text:
                    raw = reranker.predict([(candidate_text, other_text)])
                    reranker_score = float(raw[0] if hasattr(raw, "__iter__") else raw)
                    if reranker_score >= self.config.reranker_promotion_threshold:
                        outcome = DedupOutcome.NEAR_DUPLICATE

        neighbours = [
            Neighbour(
                doc_id=doc_id,
                cosine=cos,
                reranker_score=reranker_score if doc_id == best_id else None,
            )
            for doc_id, cos in top
        ]
        duplicate_of = best_id if outcome is DedupOutcome.NEAR_DUPLICATE else None
        return DedupResult(
            outcome=outcome,
            duplicate_of=duplicate_of,
            neighbours=neighbours,
            candidate_id=candidate_id,
        )
