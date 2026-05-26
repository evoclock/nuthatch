# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Vector retrieval over a corpus's `VectorStore`.

Purpose: query the corpus by similarity and return ranked chunks
    with metadata for an MCP server, a CLI report, or downstream
    LLM synthesis. Supports an optional keyword overlay for short
    noun-phrase queries that pure dense vector search underweights.

Inputs: an active `VectorStore` (typically `ChromaVectorStore`
    backing `<corpus>/.kg/embeddings/`) and an `Embedder` for the
    query encoding step.

Outputs: a list of `RetrievedChunk` dicts with text, metadata,
    score, and source pointers.

Pattern reused from `~/PhD-knowledge-base/src/retrieve.py`:
the singleton model + collection load with warning suppression,
the optional keyword overlay for short queries with phrase + token
overlap scoring, and the metadata-shape contract for the result
dict. nuthatch's implementation is a fresh write shaped by that
pattern: per-corpus scoping via `VectorStore` injection, no
threading singletons (the `Embedder` instance is the singleton at
the caller's level), and a single-stage default with the keyword
overlay as an opt-in.

Assumptions: the corpus has already been embedded via
    `nuthatch.embed.embed.embed_document`. Empty stores return an
    empty list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nuthatch.embed.embed import Embedder
from nuthatch.embed.store import Neighbour, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieval hit ready for downstream consumption."""

    chunk_id: str
    text: str
    score: float  # similarity in [0, 1]; higher is closer
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Tokens we strip from queries when computing keyword overlap. The
# list is short on purpose: stopwords vary across domains, so over-
# aggressive stripping hurts more than it helps on a paper corpus.
_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "have", "how", "in", "into", "is", "it", "of",
        "on", "or", "the", "to", "vs", "with", "what", "which", "why",
        "when", "do", "does",
    }
)


def _tokenize(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 1
    }


def _neighbour_to_chunk(n: Neighbour) -> RetrievedChunk:
    meta = dict(n.metadata)
    # Distance → similarity (cosine distance ∈ [0, 2]; flip to [-1, 1])
    similarity = 1.0 - float(n.distance)
    return RetrievedChunk(
        chunk_id=n.chunk_id,
        text=str(meta.pop("document", "")),
        score=round(similarity, 4),
        doc_id=str(meta.get("doc_id", "")),
        metadata=meta,
    )


class VectorRetriever:
    """Per-corpus dense vector retrieval.

    Construction parameters:
        store: the corpus's `VectorStore` instance.
        embedder: an `Embedder` for query encoding. Reuse one
            instance across queries to amortise model load.

    Methods:
        search(query, k, where): single-stage dense retrieval.
        search_with_keyword_overlay(query, k, where): merges
            keyword-overlap hits ahead of dense hits for short
            noun-phrase queries that dense models underweight.
    """

    __slots__ = ("_embedder", "_store")

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder or Embedder()

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        embedding = self._embedder.encode([query])[0]
        neighbours = self._store.query(embedding=embedding, k=k, where=where)
        return [_neighbour_to_chunk(n) for n in neighbours]

    def search_with_keyword_overlay(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Single-stage dense search, with chunks whose metadata title
        contains all query tokens promoted ahead of dense hits.

        Pattern reused from PhD KB's `retrieve.search(wiki_only=True)`
        keyword overlay, simplified for a single-stage corpus: the
        promotion is by title-token overlap only (no separate wiki
        partition).
        """
        dense = self.search(query, k=max(k * 3, 10), where=where)
        if not dense:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return dense[:k]

        def overlap_score(chunk: RetrievedChunk) -> float:
            title_tokens = _tokenize(str(chunk.metadata.get("title", "")))
            if not title_tokens:
                return 0.0
            overlap = q_tokens & title_tokens
            if not overlap:
                return 0.0
            base = len(overlap) / len(q_tokens)
            if query.lower().strip() in str(chunk.metadata.get("title", "")).lower():
                base += 1.0
            return base

        promoted: list[RetrievedChunk] = []
        rest: list[RetrievedChunk] = []
        for chunk in dense:
            if overlap_score(chunk) > 0:
                promoted.append(chunk)
            else:
                rest.append(chunk)
        promoted.sort(key=overlap_score, reverse=True)
        return (promoted + rest)[:k]
