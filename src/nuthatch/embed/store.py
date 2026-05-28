# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""VectorStore protocol + ChromaDB implementation.

Purpose: pluggable vector-store interface so the chunking + embedding
    pipeline isn't coupled to ChromaDB. ChromaDB is the default OSS
    backend (embedded, SQLite-backed, no GPU); FAISS and Pinecone
    drop in behind the same protocol for the scale and SaaS lanes
    described in DECISIONS.md.

Inputs at add time: chunk text, embedding vector, metadata dict
    (doc_id, ordinal, offsets, extractor_version, etc.).

Outputs at query time: list of `Neighbour(chunk_id, distance,
    metadata)`.

Assumptions: embeddings are pre-computed by the caller (the
    embedder module owns model loading). The store is per-corpus
    and stays under `<corpus>/.kg/embeddings/` for the default
    Chroma backend.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Neighbour:
    """A single query result: id + distance + arbitrary metadata."""

    chunk_id: str
    distance: float
    metadata: dict[str, Any]


@runtime_checkable
class VectorStore(Protocol):
    """Pluggable backend for chunk-level vector storage and retrieval."""

    def add(
        self,
        *,
        chunk_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None: ...

    def query(
        self,
        *,
        embedding: Sequence[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Neighbour]: ...

    def delete_by_doc(self, doc_id: str) -> int: ...

    def count(self) -> int: ...

    def iter_chunks(self) -> Iterator[tuple[str, str]]: ...


class ChromaVectorStore:
    """ChromaDB-backed `VectorStore` for the OSS default path.

    Persists to `<root>/embeddings/` (typically `<corpus>/.kg/embeddings/`).
    One collection per corpus; per-chunk metadata carries `doc_id` so
    delete-by-doc works as a metadata-filter delete.
    """

    __slots__ = ("_client", "_collection", "_collection_name", "_root")

    def __init__(
        self,
        root: Path,
        *,
        collection_name: str = "chunks",
    ) -> None:
        self._root = root
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        import chromadb

        self._root.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._root))
        # `hnsw:space=cosine` matches PhD KB's collection config so
        # retrieval ranking and similarity scores are directly
        # comparable across the two projects. Default Chroma distance
        # is L2; nuthatch is a cosine-similarity tool.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def add(
        self,
        *,
        chunk_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Upsert chunks. Re-using existing chunk_ids overwrites.

        Uses `coll.upsert()` (not `coll.add()`) so the `--force`
        re-embed path and any other re-ingestion of an existing
        doc_id does not raise on duplicate IDs.
        """
        if not chunk_ids:
            return
        coll = self._ensure_collection()
        coll.upsert(
            ids=list(chunk_ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(texts),
            metadatas=list(metadatas),
        )

    def query(
        self,
        *,
        embedding: Sequence[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Neighbour]:
        coll = self._ensure_collection()
        result = coll.query(
            query_embeddings=[list(embedding)],
            n_results=k,
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0] or [{} for _ in ids]
        return [
            Neighbour(chunk_id=cid, distance=float(d), metadata=dict(m or {}))
            for cid, d, m in zip(ids, distances, metadatas, strict=True)
        ]

    def delete_by_doc(self, doc_id: str) -> int:
        coll = self._ensure_collection()
        existing = coll.get(where={"doc_id": doc_id})
        ids = existing.get("ids", [])
        if not ids:
            return 0
        coll.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        coll = self._ensure_collection()
        return int(coll.count())

    def iter_chunks(self) -> Iterator[tuple[str, str]]:
        """Yield `(chunk_id, text)` for every stored chunk.

        Used by the token-economy counterfactual layer to build a BM25
        index over the same chunks Chroma serves. `coll.get()` with no
        `where` returns the whole collection; for very large corpora a
        streaming `limit/offset` loop would be cheaper, but Chroma's
        get() is already incremental under the hood and corpora at the
        scale this is used for (cards, papers, internal docs) fit in
        memory comfortably.
        """
        coll = self._ensure_collection()
        got = coll.get(include=["documents"])
        ids = got.get("ids", []) or []
        docs = got.get("documents", []) or []
        for cid, text in zip(ids, docs, strict=True):
            yield cid, text or ""
