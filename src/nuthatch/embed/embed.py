# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-corpus embedding pipeline.

Purpose: chunk papers, embed chunks, persist to the corpus's vector
    store. Supports incremental mode (skip already-embedded `doc_id`s)
    and full reset.

Inputs: a `CorpusLayout` + a `VectorStore` (defaults to ChromaDB at
    `<corpus>/.kg/embeddings/`). Optional embedder, model name, and
    chunking parameters.

Outputs: per-document `EmbedResult` (doc_id, n_chunks, coverage_pct,
    chunk_ids). Side effects: rows persisted to the store.

Pattern reused from `~/PhD-knowledge-base/scripts/embed.py` (the
embedding pipeline driver in the PhD knowledge-base). nuthatch's
implementation is a fresh write shaped by that pattern; literal
copy was avoided. Adaptations for nuthatch:

- Per-corpus scoping via `CorpusLayout`; no module-level
  `PROJECT_ROOT` / `CHROMA_DIR` constants.
- `VectorStore` protocol indirection (via `nuthatch.embed.store`) so
  the storage backend can swap to FAISS or Pinecone per
  DECISIONS.md without touching this driver.
- Structure-aware chunking with full-document-coverage invariant
  via `nuthatch.embed.chunk.chunk_text` instead of the PhD KB's
  word-window chunker. Coverage failure raises; partial coverage is
  the load-bearing rule from DECISIONS.md.
- Default model `BAAI/bge-m3` (matches the dedup default, multilingual)
  instead of PhD KB's `allenai/specter2_base` (English papers only).
- Reset / status / query helpers analogous to the PhD KB driver's
  `--reset`, `--status`, `--query` flags.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from nuthatch.embed.chunk import Chunk, check_coverage, chunk_text
from nuthatch.embed.store import ChromaVectorStore, VectorStore

DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
DEFAULT_CHUNK_MAX_CHARS: int = 3000
DEFAULT_CHUNK_OVERLAP_CHARS: int = 400
DEFAULT_BATCH_SIZE: int = 32


@dataclass(frozen=True)
class EmbedResult:
    doc_id: str
    n_chunks: int
    coverage_pct: float
    chunk_ids: list[str] = field(default_factory=list)


class Embedder:
    """Lazy-loaded sentence-transformers wrapper.

    Singleton-style: model loads on first `encode` call and is held
    for the lifetime of the instance. Suppresses transformers warning
    noise the way PhD KB's `retrieve._get_model` does.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        # "auto" picks the best available device: CUDA on Nvidia,
        # MPS on Apple Silicon, CPU as fallback. Mirrors the ingest
        # --accelerator pattern; CLI honours `--accelerator` and
        # `NUTHATCH_ACCELERATOR` env var. Explicit values pin the
        # device. Default flipped from "cpu" to "auto" in 2026-05
        # so embed uses the GPU without extra flags when present.
        if device == "auto":
            import os

            env_override = os.environ.get("NUTHATCH_ACCELERATOR", "").strip().lower()
            if env_override in ("cpu", "cuda", "mps", "xpu"):
                device = env_override
            else:
                try:
                    import torch

                    if torch.cuda.is_available():
                        device = "cuda"
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                except ImportError:
                    device = "cpu"
        self.device = device
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            import logging
            import os
            import warnings

            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            logging.disable(logging.WARNING)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_id, device=self.device)
            logging.disable(logging.NOTSET)
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        model = self._ensure()
        vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs.tolist()]


def embed_document(
    *,
    doc_id: str,
    markdown: str,
    store: VectorStore,
    embedder: Embedder | None = None,
    metadata: dict[str, Any] | None = None,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbedResult:
    """Chunk + embed + persist a single document.

    Raises `RuntimeError` if the chunking pass fails the coverage
    invariant from DECISIONS.md (every byte of `markdown` must land
    in at least one chunk).

    The PhD KB pattern prepends `title` to the body before chunking
    for better retrieval; expose that via `metadata['title']` if the
    caller wants it. Otherwise the markdown is chunked as-is.
    """
    if metadata is not None and metadata.get("title"):
        full_text = f"{metadata['title']}\n\n{markdown}"
    else:
        full_text = markdown

    chunks: list[Chunk] = chunk_text(full_text, max_chars=max_chars, overlap_chars=overlap_chars)

    coverage = check_coverage(full_text, chunks)
    if not coverage.passed:
        raise RuntimeError(
            f"chunking failed coverage invariant for {doc_id}: "
            f"{coverage.coverage_pct:.1f}% covered, gaps={coverage.gaps[:3]}"
        )

    if not chunks:
        return EmbedResult(doc_id=doc_id, n_chunks=0, coverage_pct=100.0, chunk_ids=[])

    embedder = embedder or Embedder()

    base_meta = dict(metadata or {})
    chunk_ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    texts: list[str] = []
    for c in chunks:
        chunk_id = f"{doc_id}::chunk_{c.ordinal}"
        chunk_ids.append(chunk_id)
        texts.append(c.text)
        metadatas.append(
            {
                **base_meta,
                "doc_id": doc_id,
                "chunk_index": c.ordinal,
                "total_chunks": len(chunks),
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
            }
        )

    # Embed in batches (matches PhD KB pattern; lets us show progress
    # downstream and respects model context limits).
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(embedder.encode(batch))

    store.add(
        chunk_ids=chunk_ids,
        embeddings=all_embeddings,
        texts=texts,
        metadatas=metadatas,
    )

    return EmbedResult(
        doc_id=doc_id,
        n_chunks=len(chunks),
        coverage_pct=coverage.coverage_pct,
        chunk_ids=chunk_ids,
    )


def already_embedded_doc_ids(store: VectorStore) -> set[str]:
    """Return the set of `doc_id` values already present in `store`.

    Used by the incremental driver to skip re-embedding. Mirrors PhD
    KB embed.py's incremental logic, which queries `collection.get()`
    for existing metadata.

    Duck-typed: any store exposing a `doc_ids_present() -> Iterable[str]`
    method gets used directly (lets test fakes participate in the
    incremental skip). The `ChromaVectorStore` path is the production
    fallback; a `VectorStore` protocol implementation that exposes
    neither method gets an empty set returned and the caller will
    re-embed everything (correct but slow).
    """
    doc_ids_fn = getattr(store, "doc_ids_present", None)
    if callable(doc_ids_fn):
        return {str(d) for d in doc_ids_fn() if d}
    if not isinstance(store, ChromaVectorStore):
        return set()
    coll = store._ensure_collection()
    if coll.count() == 0:
        return set()
    raw = coll.get(include=["metadatas"])
    metas = raw.get("metadatas") or []
    return {m.get("doc_id", "") for m in metas if m.get("doc_id")}
