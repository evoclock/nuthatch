# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Corpus-wide embed orchestrator: incremental by default, `--force` overrides.

Walks `<corpus>/.kg/extracted/*.md` (written by `IngestOrchestrator`
after schema validation), chunks each doc's markdown via the
hybrid chunker, embeds the chunks via BGE-M3 in batches, and
persists them in `<corpus>/.kg/embeddings/` (Chroma).

Incremental semantics: by default, skip any doc_id already present
in the vector store (per-doc check via Chroma metadata filter).
Re-running on an unchanged corpus is a near-no-op (model load +
quick presence check; no re-embedding).

`force=True` deletes existing chunks per doc_id before re-embedding.
Documented use case: bumping `_DEFAULT_MAX_CHARS` or
`_DEFAULT_OVERLAP_CHARS` in `nuthatch.embed.chunk`, which would
otherwise leave a mixed store (old papers under old strategy, new
papers under new). Per-corpus overrides also live in
`<corpus>/.kg/config.yaml` under the `embedding:` section.

The defaults shipped (`max_chars=3000`, `overlap_chars=400`,
`batch_size=32`, model=`BAAI/bge-m3`) are tuned for academic
paper / preprint / patent / internal-doc corpora at ~50-200
chunks per paper. Users overriding these values via config are
responsible for tuning; nuthatch will accept their values
without protest, but retrieval quality is on them.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nuthatch.corpus.config import CorpusConfig, EmbeddingConfig
from nuthatch.corpus.layout import CorpusLayout
from nuthatch.embed.chunk import (
    _DEFAULT_MAX_CHARS,
    _DEFAULT_OVERLAP_CHARS,
)
from nuthatch.embed.embed import (
    DEFAULT_BATCH_SIZE,
    Embedder,
    already_embedded_doc_ids,
    embed_document,
)
from nuthatch.embed.store import ChromaVectorStore, VectorStore

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusEmbedResult:
    """Aggregate outcome of `embed_corpus`."""

    n_docs_scanned: int
    n_docs_embedded: int
    n_docs_skipped: int  # already in store, not forced
    n_chunks_added: int
    failed: list[tuple[str, str]]  # (doc_id, error_message)
    forced: bool


def embed_corpus(
    layout: CorpusLayout,
    *,
    config: CorpusConfig | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    force: bool = False,
) -> CorpusEmbedResult:
    """Embed every extracted-markdown doc in the corpus into Chroma.

    `force=True` re-embeds every doc (deletes existing chunks first).
    Otherwise per-doc presence check via Chroma metadata.
    """
    extracted_dir = layout.extracted_dir
    if not extracted_dir.is_dir():
        return CorpusEmbedResult(
            n_docs_scanned=0,
            n_docs_embedded=0,
            n_docs_skipped=0,
            n_chunks_added=0,
            failed=[],
            forced=force,
        )

    embed_cfg = (config or CorpusConfig()).embedding
    max_chars = embed_cfg.max_chars or _DEFAULT_MAX_CHARS
    overlap_chars = embed_cfg.overlap_chars or _DEFAULT_OVERLAP_CHARS
    batch_size = embed_cfg.batch_size or DEFAULT_BATCH_SIZE

    store = store or ChromaVectorStore(root=layout.embeddings_dir)
    embedder = embedder or _build_embedder(embed_cfg)

    md_paths = sorted(extracted_dir.glob("*.md"))
    if not md_paths:
        return CorpusEmbedResult(
            n_docs_scanned=0,
            n_docs_embedded=0,
            n_docs_skipped=0,
            n_chunks_added=0,
            failed=[],
            forced=force,
        )

    already = set() if force else already_embedded_doc_ids(store)

    n_embedded = 0
    n_skipped = 0
    n_chunks = 0
    failed: list[tuple[str, str]] = []
    for md_path in md_paths:
        doc_id = md_path.stem
        if doc_id in already and not force:
            n_skipped += 1
            continue

        meta = _load_meta_sidecar(extracted_dir / f"{doc_id}.meta.json")
        per_chunk_meta = _build_per_chunk_meta(doc_id, meta)
        markdown = md_path.read_text(encoding="utf-8")

        if force:
            # Remove any prior chunks for this doc so old + new chunks
            # don't coexist after the re-embed.
            try:
                store.delete_by_doc(doc_id)
            except Exception:
                _LOG.exception("delete_by_doc failed for %s; continuing", doc_id)

        try:
            result = embed_document(
                doc_id=doc_id,
                markdown=markdown,
                store=store,
                embedder=embedder,
                metadata=per_chunk_meta,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                batch_size=batch_size,
            )
            n_embedded += 1
            n_chunks += result.n_chunks
        except Exception as exc:
            _LOG.exception("embed_document failed for %s", doc_id)
            failed.append((doc_id, str(exc)))

    return CorpusEmbedResult(
        n_docs_scanned=len(md_paths),
        n_docs_embedded=n_embedded,
        n_docs_skipped=n_skipped,
        n_chunks_added=n_chunks,
        failed=failed,
        forced=force,
    )


def _build_embedder(embed_cfg: EmbeddingConfig) -> Embedder:
    """Build an Embedder honouring per-corpus model override if present."""
    if embed_cfg.model:
        return Embedder(model_id=embed_cfg.model)
    return Embedder()


def _load_meta_sidecar(meta_path: Path) -> dict[str, Any]:
    if not meta_path.is_file():
        return {}
    try:
        result: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        return result
    except json.JSONDecodeError:
        return {}


def _build_per_chunk_meta(doc_id: str, sidecar: dict[str, Any]) -> dict[str, Any]:
    """Pluck the fields per-chunk metadata wants from the sidecar.

    Chroma stores per-chunk metadata as a flat string-keyed dict
    (str / int / float / bool only). The field schema below MATCHES
    PhD KB's `scripts/embed.py` so retrieval-time consumers can use
    the same field names across both projects:

        path, filename, title, type, year, topics,
        committee_member, source

    Plus nuthatch-specific extras:

        doc_id, source_filename, extractor_version, authors

    Lets a search hit render an Obsidian-style link straight to
    `cards/<doc_id>.md` and surface the high-value retrieval-time
    fields (title, year, topics) in the response without a second
    lookup.
    """
    metadata = sidecar.get("metadata") or {}
    source_filename = str(sidecar.get("source_filename") or "")
    out: dict[str, Any] = {
        "doc_id": doc_id,
        "filename": f"{doc_id}.md",  # PhD KB convention (card filename)
        "path": f"cards/{doc_id}.md",  # for Obsidian-style cross-ref
        "source_filename": source_filename,
        "extractor_version": str(sidecar.get("extractor_version") or ""),
    }
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if title:
            out["title"] = str(title)
        doc_type = metadata.get("type")
        if doc_type:
            out["type"] = str(doc_type)
        year = metadata.get("year")
        if year is not None:
            with contextlib.suppress(TypeError, ValueError):
                out["year"] = int(year)
        topics = metadata.get("topics")
        if isinstance(topics, list) and topics:
            out["topics"] = ", ".join(str(t) for t in topics[:20])
        authors = metadata.get("authors")
        if isinstance(authors, list) and authors:
            out["authors"] = ", ".join(str(a) for a in authors[:20])
        committee = metadata.get("committee_member")
        if isinstance(committee, list) and committee:
            out["committee_member"] = ", ".join(str(m) for m in committee[:20])
        source = metadata.get("source")
        if isinstance(source, list) and source:
            out["source"] = ", ".join(str(s) for s in source[:20])
    return out
