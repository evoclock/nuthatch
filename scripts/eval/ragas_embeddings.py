# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Langchain-compatible embeddings adapter backed by nuthatch's Embedder.

RAGAS computes cosine similarities for `context_precision`,
`answer_relevancy`, and `answer_correctness` and reaches for
OpenAI's `text-embedding-ada-002` by default. Without an
`OPENAI_API_KEY`, `evaluate()` errors out before it scores anything.

This adapter wraps `nuthatch.embed.embed.Embedder` (BGE-M3) so
RAGAS judges in the same vector space the retriever was indexed
in. Avoids a second model download and avoids any vector-space
mismatch between eval and retrieval.

Implements only the two methods RAGAS / langchain need:
    `embed_query(text) -> list[float]`
    `embed_documents(texts) -> list[list[float]]`
"""

from __future__ import annotations

from typing import Any


class NuthatchEmbeddings:
    """Langchain-style embeddings adapter using nuthatch's Embedder.

    Lazy: the underlying SentenceTransformer model loads on first call.
    Honours `NUTHATCH_ACCELERATOR` via `Embedder.__init__(device="auto")`.
    """

    def __init__(self, embedder: Any | None = None) -> None:
        from nuthatch.embed.embed import Embedder

        self._embedder = embedder if embedder is not None else Embedder()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)

    # RAGAS calls these async variants from inside an executor. We
    # delegate to the sync paths: SentenceTransformer is CPU/GPU-bound
    # so async-vs-sync makes no difference here, and skipping the
    # asyncio dance keeps the adapter readable.
    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
