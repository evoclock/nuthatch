# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""BaseRagasEmbedding subclass backed by nuthatch's Embedder.

RAGAS computes cosine similarities for `context_precision`,
`answer_relevancy`, and `answer_correctness` and reaches for
OpenAI's `text-embedding-ada-002` by default. Without an
`OPENAI_API_KEY`, `evaluate()` errors out before it scores anything.

This adapter is a direct subclass of RAGAS's modern
`BaseRagasEmbedding` (the post-langchain interface) wrapping
`nuthatch.embed.embed.Embedder` (BGE-M3). RAGAS judges in the same
vector space the retriever was indexed in. Avoids a second model
download and avoids any vector-space mismatch between eval and
retrieval.

Implements the four methods RAGAS dispatches under asyncio:
    `embed_text(text) -> list[float]`
    `embed_texts(texts) -> list[list[float]]`
    `aembed_text(text) -> list[float]`        (delegates to sync)
    `aembed_texts(texts) -> list[list[float]]` (delegates to sync)

The async delegation is fine because SentenceTransformer is
CPU/GPU-bound, not I/O-bound; wrapping in `asyncio.to_thread`
would add overhead without parallelism gain.
"""

from __future__ import annotations

from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding


class NuthatchEmbeddings(BaseRagasEmbedding):
    """RAGAS-native embeddings adapter using nuthatch's Embedder.

    Lazy: the underlying SentenceTransformer model loads on first call.
    Honours `NUTHATCH_ACCELERATOR` via `Embedder.__init__(device="auto")`.
    """

    def __init__(self, embedder: Any | None = None, cache: Any = None) -> None:
        super().__init__(cache=cache)
        from nuthatch.embed.embed import Embedder

        self._embedder = embedder if embedder is not None else Embedder()

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self._embedder.encode([text])[0]

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self._embedder.encode(texts)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text)

    async def aembed_texts(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        return self.embed_texts(texts)

    # ragas/_answer_relevance.py calls embed_query / aembed_query directly
    # (LangChain-style API) rather than going through BaseRagasEmbedding.
    # Delegate to embed_text so RAGAS 0.4.x works regardless of which
    # internal path the metric takes.
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self.embed_texts(texts, **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text, **kwargs)

    async def aembed_documents(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        return self.embed_texts(texts, **kwargs)
