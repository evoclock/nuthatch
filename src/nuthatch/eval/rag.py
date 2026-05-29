# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Thin RAG pipeline for the eval harness.

Runs each test question through:
    1. nuthatch's `VectorRetriever` against the live Chroma store
       (same retriever the MCP server uses, so the eval measures the
       agent-facing retrieval path).
    2. The configured LLM to synthesise an answer from the top-k
       retrieved contexts.

Output per question:
    `RAGResult(question, retrieved_contexts, retrieved_doc_ids, answer)`

Both retrieved contexts and the synthesised answer are fed into
RAGAS in `ragas_eval`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from nuthatch.corpus.layout import CorpusLayout


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Outcome of one RAG turn: retrieved contexts + synthesised answer."""

    question: str
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_scores: list[float]
    answer: str


_ANSWER_PROMPT = """\
You answer questions from a research-paper corpus using ONLY the \
context passages below.

Abstention is a first-class option. You MUST abstain (not answer) when:
1. The passages do not contain the information needed to answer.
2. The passages contradict each other on the relevant point.
3. The passages are ambiguous and could support multiple answers.
4. The question itself is unclear and cannot be answered as posed.

When abstaining, reply exactly with:
  "I cannot answer from the provided context: <one-line reason>."

Partial answers are also allowed and preferred over hallucinated \
completeness. If the passages support PART of the answer but not all \
of it, say what you can support, then explicitly mark the gap, e.g.:
  "The passages state X, but do not provide Y."

If you can answer fully and faithfully, do so concisely (1-4 sentences). \
Do not speculate. Do not introduce facts not present in the context. \
Do not pad an uncertain answer with hedges; abstain or answer partially \
instead.

CONTEXT PASSAGES:
{contexts}

QUESTION: {question}

ANSWER:"""


def _format_contexts(contexts: Iterable[str]) -> str:
    return "\n\n".join(f"[passage {i + 1}]\n{text}" for i, text in enumerate(contexts))


def run_rag_turn(
    question: str,
    retriever: Any,
    answerer_llm: Any,
    *,
    k: int = 5,
    rerank: bool = False,
) -> RAGResult:
    """Retrieve k contexts + synthesise an answer."""
    hits = retriever.search(question, k=k, rerank=rerank)
    contexts = [h.text for h in hits]
    chunk_ids = [h.chunk_id for h in hits]
    doc_ids = [h.doc_id for h in hits]
    scores = [float(h.score) for h in hits]

    prompt = _ANSWER_PROMPT.format(
        contexts=_format_contexts(contexts),
        question=question,
    )
    try:
        resp = answerer_llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp)).strip()
    except Exception as exc:
        answer = f"[LLM error: {exc!s}]"

    return RAGResult(
        question=question,
        retrieved_contexts=contexts,
        retrieved_chunk_ids=chunk_ids,
        retrieved_doc_ids=doc_ids,
        retrieved_scores=scores,
        answer=answer,
    )


def retrieve_only(
    question: str,
    retriever: Any,
    *,
    k: int = 5,
    rerank: bool = False,
) -> list[str]:
    """Retrieve chunk_ids only, no answer synthesis.

    Used by the rerank-delta metric: we want to know what retrieval
    returned with and without rerank for the same question; we do
    NOT need an LLM-synthesised answer for either pass.
    """
    hits = retriever.search(question, k=k, rerank=rerank)
    return [h.chunk_id for h in hits]


def build_retriever(layout: CorpusLayout) -> Any:
    """Wire nuthatch's VectorRetriever to the live corpus.

    Single source of truth for the eval-side retriever construction.
    Uses the same defaults the MCP server uses so the eval measures
    the agent-facing retrieval path.
    """
    from nuthatch.embed.embed import Embedder
    from nuthatch.embed.store import ChromaVectorStore
    from nuthatch.retrieve import VectorRetriever

    store = ChromaVectorStore(layout.embeddings_dir, collection_name="chunks")
    embedder = Embedder()  # device="auto" -> CUDA when available
    return VectorRetriever(store=store, embedder=embedder)
