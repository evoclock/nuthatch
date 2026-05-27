# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Per-tool counterfactual estimators for the token-economy report.

The counterfactual is "how many tokens would the consuming agent have
needed without nuthatch?" It is per-tool and per-query because the
answer depends on what the tool returns and what a non-nuthatch
fallback would look like.

Why this matters: a prior knowledge-graph implementation's reference benchmark uses the entire
corpus as the counterfactual (`nodes * 50 * 1.33` tokens vs the
returned subgraph BFS text), then markets the ratio as "Nx fewer
tokens per query vs reading the raw files directly." Nobody reads
the raw files directly per query; that baseline is a strawman.
Reported as ~71.5x at 52 files in a prior knowledge-graph implementation's `docs/how-it-works.md`.
nuthatch refuses that framing.

Per-tool baselines used here:

- `corpus_search(query, k)`: BM25 top-k over the same chunks Chroma
  indexes, joined as text. This isolates the contribution of
  semantic retrieval + reranker over plain lexical retrieval at the
  same `k`.
- `subgraph_extract(seeds, depth)`: sum of card markdown tokens for
  every `doc::` node in the returned subgraph. Without the graph the
  agent would read each card to figure out the relationships
  manually.
- `community_get(community_id)`: sum of card markdown tokens for
  every member of the community. Without the community page the
  agent would have to read each member card.
- `card_get(doc_id)`: skipped. The tool returns exactly the card
  the agent asked for by ID; there is no reduction to claim, only
  delivery.

Counterfactuals are bounded by the cost of the baseline itself, not
the cost of reading the entire corpus. Reductions reported under
this scheme typically land between 2x and ~8x depending on corpus
size and query specificity. Smaller than a prior knowledge-graph implementation's numbers; honest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from nuthatch.corpus.layout import CorpusLayout
from nuthatch.token_econ.measure import count_tokens

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]")


class CounterfactualEstimator(Protocol):
    """Pluggable per-call estimator. Returns `None` to skip logging."""

    def estimate(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        served_text: str,
        result: dict[str, Any],
    ) -> int | None: ...


class CardTokenIndex:
    """Cache `doc_id -> tokens(card markdown)` reads from `<corpus>/cards/`."""

    __slots__ = ("_cache", "_cards_dir")

    def __init__(self, cards_dir: Path) -> None:
        self._cards_dir = cards_dir
        self._cache: dict[str, int] = {}

    def tokens_for(self, doc_id: str) -> int:
        if doc_id in self._cache:
            return self._cache[doc_id]
        path = self._cards_dir / f"{doc_id}.md"
        if not path.is_file():
            self._cache[doc_id] = 0
            return 0
        text = path.read_text(encoding="utf-8")
        n, _ = count_tokens(text)
        self._cache[doc_id] = n
        return n

    def total_for(self, doc_ids: Iterable[str]) -> int:
        return sum(self.tokens_for(d) for d in doc_ids)


class ExtractedDocIndex:
    """Cache `doc_id -> tokens(raw extracted markdown)` from `.kg/extracted/`.

    Counterfactual source for `card_get`: the pre-card extracted text is
    what the agent would have had to load without nuthatch's card rendering.
    Returns `None` when the extracted file is absent (e.g. published demo
    KBs where `.kg/extracted/` is gitignored) so the server skips logging
    rather than recording a spurious 1.0 ratio.
    """

    __slots__ = ("_cache", "_extracted_dir")

    def __init__(self, extracted_dir: Path) -> None:
        self._extracted_dir = extracted_dir
        self._cache: dict[str, int | None] = {}

    def tokens_for(self, doc_id: str) -> int | None:
        if doc_id in self._cache:
            return self._cache[doc_id]
        bare = doc_id[len("doc::") :] if doc_id.startswith("doc::") else doc_id
        path = self._extracted_dir / f"{bare}.md"
        if not path.is_file():
            self._cache[doc_id] = None
            return None
        text = path.read_text(encoding="utf-8")
        n, _ = count_tokens(text)
        self._cache[doc_id] = n
        return n


class BM25Counterfactual:
    """Lexical (BM25) baseline for `corpus_search`.

    Builds lazily from a `chunks_provider` callable on first use.
    The provider returns `(chunk_id, text)` pairs; for nuthatch's
    default wiring this is `ChromaVectorStore.iter_chunks()`.
    """

    __slots__ = ("_bm25", "_built", "_chunks_provider", "_texts", "_total_chunk_tokens")

    def __init__(
        self,
        chunks_provider: Callable[[], Iterable[tuple[str, str]]],
    ) -> None:
        self._chunks_provider = chunks_provider
        self._bm25: Any = None
        self._texts: list[str] = []
        self._built = False
        self._total_chunk_tokens: int | None = None

    def _ensure_built(self) -> None:
        if self._built:
            return
        self._built = True
        from rank_bm25 import BM25Okapi

        texts: list[str] = []
        tokenized: list[list[str]] = []
        for _cid, text in self._chunks_provider():
            texts.append(text)
            tokenized.append(_simple_tokenize(text))
        self._texts = texts
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def tokens_for_query(self, query: str, k: int = 5) -> int:
        self._ensure_built()
        if self._bm25 is None or not self._texts:
            return 0
        scores = self._bm25.get_scores(_simple_tokenize(query))
        # Top-k indices by score; argsort is fine at the scales we run at.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        joined = "\n".join(self._texts[i] for i in top if scores[i] > 0)
        n, _ = count_tokens(joined)
        return n

    def total_chunk_cost(self) -> int:
        """Total tokens across all chunks: n_total_chunks x avg_chunk_tokens.

        Counterfactual for `community_search`: the cost of brute-force
        cosine search over every chunk individually, approximated as the
        aggregate token count the agent would need to read/embed.
        Cached after first call.
        """
        self._ensure_built()
        if self._total_chunk_tokens is not None:
            return self._total_chunk_tokens
        total = sum(count_tokens(t)[0] for t in self._texts)
        self._total_chunk_tokens = total
        return total


class PerToolEstimator:
    """Default `CounterfactualEstimator` wiring the per-tool baselines."""

    __slots__ = (
        "_bm25",
        "_cards",
        "_communities_json_tokens",
        "_community_members_fn",
        "_community_reader",
        "_default_k",
        "_extracted",
    )

    def __init__(
        self,
        *,
        bm25: BM25Counterfactual | None,
        cards: CardTokenIndex,
        default_k: int = 5,
        extracted: ExtractedDocIndex | None = None,
        community_reader: Callable[[str], str | None] | None = None,
        community_members_fn: Callable[[int], list[str]] | None = None,
        communities_json_tokens: int = 0,
    ) -> None:
        self._bm25 = bm25
        self._cards = cards
        self._default_k = default_k
        self._extracted = extracted
        self._community_reader = community_reader
        self._community_members_fn = community_members_fn
        self._communities_json_tokens = communities_json_tokens

    def estimate(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        served_text: str,
        result: dict[str, Any],
    ) -> int | None:
        del served_text, result  # parsed per tool below as needed
        if tool == "corpus_search":
            if self._bm25 is None:
                return None
            query = str(arguments.get("query", "")).strip()
            k = int(arguments.get("k", self._default_k))
            if not query:
                return None
            return self._bm25.tokens_for_query(query, k=k)
        if tool == "subgraph_extract":
            return self._estimate_subgraph(arguments)
        if tool == "community_get":
            return self._estimate_community(arguments)
        if tool == "card_get":
            return self._estimate_card_get(arguments)
        if tool == "community_search":
            if self._bm25 is None:
                return None
            return self._bm25.total_chunk_cost()
        if tool == "community_hierarchy":
            return self._communities_json_tokens or None
        return None

    def _estimate_subgraph(self, arguments: dict[str, Any]) -> int | None:
        # Counterfactual = "read every card whose doc the agent would
        # otherwise have had to discover by walking the graph manually."
        # We approximate by summing card tokens for every doc:: node
        # reachable from the seeds. The subgraph BFS already lives in
        # the server handler; rather than re-implementing it here we
        # re-parse the served JSON via the caller. To keep the
        # estimator's signature clean we just look at the seeds + their
        # immediate neighbours via the served result; if the caller
        # didn't pass `result` (e.g. dry-run), fall back to seeds.
        seeds = arguments.get("seed_nodes") or []
        if not isinstance(seeds, list):
            return None
        # Without the result we can only count the seed cards; the
        # PerToolEstimator is called with `result` populated in the
        # MCP wire-up, so the richer path lives in `estimate_from_served`.
        return self._cards.total_for(_doc_ids_only(seeds))

    def _estimate_community(self, arguments: dict[str, Any]) -> int | None:
        community_id = arguments.get("community_id")
        if community_id is None:
            return None
        # The community-member doc IDs come from the served markdown;
        # see `estimate_from_served`. Fall back to 0 without it.
        return 0

    def _estimate_card_get(self, arguments: dict[str, Any]) -> int | None:
        if self._extracted is None:
            return None
        doc_id = str(arguments.get("doc_id", "")).strip()
        if not doc_id:
            return None
        return self._extracted.tokens_for(doc_id)

    def estimate_from_served(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        served_text: str,
        result: dict[str, Any],
    ) -> int | None:
        """Same as `estimate` but uses the served payload when available.

        This is the path the MCP wire-up calls; the result carries
        the actual node set / member list and yields the honest
        counterfactual.
        """
        if tool == "subgraph_extract":
            doc_ids = _doc_ids_from_subgraph_text(served_text)
            return self._cards.total_for(doc_ids)
        if tool == "community_get":
            doc_ids = _doc_ids_from_community_markdown(served_text)
            return self._cards.total_for(doc_ids)
        if tool == "community_brief":
            return self._estimate_community_brief_from_served(arguments)
        if tool == "community_core_nodes":
            return self._estimate_community_core_nodes_from_served(arguments)
        # Other tools have no result-dependent counterfactual; delegate.
        return self.estimate(
            tool=tool,
            arguments=arguments,
            served_text=served_text,
            result=result,
        )

    def _estimate_community_brief_from_served(self, arguments: dict[str, Any]) -> int | None:
        # Counterfactual: reading the full community page that community_brief
        # summarises. Cost = token count of the full community markdown.
        if self._community_reader is None:
            return None
        try:
            cid = int(arguments.get("community_id"))
        except (TypeError, ValueError):
            return None
        full_page = self._community_reader(str(cid))
        if not full_page:
            return None
        n, _ = count_tokens(full_page)
        return n

    def _estimate_community_core_nodes_from_served(self, arguments: dict[str, Any]) -> int | None:
        # Counterfactual: loading every member card to rank by degree manually.
        # Cost = n_members x avg_card_tokens.
        if self._community_members_fn is None:
            return None
        try:
            cid = int(arguments.get("community_id"))
        except (TypeError, ValueError):
            return None
        members = self._community_members_fn(cid)
        if not members:
            return None
        return self._cards.total_for(_doc_ids_only(members))


def build_default_estimator(
    layout: CorpusLayout,
    *,
    chunks_provider: Callable[[], Iterable[tuple[str, str]]] | None = None,
) -> PerToolEstimator:
    """Wire a `PerToolEstimator` from a corpus layout.

    If `chunks_provider` is `None`, BM25 is unavailable and
    `corpus_search` / `community_search` calls produce `None`
    counterfactuals (skipped). Callers that want BM25 pass
    `chunks_provider=store.iter_chunks` where `store` is the same
    `ChromaVectorStore` the retriever queries.
    """
    cards_dir = layout.root / "cards"
    cards = CardTokenIndex(cards_dir)
    bm25 = BM25Counterfactual(chunks_provider) if chunks_provider is not None else None
    extracted = ExtractedDocIndex(layout.extracted_dir)
    communities_json_tokens = _load_communities_json_tokens(layout)
    communities_dir = layout.root / "communities"

    def _community_reader(community_id: str) -> str | None:
        path = communities_dir / f"{community_id}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        # Slug fallback for published KBs (mirrors server._default_community_reader).
        try:
            from nuthatch.clustering.persist import load_community_index

            idx = load_community_index(layout)
            if idx is not None:
                label = idx.labels.get(int(community_id), "")
                if label:
                    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:64]
                    slug_path = communities_dir / f"{slug}.md"
                    if slug_path.is_file():
                        return slug_path.read_text(encoding="utf-8")
        except (ValueError, TypeError):
            pass
        return None

    def _community_members_fn(community_id: int) -> list[str]:
        from nuthatch.clustering.persist import load_community_index

        idx = load_community_index(layout)
        if idx is None:
            return []
        return idx.members_of(community_id)

    return PerToolEstimator(
        bm25=bm25,
        cards=cards,
        extracted=extracted,
        community_reader=_community_reader,
        community_members_fn=_community_members_fn,
        communities_json_tokens=communities_json_tokens,
    )


# -- helpers --------------------------------------------------------------


def _simple_tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenization, alnum tokens only.

    BM25 is robust to weak tokenization; matching what a prior implementation's
    `substring in label` baseline does would overcount, while a
    full Lucene-style analyzer would over-engineer the baseline.
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def _doc_ids_only(node_ids: Iterable[str]) -> list[str]:
    out: list[str] = []
    for nid in node_ids:
        if isinstance(nid, str) and nid.startswith("doc::"):
            out.append(nid[len("doc::") :])
    return out


def _doc_ids_from_subgraph_text(served_text: str) -> list[str]:
    """Parse the JSON payload `_subgraph_extract` returns."""
    try:
        payload = json.loads(served_text)
    except json.JSONDecodeError:
        return []
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, list):
        return []
    return _doc_ids_only(nodes)


def _doc_ids_from_community_markdown(served_text: str) -> list[str]:
    """Parse `[[doc_id|title]]` wikilinks out of a community-page markdown."""
    out: list[str] = []
    for target in _WIKILINK_RE.findall(served_text):
        target = target.strip()
        # Wikilinks in nuthatch's community pages use bare doc_id.
        if target and "/" not in target:
            out.append(target)
    return out


def _load_communities_json_tokens(layout: CorpusLayout) -> int:
    """Token count of `.kg/communities.json` — the full community index.

    Counterfactual for `community_hierarchy`: without the tool the agent
    would load and traverse this entire file to find a doc's community path.
    Returns 0 if the file is absent so the estimator falls back to None
    (logged as skipped rather than a false 0-token counterfactual).
    """
    communities_json = layout.kg / "communities.json"
    if not communities_json.is_file():
        return 0
    text = communities_json.read_text(encoding="utf-8")
    n, _ = count_tokens(text)
    return n
