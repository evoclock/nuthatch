<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: Apache-2.0

This is the XML-tagged AGENTS doc, primary surface for Claude /
Anthropic and other modern LLM agents that parse XML reliably.
A plain-markdown sibling lives at AGENTS-MARKDOWN.md for agents
that prefer markdown sections (older / smaller / local models).
Both documents carry the same content; keep them in sync.
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for document corpora (papers, technical reports, patents, internal docs, notes, any text-based source) with principled clustering and a token-economy-aware MCP query surface.
trigger: nuthatch serve --corpus <name>
---

<purpose>
Nuthatch turns a curated corpus of documents (research papers,
technical reports, patents, internal company docs, notes, books
, any text-based source) into a navigable knowledge graph,
surfaces query-scoped subgraphs to the consuming LLM agent to
cut context usage, and keeps the graph honest with a strict
ingest-time metadata schema. This MCP surface exposes the
graph + retrieval primitives an agent needs to answer questions
grounded in the corpus without burning tokens on irrelevant context.

The corpus is single-user, on-disk, scoped to one `CorpusLayout`
(one `.kg/` marker directory). Multi-corpus support is in the
storage layer; the MCP server is scoped to one corpus at a time
via `nuthatch serve --corpus <name>`. The active SchemaProfile
determines which document type the corpus is configured for
(`arxiv_paper`, `biorxiv_paper`, `patent`, `internal_doc`, or
user-defined); the rest of the surface is type-agnostic.
</purpose>

<corpus_shape>
A corpus carries these on-disk directories under its root:

- `cards/`. One markdown card per document. YAML frontmatter
  aligns with the kb-reports.md schema (title, id, type, year,
  authors, doi, topics, status, relevance, half_life_days,
  ingested, last_touched). The `type` field reflects the active
  SchemaProfile.
- `communities/`. One markdown page per community (cluster).
  Lists member documents as Obsidian wikilinks to the cards.
- `.kg/`. Internal state. `manifest.jsonl` (ingest log),
  `graph/graph.json` (corpus graph), `embeddings/` (ChromaDB),
  `cache/` (semantic cache), `mcp/mcp.log` (this server's logs).
- `inbox/` / `quarantine/`. Ingest staging; not user-facing for
  query-time agents.
- `dashboard.md`, `index.md`, `log.md`. Obsidian-Dataview
  navigation surface for humans.

Node-id conventions for `subgraph_extract`:

- Documents: `doc::<doc_id>` (papers, patents, internal docs, notes all share this prefix)
- Authors: `author::<slug>`
- Citations: `citation::<surname>_<year>` or `citation::doi_<slug>`
- Topics: `topic::<slug>`
</corpus_shape>

<tools>
The server exposes 5 tools. Names and contracts match the
JSON-Schema in `tools/list` exactly; use that for argument
validation. Brief operational guidance for each below.

<tool name="corpus_search">
Dense vector retrieval over the corpus chunks. Use this first when
the user's question is open-ended ("what does the corpus say about
X"). Returns top-`k` chunks with similarity score and a 500-char
preview. Default `k=5`; bump to 10-20 for survey questions, drop
to 3 for tight follow-ups. Always inspect the `doc_id` and
`metadata.title` fields to decide whether to follow up with
`card_get` or `subgraph_extract`.
</tool>

<tool name="subgraph_extract">
BFS subgraph around the given seed node IDs. Use this when the
user asks "what's connected to X" or "trace the lineage of Y".
Seeds must be valid node IDs from `corpus_shape` (use
`corpus_search` first to discover seed IDs). `depth` defaults
to 2; depth 1 = direct neighbours only, depth 3 is the practical
maximum before the result blows up token budgets.
</tool>

<tool name="card_get">
Fetch the full per-paper card markdown. Use this when the user
asks for a specific paper's summary, claims, or methods (the
card carries Abstract, Key Claims, Methods, Topics sections).
Cheap; safe to call multiple times in a session.
</tool>

<tool name="community_get">
Fetch the per-community page. Use this when the user asks
"what's in cluster N" or wants a thematic summary. Community
IDs are stable across refits (greedy overlap remap); a
`community_id` from a saved query still resolves today.
</tool>

<tool name="token_econ_report">
Aggregate token-economy stats over a time range. Sprint 7
deliverable; may return "not configured" if the token-economy
instrumentation is not yet wired in this build. Safe to skip
when unavailable.
</tool>
</tools>

<usage_patterns>
Common agent flows:

<pattern name="open_question">
1. `corpus_search(query=<user question>, k=10)`
2. From the hits, pick the 2-3 most relevant `doc_id`s.
3. `card_get(doc_id=...)` for each to get full context.
4. Compose the answer citing the papers by `title` + `year`.
</pattern>

<pattern name="lineage_trace">
1. `corpus_search(query=<topic or paper title>, k=3)`
2. Take the top hit's `doc_id`; build seed `paper::<doc_id>`.
3. `subgraph_extract(seed_nodes=[seed], depth=2)`
4. Walk the returned nodes for cites / authors / co-mentions.
</pattern>

<pattern name="cluster_summary">
1. (Optional) `corpus_search` to discover a representative paper.
2. `community_get(community_id=<id>)` to read the cluster page.
3. Compose a synthesis from the community page's member list.
</pattern>
</usage_patterns>

<limits>
- The server is **read-only** with respect to the corpus from the
  agent's perspective. Ingest, schema validation, and clustering
  happen out-of-band via the CLI. No tool here mutates corpus
  state.
- Nuthatch **does not call LLMs internally**. The consuming agent
  (you) calls the LLM with the subgraph or chunks this server
  returns. The token-economy report measures how much context
  this saves you per query.
- The retriever embeds queries with the same model used to embed
  the corpus chunks (default `BAAI/bge-m3`). Queries that need a
  different embedding tier are out of scope; ask the user.
- `subgraph_extract` with `depth > 3` is typically wasteful;
  truncate or summarise rather than feeding raw subgraphs above
  ~1000 edges into an LLM.
- All file paths the server returns are relative to the corpus
  root. Resolving them to absolute paths is the agent's job.
</limits>

<errors>
Tool responses follow the MCP convention:

- Success: `{"content": [{"type": "text", "text": "..."}]}`
- Error: `{"isError": true, "content": [{"type": "text", "text": "..."}]}`

When you see `isError: true`, surface the error text to the
user verbatim. It carries the operational reason (missing
config, unknown ID, malformed argument) the user needs to fix.
</errors>
