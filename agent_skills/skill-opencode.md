<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for paper corpora; MCP query surface over corpus search + subgraph extraction + per-paper cards + community pages.
trigger: nuthatch serve --corpus <name>
---

# Nuthatch — OpenCode integration

> **Tool spec**: load `AGENTS.md` from the repo root for full tool
> semantics. This skill file covers OpenCode-specific wiring only.

OpenCode supports MCP servers natively via its config file.

## One-time setup

1. Install Nuthatch into a uv venv and initialise a corpus
   (`nuthatch init ~/my-corpus --register-as my-corpus --set-default`,
   then `nuthatch ingest`).

2. Register Nuthatch's MCP server in OpenCode's config. OpenCode
   reads `~/.config/opencode/config.json` (path may vary by
   version). Add:

   ```json
   {
     "mcp_servers": {
       "nuthatch": {
         "command": "nuthatch",
         "args": ["serve", "--corpus", "my-corpus"],
         "transport": "stdio"
       }
     }
   }
   ```

3. Restart OpenCode. The 5 Nuthatch tools (`corpus_search`,
   `subgraph_extract`, `card_get`, `community_get`,
   `token_econ_report`) appear in the tool list.

## Build-pipeline operations (when to run what)

Nuthatch's build pipeline is four sequential stages. The agent does
NOT orchestrate these. The user runs them out-of-band via the
CLI. Rationale pinned in `docs/DECISIONS.md` § "Execution model:
build out-of-band, query via MCP".

| Stage | When to run | Incremental? |
| --- | --- | --- |
| `nuthatch ingest --corpus N` | new files dropped anywhere under the corpus root | yes (hash-based file dedup) |
| `nuthatch embed --corpus N` | after ingest produces new `papers/` | yes (per-doc Chroma presence check) |
| `nuthatch cluster --corpus N` | after a batch of new embeddings | no (full SBM refit each call; correct for the method) |
| `nuthatch render --corpus N` | after cluster, or when card metadata refreshes | byte-stable (unchanged cards re-write identically) |

**`nuthatch embed --force`**: re-embed every paper from scratch,
deleting old chunks first. Required when `_DEFAULT_MAX_CHARS` or
`_DEFAULT_OVERLAP_CHARS` in `src/nuthatch/embed/chunk.py` has
changed, OR when switching the embedding model (per-corpus override
in `<corpus>/.kg/config.yaml` under `embedding.model`).

**Watch mode (`nuthatch watch --corpus N`)**: long-running, picks
up new files via filesystem events, ingests them. Skip if the
user is running ingests on a cron or batch script.

## Multi-step query flows

The agent composes the 5 MCP tools into useful flows. Four
worked examples:

### Open-ended question

> "What does the corpus say about transformer attention mechanisms?"

1. `corpus_search(query="transformer attention mechanisms", k=10)`
2. Pick the 2-3 top hits by score + by `metadata.title` relevance.
3. `card_get(doc_id=<top_hit_doc_id>)` for each (parallel calls).
4. Compose the answer citing papers by `title` + `year`, with
   Obsidian `[[<doc_id>]]` wikilinks.

### Lineage trace

> "How does paper X relate to paper Y?"

1. `corpus_search(query="<paper X title>", k=3)` -> X's `doc_id`
2. `corpus_search(query="<paper Y title>", k=3)` -> Y's `doc_id`
3. `subgraph_extract(seed_nodes=["doc::<X>", "doc::<Y>"], depth=2)`
4. Walk returned nodes for `cites`, `authored_by`, `co_mentioned_in`
   edges; describe the connection path.

### Cluster exploration

> "What themes are in the corpus?"

1. `community_get(community_id=0)` to start with cluster 0.
2. Read members; for an outlier-looking member,
   `card_get(doc_id=<it>)` to see why it landed in that cluster.
3. Iterate. Cluster IDs are stable across refits via greedy
   overlap remap, so "cluster 3" today is "cluster 3" after the
   next `nuthatch cluster` run.

### Cost check (token economy)

> "How much context did I save on this task?"

1. `token_econ_report(group_by="tool")` after the work session.
2. Nuthatch's counterfactual is per-tool BM25 (for `corpus_search`)
   / card-sum (for subgraph + community), NOT "the whole corpus"
   (which would be the prior-implementation strawman pinned in
   `docs/DECISIONS.md` § Token-economy methodology). Reported
   ratios typically 2-8x, not 50-100x. Smaller, defensible.

## When to ask the user to rebuild

- **`corpus_search` returns no hits** for an obvious query right
  after the user mentioned new papers -> ask them to run
  `nuthatch ingest --corpus N && nuthatch embed --corpus N`.
- **`community_get` returns "community not found"** for a known ID
  -> corpus grew, cluster IDs may have shifted at boundaries. Ask
  for `nuthatch cluster --corpus N`.
- **`card_get` returns stale frontmatter** -> ask for
  `nuthatch render --corpus N`.

Do NOT invoke the build CLI yourself. The agent queries; the user
builds. (`docs/DECISIONS.md` § "Execution model" pins this.)

## See also

- `AGENTS.md`. Full tool semantics and usage patterns (XML-tagged)
