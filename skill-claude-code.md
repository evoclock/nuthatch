<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for paper corpora; MCP query surface over corpus search + subgraph extraction + per-paper cards + community pages.
trigger: nuthatch serve --corpus <name>
---

# nuthatch — Claude Code integration

> **Tool spec**: load `AGENTS.md` (XML-tagged) **or**
> `AGENTS-MARKDOWN.md` (plain markdown) for full tool semantics.
> Both carry the same content; pick whichever your parser handles
> best. Claude Code consumes `AGENTS.md` natively from the repo
> root, so no extra configuration is required to surface the
> tool semantics. This skill file covers Claude-Code-specific
> wiring only.

## One-time setup

1. Install nuthatch (once published to PyPI):

   ```bash
   sfw pipx install nuthatch
   # or, until published:
   sfw pipx install git+https://github.com/evoclock/nuthatch.git@main
   ```

2. Initialise a corpus, register it as default:

   ```bash
   nuthatch init ~/my-corpus --register-as my-corpus --set-default
   ```

3. Drop sources anywhere under the corpus root (`inbox/`,
   `arxiv/`, `bioarxiv/`, `notes/`, root-level — your choice; the
   ingest pipeline scans the tree). Then run the four-stage
   pipeline:

   ```bash
   nuthatch ingest  --corpus my-corpus
   nuthatch embed   --corpus my-corpus
   nuthatch cluster --corpus my-corpus
   nuthatch render  --corpus my-corpus
   ```

4. Register the MCP server with Claude Code. Add to your
   `~/.claude.json` `mcpServers` block:

   ```json
   {
     "mcpServers": {
       "nuthatch": {
         "command": "nuthatch",
         "args": ["serve", "--corpus", "my-corpus"]
       }
     }
   }
   ```

   Restart Claude Code; the five nuthatch tools
   (`corpus_search`, `subgraph_extract`, `card_get`,
   `community_get`, `token_econ_report`) appear in the tool list.

## Build-pipeline operations (when to run what)

nuthatch's build pipeline is four sequential stages. The agent does
NOT orchestrate these — the user runs them out-of-band via the
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
changed, OR when switching the embedding model (per-corpus
override in `<corpus>/.kg/config.yaml` under `embedding.model`).

**Watch mode (`nuthatch watch --corpus N`)**: long-running, picks
up new files via filesystem events. Skip if the user runs
ingests on a cron or batch script.

## Multi-step query flows

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
4. Walk returned edges (`cites`, `authored_by`,
   `co_mentioned_in`); describe the connection path.

### Cluster exploration

> "What themes are in the corpus?"

1. `community_get(community_id=0)` to start with cluster 0.
2. For an outlier-looking member, `card_get(doc_id=<it>)` to see
   why it landed there.
3. Iterate. Cluster IDs are stable across refits (greedy overlap
   remap), so "cluster 3" today is "cluster 3" after the next
   `nuthatch cluster` run.

### Cost check (token economy)

> "How much context did I save by using nuthatch on this task?"

1. `token_econ_report(group_by="tool")` after the work session.
2. nuthatch's counterfactual is per-tool BM25 / card-sum, NOT
   "the whole corpus" (the kestrel-style strawman pinned in
   `docs/DECISIONS.md` § Token-economy methodology). Expect
   2-8x, not 50-100x. Honest numbers.

## When to ask the user to rebuild

- **`corpus_search` returns no hits** for an obvious query and the
  user just said they added new papers -> ask them to run
  `nuthatch ingest --corpus N && nuthatch embed --corpus N`.
- **`community_get` returns "community not found"** for an ID the
  user remembers -> the corpus grew, cluster IDs may have shifted.
  Ask for `nuthatch cluster --corpus N`.
- **`card_get` returns stale frontmatter** -> ask for
  `nuthatch render --corpus N`.

Do NOT try to invoke the build CLI yourself. The agent queries;
the user builds. (`docs/DECISIONS.md` § "Execution model" pins
this — and explains why the user-supervised model is honest about
ingestion failures, while kestrel's agent-driven pipeline hides
them.)

## See also

- `AGENTS.md` — full tool semantics and usage patterns (XML-tagged)
- `AGENTS-MARKDOWN.md` — same content, markdown sections
- `docs/SPEC.md` — architecture
- `docs/DECISIONS.md` — locked design decisions
