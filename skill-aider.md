<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for paper corpora; MCP query surface over corpus search + subgraph extraction + per-paper cards + community pages.
trigger: nuthatch serve --corpus <name>
---

# nuthatch — Aider integration

> **Tool spec**: load `AGENTS.md` (XML-tagged) **or**
> `AGENTS-MARKDOWN.md` (plain markdown) for full tool semantics.
> Pick whichever format your parser handles best — both carry the
> same content and stay in sync. This skill file covers
> Aider-specific wiring only.

## One-time setup

1. Install nuthatch into a uv venv:

   ```bash
   sfw uv add nuthatch         # once published; until then install editable from source
   ```

2. Initialise a corpus and register it:

   ```bash
   nuthatch init ~/my-corpus
   nuthatch init ~/my-corpus --register-as my-corpus --set-default
   ```

3. Drop papers into `~/my-corpus/inbox/` and run ingest:

   ```bash
   nuthatch ingest --corpus my-corpus
   ```

4. Configure Aider to launch the MCP server on demand. Aider does
   not yet have first-class MCP support; the workaround is to run
   the server in a side-terminal:

   ```bash
   nuthatch serve --corpus my-corpus
   ```

   And use Aider's `/run` command to invoke nuthatch CLI helpers
   inline (`nuthatch query "..."`, `nuthatch card-get <doc_id>`).

## Aider usage primer

When Aider asks "what should I do?", reach for nuthatch when:

- You need to ground a code change in a paper or doc the user has
  ingested. `nuthatch query "..."` returns the relevant chunks;
  paste them into the Aider context with `/paste`.
- You want to summarise a paper before writing code based on it.
  `nuthatch card-get <doc_id>` returns the per-paper card; that
  goes into Aider's context.

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
up new files via filesystem events. Skip if running ingests on a
cron or batch script.

## Multi-step query flows

### Open-ended question

> "What does the corpus say about X?"

1. `nuthatch query "X"` (or via the MCP `corpus_search` tool when
   wired) -> top chunks.
2. Pick the 2-3 most relevant `doc_id`s from the metadata.
3. `nuthatch card-get <doc_id>` for each -> full per-paper card.
4. Paste into Aider's context.

### Lineage trace

> "How does paper X relate to paper Y?"

1. Resolve `doc_id`s for X and Y via `nuthatch query`.
2. `nuthatch subgraph --seeds doc::X doc::Y --depth 2`.
3. Walk returned edges (`cites`, `authored_by`, `co_mentioned_in`).

### Cost check

> "How much context did the corpus query save?"

1. `nuthatch token-report --group-by tool` after a session.
2. nuthatch counterfactual is per-tool BM25 / card-sum, NOT
   "whole corpus" (per `docs/DECISIONS.md` § Token-economy
   methodology). Expect 2-8x, not 50-100x. Honest numbers.

## When to ask the user to rebuild

- **No hits for an obvious query** after they mentioned new files
  -> ask them to run
  `nuthatch ingest --corpus N && nuthatch embed --corpus N`.
- **Stale community membership** after a corpus grew -> ask for
  `nuthatch cluster --corpus N`.
- **Stale card frontmatter** -> ask for
  `nuthatch render --corpus N`.

Do NOT invoke the build CLI yourself.

## See also

- `AGENTS.md` — full tool semantics and usage patterns (XML-tagged)
- `AGENTS-MARKDOWN.md` — same content, markdown sections
- `docs/SPEC.md` — architecture
