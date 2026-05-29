<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: AGPL-3.0-or-later
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for paper corpora; MCP query surface over corpus search + subgraph extraction + per-paper cards + community pages.
trigger: nuthatch serve --corpus <name>
---

# Nuthatch — Hermes integration

> **Tool spec**: load `AGENTS.md` from the repo root for full tool
> semantics. This skill file covers Hermes-specific wiring only.

Hermes (Nous Research) supports MCP servers as multi-provider
endpoints. This skill walks through registration in either the
host-supervised lane or the VM-resident autonomous lane.

## One-time setup (host-supervised Hermes)

1. Install Nuthatch into a uv venv and initialise a corpus:

   ```bash
   sfw uv add nuthatch         # once published
   nuthatch init ~/my-corpus --register-as my-corpus --set-default
   nuthatch ingest --corpus my-corpus
   ```

2. Register Nuthatch's MCP server in Hermes's config
   (`~/.hermes/config.yaml`):

   ```yaml
   mcp_servers:
     nuthatch:
       command: nuthatch
       args: [serve, --corpus, my-corpus]
       transport: stdio
   ```

3. Restart Hermes. The 5 Nuthatch tools register against the
   active Hermes session and any subsequent task can call them.

## VM-resident Hermes (autonomous lane)

In the persistent agent VM (`hermes-agent-vm`), Nuthatch must be
installed inside the VM and the MCP server registered in the
VM-side `~/.hermes/config.yaml`. The host-side Hermes does NOT
proxy MCP calls into the VM. Each lane runs its own server.

Per `~/project-planning-agent/strands/hermes-interim.md`, the VM
lane is gated on Phases 2-14 of the build plan; until those land,
host-supervised is the only Hermes path.

## Build-pipeline operations (when to run what)

Nuthatch's build pipeline is four sequential stages. Hermes
worker tasks do NOT orchestrate the build. The user (or a
scheduled job inside the VM) runs the four stages out-of-band
via the CLI. Rationale pinned in `docs/DECISIONS.md` § "Execution
model: build out-of-band, query via MCP".

| Stage | When to run | Incremental? |
| --- | --- | --- |
| `nuthatch ingest --corpus N` | new files dropped anywhere under the corpus root | yes (hash-based file dedup) |
| `nuthatch embed --corpus N` | after ingest produces new `papers/` | yes (per-doc Chroma presence check) |
| `nuthatch cluster --corpus N` | after a batch of new embeddings | no (full SBM refit each call; correct for the method) |
| `nuthatch render --corpus N` | after cluster, or when card metadata refreshes | byte-stable (unchanged cards re-write identically) |

In the VM-resident lane, the natural cadence is a cron'd
`nuthatch ingest && nuthatch embed` every 6 hours and a manual
`nuthatch cluster && nuthatch render` on a weekday morning. The
worker tasks query the corpus state as it stands at task-dispatch
time; they don't wait for the next rebuild.

**`nuthatch embed --force`**: re-embed every paper from scratch,
deleting old chunks first. Required when `_DEFAULT_MAX_CHARS` or
`_DEFAULT_OVERLAP_CHARS` in `src/nuthatch/embed/chunk.py` has
changed, OR when switching the embedding model (per-corpus
override in `<corpus>/.kg/config.yaml` under `embedding.model`).

## Multi-step query flows for Hermes workers

### Open-ended question delegated to a worker

> "Summarise what the corpus says about <topic> in 200 words."

1. Worker calls `corpus_search(query="<topic>", k=10)`.
2. Worker picks the 2-3 top hits, calls `card_get` on each.
3. Worker composes the summary citing papers by `title` + `year`.

### Lineage trace from an orchestrator task

> "Trace the citation lineage between paper X and paper Y."

1. Orchestrator resolves `doc_id`s via `corpus_search`.
2. Delegates `subgraph_extract(seed_nodes=["doc::X", "doc::Y"],
   depth=2)` to a worker.
3. Worker returns the subgraph; orchestrator summarises the
   connection path for the human.

### Cluster-summary delegation

1. `community_get(community_id=N)` returns the member list.
2. Per-member `card_get` calls fan out across workers in parallel.
3. Orchestrator stitches the results into a thematic overview.

### Cost-accounting per delegated task

Track per-worker savings by calling `token_econ_report(group_by="tool",
surface_id="hermes-worker-<id>")` after a batch of tasks. The
counterfactual is per-tool BM25 / card-sum, NOT "the whole corpus"
(the prior-implementation strawman pinned in `docs/DECISIONS.md` §
Token-economy methodology). Expect 2-8x reduction ratios per
worker, summed across all five MCP tools.

## When to ask the user to rebuild

- **`corpus_search` returns no hits** for a known query after a
  recent ingest in the VM cadence -> verify the cron actually ran;
  ask the human to inspect `<corpus>/.kg/audit/` for the ingest
  audit log.
- **`community_get` returns "community not found"** for an ID the
  orchestrator cached from a prior session -> the corpus grew,
  cluster IDs may have shifted. Trigger or request a
  `nuthatch cluster --corpus N`.
- **A worker reports stale frontmatter on a card** -> request
  `nuthatch render --corpus N`.

A Hermes orchestrator does NOT invoke the build CLI directly
inside a worker dispatch. Builds run under their own supervision
(cron + tmux log + the operator's eye), so quality issues like
low extraction yield, schema-validation failures, and clustering
downgrades are surfaced in the audit log where the operator sees
them. Not silently absorbed into the agent loop.

## See also

- `AGENTS.md`. Full tool semantics and usage patterns (XML-tagged)
- `~/project-planning-agent/strands/hermes-interim.md`. VM lane build plan
