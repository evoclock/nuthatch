<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for paper corpora; MCP query surface over corpus search + subgraph extraction + per-paper cards + community pages.
trigger: nuthatch serve --corpus <name>
---

# nuthatch — Hermes integration

> **Tool spec**: load `AGENTS.md` (XML-tagged) **or**
> `AGENTS-MARKDOWN.md` (plain markdown) for full tool semantics.
> Pick whichever format your parser handles best — both carry the
> same content and stay in sync. This skill file covers
> Hermes-specific wiring only.

Hermes (Nous Research) supports MCP servers as multi-provider
endpoints. This skill walks through registration in either the
host-supervised lane or the VM-resident autonomous lane.

## One-time setup (host-supervised Hermes)

1. Install nuthatch into a uv venv and initialise a corpus:

   ```bash
   sfw uv add nuthatch         # once published
   nuthatch init ~/my-corpus --register-as my-corpus --set-default
   nuthatch ingest --corpus my-corpus
   ```

2. Register nuthatch's MCP server in Hermes's config
   (`~/.hermes/config.yaml`):

   ```yaml
   mcp_servers:
     nuthatch:
       command: nuthatch
       args: [serve, --corpus, my-corpus]
       transport: stdio
   ```

3. Restart Hermes. The 5 nuthatch tools register against the
   active Hermes session and any subsequent task can call them.

## VM-resident Hermes (autonomous lane)

In the persistent agent VM (`hermes-agent-vm`), nuthatch must be
installed inside the VM and the MCP server registered in the
VM-side `~/.hermes/config.yaml`. The host-side Hermes does NOT
proxy MCP calls into the VM — each lane runs its own server.

Per `~/project-planning-agent/strands/hermes-interim.md`, the VM
lane is gated on Phases 2-14 of the build plan; until those land,
host-supervised is the only Hermes path.

## Usage primer

Hermes worker tasks that should reach the corpus use the standard
flow: `corpus_search` first to discover relevant `doc_id`s, then
`card_get` or `subgraph_extract` to deepen. The token-economy
report (`token_econ_report`) lets a Hermes orchestrator measure
how much subgraph extraction saved per delegated task.

## See also

- `AGENTS.md` — full tool semantics and usage patterns (XML-tagged)
- `AGENTS-MARKDOWN.md` — same content, markdown sections
- `~/project-planning-agent/strands/hermes-interim.md` — VM lane build plan
