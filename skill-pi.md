<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for paper corpora; MCP query surface over corpus search + subgraph extraction + per-paper cards + community pages.
trigger: nuthatch serve --corpus <name>
---

# nuthatch — Pi integration

> **Tool spec**: load `AGENTS.md` (XML-tagged) **or**
> `AGENTS-MARKDOWN.md` (plain markdown) for full tool semantics.
> Pick whichever format your parser handles best — both carry the
> same content and stay in sync. This skill file covers Pi-specific
> wiring only.

## One-time setup

1. Install nuthatch into a uv venv and initialise a corpus:

   ```bash
   sfw uv add nuthatch         # once published
   nuthatch init ~/my-corpus --register-as my-corpus --set-default
   nuthatch ingest --corpus my-corpus
   ```

2. Register the MCP server in the Pi agent config. The expected
   shape is the same stdio-MCP launcher pattern other agents use:

   ```yaml
   mcp_servers:
     nuthatch:
       command: nuthatch
       args: [serve, --corpus, my-corpus]
       transport: stdio
   ```

   Confirm the exact path / file name against the Pi version you
   have installed; this convention follows kestrel's pattern that
   OpenClaw inherits.

3. Restart the agent. The 5 nuthatch tools are now available.

## Usage primer

Standard MCP-tool flow: `corpus_search` for open-ended questions,
`card_get` for specific paper deep-dives, `subgraph_extract` for
"what's connected to X" queries, `community_get` for thematic
cluster summaries. See `AGENTS.md` for argument schemas and
recommended `k` / `depth` defaults.

## See also

- `AGENTS.md` — full tool semantics and usage patterns (XML-tagged)
- `AGENTS-MARKDOWN.md` — same content, markdown sections
