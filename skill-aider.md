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

## See also

- `AGENTS.md` — full tool semantics and usage patterns (XML-tagged)
- `AGENTS-MARKDOWN.md` — same content, markdown sections
- `docs/SPEC.md` — architecture
