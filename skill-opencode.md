<!--
SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

---
name: nuthatch
description: Local-first knowledge-graph tool for paper corpora; MCP query surface over corpus search + subgraph extraction + per-paper cards + community pages.
trigger: nuthatch serve --corpus <name>
---

# nuthatch — OpenCode integration

> **Tool spec**: load `AGENTS.md` (XML-tagged) **or**
> `AGENTS-MARKDOWN.md` (plain markdown) for full tool semantics.
> Pick whichever format your parser handles best — both carry the
> same content and stay in sync. This skill file covers
> OpenCode-specific wiring only.

OpenCode supports MCP servers natively via its config file.

## One-time setup

1. Install nuthatch into a uv venv and initialise a corpus
   (`nuthatch init ~/my-corpus --register-as my-corpus --set-default`,
   then `nuthatch ingest`).

2. Register nuthatch's MCP server in OpenCode's config. OpenCode
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

3. Restart OpenCode. The 5 nuthatch tools (`corpus_search`,
   `subgraph_extract`, `card_get`, `community_get`,
   `token_econ_report`) appear in the tool list.

## OpenCode usage primer

When a user asks a corpus-grounded question, the typical flow:

1. Call `corpus_search` with their question (k=5-10).
2. Read the top hits' `doc_id` + `metadata.title`.
3. Call `card_get` on the 2-3 most relevant `doc_id`s.
4. Compose the answer, citing papers by title + year.

For "what's connected to X" questions, use `subgraph_extract`
after `corpus_search` resolves the seed `paper::<doc_id>`.

## See also

- `AGENTS.md` — full tool semantics and usage patterns (XML-tagged)
- `AGENTS-MARKDOWN.md` — same content, markdown sections
