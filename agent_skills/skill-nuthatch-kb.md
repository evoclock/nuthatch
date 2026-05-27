---
name: nuthatch-kb
description: Query a nuthatch-published knowledge base. Pick the right MCP tool for the kind of question (chunk hit vs community jump vs subgraph walk), dedupe correctly, separate lexical hits from semantic neighbours, and fall back to the on-disk Obsidian surface when MCP is not available.
trigger: /nuthatch-kb
---

# /nuthatch-kb

Recipe for an agent working against a nuthatch-published KB. The KB
ships with both an MCP server (`mcp__nuthatch__*` tools) and an
on-disk Obsidian-compatible surface (`cards/`, `communities/`,
`.kg/communities.json`, `.kg/graph/graph.json`). The two surfaces are
internally consistent: every fact you can reach through MCP is also
reachable by reading a card or a community page directly.

## When to use

- The user asks a question that begins with "what does the corpus say
  about...", "find me papers on...", "which cluster contains...", or
  similar; AND
- The KB at hand is a nuthatch-published KB (look for `.kg/communities.json`,
  `.kg/community_centroids.npy`, and a top-level `CAPABILITIES.json`).
- Do NOT invoke this skill for general web search, code-base navigation,
  or non-nuthatch knowledge bases.

## Surfaces available

### MCP (preferred when available)

| Tool | Use when |
|------|----------|
| `mcp__nuthatch__corpus_search` | Lexical or semantic search at the chunk level. Returns top-k chunks with score + doc_id + title. Best for "show me where the corpus mentions X". |
| `mcp__nuthatch__community_search` | Same idea but at the cluster level. Ranks communities by query-to-centroid cosine. Best for "what topical area covers X" or "jump me to the relevant cluster". |
| `mcp__nuthatch__community_brief` | Cheap structured preamble for one community: label, member count, top-N representatives. Use BEFORE fetching the full community page. |
| `mcp__nuthatch__community_get` | Full per-community markdown page. Use after `community_brief` decided the cluster is worth a deeper look. |
| `mcp__nuthatch__community_hierarchy` | Walk the nested SBM hierarchy (leaf → super-communities). Use when the user asks "what bigger theme contains cluster X". |
| `mcp__nuthatch__community_core_nodes` | High-degree members within one community's induced subgraph. Use for "what are the key papers in cluster X". |
| `mcp__nuthatch__card_get` | Full per-paper markdown card (frontmatter + abstract + topics). Use to drill into a specific paper after retrieval. |
| `mcp__nuthatch__subgraph_extract` | BFS from one or more seed nodes through the graph. Use for "show me everything connected to paper X up to 2 hops" or relationship questions. |
| `mcp__nuthatch__token_econ_report` | Aggregated MCP query log. Use for "how much have I queried this KB, and which tools dominate".|

### On-disk Obsidian surface (fallback / browser-only use)

| Path | Use when |
|------|----------|
| `cards/<doc_id>.md` | One markdown card per paper with frontmatter (title, authors, year, tags, topics, `cluster/<cid>`, `community_label`). Grep-friendly. |
| `communities/<slug>.md` | One page per community with `Core papers` + `Members` lists. The slug derives from the community label. |
| `dashboard.md`, `index.md` | Discovery entry points. Open these first if exploring. |
| `.kg/communities.json` | Canonical community index. Use `community_id → members` for programmatic lookup without the MCP server. |
| `.kg/graph/graph.json` | NetworkX-shape graph with per-node `community: <cid>` and `community_label` embedded. Self-contained; no joins needed. |
| `docs/graph.html` | Static D3 viz; opens in any browser, no backend. |

## Query recipes

### Recipe 1: "how many papers mention X, and what's the context?"

1. `corpus_search(query="X", k=20)` — get top-k chunk hits.
2. Dedupe the results by `doc_id` (a single paper can produce multiple
   chunk hits; do not double-count papers).
3. For each distinct `doc_id`, call `card_get(doc_id)` to read the
   abstract + topics + methods.
4. Separate **lexical hits** (the paper actually names X in title /
   methods / abstract) from **semantic neighbours** (the embedder
   ranks them adjacent because they discuss related material). Only
   the lexical hits count toward "papers about X"; the neighbours are
   useful as adjacent context.
5. Report distinct count + context per paper.

### Recipe 2: "what topical area does my question belong to?"

1. `community_search(query="...", k=3)` — get top-k community matches
   by centroid cosine.
2. For the top hit, `community_brief(community_id=<cid>, top_n=5)` to
   see the representative members at a glance.
3. If the user wants the full picture, `community_get(community_id=<cid>)`
   for the full member list + label.
4. If the user asks "what bigger theme contains this", call
   `community_hierarchy(doc_id=<any-member>)` to walk up the nested
   SBM levels.

### Recipe 3: "show me everything connected to paper X"

1. `card_get(doc_id="X")` — confirm the paper exists and read its
   frontmatter.
2. `subgraph_extract(seeds=["doc::X"], depth=2)` — BFS through the
   graph (relationships are typed: `authored_by`, `cites`,
   `mentions_method`, etc.).
3. Group results by relation type before reporting.

### Recipe 4: "key papers in cluster N"

1. `community_brief(community_id=N)` — cheap lookup with top reps.
2. `community_core_nodes(community_id=N)` — high-degree members
   within the community's induced subgraph (the "important ones" by
   internal connectivity).

### Recipe 5: air-gapped / corporate environments (no MCP)

When MCP is unavailable because the runtime cannot reach the
nuthatch server (corporate firewall blocks outbound traffic,
locked-down browser, claude.ai web with no custom-connector
permission), the KB is still queryable as static text.

Three working routes, in order of fidelity:

1. **Self-hosted MCP inside the corporate network (highest fidelity).**
   Ship the KB as a containerised artifact (zip or container image)
   into an internal runtime such as Microsoft Copilot Studio
   knowledge tool, or a team-internal container host. If `nuthatch
   serve` runs inside that container, full MCP semantics
   (`corpus_search`, `community_search`, `token_econ_report`) are
   available without any external traffic.
2. **claude.ai Project with cards uploaded (medium fidelity).** Drop
   the `cards/` directory into a claude.ai Project. Claude's
   project-file retrieval handles lexical / semantic recall over the
   markdown. No `community_search`, no `subgraph_extract`, no
   community-centroid jumps; the `cluster/<cid>` tag in card
   frontmatter still works as a coarse group-by handle.
3. **Plain markdown navigation (always works).** Open `dashboard.md`
   for an overview. Lexical search across `cards/*.md` for the
   query term. Use `cluster/<cid>` to find sibling papers in the
   same community. Community pages under `communities/` carry the
   topical theme and member list.

Route 1 needs a one-time deployment artifact (Dockerfile +
container entrypoint that runs `nuthatch serve`). Route 2 needs
only file upload. Route 3 needs only the published KB directory.

## Anti-patterns to avoid

- **Treating semantic neighbours as positive hits.** `corpus_search`
  uses dense embedding cosine; a paper about gradient-based
  optimization for neural networks scores adjacent to papers about
  XGBoost (gradient boosting), even when XGBoost is never mentioned.
  Always confirm via `card_get` before counting.
- **Trusting the community label as ground truth.** Labels are
  produced by an LLM-naming pass over member titles; for clusters with
  off-topic members, the label can mislead (cluster 10 in the demo
  was labelled "Deep Learning in Genomics and Proteomics" but
  contained a KAN+XGBoost paper on Australian energy markets). The
  numeric `cluster/<cid>` handle is the safe primary key; the label
  is advisory metadata.
- **Calling community_search expecting tool-level filtering by label.**
  community_search ranks by *centroid cosine*, not label string match.
  If you want a label match, read the labels block from
  `.kg/communities.json` directly.
- **Skipping community_brief and going straight to community_get.**
  community_brief is cheap; community_get returns a multi-KB markdown
  page. Drill incrementally.
- **Forgetting that doc_ids carry a `doc::` graph-node prefix.** The
  community index and the graph use `doc::<bare_id>`; card filenames
  and `corpus_search` results use the bare `<doc_id>`. Strip / add
  the prefix as needed when crossing surfaces.

## Diagnostics

If a tool errors with "community index not built" or "community
centroids not built", the published KB is missing canonical
artefacts:

- `.kg/communities.json` (powers community_brief, community_get,
  community_hierarchy, community_core_nodes)
- `.kg/community_centroids.npy` (powers community_search)

Workarounds:

1. If suffixed copies exist (`.kg/communities_sbm.json` etc.), the KB
   was published with an older nuthatch; ask the maintainer to
   re-publish with the current version (the publish step now promotes
   to canonical automatically).
2. If the chroma store is missing or compressed (`embeddings.tar.gz`
   at KB root with no unpacked `.kg/embeddings/`), unpack:

   ```bash
   mkdir -p .kg && tar xzf embeddings.tar.gz -C .kg/
   ```

3. Fall back to the on-disk Obsidian surface (recipe 5).

## Cross-references

- Tool repo (nuthatch itself): documentation under `docs/HOWTO.md`,
  `docs/SPEC.md`, `docs/Design_Decisions.md`.
- In-tool agent skills: `agent_skills/skill-*.md` carry the
  per-agent-host instructions (Aider, Claude Code, Codex, Hermes,
  Opencode, Pi).
- This skill's tool-shipped copy: `<nuthatch-repo>/agent_skills/skill-nuthatch-kb.md`.
