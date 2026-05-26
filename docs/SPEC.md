# Nuthatch: architecture spec

The architectural shape of the system. The reasoning behind each
choice lives in [`Design_Decisions.md`](Design_Decisions.md);
the operator workflow lives in [`HOWTO.md`](HOWTO.md).

## Purpose

Nuthatch turns a curated corpus of text-based sources (papers,
patents, internal documents, technical reports, notes,
Repomix-preprocessed codebases, or any user-defined structure)
into a navigable knowledge graph and exposes it to an MCP-aware
agent through a read-only query surface.

The corpus type is a schema profile, not a category constraint.
Built-in profiles ship for `arxiv_paper`, `biorxiv_paper`,
`patent`, `internal_doc`; users define their own via YAML or a
small Python plugin.

## Read / write boundaries

Nuthatch enforces a strict separation between **building** the
corpus and **querying** it:

- **Operator writes** through a five-stage CLI pipeline. Each
  stage is a separate process triggered by the operator with
  inspection gates between stages. There is no end-to-end
  unattended automation, by design.
- **Agent reads** through the MCP server. Five tools, all
  read-only. No tool mutates corpus state, fetches new sources,
  triggers re-ingestion, or rewrites cards.

This split exists so long-running stages are operator-supervised
(quality issues surface in the audit log) and the agent loop
stays fast and predictable. The rationale is documented in
[`Design_Decisions.md`](Design_Decisions.md) under "Execution
model: build out-of-band, query via MCP".

## Pipeline shape

```text
inputs/                                   (user-organised source files)
   |
   v
ingest    -> .kg/manifest.jsonl           (hash-based dedup)
             .kg/extracted/<doc_id>.md    (extracted body markdown)
             .kg/extracted/<doc_id>.meta.json  (publisher-API metadata
                                          for arxiv / bioRxiv, body
                                          heuristic for others)
             .kg/metadata_cache/          (cached API responses)
             papers/                       (originals after success)
             quarantine/<reason>/         (failures, with .reason.json
                                          sidecars)
             .kg/math_retry.jsonl         (math-heavy docs flagged
                                          under --skip-chandra for
                                          later patching)
   |
   v
embed     -> .kg/embeddings/              (Chroma vector store, cosine
                                          HNSW; per-doc incremental skip)
   |
   v
graph     -> .kg/graph/graph.json         (NetworkX MultiDiGraph;
                                          doc + entity nodes;
                                          authored_by / cites /
                                          co_mentioned_in edges)
   |
   v
cluster   -> partition written back onto graph nodes
             (SBM via graph-tool when available; Leiden fallback;
              embeddings k-means as floor)
   |
   v
render    -> cards/<doc_id>.md            (Obsidian-compatible
                                          kb-reports-aligned frontmatter)
             communities/<id>.md          (one page per cluster, wikilinks
                                          to member cards)
             dashboard.md, index.md, log.md
```

Each stage is independently re-runnable. Ingest is hash-deduped at
the file level. Embed is per-doc-presence-checked against the
vector store. Graph rebuilds deterministically from the extracted
sidecars. Cluster always re-fits the whole graph (correct for the
SBM / Leiden methods). Render is byte-stable per card.

A sixth stage, `decay`, runs independently on demand to compute
the relevance / supersession pass over the graph and cards.

### Convention: `benchmark_test/` for reference PDFs

`benchmark_test/` is a reserved directory name: PDFs placed there
are kept on disk for OCR / extraction benchmarks but are NOT
auto-scanned into the corpus. Use it for canonical ground-truth
scans (Mendel 1866, McDonald-Kreitman 1991, etc.) that you want
the OCR benchmark scripts (`scripts/bench/`) to read but the
ingest pipeline to leave alone.

## Per-corpus directory layout

Each corpus is a self-contained directory. The `.kg/` subdirectory
marks the corpus root (analogous to `.git/`). Discovery walks up
from the current working directory looking for `.kg/`; the named
registry at `~/.config/nuthatch/registry.toml` resolves
`--corpus <name>` to a path.

```text
<corpus-root>/
|-- .kg/                          tool-managed state (gitignored when sharing)
|   |-- audit/                    per-stage run logs
|   |-- manifest.jsonl            ingest record: hash, source, status, timestamp
|   |-- config.yaml               schema profile, embedding overrides
|   |-- extracted/                <doc_id>.md + <doc_id>.meta.json per ingested doc
|   |-- embeddings/               Chroma vector store
|   |-- graph/graph.json          NetworkX MultiDiGraph (node-link JSON)
|   |-- metadata_cache/           cached publisher-API responses
|   |-- mcp/mcp.log               MCP server log
|   `-- math_retry.jsonl          (optional) docs flagged for Chandra patch
|-- inputs are user-organised: inbox/, arxiv/, bioarxiv/, notes/, root-level...
|-- papers/                       post-ingest originals
|-- quarantine/<reason>/<file>    failed files + .reason.json sidecars
|-- cards/<doc_id>.md             rendered per-doc cards
|-- communities/<id>.md           rendered per-cluster pages
|-- reports/                      generated reports (token-economy, decay)
|-- dashboard.md                  Dataview-driven landing page
|-- index.md                      navigation index
`-- log.md                        unified timeline
```

The pipeline scans the corpus root recursively, skipping the
nuthatch-managed subdirs above (`.kg/`, `papers/`, `quarantine/`,
`cards/`, `communities/`, `reports/`, `html/`, `graph/`,
`exports/`). User-organised input subdirs (`inbox/`, `arxiv/`,
`bioarxiv/`, `notes/`, or anything else) are picked up.

## MCP query surface

The server exposes nine tools over stdio JSON-RPC 2.0. None mutate
corpus state.

| Tool | Returns |
| --- | --- |
| `corpus_search(query, k)` | Top-`k` chunk hits with doc_id, score, text preview, metadata + `community_id` / `community_path` / `community_label` so agents can route directly into community tools |
| `subgraph_extract(seed_nodes, depth)` | BFS subgraph (nodes + edges) around the seeds |
| `card_get(doc_id)` | Full rendered per-doc card markdown (frontmatter carries community fields) |
| `community_get(community_id)` | Rendered per-cluster page markdown |
| `community_brief(community_id, top_n)` | Short structured preamble: label, n_members, top-N representative doc_ids. Cheap lead-in before drilling into a full card / community page |
| `community_search(query, k)` | Semantic search at the community level. Ranks communities by query-to-centroid cosine. Lets agents jump straight to the relevant cluster without a chunk-level intermediate |
| `community_core_nodes(community_id)` | High-degree members within a community's induced subgraph — the "key papers" of the cluster |
| `community_hierarchy(doc_id)` | Full nested-SBM path from leaf community up through super-communities. Single-level for flat backends; multi-level for the principled SBM tier |
| `token_econ_report(group_by, since, until, ...)` | Aggregate token-economy stats with per-tool BM25 / card-sum counterfactual baselines |

The community tools together make Nuthatch's headline feature
work: chunk-level retrieval surfaces routing keys, and the agent
can choose chunk-level / community-level / hierarchy-level zoom
without ever having to walk the global graph. See
`docs/Design_Decisions.md` § *Community-aware retrieval* for the
full rationale.

Per-agent registration recipes live in the `skill-*.md` files at
the repo root (one per supported host: `skill-claude-code.md`,
`skill-codex.md`, `skill-aider.md`, `skill-opencode.md`,
`skill-pi.md`, `skill-hermes.md`).

## Module layout in `src/nuthatch/`

```text
src/nuthatch/
|-- cli.py                  CLI subcommand dispatch
|-- __main__.py             `python -m nuthatch` entry
|-- corpus/                 layout + registry + per-corpus config loader
|-- ingest/                 source-walk + extract + metadata + schema gate
|   |-- state_machine.py    IngestOrchestrator
|   |-- extract.py          PDF routing: Docling / Chandra / Granite / EasyOCR
|   |-- metadata.py         schema-validated extraction
|   |-- source_metadata.py  publisher-API metadata fetchers (arxiv, bioRxiv)
|   |-- math_validator.py   broken-LaTeX classifier (drives --skip-chandra retry)
|   |-- security.py         SSRF / loopback / metadata-IP URL gate
|   |-- cache.py            semantic cache for re-ingest
|   |-- hooks.py            pluggable pre / post hooks
|   `-- watch.py            long-running filesystem watcher
|-- schema/                 SchemaProfile abstraction + built-in profiles
|-- dedup/semantic.py       embedding-based near-duplicate detection + reranker
|-- embed/                  chunking + embedding + Chroma store
|   |-- chunk.py            hybrid chunker with full-doc coverage invariant
|   |-- embed.py            Embedder + single-doc embed_document
|   |-- store.py            VectorStore protocol + ChromaVectorStore
|   `-- orchestrator.py     corpus-wide embed loop (incremental + --force)
|-- graph/                  NetworkX graph layer
|   |-- build.py            assemble graph from doc contributions
|   |-- entities.py         regex-based entity extraction
|   |-- edges.py            Edge dataclass + confidence enum
|   |-- io.py               schema-versioned JSON serialisation
|   |-- decay.py            relevance(t) = max(backlinks,1) * exp(-ln2*dt/half_life)
|   `-- orchestrator.py     corpus-wide graph build
|-- clustering/             rigor ladder (SBM / Leiden / embeddings)
|   |-- protocol.py         ClusteringBackend protocol + Rigor enum
|   |-- router.py           highest-rigor-available dispatcher
|   |-- hub_exclusion.py    core_nodes excluded + majority-neighbour reattach
|   |-- stable_ids.py       greedy overlap remap for cluster-ID continuity
|   `-- backends/{sbm,leiden,embeddings}.py
|-- decay/                  decay + supersession pass over cards + graph
|-- render/                 Obsidian cards, communities, dashboard, HTML companions
|-- retrieve/vector.py      query-side embedding + Chroma retrieval
|-- mcp/server.py           stdio JSON-RPC server + 5 tools
`-- token_econ/             per-tool counterfactual + JSONL log + markdown report
```

## See also

- [`Design_Decisions.md`](Design_Decisions.md): the reasoning
  behind every choice in this spec
- [`HOWTO.md`](HOWTO.md): operator runbook for the five-stage
  pipeline plus MCP registration per agent host
- [`extraction-benchmarks/ocr-comparison.md`](extraction-benchmarks/ocr-comparison.md):
  evidence for the OCR routing decisions
