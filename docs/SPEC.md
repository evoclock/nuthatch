# nuthatch: architecture spec

Working spec as of the project scaffolding. Settled choices land in
`DECISIONS.md`; this file describes the architecture they shape.

## Purpose

Knowledge-graph tool over a user-curated corpus of papers and notes.
Three differentiators in combination (none novel alone):

1. **Stochastic Block Model** clustering (Tiago Peixoto, via `graph-tool`)
   as the principled default; Leiden / vector fallback when SBM is
   unavailable.
2. **Strict ingest-time schema gate**: papers whose metadata can't
   be filled go to `quarantine/`, not to the graph.
3. **Explicit token-economy instrumentation**: every LLM query
   surfaces actual tokens used vs counterfactual full-context.

## Architecture spine

Two flows + one orthogonal instrumentation layer.

### Inbound (corpus → graph)

```text
inbox/ -> identify+hash -> extract (Docling/Chandra-OCR) -> schema validate
                                                              |
                                                  PASS--------+--------FAIL
                                                    |                    |
                                                  chunk + embed       quarantine/
                                                    |                  (await
                                                  MD card               manual
                                                  + HTML companion      metadata
                                                    |                   fix)
                                                  entity extract
                                                    |
                                                  graph integrate
                                                  (SBM re-fit; user-triggered)
                                                    |
                                                  surface (Obsidian + MCP)
```

### Outbound (decay + supersession)

```text
periodic pass -> recompute relevance(half_life, backlinks, age)
                          |
              below threshold        marked superseded
                  |                       |
            flag for archive        edge to successor +
            (manual confirm)        downweight predecessor
```

### Token-econ layer (orthogonal)

```text
LLM query -> compute full-context tokens (counterfactual)
          -> extract subgraph (community summary + k-hop, hybrid)
          -> compute actual tokens
          -> log delta, surface in dashboard
```

## Per-corpus directory layout

Each corpus is a self-contained directory (the `.kg/` subdirectory
is the marker, analogous to `.git/`). Multiple corpora live anywhere
on disk; a small registry at `~/.config/nuthatch/registry.toml`
resolves named corpora to filesystem paths.

```text
my-corpus/
├── .kg/                         # tool bookkeeping (gitignored when sharing)
│   ├── config.yaml              # schema profile, decay params, backend choice
│   ├── manifest.jsonl           # ingest log: what, when, by whom, extractor version
│   ├── state.db                 # runtime state (SQLite)
│   ├── chroma/                  # vector store
│   └── audit/                   # per-run audit logs
├── inbox/                       # drop files here; watcher picks them up
├── quarantine/                  # schema-failed; awaiting metadata fix
├── papers/                      # original PDFs / sources
├── cards/                       # per-paper MD summary (schema-compliant)
├── html/                        # per-paper HTML companion (figures, equations)
├── notes/                       # user-imported notes (multi-format)
├── graph/                       # graph state + serialized SBM BlockState
└── exports/                     # generated outputs (Obsidian vault, SVG, reports)
```

## Clustering backend abstraction

See `src/nuthatch/clustering.py`. Three rigor tiers
(`PRINCIPLED` / `HEURISTIC` / `EMBEDDINGS_ONLY`) × three compute
locations (`LOCAL` / `REMOTE_SAAS` / `REMOTE_CUSTOMER_CLOUD`).
Every clustering response declares the rigor it actually delivered
so consumers can honestly surface downgrades to the user.

This is the first concrete artifact of the scaffold. Implementations
land separately as `nuthatch_sbm`, `nuthatch_leiden`, etc.

## Open questions

- Vector store: ChromaDB vs. simpler embedded option (sqlite-vec).
  Default leans ChromaDB for ecosystem parity with adjacent tools.
- Embedding model: BGE-large for English-only? Multilingual e5 for
  papers with non-English abstracts? Defer until real corpus.
- MCP server: stdio-only at first; HTTP-SSE if/when an external
  agent client needs it.
- Surface for the token-econ dashboard: TUI like thermall, or a
  small web UI? TUI bias for the v1 since most nuthatch users will
  already have a terminal open.

## References

- Strand doc (planning history):
  `~/project-planning-agent/strands/nuthatch.md`
- DECISIONS.md (this directory) for the locked-in choices.
