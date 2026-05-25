# nuthatch

> Knowledge-graph tool for paper corpora and notes, with principled
> clustering and explicit LLM context-token economy.

**Status**: Planning / scaffold only. No usable functionality yet. See
`docs/SPEC.md` for the architecture spine.

## What this is

A local-first tool that turns a curated corpus of papers (PDFs, HTML,
plain text) and freeform notes into a navigable knowledge graph,
surfaces query-scoped subgraphs to LLMs to cut context usage, and
keeps the graph honest with a strict ingest-time metadata schema.

The differentiating choices, none of which is novel alone but the
combination of which is under-served:

1. **Stochastic Block Model** (Tiago Peixoto's degree-corrected nested SBM
   via [graph-tool](https://graph-tool.skewed.de/)) as the principled
   default clustering backend. Falls back to Leiden + vector
   similarity when graph-tool is unavailable or the corpus exceeds
   local-SBM compute capacity. The user is told which backend is
   active and what trade-offs apply.

2. **Hard schema gate on ingest.** Papers whose metadata cannot be
   extracted into the required schema go to `quarantine/` with a
   documented reason, not into the graph. The graph stays clean.

3. **Explicit token-economy instrumentation.** Every LLM query
   surfaces actual tokens used vs. the counterfactual full-context
   size, so users see what the subgraph extraction is saving.

## Status / roadmap

This repo currently contains only the package skeleton and the first
concrete spec artifact (the `ClusteringBackend` protocol). No ingest,
no graph, no UI yet.

See `docs/SPEC.md` for the architecture and `docs/DECISIONS.md` for
the locked-in choices that shape the build.

## Licence

Apache 2.0. See `LICENSE`.

## Acknowledgements

Built by Julen Gamboa with some agent-assisted spec-driven
development. Claude Code drove orchestration, design discussion, and
most of the implementation work; Hermes (using GPT-5.5 and Minimax
M2.5) handled additional review and asset generation. Specs and
decisions are pinned in `docs/SPEC.md` and `docs/DECISIONS.md` to
hold the agent loop accountable: every implementation must point
back to a spec entry.
