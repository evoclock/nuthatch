# nuthatch — locked-in decisions

Choices that should not be relitigated without specific new evidence.
Decisions worth changing get a new entry below the originals with the
reason for the change; the old entry stays for the audit trail.

## Identity

- **Name**: nuthatch (working name as of 2026-05-24, from the bird
  genus *Sitta* that navigates trees head-first in any direction).
- **Licence**: Apache 2.0 (enterprise-friendly; AGPL excluded
  because some enterprise procurement bans it outright).

## Architecture

- **Storage**: fully isolated per-corpus directory tree with `.kg/`
  as the marker (analogous to `.git/`). A small registry at
  `~/.config/nuthatch/registry.toml` resolves names to paths. No
  co-located multi-corpus root.
- **Schema profiles**: user-extensible from day one. Built-in
  profiles for `arxiv_paper`, `biorxiv_paper`, `patent`,
  `internal_doc`; users define their own via YAML or Python plugin.
- **Schema gate**: strict. Papers that can't fill the schema go to
  `quarantine/`, NOT to the graph.
- **Multi-corpus from day one** in the data model. The OSS may
  default to a single-corpus UX but the underlying storage layout
  and API support N from the start.
- **Auth**: OSS is single-user (filesystem-permissions only). Paid
  tier adds SSO / RBAC / audit logs on top via the same `actor`
  abstraction surfaced at every state-changing call site.

## Clustering

- **SBM is the principled default**: Peixoto's nested
  degree-corrected SBM via `graph-tool` (conda-only; no pure-Python
  reimplementation). Pluggable backend via the `ClusteringBackend`
  protocol in `src/nuthatch/clustering.py`.
- **Leiden + vector fallback** when `graph-tool` is unavailable or
  the graph exceeds local SBM compute capacity. Rigor downgrade is
  surfaced to the user via the `rigor_used` field on every response,
  never silently accepted.
- **Hub exclusion** in clustering (ported pattern: high-degree
  super-hub nodes excluded from partition, reattached by
  majority-vote neighbour community). Essential for paper KGs where
  a 200k-citation paper would otherwise distort the partition.
- **SBM refit is user-triggered**, not on every ingest. ChromaDB +
  embeddings serve as the bridge between refits; users see which
  view they're querying (the graph as of refit N, plus M new
  papers via vector search).

## Ingest

- **Hash-based file dedup** at inbox (same paper dropped twice).
- **Semantic dedup** for papers (preprint vs published, multiple
  PDF revisions, OCR rebuild of same paper).
- **Watched directory** for ingest, plus an explicit `nuthatch
  import` command for batch / scripted use.
- **Manifest file** as authoritative ingest record: `.kg/manifest.jsonl`
  with one line per (file, extractor_version, timestamp).
- **Edge confidence labels** on every edge:
  `EXTRACTED | INFERRED | AMBIGUOUS`. Surfaced in the graph viewer
  and in the MCP-served subgraphs.
- **Decay function** for stale notes / superseded papers; formula
  ported from the PhD KB's `kb-reports.md` schema:
  `relevance(t) = max(backlinks, 1) · exp(-ln2 · Δt / half_life)`.

## Surfacing

- **MCP stdio server** for agents (`nuthatch serve --corpus <name>`
  scopes the agent to one corpus).
- **Per-paper MD card** (schema-compliant, queryable).
- **Per-paper HTML companion** (figures, equations, rich content).
- **Obsidian render** for humans.
- **Token-economy instrumentation** shown live as actual vs
  counterfactual full-context tokens per query.

## Monetisation

- **Open-core**: algorithm always free, large-scale compute
  optionally paid.
- **Phase 1**: OSS only. Build adoption. No paid layer.
- **Phase 2 (when usage demands it)**: Hosted SaaS for users who
  exceed local SBM compute. "Send us your graph, get back a refit."
  Manual at first; API once usage validates.
- **Phase 3 (when an enterprise prospect asks)**: Managed-on-
  customer-cloud (Model C). Runtime image deployed via cross-account
  IAM into customer cloud; their data never leaves their VPC.
- **Explicitly NOT a target**: deploy-yourself + sell-support
  (Model B). Doesn't fit the tool's complexity profile; conversion
  would be poor.

## Out of scope for v1

- Cross-corpus queries (corpus-scoped only initially; registry
  shape leaves the door open for later).
- Audio / video transcription (NeurIPS / ICML talk ingest).
- Google Drive / cloud-folder ingest.
- Pretty UI for adding custom schema profiles (YAML-by-hand is the
  v1 path).

## References

- `docs/SPEC.md` — architecture spine.
- `~/project-planning-agent/strands/nuthatch.md` — planning history
  with full design discussion.
