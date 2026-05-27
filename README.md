# Nuthatch

<p align="center">
  <img src="assets/Nuthatch_bgrm.png" alt="Nuthatch logo" width="200">
</p>

> Local-first knowledge-graph tool for structured corpora, with
> principled clustering, strict ingest-time schema validation, and
> honest LLM token-economy accounting.

## What this is

Nuthatch turns a curated corpus (research papers, patents, internal
documents, technical reports, notes, Repomix-preprocessed codebases,
or any text-based source) into a navigable knowledge graph and
serves it to an MCP-aware agent (Claude Code, Codex, OpenCode,
Aider, Pi, Hermes, an Obsidian plugin, or your own client). The
agent queries the graph through five read-only MCP tools; the
operator drives the build pipeline through a five-stage CLI.

The rendered corpus opens as an Obsidian-compatible vault: per-paper
cards live under `cards/`, per-community pages under `communities/`,
with `dashboard.md` / `index.md` / `log.md` at the top. Wikilinks
between cards drive Obsidian's graph view; Dataview queries in the
dashboard filter by tag / year / community.

The architecture is documented in
[`docs/Design_Decisions.md`](docs/Design_Decisions.md) and the
operator workflow in [`docs/HOWTO.md`](docs/HOWTO.md). The diagram
below is the data lifecycle; the corresponding module graph and the
full architecture set are at
[`docs/architecture/`](docs/architecture/).

<p align="center">
  <img src="docs/architecture/nuthatch_data_lifecycle.svg" alt="Nuthatch data lifecycle" width="720">
</p>

<p align="center">
  <img src="assets/screenshots/sbm_full.png" alt="Nuthatch D3 topology viz: full SBM partition with some filters dropped" width="720">
  <br/>
  <em>SBM partition with a baseline subset of filters active. Communities coloured via Catppuccin Mocha; floating boxes (Communities, Filters) are draggable and persist position to <code>localStorage</code>. Interactive version lives at <code>docs/graph.html</code> in every nuthatch-published KB; open it in any browser, no backend.</em>
</p>

<p align="center">
  <img src="assets/screenshots/sbm_no_citations.png" alt="Same partition, most filters off, citations off" width="720">
  <br/>
  <em>Most relations toggled off and the <code>cites</code> relation off: the visible structure is the SBM community as the clustering inferred it from non-citation edges (authorship, topic / method co-mentions, embedding-driven adjacency).</em>
</p>

<p align="center">
  <img src="assets/screenshots/sbm_citations.png" alt="Same partition, most filters off, citations on" width="720">
  <br/>
  <em>Same filter set as above, but with <code>cites</code> toggled on. The new edges show how one paper citing another can extend a community beyond what semantic / co-mention signals alone would produce. The contrast between the previous frame and this one is the citation-extension effect made visible.</em>
</p>

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=BlyR4Qi6fsI">
    <img src="https://img.youtube.com/vi/BlyR4Qi6fsI/maxresdefault.jpg" alt="Nuthatch demo" width="720">
  </a>
</p>

## Why it exists

The graph-augmented retrieval space has working open-source
implementations (Microsoft GraphRAG, graphify, LightRAG, HippoRAG,
nano-graphrag) plus adjacent tools (Cognee, PaperQA2, Khoj, Verba).
We surveyed them and built Nuthatch anyway because we wanted a
specific combination of choices none of them makes:

- **Community-aware retrieval, not just clustering.** Every chunk
  hit carries `community_id`, `community_path` (the nested SBM
  chain), and `community_label` so an agent can route directly
  into the relevant cluster without paying for a card fetch first.
  Plus `community_search` ranks communities semantically by
  query-to-centroid cosine; flat modularity-based clustering
  (Leiden, Louvain) cannot do this. See
  [`docs/Design_Decisions.md`](docs/Design_Decisions.md) §
  *Community-aware retrieval*.
- Bayesian Stochastic Block Model (Peixoto, via
  [graph-tool](https://graph-tool.skewed.de/)) as the principled
  clustering ceiling, with Leiden as a graceful fallback when
  graph-tool is unavailable. The SBM tier emits a nested hierarchy
  that flat methods cannot, and Nuthatch persists every level so
  agents can zoom from leaf clusters up to coarser super-clusters.
- Honest per-tool token-economy accounting (BM25 baseline for
  search, card-token-sum baseline for subgraph and community)
  instead of whole-corpus headline ratios
- Build pipeline triggered out-of-band by the operator with quality
  gates between stages, never end-to-end unattended
- Strict schema quarantine on ingest with reasoned per-file failure
  sidecars
- Authoritative metadata fetched from publisher APIs (arxiv,
  bioRxiv) rather than parsed from PDF body text
- User-extensible schema profiles supporting papers, patents,
  internal documents, and arbitrary user-defined types
- A read-only MCP query surface that the agent cannot mutate

<p align="center">
  <img src="assets/Token_economy.png" alt="Nuthatch token-economy report: per-tool actual cost vs BM25 and card-sum baselines" width="720">
  <br/>
  <em>Per-tool token-economy report from <code>token_econ_report</code>. Each tool is measured against two honest baselines: BM25 (what a flat keyword search over the whole corpus would cost) for search tools, and card-token-sum (the cost of fetching every card) for subgraph and community tools. The ratio shows how much of the corpus a query actually touches. Whole-corpus headline ratios inflate the savings figure by comparing against a ceiling nobody would pay; Nuthatch compares against what a reasonable alternative would actually cost.</em>
</p>

The corpus type is a schema profile, not a category constraint.
Nuthatch is designed for any structured-corpus problem where
graph-augmented retrieval helps.

## Status

The build pipeline is operational. The five build stages
(`ingest`, `embed`, `graph`, `cluster`, `render`) chain into a
queryable corpus, plus three post-pipeline commands serve and
ship it: `serve` (the MCP server), `publish` (export a shareable
KB directory), and `viz d3` (interactive force-atlas2 + Catppuccin
topology viz). Clustering supports an optional `--relabel-llm`
pass that replaces the heuristic word-frequency labels with
topical 2-4 word names produced by an LLM (default
`granite3-dense:8b` via Ollama).

The MCP server exposes nine read-only tools: `corpus_search`,
`subgraph_extract`, `card_get`, `community_get`, `community_brief`,
`community_search`, `community_core_nodes`, `community_hierarchy`,
and `token_econ_report`. The community tools, plus the routing
keys (`community_id`, `community_path`, `community_label`)
carried inside every search hit and every card's frontmatter,
are what make graph-RAG actually work without per-card lookups.

The test suite covers every contract (routing thresholds, model
defaults, chunk coverage, reranker invocation, schema validation,
end-to-end pipeline integration, suffix-rename invariants,
publish-side canonical promotion, `--relabel-llm` round-trip with
mock LLM).

A pre-built reference corpus
([nuthatch-kb-demo](https://github.com/evoclock/nuthatch-kb-demo))
and a short demo are shipped. PyPI publication is in progress.

## Roadmap

Planned work, not yet landed:

- **Air-gapped / corporate environments.** A Dockerfile + container
  entrypoint so the published KB can be lifted into internal
  runtimes (Microsoft Copilot Studio knowledge tool, team-internal
  container hosts such as Testudo) and queried via MCP without any
  outbound traffic. The published KB is already a self-contained
  bundle (markdown cards + JSON indexes + chroma archive); this
  task wraps it in a runtime so a corporate browser can reach a
  full MCP server inside the firewall.
- **Routing hardening for heterogeneous inputs.** The current
  ingest pipeline routes by filename profile (arXiv preprint,
  bioRxiv, internal doc, patent); next is per-content-type
  detection and per-profile thresholding so mixed corpora
  (scanned PDFs + born-digital papers + plain-text notes) route
  cleanly without manual triage.
- **Chandra for math-heavy text.** The Chandra OCR + math-aware
  extraction path is in `--skip-chandra`-style optional form; the
  next sprint promotes it from optional to first-class for any
  paper the triage step flags as math-heavy, with `sympy`-based
  per-equation validation against a known-equation checklist
  (already exists for the OCR benchmark; needs lifting into the
  ingest path).
- **Granite-Docling as the general extraction default.** Per the
  OCR benchmark (`docs/extraction-benchmarks/ocr-comparison.md`),
  Granite-Docling matched Chandra on key facts at much lower
  output volume; it becomes the default for born-digital
  scientific papers. Chandra stays the default for math-heavy
  papers; SmolDocling and EasyOCR stay as documented fallbacks.
- **SPECTER2 as the scientific-paper embedding default.** The
  current default embedding model is `BAAI/bge-m3` (general
  purpose). For the `scientific_paper` profile, SPECTER2
  (AllenAI; trained on the scientific-citation graph and
  identified during the PhD knowledge-base build as superior
  to Docling-derived embeddings for paper retrieval) becomes
  the documented default. Other profiles keep BGE-M3.

## Quick start

See [`docs/HOWTO.md`](docs/HOWTO.md) for the recipe-style operator
guide covering install, corpus initialisation, the five-stage build
pipeline, MCP registration per agent host, and stop / resume
semantics. The short version:

```bash
pipx install git+https://github.com/evoclock/nuthatch.git@main
nuthatch init ~/my-corpus --register-as my-corpus --set-default

# Drop sources anywhere under the corpus root (inbox/, arxiv/,
# bioarxiv/, notes/, root-level: your choice). Then trigger each
# pipeline stage in sequence and inspect between stages.

scripts/ops/launch-stage.sh ingest  my-corpus
scripts/ops/launch-stage.sh embed   my-corpus
scripts/ops/launch-stage.sh graph   my-corpus
scripts/ops/launch-stage.sh cluster my-corpus
nuthatch render --corpus my-corpus

# Then serve it to an MCP-aware agent:
nuthatch serve --corpus my-corpus
```

## Documentation

- [`docs/Design_Decisions.md`](docs/Design_Decisions.md): the
  architectural choices and the reasoning behind each, including
  the peer-landscape survey and the token-economy methodology
- [`docs/HOWTO.md`](docs/HOWTO.md): operator runbook for the
  five-stage pipeline plus the MCP query surface
- [`docs/SPEC.md`](docs/SPEC.md): the architecture spine
- [`docs/extraction-benchmarks/ocr-comparison.md`](docs/extraction-benchmarks/ocr-comparison.md):
  evidence for the OCR routing decisions (Chandra vs Docling vs
  EasyOCR vs Granite vs SmolDocling across three representative
  scanned papers)
- Per-agent skill files under `agent_skills/` (`skill-claude-code.md`,
  `skill-codex.md`, `skill-aider.md`, `skill-opencode.md`,
  `skill-pi.md`, `skill-hermes.md`) for MCP registration details and
  recommended multi-step query flows

## Licence

Apache 2.0. See [`LICENSE`](LICENSE).

## Acknowledgements

Designed by Julen Gamboa, who drove orchestration, design
discussion, and implementation decisions, with Claude Code
and Hermes (using GPT-5.5 and Minimax M2.5) as planning
collaborators. Claude Code executed much of the implementation
tasks under that direction.

Specs and decisions are pinned in `docs/SPEC.md` and
`docs/Design_Decisions.md` so every implementation points back
to a spec entry.
