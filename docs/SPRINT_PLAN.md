# nuthatch: prospective sprint plan

Sprint plan from planning-stage scaffold (`v0.0.1`) to first public
release candidate (`v0.1.0`). Each sprint is sized roughly 1–2 weeks
of focused work and delivers something vertically usable.

The plan deliberately leans on existing code from two sources:

- **PhD KB** (`/home/jgamboa/PhD-knowledge-base/`): production-tested
  patterns for an in-house knowledge-base / RAG store.
- **Graphify** (pip-installed at
  `/home/jgamboa/.pyenv/versions/3.11.8/lib/python3.11/site-packages/graphify/`,
  source at <https://github.com/safishamsi/graphify>): patterns
  only; **not** a fork. License is MIT so the patterns can be
  re-implemented cleanly.

What's reused vs. what's new is called out per sprint so the
agent loop can be held accountable to the inventory.

## Reusable inventory at a glance

### From PhD KB

| Module | Reused as | Notes |
| --- | --- | --- |
| `scripts/00_extract_text.py` | basis for `nuthatch/ingest/extract.py` | Pdfminer / Docling adapter pattern; needs Chandra-OCR adapter added |
| `scripts/01_extract_metadata.py` | basis for `nuthatch/ingest/metadata.py` | Schema validation step |
| `scripts/02_assemble_wiki_page.py` | basis for `nuthatch/render/card.py` | MD-card-per-paper rendering |
| `scripts/embed.py` | basis for `nuthatch/embed/embed.py` | ChromaDB + sentence-transformers |
| `src/retrieve.py` | basis for `nuthatch/retrieve/vector.py` | Vector-similarity retrieval for the bridge mode between SBM refits |
| `src/ask.py` | basis for `nuthatch/llm/local.py` | Ollama-driven local synthesis (Jan-Code-4B / Devstral-small-2) |
| `workflows/incremental_embed.json` | basis for the incremental ingest path | Hillstar workflow shape |
| `workflows/ocr_reextract.json` | basis for re-extraction on Docling/Chandra version bump | Hillstar workflow shape |
| `scripts/validate_step01.py` + `verify_coverage*.py` | basis for `nuthatch/ingest/qc.py` | Coverage-verification per stage |
| Frontmatter schema from `~/project-planning-agent/conventions/kb-reports.md` | basis for `nuthatch/schema/*.py` profiles | Same metadata-contract pattern, applied per `SchemaProfile` |
| Decay formula from `kb-reports.md` | basis for `nuthatch/graph/decay.py` | `relevance(t) = max(backlinks, 1) · exp(-ln2 · Δt / half_life)` |

### Patterns to lift from Graphify (re-implement, not fork)

| Pattern | Where it shows up in nuthatch | Notes |
| --- | --- | --- |
| Pipeline shape (`detect → extract → build_graph → cluster → analyze → report → export`) | `nuthatch/ingest/pipeline.py` | Same coarse stages; each is its own module, communicating through plain dicts / NetworkX graphs |
| `manifest.py` (authoritative ingest record) | `nuthatch/ingest/manifest.py` | One line per `(file, extractor_version, timestamp)` in `.kg/manifest.jsonl` |
| `cache.py` (semantic cache) | `nuthatch/ingest/cache.py` | Skip re-extraction of unchanged files |
| `watch.py` (filesystem-event triggered ingest) | `nuthatch/ingest/watch.py` | Reads `inbox/`, writes a flag file |
| `security.py` (URL validation, SSRF protection) | `nuthatch/ingest/security.py` | If we ever accept URL-based ingest |
| `cluster.py:remap_communities_to_previous` (stable community IDs across refits) | `nuthatch/graph/cluster.py` | Critical for the bridge-mode UX between SBM refits |
| Hub exclusion (`cluster.py:exclude_hubs_percentile`) | `nuthatch/graph/cluster.py` | Essential for paper KGs where one cited-everywhere paper dominates |
| Edge confidence labels (`EXTRACTED \| INFERRED \| AMBIGUOUS`) | `nuthatch/graph/edges.py` | On every edge |
| `serve.py` (MCP stdio server) | `nuthatch/mcp/server.py` | Standalone MCP surface scoped to one corpus |
| `benchmark.py` (corpus-vs-subgraph token comparison) | `nuthatch/token_econ/benchmark.py` + dashboard | Pattern is right; we surface it live rather than as a one-shot report |
| `hooks.py` (pre/post pipeline hooks) | `nuthatch/ingest/hooks.py` | User-extensibility for metadata enrichers |
| `AGENTS.md` convention | `nuthatch/AGENTS.md` | Documentation for AI agents using the MCP surface |

### Explicitly NOT reused

- Tree-sitter language extractors (Graphify's `extract.py` is 329KB
  of these; not transferable to a paper-KG context).
- SCIP ingestion (code-focused).
- Call-flow HTML (code-focused).
- Symbol resolution (code-focused).
- PR / GitHub-integration (code-focused).

## Sprint 0: Foundations (DONE: in initial scaffold)

**Goal**: planning-stage skeleton + first concrete spec artifact.

**Deliverables (already shipped in `v0.0.1`)**

- README, LICENSE (Apache-2.0), pyproject.toml.
- `src/nuthatch/clustering.py`: the `ClusteringBackend` protocol
  with `Rigor` and `BackendLocation` enums.
- `tests/test_clustering.py`: shape tests pinning the protocol.
- `docs/SPEC.md`, `docs/DECISIONS.md`.
- `corpus/{arxiv,bioarxiv}/`: 147 seed PDFs (gitignored).
- SPDX Apache-2.0 headers on all Python files.
- CI-ready (uv + ruff + mypy strict + pytest infrastructure pinned
  in pyproject.toml; no `.github/workflows/` yet).

**Status**: complete.

## Sprint 1: Corpus state machine

**Goal**: an unattended ingest pipeline that watches `inbox/`,
hashes + dedupes incoming files, writes a manifest, and routes
unsupported / corrupt files to `quarantine/` with a reason.

No extraction yet; this sprint validates the spine end-to-end with
trivial "extracted" payloads (e.g., file size + mtime).

**Deliverables**

- `src/nuthatch/ingest/watch.py`: filesystem watcher
  (`watchdog` or polling fallback). Pattern from Graphify's
  `watch.py`. Emits events to an in-process queue.
- `src/nuthatch/ingest/dedup.py`: SHA256 hashing + lookup against
  the manifest. Pattern from Graphify's `dedup.py`.
- `src/nuthatch/ingest/manifest.py`: `.kg/manifest.jsonl` writer +
  reader. One line per ingest event: `{path, hash, status,
  reason, extractor_version, timestamp_utc}`. Pattern from
  Graphify's `manifest.py`.
- `src/nuthatch/ingest/state_machine.py`: orchestrator that walks
  files through `inbox → identify → hash dedup → (placeholder
  extract) → route(pass/fail) → manifest log`.
- `src/nuthatch/corpus/registry.py`: `~/.config/nuthatch/registry.toml`
  reader + writer.
- `src/nuthatch/corpus/layout.py`: discovers `.kg/` markers,
  creates the standard subdirs (`inbox/`, `quarantine/`, `papers/`,
  `cards/`, etc.) on `nuthatch init`.
- CLI subcommand: `nuthatch init <path>`, `nuthatch ingest`,
  `nuthatch status`. Replaces the planning-stage `__main__.py`
  placeholder.

**Definition of done**

- Drop a PDF into `corpus/arxiv/inbox/`, run `nuthatch ingest`,
  observe it move to `papers/` (with placeholder metadata in
  `manifest.jsonl`) or `quarantine/`.
- Re-ingesting the same file is a no-op (hash dedup).
- Tests cover: empty inbox, single file, duplicate file, malformed
  file, manifest round-trip.

**Reused / new split**

- Reuse: Graphify watch/dedup/manifest patterns; PhD KB workflow
  shape (`workflows/incremental_embed.json`).
- New: registry, corpus layout, state machine, CLI surface.

## Sprint 2: Extract + schema validate

**Goal**: replace the placeholder extraction with real OCR-backend
routing; gate ingest on metadata schema; add embedding-based
semantic dedup with reranker.

**Deliverables**

- [DONE] `src/nuthatch/ingest/extract.py`: routing extractor.
  Detects scanned vs digital PDFs via pdfminer text-yield
  (threshold 200 chars/page) and dispatches to one of four
  backends: Docling (digital), Chandra-OCR-2 (scanned + high
  quality preferred), Docling+Granite-Docling VLM (scanned + GPU
  + smaller model preferred), Docling+EasyOCR (scanned + no GPU).
  Empirical benchmark in
  `docs/extraction-benchmarks/ocr-comparison.md`.
- [DONE] `src/nuthatch/dedup/semantic.py`: sentence-transformers
  `BAAI/bge-m3` for bi-encoder similarity; `BAAI/bge-reranker-v2-m3`
  cross-encoder for borderline pairs. Both model IDs configurable
  via `dedup.embedding_model` and `dedup.reranker` in corpus
  config. No chat LLM in the dedup path.
- [DONE] `src/nuthatch/schema/profile.py` + `schema/profiles/*.py`:
  `SchemaProfile` base + four built-in profiles
  (`arxiv_paper`, `biorxiv_paper`, `patent`, `internal_doc`).
  Pattern from `~/project-planning-agent/conventions/kb-reports.md`
  frontmatter.
- [DONE] `src/nuthatch/ingest/metadata.py`: heuristic metadata
  lift (title from H1, year, arXiv id, DOI, patent number) plus
  `extract_and_validate(markdown, profile)` for the gate.
- [DONE] `src/nuthatch/ingest/quarantine.py`: moves schema-failed
  files to `quarantine/<reason-slug>/` with a sidecar
  `.reason.json` capturing reason + details (UTC timestamp, stage,
  original filename, extracted metadata).
- [DONE] `src/nuthatch/ingest/qc.py`: `check_extract_yield`
  asserts the extractor produced enough text (absolute floor +
  per-page floor) before the schema gate runs.
- [DONE] `IngestOrchestrator` wired through extract + qc +
  metadata + schema + quarantine. Dependency-injection for
  `extractor` and `schema_profile` so tests substitute fast mocks.
- [DONE] `scripts/bench/math_recall.py`: math-notation metric
  (inline math, block math, math symbols, per-paper equation
  checklist). Shows Chandra emits 10-180× more inline math than
  Docling backends on equation-heavy material; pinned in
  DECISIONS.md routing.

**Definition of done**

- Drop a real arxiv PDF into inbox, see it land in `papers/` with
  a fully-populated metadata sidecar.
- Drop a low-quality scanned PDF, see it route to Chandra-OCR-2
  fallback (or to quarantine with the missing-field reason).
- Schema profile is hot-swappable: switching corpus from `arxiv_paper`
  to `internal_doc` changes which fields are required.
- Drop a near-duplicate revision of an already-ingested paper, see
  the semantic-dedup pipeline catch it: bi-encoder cosine flags
  the candidate, reranker confirms, manifest records both scores
  and the dedup decision.

**Reused / new split**

- Reuse: PhD KB's three-stage extract pattern; Hillstar workflow
  `workflows/ocr_reextract.json` shape.
- New: `SchemaProfile` abstraction, the four built-in profiles, the
  quarantine path with sidecar reasons, the bi-encoder + reranker
  semantic-dedup pipeline.

## Sprint 3: Chunk + embed + per-paper artifacts

**Goal**: each ingested paper produces a queryable MD card + an HTML
companion + chunks in a vector store.

**Deliverables**

- `src/nuthatch/embed/embed.py`: ChromaDB + sentence-transformers
  (BGE-large default). Pattern from PhD KB's `embed.py`.
- `src/nuthatch/embed/chunk.py`: hybrid chunking (semantic +
  structure-aware: section / paragraph boundaries from Docling).
  **Full-document coverage is mandatory** per `docs/DECISIONS.md`:
  every byte of extracted text lands in at least one chunk
  (abstract, body, references, appendices, footnotes, captions);
  chunks may overlap but the *union* must cover the document. A
  coverage check at the end of each ingest pass asserts this and
  flags missing coverage to `.kg/audit/coverage_misses.jsonl`.
- `src/nuthatch/render/card.py`: MD card with the canonical
  frontmatter from the schema profile + a short summary section.
  Pattern from PhD KB's `02_assemble_wiki_page.py`.
- `src/nuthatch/render/html.py`: HTML companion: rich figures +
  equations + reference list. Uses Docling's structured output.
- `src/nuthatch/retrieve/vector.py`: vector-similarity query
  against the ChromaDB store. The "bridge mode" retrieval path
  that runs without a graph. Pattern from PhD KB's `retrieve.py`.

**Definition of done**

- Per ingested paper: one `cards/<id>.md` and one `html/<id>.html`.
- `nuthatch query "transformers in single-cell RNA"` returns the
  top-k chunks via ChromaDB without any graph involvement.
- Embedding store survives a restart (persisted to `.kg/chroma/`).

**Reused / new split**

- Reuse: PhD KB embed + retrieve scripts almost wholesale; PhD KB
  wiki-page assembly.
- New: HTML companion render, hybrid-chunking refinement, the
  cards/ + html/ split.

## Sprint 4: Graph integration

**Goal**: extracted entities + relations land in a NetworkX
graph that can be serialised + reloaded across runs.

**Deliverables**

- `src/nuthatch/graph/entities.py`: entity extraction from chunks
  (NER via sentence-transformers / spaCy; citation extraction from
  Docling's reference list).
- `src/nuthatch/graph/edges.py`: edge dataclasses + the
  `EXTRACTED | INFERRED | AMBIGUOUS` confidence label on every edge.
  Pattern from Graphify's edge schema.
- `src/nuthatch/graph/build.py`: `build_graph(corpus)` ->
  `nx.MultiDiGraph`. Pattern from Graphify's `build.py`.
- `src/nuthatch/graph/io.py`: graph (de)serialisation to
  `.kg/graph/graph.json`. Round-trippable.
- `src/nuthatch/graph/decay.py`: relevance decay formula from PhD
  KB's kb-reports schema.

**Definition of done**

- After ingest, `nuthatch graph build` produces a NetworkX graph
  with citation + entity-cooccurrence edges.
- Every edge carries a confidence label.
- Graph re-loads from `.kg/graph/graph.json` without semantic loss.
- Decay pass downweights nodes per the formula.

**Reused / new split**

- Reuse: Graphify edge-schema + build pattern; PhD KB decay formula.
- New: entity extraction layer (Graphify has tree-sitter for code;
  for papers we need NER + citation parser).

## Sprint 5: Clustering (the principled-default differentiator)

**Goal**: ship the three clustering backends behind the
`ClusteringBackend` protocol from Sprint 0. SBM via `graph-tool`
locally; Leiden via `graspologic` or `networkx`; vector-similarity
fallback when neither runs.

**Deliverables**

- `src/nuthatch/clustering/backends/sbm.py`: Tiago Peixoto's nested
  degree-corrected SBM via `graph-tool`. Conda-only; documented in
  README install section.
- `src/nuthatch/clustering/backends/leiden.py`: Leiden via
  `graspologic`. Fallback when SBM unavailable.
- `src/nuthatch/clustering/backends/embeddings.py`: vector-based
  cluster (k-means on embeddings) for the bridge mode.
- `src/nuthatch/clustering/router.py`: picks the highest-rigor
  available backend; surfaces downgrade-notes in the response.
- `src/nuthatch/clustering/hub_exclusion.py`: port of Graphify's
  `exclude_hubs_percentile` pattern. Essential.
- `src/nuthatch/clustering/stable_ids.py`: community ID remapping
  across refits via greedy overlap match. Pattern from Graphify's
  `remap_communities_to_previous`.

**Definition of done**

- On a corpus of 100+ papers, all three backends produce a
  partition; `clustering_rigor` field accurately reflects which
  ran.
- User-triggered refit (`nuthatch graph refit`) re-clusters and
  preserves stable community IDs where possible.
- Hub exclusion is on by default and parameterisable.

**Reused / new split**

- Reuse: Graphify's hub-exclusion + stable-ID patterns; the
  ClusteringBackend protocol from Sprint 0.
- New: the three concrete backend implementations + the router.

## Sprint 6: Surfaces (Obsidian + MCP)

**Goal**: two human / agent surfaces: an Obsidian vault that
renders the corpus + graph, and an MCP stdio server scoped to a
corpus for agent integration.

**Deliverables**

- `src/nuthatch/render/obsidian.py`: exports cards + community
  pages + graph view + Dataview queries. Pattern from PhD KB's
  Obsidian vault layout.
- `src/nuthatch/mcp/server.py`: MCP stdio server. Tools:
  `corpus_search`, `subgraph_extract`, `card_get`, `community_get`,
  `token_econ_report`. Raw STDIO JSON-RPC handler, no `mcp` SDK
  dependency; pattern from Hillstar Orchestrator's
  `mcp-server/minimax_server.py` and Graphify's `serve.py`.
- `nuthatch/AGENTS.md`: agent-facing docs for the MCP surface.
  Pattern from Graphify's AGENTS.md convention.
- CLI subcommands: `nuthatch obsidian export <corpus>`,
  `nuthatch serve --corpus <name>`.

**Definition of done**

- Generated Obsidian vault opens cleanly; graph view works;
  Dataview queries against the frontmatter return.
- An MCP-aware agent (Claude Code via stdio, or `mcp-inspector`)
  can call all five tools and get sensible responses.

**Reused / new split**

- Reuse: PhD KB Obsidian vault patterns; Graphify MCP server
  shape.
- New: the per-tool implementations + the AGENTS doc.

## Sprint 7: Token-economy instrumentation

**Goal**: every MCP query the surface makes surfaces actual vs
counterfactual tokens, so users see what the subgraph extraction is
saving. The user-facing differentiator.

nuthatch never calls an LLM itself; the consuming surface (Claude
Code / Codex / Hermes / Obsidian plugin / Vogelkop UI) calls its
own LLM with the subgraph nuthatch served. Instrumentation lives
at the MCP boundary, not inside an LLM wrapper.

**Deliverables**

- `src/nuthatch/token_econ/measure.py`: per-MCP-query measurement.
  Records `tokens_subgraph` (what the surface actually received),
  `tokens_full_corpus` (counterfactual), `tool`, `timestamp`,
  `surface_id`. Uses `tiktoken` cl100k as the universal tokenizer
  approximation (documented as such; per-model precision is a later
  extension).
- `.kg/token_log.jsonl`: append-only per-query log.
- `src/nuthatch/token_econ/report.py`: aggregates the log into
  summary stats (per-day, per-tool, per-corpus). Exports markdown
  to `<corpus>/reports/token-economy-<date>.md` for Obsidian
  Dataview rendering. JSON-shaped for programmatic consumers.
- MCP tool `token_econ_report(time_range, group_by)` registered in
  the Sprint 6 server; returns the same aggregates as JSON.
- CLI: `nuthatch token-report [--since DATE] [--out PATH]`.

**Definition of done**

- After a session of MCP queries, `nuthatch token-report` shows
  a real reduction: "1,240 tokens served vs 84,200 full-corpus
  tokens over 18 queries; 98.5% saved."
- Obsidian renders the markdown report with a Dataview table.
- The MCP `token_econ_report` tool returns the same data as JSON.

**Reused / new split**

- Reuse: Graphify's `benchmark.py` measurement pattern (rewritten
  per-query, not one-shot).
- New: the JSONL log + aggregation layer + markdown export + MCP
  tool surface.

## Sprint 8: Decay + supersession

**Goal**: the outbound flow from SPEC.md: relevance decays over
time; user-marked supersedes downweight predecessors.

**Deliverables**

- `src/nuthatch/graph/decay_pass.py`: periodic pass recomputes
  relevance; flags low-score / zero-backlink nodes for archive.
- `src/nuthatch/graph/supersede.py`: `nuthatch supersede <old>
  --by <new>` adds the supersedes edge + downweights predecessor.
- CLI: `nuthatch decay run`, `nuthatch supersede`.

**Definition of done**

- Run the decay pass on the seed corpus; verify cards older than
  the half-life get downweighted but not deleted.
- Supersede flow: `nuthatch supersede <id-old> --by <id-new>`
  produces a visible edge in the graph + an entry in the manifest.

**Reused / new split**

- Reuse: PhD KB decay formula (already ported in Sprint 4).
- New: the decay pass orchestrator + the supersede command.

## Sprint 9: Polish + first release

**Goal**: ship `v0.1.0` to the public repo with screenshots, docs,
and CI green on Python 3.11/3.12.

**Deliverables**

- `.github/workflows/{ci,release}.yml` (mirroring thermall).
- README updated with full install + usage + screenshots.
- CHANGELOG.md with v0.1.0 surface.
- SQUASH_PLAN.md (mirroring thermall) for the public mirror.
- Squash + mirror to `evoclock/nuthatch` (public).
- GitHub Release v0.1.0 with the CHANGELOG body.

**Definition of done**

- Anyone can `pipx install` from the public repo with
  Python 3.11+ and conda-installed graph-tool, drop papers into an
  inbox, and see them ingested + clustered + queryable.

## Out of scope for v0.1.x (deferred to v0.2+)

These are real backlog items, NOT v0.1 cuts:

- Cross-corpus queries (corpus-scoped only in v0.1).
- Audio / video transcription (NeurIPS / ICML talks).
- Google Drive / cloud-folder ingest.
- Pretty UI for adding custom schema profiles (YAML-by-hand in v0.1).
- Hosted SaaS or managed-on-customer-cloud (Model A / C from the
  strand). Phase 1 is OSS-only.

## Critical path

`Sprint 0` (done) → `Sprint 1` (state machine; unblocks everything
else) → `Sprint 2` (extract; gates content into the rest of the
pipeline) → `Sprint 3` (chunk/embed/cards; gives the bridge mode +
the queryable surface).

After Sprint 3 there's parallelism:

- Sprint 4 (graph) + Sprint 5 (clustering) chain in series: both
  required for the differentiator story but neither blocks the
  user-facing card / embed surface from Sprint 3.
- Sprint 6 (surfaces) can start after Sprint 3 (Obsidian works on
  cards alone) and finish in parallel with Sprint 5 (MCP integration
  needs the graph, so the MCP half lands later).
- Sprint 7 (token-econ) requires Sprint 6's MCP surface for the
  full story but the wrapper + dashboard parts can land standalone.
- Sprint 8 (decay) needs Sprint 4's graph; otherwise standalone.

## Sprint sizing reality check

These are 1–2 week sprints in *focused* work time. Realistic
solo-with-agent-help calendar time: probably 4–6 weeks of
elapsed time per sprint when accounting for the spike work
(Sprint 2's Docling/Chandra adapters in particular will need a
real-PDF spike before scaffolding).

Total: ~6 calendar months to `v0.1.0` at a steady-but-not-frantic
cadence. Faster if more focused; slower if Sprint 2 surfaces a
Docling/Chandra quality problem that forces a different extractor
approach.

## References

- Spec: `docs/SPEC.md`.
- Decisions: `docs/DECISIONS.md`.
- Planning history: `~/project-planning-agent/strands/nuthatch.md`.
- PhD KB source patterns: `/home/jgamboa/PhD-knowledge-base/`.
- Graphify source: <https://github.com/safishamsi/graphify> (MIT;
  patterns only, not a fork).
- thermall as a precedent for the agent-assisted spec-driven
  development style: `<https://github.com/evoclock/thermall>`.
