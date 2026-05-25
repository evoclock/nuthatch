# nuthatch: locked-in decisions

Choices that should not be relitigated without specific new evidence.
Decisions worth changing get a new entry below the originals with the
reason for the change; the old entry stays for the audit trail.

## Identity

- **Name**: nuthatch (working name as of 2026-05-24).
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
- **Supported platforms**: macOS (Apple Silicon), Linux (any),
  Windows (only with a CUDA GPU). Windows CPU-only is unsupported:
  the Python + ML stack on Windows without GPU acceleration is too
  slow for the local-inference path and Windows packaging fragility
  is not worth the maintenance cost.
- **LLM boundary**. nuthatch never calls LLMs internally. Any
  LLM work happens at the consuming surface: any process speaking
  MCP consumes nuthatch's tools. Claude Code Opus 4.7, Codex,
  Hermes, a user's own local model on a 128 GB workstation are
  all equivalent from nuthatch's perspective. No special
  integration required or planned. Semantic dedup uses embedding
  models, not chat LLMs (see Ingest).

## Clustering

- **SBM is the principled default**: Tiago Peixoto's nested
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
  PDF revisions, OCR rebuild of same paper). Embedding-model
  cosine similarity via `sentence-transformers`; no chat LLM
  involved.
  - *Default embedding model*: `BAAI/bge-m3` (~2 GB, multilingual,
    state-of-the-art on retrieval benchmarks). Picked as default
    because paper corpora may have any non-English content and the
    multilingual coverage is free.
  - *Configurable*: user overrides `dedup.embedding_model` to any
    sentence-transformers model. Alternatives worth knowing:
    `sentence-transformers/all-MiniLM-L6-v2` (~22 MB, CPU,
    English-leaning) for the lightest path;
    `intfloat/e5-mistral-7b-instruct` (~7 GB Q4, GPU) for
    English-only maximum-accuracy paths.
  - *Reranker* (enabled by default): `dedup.reranker` =
    `BAAI/bge-reranker-v2-m3` rescores borderline pairs the
    bi-encoder cosine flags in a "maybe" band. Cross-encoder,
    multilingual, matches BGE-M3 language coverage. Adds latency
    only on borderline pairs (the rest pass / fail on cosine
    alone); accuracy gain on dedup is worth the cost.
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

## Extraction (Sprint 2)

- **Multi-backend orchestrator.** nuthatch routes between Docling
  (for digital-text PDFs and most scanned papers via EasyOCR) and
  Chandra-OCR-2 (for difficult scanned source material). Granite-
  Docling 258M VLM is a GPU-accelerated middle option. SmolDocling
  256M preview is not recommended (hallucinated `GLYPH` tokens,
  inconsistent quality).
- **Routing criteria** (live in `src/nuthatch/ingest/extract.py`):
  1. Detect text yield via `pdfminer.six`. Threshold:
     `_SCANNED_TEXT_YIELD_PER_PAGE = 200.0` chars/page.
  2. Above threshold: Docling without OCR (digital path).
  3. Below threshold: route by user preference and host
     capabilities. Default with GPU + `prefer_max_quality=True`:
     Chandra-OCR-2. Default with GPU + `prefer_max_quality=False`:
     Granite-Docling VLM. No GPU: Docling + EasyOCR.
  4. Explicit override available via per-file `.kg/overrides.yaml`.
- **Routing recommendation** (evidence in
  `docs/extraction-benchmarks/ocr-comparison.md`):
  - **Default for scientific papers**: **Chandra-OCR-2**. Most
    research papers contain at least some math, scientific notation,
    sub/superscripts, or formal tabular data. The math-recall
    metric shows Chandra emits 10-180× more inline math than the
    Docling backends on equation-heavy material; on body-text-only
    papers it still produces the cleanest, most human-readable
    output with the richest sidecar bundle. Slow (~30 s/page) but
    the right choice when accuracy matters.
  - **GPU available, throughput matters, content has math**: fall
    back to **Granite-Docling 258M**. About 4× faster than Chandra
    with usable math preservation (10× less than Chandra but still
    real). The smaller-model option when paying Chandra's wall-clock
    cost is not viable.
  - **No GPU, content is plain prose** (no equations, no scientific
    notation, no tables that matter): **Docling + EasyOCR**.
    ~14× faster than Chandra; ties on key-facts on plain papers.
    Skip for anything with equations: EasyOCR emits zero math
    symbols and zero block math on equation-heavy material.
  - **Never default**: SmolDocling 256M preview is unrecommended
    (hallucinated GLYPH tokens, inconsistent quality across paper
    types).
- **Backend licensing posture.** nuthatch is a routing layer that
  calls user-installed OCR backends. nuthatch does not redistribute
  model weights and does not run inference as a service. Backend
  licences attach to the end user's runtime:
  - Chandra-OCR-2: OpenRAIL (responsible-AI; permits commercial +
    OSS use, prohibits surveillance / weapons / illegal use). End
    user accepts these terms by installing and running Chandra.
  - Docling + EasyOCR / Granite-Docling / SmolDocling: permissive
    (MIT / Apache / similar). End user accepts each by installing.
  Users install backends themselves (`sfw uv add chandra-ocr[hf]`,
  etc.) and choose their stack. nuthatch stays Apache 2.0
  regardless of backend choice.
- **Monetisation impact.** None. All backends stay in the OSS path.
  Paid tier remains at the graph-tool / SBM refit step.

## Chunking + embedding (Sprint 3)

- **VectorStore protocol** abstracts the embedding store. ChromaDB
  is the default (embedded, SQLite-backed, batteries-included
  with sentence-transformers, local-first). FAISS and Pinecone
  drop in behind the same protocol when they make sense: FAISS
  for single-corpus scale beyond ChromaDB's practical limits
  (~millions of vectors, strict latency); Pinecone for the
  future hosted-SaaS multi-tenant path. No FAISS or Pinecone
  install in the OSS default.
- **Full-document coverage is mandatory**, NOT partial / sampled /
  summary-only. Every byte of text the extractor produces lands in
  at least one chunk. This applies to: abstract, full body
  (every section), references, appendices, footnotes, figure +
  table captions, supplementary text when present.
- Chunks may overlap (and should, per hybrid-chunking best practice
  for retrieval recall), but the *union* of all chunks must cover
  the document. A coverage check at the end of each ingest pass
  asserts this invariant; missing-coverage events go to
  `audit/coverage_misses.jsonl` and the offending file is flagged.
- Rationale: a knowledge graph that surfaces "the relevant section"
  to an LLM is only as honest as the chunks. Partial coverage means
  a query could miss the one paragraph where the paper's actual
  contribution is stated. The cost (more chunks, larger vector
  store) is much lower than the cost (silently wrong answers from
  the LLM) of partial coverage.

## Surfacing

- **MCP stdio server** for agents (`nuthatch serve --corpus <name>`
  scopes the agent to one corpus). Raw STDIO JSON-RPC handler,
  no `mcp` SDK dependency; pattern from Hillstar Orchestrator's
  `mcp-server/minimax_server.py`.
- **Any MCP-speaking process is a first-class consumer.** Claude
  Code, Codex, Hermes, an Obsidian plugin, the future Vogelkop UI,
  and a user-run local LLM script all interact through the same
  MCP tools. No special integration path for any of them.
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

- `docs/SPEC.md`: architecture spine.
- `~/project-planning-agent/strands/nuthatch.md`: planning history
  with full design discussion.
