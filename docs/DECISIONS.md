# nuthatch: locked-in decisions

Choices that should not be relitigated without specific new evidence.
Decisions worth changing get a new entry below the originals with the
reason for the change; the old entry stays for the audit trail.

## Identity

- **Name**: nuthatch (working name as of 2026-05-24).
- **Licence**: Apache 2.0 (enterprise-friendly; AGPL excluded
  because some enterprise procurement bans it outright).
- **External-library alias**: `kestrel` is the coded alias used
  throughout nuthatch's docs and code to refer to an MIT-licensed
  external pipeline library nuthatch borrows architectural patterns
  from. Real upstream identity is captured outside the public
  surface; the alias keeps nuthatch's references neutral. Reuse
  policy: pattern-only, never literal copy; credit `kestrel` in
  docstrings.

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
  - **Digital-born papers** (modern arXiv / bioRxiv / LaTeX-source
    PDFs): pdfminer text yield > 200 chars/page → Docling without
    OCR. Reads embedded text + structure directly, including math
    typeset as text. No GPU, fast (<1 s/page), full math preserved.
    The common case for typical modern preprint corpora.
  - **Scanned papers with math content** (equations, sub/superscripts,
    formal scientific notation): **Chandra-OCR-2**. The math-recall
    metric shows Chandra produces 42× more real equations than
    Granite-Docling on Wright 1931 (513 vs 12). Docling-VLM
    backends emit mostly broken LaTeX fragments (`\_{s}`, `^{6}`);
    EasyOCR emits no LaTeX. There is no middle option that closes
    this gap. Chandra costs ~30 s/page on the test hardware; a
    100-page paper takes ~55 minutes. Pay the time when math
    matters.
  - **Scanned papers with no math** (plain-prose archival material,
    historical letters, body-text-only old books): **Docling +
    EasyOCR**. ~14× faster than Chandra, ties on key-facts on
    plain-prose papers, no GPU. Skip Chandra here; the time cost
    buys nothing.
  - **Commercial-publisher PDFs with image-embedded math** (Nature,
    Cell, Science occasionally embed equations as PNG/SVG in
    otherwise-digital PDFs): the router currently sends these to
    Docling-without-OCR because text yield passes the threshold,
    but the image-only equations are silently lost. Detect-and-
    escalate to Chandra for these cases is a Sprint 4+ refinement.
  - **Never default**: SmolDocling 256M preview is unrecommended
    (hallucinated GLYPH tokens, broken LaTeX, inconsistent quality
    across paper types).
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
- **Chunk size**: 3000 chars target, 400 chars overlap. Larger
  than typical to preserve more context per chunk for retrieval
  (matches PhD KB's ~512-word chunks at ~6 chars/word). Hybrid
  chunker prefers structural boundaries (headings, paragraphs,
  sentence ends) when they fit within the window.
- **Embedding model default**: `BAAI/bge-m3` (multilingual, ~2 GB,
  matches the semantic-dedup default). Alternative pinned for
  English-paper corpora: `allenai/specter2_base` (PhD KB default;
  paper-specific embeddings tuned for scientific abstracts/titles).
  Users override via the corpus config `embedding.model` field.
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

## Token-economy methodology (Sprint 7)

The token-economy report measures how many tokens the consuming
agent saved by using nuthatch instead of a non-nuthatch fallback.
The choice of fallback is the entire methodology; everything else
is bookkeeping. nuthatch uses a **per-tool, per-query** fallback,
not a single full-corpus number.

### Why per-tool

A static "full corpus" counterfactual overstates savings by
orders of magnitude because nobody pastes an entire corpus into
context per query. The honest comparison is what a non-nuthatch
agent would actually do for each tool call.

| Tool | Counterfactual |
| --- | --- |
| `corpus_search(query, k)` | BM25 top-`k` text over the same chunks Chroma indexes |
| `subgraph_extract(seeds, depth)` | Sum of card-markdown tokens for every `doc::` node in the returned subgraph |
| `community_get(community_id)` | Sum of card-markdown tokens for every member of the community |
| `card_get(doc_id)` | Skipped entirely. Pure delivery, no honest reduction to claim |

Implementation: `src/nuthatch/token_econ/counterfactual.py`. The
MCP server logs `None` returns as "skip" rather than as `1.0×`
ratios; otherwise `card_get` traffic would dilute the per-tool
report.

### Calling out kestrel's strawman

The reference library nuthatch borrowed pipeline patterns from
(coded alias `kestrel`) markets a "71.5× fewer tokens per query
vs reading the raw files directly" headline in
`docs/how-it-works.md`. The methodology behind this claim is:

- Counterfactual = `nodes × 50 words × 1.33` tokens (a synthetic
  estimate of "the whole corpus as raw text"). At a 1000-node
  graph this puts the baseline at ~67k tokens regardless of
  query.
- Served = the BFS subgraph text after keyword-substring matching
  three seed nodes.
- Ratio = corpus / served, char/4 throughout.

Source: `graphify/benchmark.py` in
`/home/jgamboa/kestrel-latest-audit/` (HEAD `3efae38` 2026-05-25,
byte-identical to the v8 snapshot). Their own `how-it-works.md`
admits the comparison breaks at small corpora ("Six files already
fits in a context window") but the headline number is computed
this way.

nuthatch rejects this framing. Reported ratios under the per-tool
scheme typically land between ~2× and ~8× depending on corpus
size and query specificity. Smaller numbers, defensible numbers.

### What counts as a fair claim

Any token-economy figure shipped in nuthatch docs, READMEs, or
the dashboard MUST cite:

1. Which tool's counterfactual was used.
2. The corpus size + the query (or the sample query set).
3. The reduction ratio computed from per-tool baselines, never
   from a corpus-wide estimate.

If a future contributor proposes a "headline X× reduction"
number, they reread this section first.

## Execution model: build out-of-band, query via MCP

nuthatch separates **building the corpus** from **querying it**.

- **Build** happens out-of-band via the CLI:
  `nuthatch ingest` -> `nuthatch embed` -> `nuthatch cluster` ->
  `nuthatch render`. Each stage is incremental (hash-based file
  dedup for ingest, per-doc Chroma presence check for embed, full
  SBM refit each time for cluster which is correct, byte-stable
  re-render for render). Triggered by the user (or a cron / watcher),
  runs to completion, persists artifacts under `<corpus>/.kg/` and
  `<corpus>/{cards,communities}/`.
- **Query** happens at agent runtime via the MCP server. The agent
  calls the 5 tools (`corpus_search`, `subgraph_extract`,
  `card_get`, `community_get`, `token_econ_report`) against
  already-built state. No graph mutation; no extraction; no
  embedding; no clustering at query time.

**Why:** moving heavy work out of the agent loop has five concrete
benefits.

1. **Honest latency.** Agent tool calls return in milliseconds to
   single-digit seconds (vector lookup, BFS, file read). No tool
   call hides a 5-10 minute embedding job.
2. **Predictable cost.** Build cost is fixed and amortised; query
   cost is bounded by the chunks/cards returned. The token-economy
   report compares served vs counterfactual at query time without
   the noise of "and we also ran OCR".
3. **One source of truth.** The graph + embeddings + cards on disk
   are the only state. Two agents querying the same corpus see the
   same answer. No per-session rebuild divergence.
4. **Long-running stages are user-supervised, not hidden.** When
   `nuthatch ingest` is OCR'ing 100 scanned PDFs (potentially
   hours of GPU time on Chandra), the user sees it in a tmux log
   and can pause / kill / restart. Kestrel's model hides the same
   work inside an agent invocation; if the agent session times out
   or the user disconnects, the work is lost and they have no
   feedback channel to recover. nuthatch's CLI runs in the user's
   terminal, exits with a status code, writes per-run audit logs
   to `<corpus>/.kg/audit/`.
5. **Quality issues are surfaced and actionable, not silently
   absorbed.** When extraction yield drops below the QC floor,
   `nuthatch ingest` QUARANTINES the file with a reason-tagged
   subdir and a `.reason.json` sidecar; the user inspects, fixes
   the source PDF (or marks it for manual transcription), re-runs.
   Schema validation failures log missing required fields. The
   clustering router downgrades from principled SBM to heuristic
   Leiden when graph-tool isn't available and surfaces that in the
   response's `rigor_used` field. In each case the user knows
   exactly what fell short and what their options are. Kestrel's
   per-invocation pipeline decides "good enough" silently and
   moves on; broken extractions only surface at query time as
   missing or wrong answers, by which point the audit trail of
   what failed is gone.
6. **Server-side orchestration is harness-proof.** Skill-driven
   orchestration (kestrel's `skill-*.md` model) tells the agent
   "MUST use the Agent tool", "spawn 22-file chunks in parallel",
   "use this 45s timing estimate". None of that is enforceable;
   the harness (Claude Code, Codex, OpenCode, Aider) can demote
   the directive when context is tight, substitute serial
   file-reading when an Agent tool call times out, reorder steps
   per its own planning model, or silently fall back. Documented
   in the harness vendors' own GitHub issue trackers. The skill
   author has no enforcement channel.

   nuthatch's model puts the orchestration inside the MCP server
   (Python code in this repo). The agent calls a tool
   (`corpus_search`, `subgraph_extract`, etc.), the server does
   the right thing internally — parallel work, batching, caching,
   deduplication — and the agent CANNOT deviate because the agent
   isn't doing the orchestration. Same reason a well-designed REST
   API doesn't ship a runbook telling the client "you MUST batch
   requests": the server handles batching internally, the client
   just calls the endpoint. Skill-driven orchestration is the
   anti-pattern; server-side orchestration is the contract.

**What this means for the skill files** (`skill-*.md`): they are
operator runbooks for the MCP query surface, not orchestration
runbooks for the build pipeline. They tell the agent (a) how to
register the MCP server, (b) how to use the 5 tools effectively
in common multi-step flows, and (c) when to ask the user to run
the build pipeline (e.g. after dropping new files into `inputs/`).
They do NOT spawn subagents to extract entities, run OCR, or
fit clusters. That is the CLI's job.

**Comparison point:** kestrel's `skill-*.md` files are all
orchestration runbooks (10 of 11 are 1,228-1,434 lines; average
~1,280): the agent uses `@agent` parallel dispatch (or the
host's equivalent) to spawn semantic-extraction subagents on
each invocation, then drives clustering / analysis / rendering
inline. That model fits a "build per query session" shape and
the per-tool skill file is the runbook the agent follows.

nuthatch's "build once, query many" shape means our skill files
are an order of magnitude shorter without that being a deficit:
the build runbook is the CLI itself, called by the operator
out-of-band. The skill files only document MCP registration +
how to use the 5 query tools effectively.

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
