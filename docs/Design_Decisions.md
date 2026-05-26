# Design Decisions

This document records the design choices nuthatch ships with and the
reasoning behind each. Settled choices stay here as a reference for
contributors and users. Where a decision deviates from a peer tool's
approach, the reasoning is given factually so readers can evaluate
the trade-off for themselves rather than take our word for it.

## Identity

- **Licence**: Apache 2.0.

## Architecture

### Sources are primitives; the corpus is read-only to the agent

A research corpus is only useful if every claim it surfaces can be
traced back to a source the user can re-read. The rule, adapted
from my own PhD knowledge-base practice:

> **Sources are immutable, derived state is regenerated from sources,
> agents are read-only consumers.** Agents query across the corpus,
> synthesise answers, and cite back. They do not rewrite the cards,
> the embeddings, the graph, or the source PDFs.

If an LLM (or a tool driven by an LLM) is allowed to rewrite the
knowledge base in place, every rewrite is a chance to drift from
the source. Over time the corpus becomes a hallucination the user
cannot audit because the source text is gone or has been silently
overwritten. That is a bad trade for any research tool where
every claim must be defensible.

nuthatch enforces this rule at four boundaries:

1. **Source PDFs in `papers/` are never modified after ingest.**
   The original bytes are preserved as the audit trail.
2. **Extracted markdown in `.kg/extracted/` is regenerated from
   sources, never edited by agents.** It is the canonical
   derivation that downstream stages consume.
3. **Cards in `cards/` are rendered from extracted markdown +
   cluster output**, not synthesised by an LLM at query time.
   Card content is verifiable by reading the corresponding source.
4. **The MCP query surface is read-only by design.** The five tools
   (`corpus_search`, `subgraph_extract`, `card_get`,
   `community_get`, `token_econ_report`) inspect state and return
   data; none mutate it. Ingest, embed, graph, cluster, and
   render are operator-triggered CLI commands; the agent cannot
   invoke them and cannot rewrite their outputs.

The rule survives every architectural choice in this document.
When the rule conflicts with a convenience (for example,
"wouldn't it be faster to let the agent regenerate the card after
edits?"), nuthatch chooses the rule.

### Per-corpus storage

- **Per-corpus directory tree.** Each corpus is a self-contained
  directory with a `.kg/` marker (analogous to `.git/`). Discovery
  walks up from the current working directory looking for `.kg/`,
  so `nuthatch <subcommand>` works without `--corpus` when invoked
  from inside one. A small registry at
  `~/.config/nuthatch/registry.toml` resolves named corpora to
  paths. There is no co-located multi-corpus root: each corpus
  moves, backs up, and shares independently.
- **Strict schema gate on ingest.** Files that cannot satisfy their
  active `SchemaProfile` are moved to `<corpus>/quarantine/` with
  a reason-tagged sidecar (`.reason.json`), NOT silently dropped
  into the graph. The user inspects, fixes the source (or marks it
  for manual transcription), and re-runs ingest. The pipeline
  prefers a smaller graph that is correct over a larger graph with
  hidden defects.
- **Schema profiles are user-extensible from day one.** Built-in
  profiles ship for `arxiv_paper`, `biorxiv_paper`, `patent`,
  `internal_doc`. Users define their own via YAML or a small
  Python plugin; the corpus's `.kg/config.yaml` names the active
  profile.
- **The LLM boundary.** nuthatch never calls a chat LLM internally.
  Embedding models are used for semantic dedup and corpus search;
  no chat-style inference. Any LLM consumption happens at the
  consuming agent (Claude Code, Codex, Aider, OpenCode, Pi,
  Hermes, an Obsidian plugin, the user's own local model). All of
  them interact through the same MCP tool surface — no special
  integration path per agent.
- **Supported platforms.** macOS (Apple Silicon), Linux (any),
  Windows-with-CUDA-GPU. Windows-without-GPU is unsupported: the
  Python ML stack on Windows without GPU acceleration is too slow
  for the local-inference path, and Windows packaging fragility
  is not worth the maintenance cost to me, but you are welcome to
  fork the repo and build the functionality.

## Clustering

- **Stochastic Block Model is the principled default.** nuthatch's
  highest-rigor clustering backend is Tiago Peixoto's nested
  degree-corrected SBM via `graph-tool`. The SBM is Bayesian
  (avoids the resolution-limit problem that traps modularity
  methods like Louvain and Leiden on graphs with mixed cluster
  scales) and gives partition uncertainty as a side effect of
  the model fit.
- **`graph-tool` is conda-only.** Peixoto distributes via
  conda-forge, his own Debian/Ubuntu apt repo, and Homebrew. No
  PyPI wheel exists because the C++ dep chain (boost-python3,
  expat, sparsehash, scipy headers) is unmaintainable as a
  cross-platform pip package. This is a known constraint of the
  library, not a nuthatch design choice. Users who can't or won't
  install conda get the heuristic tier instead.
- **Leiden + embeddings fallback.** When `graph-tool` is
  unavailable or the graph exceeds local SBM compute capacity,
  the clustering router downgrades to the Leiden algorithm
  (Traag, Waltman, van Eck 2019, via `leidenalg` + `python-igraph`),
  and below that to k-means over chunk embeddings. Every
  clustering response carries a `rigor_used` field stating the
  tier the partition actually came from; downgrades are surfaced
  to the user, never silently accepted.
- **Hub exclusion + reattach by majority neighbour.** Before
  clustering, high-degree super-hub nodes (the top percentile by
  degree, e.g. a 200,000-citation foundational paper) are
  excluded from the partition fit and reattached afterwards by
  majority-vote of their neighbours' community memberships. This
  is a well-known correction for graphs where a few very-popular
  nodes would otherwise distort the partition; nuthatch's
  implementation calls these nodes `core_nodes` (the term
  "high-degree node" with established meaning in graph theory,
  named here without anthropomorphisation).
- **Community-ID stability across re-fits.** A greedy set-overlap
  remap preserves community IDs across re-clusterings, so the
  Obsidian community pages and any saved queries keep referring
  to the same conceptual groupings even after new papers shift
  the partition at the boundaries.

## Ingestion

- **Hash-based file dedup at ingest.** Every source file is
  SHA-256 hashed; duplicates are recorded in the manifest and
  skipped from extraction. Files are addressed by hash in
  `<corpus>/.kg/manifest.jsonl` so re-running `nuthatch ingest`
  on the same corpus is byte-cheap.
- **Authoritative metadata from the publisher API, not the PDF
  body.** For arxiv (`2605.15308v1.pdf` filename shape) and
  bioRxiv (`2026.05.14.725010v1.full.pdf` shape), title, authors,
  abstract, year, and DOI come from the publisher's API
  (`export.arxiv.org/api/query` Atom XML; `api.biorxiv.org/details/biorxiv`
  JSON). Body-text heuristic extraction fills remaining fields
  (key claims, methods narrative, topics). The publisher record
  is the canonical source for paper metadata, and using it
  removes a large class of brittle author-line / title-heading
  regex failures that would otherwise quarantine papers with
  unconventional title pages. Fetches are cached locally under
  `<corpus>/.kg/metadata_cache/` so re-ingest is a free file
  read. Every URL is validated against the SSRF / loopback /
  metadata-IP / private-network gate before fetch, consistent
  with the rest of the ingest layer.
- **Semantic dedup for papers.** Preprint vs published, multiple
  PDF revisions, OCR rebuilds of the same paper are caught by
  cosine similarity over embedding vectors, with a three-band
  decision (near-duplicate / borderline / distinct). Default
  thresholds: 0.95 / 0.80 with a 0.85 reranker-promotion gate.
  Defaults are conservative starting points; corpora with unusual
  similarity distributions can override per-corpus.
- **BGE-M3 as the default embedding model.** Multilingual,
  ~2 GB, state-of-the-art on retrieval benchmarks. Chosen as the
  shared default for both semantic dedup and corpus search so
  cosine distances are directly comparable. Alternatives are
  documented for English-paper-only corpora (`allenai/specter2_base`)
  and lightweight CPU paths (`sentence-transformers/all-MiniLM-L6-v2`);
  users override via `embedding.model` in the corpus config.
- **BGE-reranker-v2-m3 as the default reranker.** Cross-encoder
  re-scoring of the borderline cosine band only (not every pair).
  Adds latency only where the cosine signal was uncertain;
  accuracy gain on dedup is worth the cost. Optional, defaults on.
- **Edge confidence labels.** Every edge in the corpus graph is
  tagged `EXTRACTED | INFERRED | AMBIGUOUS`. This is a long-
  established convention in knowledge-graph literature (NELL,
  YAGO, ConceptNet have variants from the 2000s and 2010s);
  nuthatch surfaces it consistently in the graph viewer and in
  MCP-served subgraphs so the consuming agent knows what is
  evidence vs what is inference.

## Extraction

nuthatch is a routing layer that calls user-installed OCR backends.
Backend licences attach to the end user's runtime; nuthatch does
not redistribute model weights or run inference as a service.

- **Routing criteria.** PDF text yield (chars per page, measured
  via `pdfminer.six`) is the primary signal. Above the threshold:
  Docling without OCR — fast, structure-aware, preserves
  text-typeset math. Below the threshold: scanned-text path,
  routed by host capabilities (GPU + max-quality preference →
  Chandra-OCR-2; GPU + smaller-model preference → Granite-Docling
  VLM; no GPU → Docling + EasyOCR).
- **`--skip-chandra` for fast first ingests.** Bypasses
  Chandra-OCR entirely; routes every PDF through Docling (with
  EasyOCR fallback for genuinely-scanned PDFs). Math-heavy docs
  whose Docling output has broken LaTeX spans are flagged in
  `<corpus>/.kg/math_retry.jsonl` for a later batch-Chandra patch
  pass. Designed for operators who want a fast first pass and
  will patch the math separately.
- **Routing recommendations validated on a published benchmark.**
  See `docs/extraction-benchmarks/ocr-comparison.md` for the
  full evidence: Chandra-OCR-2 produced 42× more real equations
  than Granite-Docling-258M on a math-heavy 1931 genetics paper
  (513 vs 12 inline equations); EasyOCR and SmolDocling-256M
  produced approximately zero usable LaTeX. The cost of Chandra
  on the test hardware is ~30 s/page, vs ~1 s/page for
  Docling-without-OCR.

## Chunking + embedding

- **VectorStore protocol.** ChromaDB is the default (embedded,
  SQLite-backed, batteries-included). FAISS and Pinecone
  drop in behind the same protocol when their trade-offs are the
  right ones (FAISS for single-corpus scale beyond ChromaDB's
  practical limits; Pinecone for hosted multi-tenant paths).
  Only ChromaDB ships in the OSS default.
- **Chunk size: 3000 chars, 400 chars overlap.** Larger than the
  retrieval-tool default (~500 tokens) to preserve more context
  per chunk for retrieval-grounded answer quality. The hybrid
  chunker prefers structural boundaries (markdown headings,
  paragraphs, sentence ends) when they fit within the window;
  falls back to fixed-window slicing otherwise. Overridable per
  corpus via `embedding.max_chars` / `embedding.overlap_chars`.
- **Full-document coverage is mandatory.** Every byte of text the
  extractor produces lands in at least one chunk. The chunker
  enforces a coverage invariant at the end of every pass and
  raises on partial coverage. A knowledge graph that surfaces
  "the relevant section" to an LLM is only as honest as its
  chunks; missing a single paragraph is a hidden retrieval gap
  whose downstream cost (silently wrong answers) substantially
  exceeds the upstream cost (more chunks, slightly larger
  vector store).
- **Cosine HNSW for vector retrieval.** Chroma collections are
  created with `metadata={"hnsw:space": "cosine"}` so retrieval
  ranking is by cosine similarity end-to-end.
- **Per-chunk metadata schema.** Each chunk carries `doc_id`,
  `filename`, `path`, `title`, `type`, `year`, `topics`,
  `committee_member`, `source`, `chunk_index`, `total_chunks`,
  plus extras (`source_filename`, `extractor_version`, `authors`).
  Field names match established academic-knowledge-base
  conventions so retrieval consumers can navigate back to the
  per-doc card with no extra lookup.

## Surfacing

- **MCP stdio server** as the agent-facing surface
  (`nuthatch serve --corpus <name>`). Raw STDIO JSON-RPC 2.0,
  no MCP SDK dependency. The five tools are:
  `corpus_search`, `subgraph_extract`, `card_get`,
  `community_get`, `token_econ_report`.
- **Per-document markdown card** (Obsidian-compatible YAML
  frontmatter, queryable via Dataview) — one per ingested doc,
  under `<corpus>/cards/`.
- **Per-community markdown page** under `<corpus>/communities/`,
  listing members as wikilinks for one-click navigation in
  Obsidian.
- **Dashboard / index / log** at the corpus root for the
  human-readable view.

## Token-economy methodology

nuthatch ships a `token_econ_report` MCP tool that measures how
many tokens the consuming agent saved by using nuthatch instead
of a non-nuthatch fallback. **The choice of fallback is the entire
methodology** — everything else is bookkeeping.

### Per-tool, per-query baselines

A static "whole corpus" counterfactual overstates savings by
orders of magnitude because nobody pastes an entire corpus into
context per query. nuthatch uses a baseline specific to what each
tool actually replaces:

| Tool | Counterfactual baseline |
| --- | --- |
| `corpus_search(query, k)` | BM25 top-`k` text over the same chunks Chroma indexes |
| `subgraph_extract(seeds, depth)` | Sum of card-markdown tokens for every `doc::` node in the returned subgraph |
| `community_get(community_id)` | Sum of card-markdown tokens for every member of the community |
| `card_get(doc_id)` | Skipped entirely (pure delivery of a card the agent asked for by ID; no reduction to claim) |

Reductions reported under this scheme typically land between
~2× and ~8× depending on corpus size and query specificity. The
smaller numbers are deliberate: they are defensible, comparable
across queries, and computed against a baseline the operator
would actually have used in the absence of nuthatch.

### Comparison with the whole-corpus framing

graphify (Safi Shamsi,
[github.com/safishamsi/graphify](https://github.com/safishamsi/graphify))
publishes a "71.5× fewer tokens per query vs reading the raw
files directly" headline in `docs/how-it-works.md`. The
methodology behind that figure (from `graphify/benchmark.py`):
counterfactual is computed as `nodes × 50 words × 1.33`
synthetic tokens — i.e. an estimate of "the whole corpus as raw
text"; served is the BFS subgraph text after keyword-substring
matching three seed nodes; ratio = corpus / served.

The honest critique is one of methodology, not effort: a 71.5×
ratio computed against "all raw files" is not a comparison to
any retrieval baseline an operator would actually have used. The
relevant question for a token-economy claim is not "how much
smaller is the served output than the corpus" but "how much
smaller is it than what I'd otherwise have pasted into the LLM",
and "what I'd otherwise have pasted" is a top-k retrieval, not
the corpus. The graphify documentation itself notes the
comparison breaks at small corpora ("Six files already fits in a
context window — the graph value there is structural clarity,
not compression"), which is correct, but the headline number is
still computed by the whole-corpus method and shown without that
caveat.

nuthatch's per-tool baselines produce smaller and defensible
ratios because they compare like-for-like. Any token-economy
figure that ships in nuthatch's documentation or dashboard cites
which tool's counterfactual was used, the corpus size, and the
query (or sample query set).

## Execution model: build out-of-band, query via MCP

nuthatch separates **building the corpus** from **querying it**.

- **Build** happens out-of-band via the CLI:
  `nuthatch ingest` → `nuthatch embed` → `nuthatch graph` →
  `nuthatch cluster` → `nuthatch render`. Each stage is
  incremental where the operation supports it (hash-based file
  dedup for ingest, per-doc Chroma presence check for embed,
  deterministic rebuild for graph, full re-fit for cluster which
  is correct for the method, byte-stable re-render for render).
  Triggered by the operator (or a cron / watcher); runs to
  completion; persists artifacts under `<corpus>/.kg/` and
  `<corpus>/{cards,communities}/`.
- **Query** happens at agent runtime via the MCP server. The
  agent calls the 5 tools against already-built state. No graph
  mutation; no extraction; no embedding; no clustering at query
  time.

### Why this split

1. **Honest latency.** Agent tool calls return in milliseconds to
   single-digit seconds (vector lookup, BFS, file read). No tool
   call hides a multi-minute embedding job.
2. **Predictable cost.** Build cost is fixed and amortised;
   query cost is bounded by the chunks or cards returned. The
   token-economy report compares served vs counterfactual at
   query time without the noise of "and we also ran OCR".
3. **One source of truth.** The graph, embeddings, and cards on
   disk are the only state. Two agents querying the same corpus
   see the same answer. No per-session rebuild divergence.
4. **Long-running stages are operator-supervised, not hidden.**
   When `nuthatch ingest` is OCR'ing 100 scanned PDFs (potentially
   hours of GPU time), the operator sees per-file progress in a
   terminal and can pause, kill, or restart. An alternative model
   that drives the same work from inside an agent invocation
   loses this feedback channel: if the agent session times out
   or the operator disconnects, the work is lost without a clean
   recovery path. nuthatch's CLI runs in the operator's terminal,
   exits with a status code, and writes per-run audit logs to
   `<corpus>/.kg/audit/`.
5. **Quality issues are surfaced and actionable, not silently
   absorbed.** When extraction yield drops below the QC floor,
   `nuthatch ingest` quarantines the file with a reason-tagged
   subdir and a `.reason.json` sidecar — the operator inspects,
   fixes the source, re-runs. Schema validation failures log
   missing required fields. The clustering router downgrades from
   principled SBM to heuristic Leiden when graph-tool is
   unavailable and surfaces that in the response's `rigor_used`
   field. In each case the operator knows exactly what fell short
   and what their options are.
6. **Server-side orchestration is harness-proof.** Skill files
   that tell the agent "MUST use the Agent tool" or "spawn N
   parallel subagents" are not enforceable: the host harness
   (Claude Code, Codex, OpenCode, Aider) can demote directives
   when context is tight, substitute serial file-reading when a
   parallel call times out, reorder steps per its own planning
   model, or fall back when it judges the skill too verbose.
   Putting the orchestration inside the MCP server (Python code
   the agent does not see) means the agent calls a tool and the
   server does the right thing internally — parallel work,
   batching, caching, deduplication. The agent cannot deviate
   because the agent is not doing the orchestration. This is the
   same reason a well-designed REST API does not ship a client
   runbook saying "you MUST batch requests": the server handles
   batching itself.

### Comparison with the build-per-query model

graphify's skill files (one per agent host: `skill-claude-code.md`,
`skill-codex.md`, `skill-opencode.md`, etc., each in the
1,200-1,400-line range) are orchestration runbooks: the agent
uses `@agent` parallel dispatch (or the host's equivalent) to
spawn semantic-extraction subagents on each invocation, then
drives clustering and analysis and rendering inline. That model
fits a "build per query session" shape, and the per-host skill
file is the runbook the agent is expected to follow.

nuthatch's "build once, query many" shape means our skill files
are an order of magnitude shorter (~150-200 lines) without that
being a deficit: the build runbook is the CLI itself, called by
the operator out-of-band. The skill files only document MCP
registration plus how to use the 5 query tools effectively in
common multi-step flows. The agent does the agent's job;
nuthatch does nuthatch's.

## What is not invented here

nuthatch composes well-established methods. Where it differs
from peer tools, the difference is in engineering decisions
about how to compose them, not in algorithmic novelty.

| Method | Origin | nuthatch's use |
| --- | --- | --- |
| Stochastic Block Model (Bayesian, nested, degree-corrected) | Karrer & Newman 2011; Peixoto 2014, 2017 | The principled clustering tier via `graph-tool` |
| Leiden community detection | Traag, Waltman, van Eck 2019 (Nature Sci. Rep. 9, 5233) | The heuristic clustering tier via `leidenalg` |
| Knowledge graphs as a concept | Cyc (1984), the Semantic Web stack (2000s), Freebase (2007), Wikidata (2012) | The data model |
| Edge confidence labels | NELL, YAGO, ConceptNet (2000s-2010s) | The `EXTRACTED` `INFERRED` `AMBIGUOUS` tags on every edge |
| Sentence-transformer embeddings | Reimers & Gurevych 2019 | Semantic dedup and corpus search |
| BM25 lexical retrieval | Robertson et al. (the BM25 family, late 1990s) | The token-economy counterfactual baseline for `corpus_search` |
| Cosine similarity for dense retrieval | Established in IR for decades | The default distance metric for the vector store |
| Graph-based retrieval over LLM-built entity graphs | Microsoft GraphRAG (April 2024), among others | The conceptual pipeline shape; we chose SBM where GraphRAG chose Leiden |
| Hybrid chunking with structural-boundary preference | Standard RAG practice | The chunker (`src/nuthatch/embed/chunk.py`) |

We evaluated existing tools in this space — graphify, Microsoft
GraphRAG, Khoj, Cognee, Verba — and chose to build nuthatch
because we wanted a different combination of trade-offs:
principled clustering as the rigor ceiling rather than Leiden
as the rigor floor; honest token-economy accounting rather than
whole-corpus headline numbers; build out-of-band rather than
build per query; strict schema-quarantine on ingest rather than
accept-and-degrade; and a five-stage CLI pipeline the operator
controls rather than a multi-thousand-line agent runbook.

The peer tools are not wrong for choosing differently. Their
choices fit their value propositions; ours fit ours. This
document records what we chose and why so the next contributor
or evaluator can judge whether the trade-offs match their
needs.

## References

- Karrer, B. & Newman, M.E.J. 2011. "Stochastic blockmodels and
  community structure in networks." *Phys. Rev. E* 83, 016107.
- Peixoto, T.P. 2014. "Hierarchical block structures and
  high-resolution model selection in large networks."
  *Phys. Rev. X* 4, 011047.
- Traag, V.A., Waltman, L., van Eck, N.J. 2019. "From Louvain to
  Leiden: guaranteeing well-connected communities."
  *Sci. Rep.* 9, 5233.
- Reimers, N. & Gurevych, I. 2019. "Sentence-BERT: Sentence
  embeddings using Siamese BERT-networks." *EMNLP*.
- Microsoft GraphRAG (2024): <https://github.com/microsoft/graphrag>
- graphify: <https://github.com/safishamsi/graphify>
- Khoj: <https://github.com/khoj-ai/khoj>
- Cognee: <https://github.com/topoteretes/cognee>
- Verba (Weaviate): <https://github.com/weaviate/Verba>
- nuthatch OCR benchmark:
  `docs/extraction-benchmarks/ocr-comparison.md`
