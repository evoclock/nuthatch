# How to run Nuthatch

Recipe-style operator guide for the five-stage build pipeline plus
the MCP query surface. Companion to `Design_Decisions.md` (the
"why") and `extraction-benchmarks/ocr-comparison.md` (the OCR
backend evidence).

## Operating principle

The build pipeline runs as **five separate stages, triggered
between each by you**:

```text
nuthatch ingest  ->  nuthatch embed  ->  nuthatch graph
                    ->  nuthatch cluster  ->  nuthatch render
```

There is no `nuthatch run-all` command, by design. Each stage's
output is what you'd want to inspect before paying for the next
stage's compute. Bad ingest quality (low extraction yield,
schema-quarantine spike, wrong routing on scanned PDFs) makes
embedding, clustering, and rendering on top of it a waste of
GPU time and storage. Inspect, then proceed.

`Design_Decisions.md` § "Execution model" explains the rationale
in more detail.

## Quick install

```bash
# Once published to PyPI:
pipx install nuthatch

# Until then, from the public repo:
pipx install git+https://github.com/evoclock/nuthatch.git@main
```

For the principled clustering tier (Bayesian SBM via graph-tool)
you also need conda. Graph-tool is conda-only by upstream's
design. Without it, Nuthatch downgrades cleanly to the Leiden
heuristic tier and surfaces that fact in every clustering
response's `rigor_used` field.

```bash
# Optional: conda env for the SBM tier
conda create -n nuthatch-gt -c conda-forge graph-tool python
```

## Stage launcher

`scripts/ops/launch-stage.sh` is the recommended path for running
each stage. It:

- Starts the stage as its own tmux session (`nuthatch-<stage>`)
- Pops a graphical terminal on your display auto-tailing the live
  log (zero-click visibility. Works on macOS Terminal/iTerm2,
  Linux gnome-terminal / konsole / kitty / alacritty / xterm /
  others, Windows Terminal via WSL)
- Logs to `<corpus>/.kg/audit/<stage>-<timestamp>.log`
- Unbuffers stdout (`PYTHONUNBUFFERED=1`) so progress appears in
  real time, not buffered until the process exits

```bash
scripts/ops/launch-stage.sh <stage> <corpus> [extra-args]
```

For headless servers / SSH sessions without a graphical
environment, the launcher detects and falls back to printing the
`tail -f` and `tmux attach` commands for you to run by hand.

You can also bypass the launcher and call `nuthatch <stage>`
directly. The launcher is a convenience, not a privileged path.

### Two ways to drive the pipeline, both gated

Whichever path you pick, **the pipeline runs one stage at a time
with operator confirmation between stages**. Nuthatch does not
ship an end-to-end automation that runs ingest → embed → graph →
cluster → render unattended. Bad ingest quality wastes the
compute of every stage that follows; the gate exists so you
catch problems before paying that cost.

**Path A: harness-driven (agent invokes the launcher on your
behalf).** If you want an MCP-connected agent (Claude Code,
Codex, OpenCode, Aider, Pi, Hermes) to handle the mechanics,
instruct it to run `scripts/ops/launch-stage.sh <stage> <corpus>`
via its Bash tool. The agent's flow should be:

> 1. Move new sources into the corpus tree (`mv` / `cp`).
> 2. Confirm with the user: "I'll run `nuthatch ingest` on
>    <N> new files. It will take roughly <est> minutes.
>    Proceed?"
> 3. On approval, invoke the launcher. A terminal window opens
>    on the user's display with live tail; the agent reads
>    progress from the log file.
> 4. After the stage finishes, summarise the result (per-status
>    counts, any quarantines, where to inspect them).
> 5. Ask before triggering the next stage. Never chain
>    `ingest && embed && graph && cluster && render` unattended.

The skill files in this repo (`skill-claude-code.md`,
`skill-codex.md`, etc.) embed this discipline so per-host
agents have it as their default. Per Design_Decisions.md §
"Execution model", server-side orchestration inside the MCP
read-only tools is harness-proof; build-stage orchestration is
operator-confirmed-per-stage and the agent is a convenience for
the user, not a substitute.

**Path B: operator-driven (you run it yourself; recommended for
the first few runs).** Open a terminal, run the launcher per
stage, inspect output between stages. Same gated semantics as
Path A, just without an agent in the loop:

```bash
scripts/ops/launch-stage.sh ingest my-corpus
# inspect output (quarantine, papers/, .kg/extracted/)
scripts/ops/launch-stage.sh embed my-corpus
# inspect chunk count
scripts/ops/launch-stage.sh graph my-corpus
# inspect graph shape
NUTHATCH_BIN=$(conda run -n nuthatch-gt which nuthatch) \
    scripts/ops/launch-stage.sh cluster my-corpus
# inspect partition
nuthatch render --corpus my-corpus
# inspect cards/, communities/, dashboard.md
```

Recommended for first runs on a new corpus and any time the
ingest config (schema profile, OCR routing, extraction backend)
has changed. The harness-driven path is useful once the
pipeline is tuned for the corpus shape and you trust the
quality gate at each stage.

## Initialise a corpus

```bash
nuthatch init ~/my-corpus --register-as my-corpus --set-default
```

This creates `~/my-corpus/.kg/` (the corpus marker), plus the
standard subdirectories. The corpus is registered under the name
`my-corpus` in `~/.config/nuthatch/registry.toml`, set as the
default so subsequent commands can omit `--corpus`.

Drop your source files anywhere under the corpus root. Suggested
shapes (the pipeline scans recursively, skipping the Nuthatch-
managed subdirs):

```text
~/my-corpus/
├── inbox/                 # the convention if you want a flat drop
├── arxiv/2026/            # or organise by source
├── bioarxiv/              # or by venue
├── notes/                 # mix prose notes in too
└── topics/genetics/       # whatever organisation fits your workflow
```

## Stage 1: ingest

Extracts text from each source file, validates against the active
`SchemaProfile`, moves successful files to
`<corpus>/processed/<original_subdir>/` (source provenance
preserved), writes the extracted markdown + metadata sidecar to
`<corpus>/.kg/extracted/<doc_id>.md` and `<doc_id>.meta.json`.
Files that fail extraction, QC, or schema validation are moved to
`<corpus>/quarantine/<reason>/` with a per-file `.reason.json`
sidecar that records the file's original subdir so a fix-pass can
route it back to `processed/<original_subdir>/` after the issue
is resolved. Quarantine is a transient state; the only legal exits
are `processed/` (fixed) or `rejected/` (declared unfixable).

```bash
scripts/ops/launch-stage.sh ingest my-corpus
```

### `--skip-chandra` for fast first ingests

Bypasses Chandra-OCR entirely; routes every PDF through Docling
(with EasyOCR fallback for genuinely-scanned PDFs). Math-heavy
docs whose Docling output has broken LaTeX spans are flagged in
`<corpus>/.kg/math_retry.jsonl` for a later batch-Chandra patch
pass (the patcher itself is a future feature; the marker file
is forward-compatible).

```bash
scripts/ops/launch-stage.sh ingest my-corpus --skip-chandra
```

### `--accelerator` for GPU / Apple Silicon

Docling's layout / table-structure / OCR models run on whatever
device you point them at. Default is `auto`, which lets Docling
detect the best available (CUDA on Nvidia, MPS on Apple Silicon,
XPU on Intel, CPU as fallback). Override per-run via the flag:

```bash
# Force CUDA (Nvidia GPU)
scripts/ops/launch-stage.sh ingest my-corpus --skip-chandra --accelerator cuda

# Apple Silicon (M1 / M2 / M3 / M4)
scripts/ops/launch-stage.sh ingest my-corpus --skip-chandra --accelerator mps

# Force CPU even when a GPU is present
scripts/ops/launch-stage.sh ingest my-corpus --skip-chandra --accelerator cpu
```

Or set persistently in the environment:

```bash
export NUTHATCH_ACCELERATOR=cuda
scripts/ops/launch-stage.sh ingest my-corpus --skip-chandra
```

The CLI flag wins when both are set. Speedup on the digital-PDF
path (Docling layout + table-structure) is roughly 5-7x on a
mid-range CUDA GPU vs. CPU; VRAM use is under 500 MB per worker,
so this is safe to run alongside other GPU work.

### Inspect ingest results

```bash
# Manifest stats:
nuthatch status --corpus my-corpus

# Counts at each stage of the funnel:
find ~/my-corpus/processed -name '*.pdf' | wc -l   # ingested + schema-passed
ls ~/my-corpus/.kg/extracted/*.md        | wc -l   # extracted markdown
ls ~/my-corpus/quarantine/               | wc -l   # quarantine-reason subdirs

# Per-reason quarantine inspection:
ls ~/my-corpus/quarantine/<reason>/
cat ~/my-corpus/quarantine/<reason>/<file>.reason.json
```

Common quarantine reasons:

| Reason | Means | Typical fix |
| --- | --- | --- |
| `unsupported_format` | Wrong file extension | Rename or convert |
| `empty_file` | 0-byte source | Re-download |
| `extract_yield_failed` | Extracted text below the QC floor | Verify the PDF opens; consider Chandra if it's a scanned doc |
| `schema_invalid` | Extracted text didn't yield required metadata | Inspect the extracted markdown; the active SchemaProfile may need to be relaxed for this corpus, or the PDF is missing the expected fields |
| `read_error` / `extract_error` | Infrastructure problem | Read the `.reason.json` sidecar; usually a backend crash worth retrying |

A quarantined file stays on disk. After fixing the source, move
it back to the corpus tree and re-run ingest.

### When ingest looks acceptable

- The vast majority of files in `processed/<subdir>/`, not `quarantine/`
- Quarantine reasons are explainable (specific corrupt files,
  not a systematic class)
- The per-file progress in the log shows the expected mix of
  Docling (fast) and Chandra (slow, math-heavy)

Then proceed to embed.

## Stage 2: embed

Chunks each extracted markdown into overlapping windows (3000
chars target, 400 chars overlap by default; structure-aware
boundary preference), embeds each chunk via BGE-M3, persists
to ChromaDB at `<corpus>/.kg/embeddings/`. Incremental: only
docs not already in the vector store get embedded.

```bash
scripts/ops/launch-stage.sh embed my-corpus

# Re-embed everything from scratch (deletes old chunks per doc
# first). Required when you change _DEFAULT_MAX_CHARS /
# _DEFAULT_OVERLAP_CHARS in src/nuthatch/embed/chunk.py or
# switch the embedding model via .kg/config.yaml:
scripts/ops/launch-stage.sh embed my-corpus --force
```

### Per-corpus overrides

Optional `~/my-corpus/.kg/config.yaml`:

```yaml
embedding:
  max_chars: 3000          # chunk window
  overlap_chars: 400       # adjacent-chunk overlap
  batch_size: 32           # BGE-M3 batch size (tune for GPU memory)
  model: BAAI/bge-m3       # any sentence-transformers model
```

Defaults are tuned for academic-paper-size corpora at ~50-200
chunks per doc. Override at your own risk; retrieval quality
becomes your tuning responsibility.

### Inspect embed results

```bash
python -c "from nuthatch.embed.store import ChromaVectorStore; \
    from pathlib import Path; \
    s=ChromaVectorStore(root=Path('~/my-corpus/.kg/embeddings').expanduser()); \
    print(f'chunks in store: {s.count()}')"
```

Check chunk count is reasonable (typically 30-200 per paper).
A 1:1 chunks-to-papers ratio means most papers single-chunked,
which is suspicious unless they're very short.

## Stage 3: graph

Walks `<corpus>/.kg/extracted/`, runs entity extraction (authors,
citations, topics) per doc, assembles the NetworkX MultiDiGraph
and saves to `<corpus>/.kg/graph/graph.json`. Fast (regex-based;
no LLM).

```bash
scripts/ops/launch-stage.sh graph my-corpus
```

### Inspect graph

```bash
python -c "from nuthatch.graph.io import load_graph; \
    from pathlib import Path; \
    g=load_graph(Path('~/my-corpus/.kg/graph/graph.json').expanduser()); \
    print(f'{g.number_of_nodes()} nodes, {g.number_of_edges()} edges'); \
    docs=[n for n,d in g.nodes(data=True) if d.get('node_type')=='document']; \
    entities=[n for n,d in g.nodes(data=True) if d.get('node_type')=='entity']; \
    print(f'  documents: {len(docs)}'); \
    print(f'  entities:  {len(entities)}')"
```

A thin graph (entity count < 2× doc count) means the entity
extractor didn't pick up much. Could be a real signal (papers
have weak metadata) or worth a closer look at the extractor.

## Stage 4: cluster

Fits communities on the corpus graph and writes the community ID
back onto each doc node. Always a full re-fit (clustering is
global by nature; correct for the SBM / Leiden methods Nuthatch
uses).

```bash
# Default venv: will fall back to Leiden if graph-tool isn't
# available.
scripts/ops/launch-stage.sh cluster my-corpus

# For the principled SBM tier:
NUTHATCH_BIN=$(conda run -n nuthatch-gt which nuthatch) \
    scripts/ops/launch-stage.sh cluster my-corpus
```

The CLI prints `rigor=principled` (SBM) or `rigor=heuristic`
(Leiden) so you know which tier you got.

### Inspect partition

```bash
python -c "from nuthatch.graph.io import load_graph; \
    from pathlib import Path; \
    from collections import Counter; \
    g=load_graph(Path('~/my-corpus/.kg/graph/graph.json').expanduser()); \
    cids=Counter(d.get('community_id') for n,d in g.nodes(data=True) \
                 if n.startswith('doc::') and d.get('community_id') is not None); \
    print(f'{len(cids)} communities; sizes:', dict(cids.most_common()))"
```

A 150-paper corpus typically partitions into 5-15 communities.
One giant community containing nearly everything is a sign of a
bad partition: re-run, switch rigor tier, or look at why the
graph has so little structural signal (often: weak entity
extraction in stage 3).

## Stage 5: render

Regenerates the Obsidian-compatible per-doc cards and per-cluster
community pages, plus the dashboard / index / log at the corpus
root.

```bash
nuthatch render --corpus my-corpus
```

Fast (sub-second to a few seconds for typical corpora). No tmux
session needed.

### Inspect render output

```bash
ls ~/my-corpus/cards/        | wc -l   # one per doc
ls ~/my-corpus/communities/  | wc -l   # one per community
head ~/my-corpus/dashboard.md
```

Open `~/my-corpus/` as a vault in [Obsidian](https://obsidian.md)
to navigate the result visually. The cards use Dataview-friendly
YAML frontmatter so you can query the corpus from inside Obsidian
("all papers tagged X", "low-relevance archive candidates",
"papers needing schema fix", etc.).

## Serve: the MCP query surface

After the build pipeline runs, you have a queryable corpus. Point
an MCP-aware agent at it:

```bash
nuthatch serve --corpus my-corpus
```

The server exposes nine tools over stdio JSON-RPC:

| Tool | Purpose |
| --- | --- |
| `corpus_search(query, k)` | Chunk-level dense retrieval. Hits include `community_id`, `community_path`, `community_label` so the agent can route directly into community tools without an extra round-trip. |
| `subgraph_extract(seed_nodes, depth)` | BFS subgraph around seeds. |
| `card_get(doc_id)` | Full per-doc card markdown (frontmatter has community fields). |
| `community_get(community_id)` | Full per-community page markdown. |
| `community_brief(community_id, top_n)` | Cheap structured preamble (label, n_members, top-N representatives) before paying card-fetch cost. |
| `community_search(query, k)` | Semantic search at the community level — ranks communities by query-to-centroid cosine. Jump straight to the relevant cluster. |
| `community_core_nodes(community_id)` | High-degree members within the community. The "key papers" of the cluster. |
| `community_hierarchy(doc_id)` | Walk the nested SBM hierarchy (leaf → super-communities). Progressive zoom for context expansion / contraction. |
| `token_econ_report(group_by, since, until)` | Aggregate per-tool counterfactual savings. |

The community tools are the headline of Nuthatch — they let
agents do graph-RAG over your corpus without paying full-card
costs to learn community membership. See
[`docs/Design_Decisions.md`](Design_Decisions.md) §
*Community-aware retrieval* for the rationale. A typical
LLM-facing recipe:

```text
1. agent calls community_search("relevant topic", k=3)
   -> three community_ids ranked by semantic match
2. agent calls community_brief(community_id, top_n=5) on the winner
   -> label + 5 representative doc_ids, ~200 tokens
3. agent decides: drill into one card (card_get) or read the
   whole community page (community_get) or walk the parent
   community (community_hierarchy)
```

That flow replaces what an Obsidian-only setup or a flat-vector-
search tool would force into N round-trips, one per hit.

Each skill file in the repo (`skill-claude-code.md`,
`skill-codex.md`, `skill-aider.md`, `skill-opencode.md`,
`skill-pi.md`, `skill-hermes.md`) shows the host-specific
registration steps.

For Claude Code specifically:

```json
{
  "mcpServers": {
    "nuthatch": {
      "command": "nuthatch",
      "args": ["serve", "--corpus", "my-corpus"]
    }
  }
}
```

added to `~/.claude.json`'s `mcpServers` block, then restart
Claude Code.

## Watch mode (continuous ingest)

For setups where files arrive continuously (e.g. a Zotero export
folder, a watched dropbox):

```bash
scripts/ops/launch-stage.sh watch my-corpus
```

The watcher runs `nuthatch ingest` on every debounced filesystem
event under the corpus root (skipping the Nuthatch-managed
subdirs). It handles ingest only; embed / graph / cluster /
render remain manual triggers so you control when those costs
are paid.

## Stop and resume

Every stage is resumable; the pipeline is incremental where the
operation supports it:

- **ingest**: hash-based file dedup in `.kg/manifest.jsonl`
- **embed**: per-doc Chroma presence check
- **graph**: deterministic rebuild from scratch (cheap; no
  resume needed)
- **cluster**: always re-fits the whole graph (correct for SBM /
  Leiden)
- **render**: byte-stable; cards whose content didn't change get
  rewritten identically

To stop any stage:

```bash
tmux kill-session -t nuthatch-<stage>
# re-run the same command later to pick up
```

## File layout after a successful build

```text
~/my-corpus/
├── .kg/                         # nuthatch-managed state
│   ├── audit/                   # per-stage run logs
│   ├── manifest.jsonl           # what's ingested, when, hash
│   ├── extracted/               # <doc_id>.md + <doc_id>.meta.json
│   ├── embeddings/              # ChromaDB vector store
│   ├── graph/graph.json         # corpus graph (NetworkX node-link)
│   ├── math_retry.jsonl         # written when --skip-chandra flagged docs
│   └── mcp/mcp.log              # MCP server log
├── inbox/                       # convention: drop new files here
├── arxiv/, bioarxiv/, ...       # your own organisation; scanned recursively
├── papers/                      # post-ingest originals
├── quarantine/                  # files that failed any stage
├── cards/                       # per-doc Obsidian cards
├── communities/                 # per-cluster pages
├── dashboard.md
├── index.md
└── log.md
```

## See also

- `Design_Decisions.md`. The "why" behind every choice in this
  document
- `extraction-benchmarks/ocr-comparison.md`. The OCR-backend
  evidence (Chandra vs Docling vs EasyOCR vs Granite vs SmolDocling
  on three representative scanned papers)
- The per-agent skill files (`skill-*.md` in the repo root) for
  MCP registration details per host
