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

The skill files under `agent_skills/` (`skill-claude-code.md`,
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

## Pre-flight: triage

Before running the expensive Docling extraction over every PDF,
`nuthatch triage` does a cheap pdftotext-only pass (no GPU,
~50 ms per PDF) and classifies each source file as:

| Class | Meaning |
| --- | --- |
| `PASS` | title + authors + abstract all detectable; should ingest cleanly |
| `FLAG` | title + authors detected but no `Abstract` keyword on page 1; relies on the paragraph fallback in the extractor |
| `DEFER` | missing title OR missing authors; likely quarantine target |
| `UNKNOWN` | pdftotext failed or timed out |

Run it any time the source set changes:

```bash
nuthatch triage --corpus my-corpus
```

The report prints per-class lists with a one-line reason. Add
`--defer` to auto-move every `DEFER` PDF to
`<corpus>/defer/<original_subdir>/`. The `defer/` subdirectory is
reserved &mdash; the recursive source-walk skips it, so the next
`nuthatch ingest` run picks up only `PASS` and `FLAG` candidates.

```bash
nuthatch triage --corpus my-corpus --defer
```

This is the recommended workflow for corpora with heterogeneous
PDF sources (notably bioRxiv preprints, where Word-submitted
manuscripts produce highly variable title-page layouts):

1. Drop PDFs into `<corpus>/<subdir>/`.
2. `nuthatch triage --corpus my-corpus --defer` to filter out
   the unparseable subset.
3. `nuthatch ingest --corpus my-corpus` runs only on the
   `PASS`/`FLAG` set.
4. Periodically inspect `<corpus>/defer/` and either extend the
   extractor heuristics (`src/nuthatch/ingest/metadata.py`) or
   move papers to `<corpus>/rejected/` if their content is
   unrecoverable.

The triage classification is an *approximation* of what the real
extractor will see (pdftotext output differs from Docling
markdown). A `PASS` here is "very likely to ingest," not
"guaranteed."

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

### Optional: LLM-named community labels

The default labels are heuristic word-frequency strings
("circadian gene expression mouse"). Pass `--relabel-llm` to
replace them with topical 2-4 word names produced by a local
LLM ("Circadian Molecular Biology"):

```bash
nuthatch cluster --corpus my-corpus --backend sbm --relabel-llm \
    --relabel-backend ollama --relabel-model granite3-dense:8b
```

The relabel pass writes back into the canonical
`communities.json` and every `communities_<suffix>.json` sibling
produced by this run, so the labels stay consistent across the
on-disk views. Requires `ollama serve` and the model already
pulled (`ollama pull granite3-dense:8b`). When `--relabel-llm`
is omitted, the heuristic labels are kept and the rendered
cards / community pages carry those instead.

### `--output-suffix` is copy-not-rename

`--output-suffix sbm` writes an additional
`communities_sbm.json` (and `community_centroids_sbm.npy`)
under `.kg/`; the canonical `communities.json` /
`community_centroids.npy` are preserved. Use this when you want
to compare backends side-by-side: run each backend with its own
suffix, and the canonical is whichever ran last without one. The
MCP server (`nuthatch serve`) reads canonical exclusively;
suffixed copies are there for ad-hoc inspection and the D3 viz's
`--communities <suffix>` overlay.

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

## Optional: visualise the graph

`nuthatch viz d3` renders an interactive D3 + HTML5-canvas view of the
corpus graph. The expensive step (force-directed layout) runs once
server-side and is reused across every overlay, so emitting one HTML
per clustering backend is cheap.

```bash
# Baseline: no community overlay, just the topology
nuthatch viz d3 --corpus my-corpus

# Color nodes by SBM community membership
nuthatch viz d3 --corpus my-corpus --communities sbm

# Emit one HTML per available communities_<backend>.json
# (sbm, leiden, embeddings - whichever you have on disk)
nuthatch viz d3 --corpus my-corpus --communities all
```

Outputs land in `pipeline_output/graph_topology_d3_<label>_<UTC>.html`.
Open any of them in a browser; drag to pan, scroll to zoom, hover for
node details, toggle entity types + relations from the in-page filter
panel.

Useful flags:

- `--filter-types document,topic,method` — keep only specific entity
  types (defaults to all)
- `--max-nodes 3000` — degree-prune to the top-N nodes if the graph is
  larger than your browser can comfortably handle
- `--layout {forceatlas2,spring,kamada_kawai}` — layout algorithm;
  forceatlas2 is the default and the best balance of speed and quality
- `--layout-iterations 200` — more iterations = better layout, slower

Not on the critical pipeline path; renders in tens of seconds for typical
corpora. No tmux / `launch-stage.sh` wrapper needed.

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
"papers needing schema fix", etc.). The dedicated setup guide is
the next section.

## Opening the corpus in Obsidian

### One-time setup

1. Install [Obsidian](https://obsidian.md) if you have not already.
2. Open Obsidian and choose **Open folder as vault**.
3. Navigate to your corpus root (e.g. `~/my-corpus/`) and click
   **Open**. Obsidian will index the tree.
4. Go to **Settings → Community plugins**, disable safe mode, and
   install the **Dataview** plugin. Enable it. That is the only
   required plugin; everything else is optional.

The vault is immediately navigable once Dataview is enabled. The
`dashboard.md` and `index.md` files at the vault root render live
Dataview queries against the `cards/` and `communities/` folders.

### Vault layout

```text
<corpus-root>/          ← open this as the Obsidian vault
├── cards/              ← one note per document
│   └── <doc_id>.md    ←   YAML frontmatter + abstract + topics
├── communities/        ← one note per community cluster
│   └── <id>.md        ←   member list, core papers, description
├── dashboard.md        ← quick-access Dataview queries
├── index.md            ← content catalog by type
└── log.md              ← append-only ingest/export history
```

`cards/` and `communities/` are populated by `nuthatch render`. The
other three files are regenerated on every render run.

### Navigating the vault

**Start at `dashboard.md`.** It has four ready-to-run Dataview
tables:

- Recently ingested papers (sorted by ingest date)
- Highest-relevance papers (sorted by relevance score)
- Papers by community (the main thematic grouping)
- All communities with member counts

**Click any wikilink** (`[[doc_id|title]]`) from a community page
to jump to that paper's card. Every card's frontmatter carries
`community_id`, `community_path`, and `community_label` so you can
follow the graph from any paper back to its cluster.

**Use `index.md`** for a flat sorted list of all documents and
communities. Useful when you want alphabetical or by-year access
rather than thematic grouping.

**Graph view** (`Ctrl+G` / `Cmd+G`) visualises the wikilink network
between cards and community pages. Community pages link to every
member card, so the graph view naturally shows community structure.
Enable **Filters → Show attachments: off** and
**Groups → color by tag** to highlight community membership.

### Dataview query examples

All queries run in a `dataview` fenced code block inside any note.

Papers in community 23:

```dataview
TABLE title, year, authors
FROM "cards"
WHERE community_id = 23
SORT year ASC
```

Papers tagged with a specific topic:

```dataview
TABLE title, year, community_label
FROM "cards"
WHERE contains(tags, "reinforcement-learning")
SORT year ASC
```

All communities sorted by size:

```dataview
TABLE n_members, backend
FROM "communities"
SORT n_members DESC
```

Papers not yet assigned to a community (extraction or clustering
incomplete):

```dataview
TABLE title, year
FROM "cards"
WHERE community_id = null
```

### Re-rendering after pipeline changes

Run `nuthatch render --corpus my-corpus` any time you:

- Re-cluster with a different backend (SBM, Leiden, embeddings)
- Update community descriptions
- Add new papers (after ingest + embed + graph + cluster)

The render stage is fast and idempotent; cards whose content did not
change are rewritten identically. After re-render, close and reopen
Obsidian (or run **Reload app without saving** from the command
palette) to pick up changes to the Dataview cache.

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

Each skill file under `agent_skills/` (`skill-claude-code.md`,
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

## Publish: a shareable KB directory

Once `render` has produced cards, communities, and dashboards
inside the corpus root, `nuthatch publish` promotes the
consumable surface (cards, communities, dashboard, graph JSON,
community indexes, optional chroma archive, AGENTS / CAPABILITIES
manifests) into a destination of your choice. The destination is
the artifact you ship: a folder an agent host can
`nuthatch serve` against, a human can open as an Obsidian vault,
or you can commit to its own git repo for distribution.

```bash
nuthatch publish --corpus my-corpus \
    --to ~/my-kb-name \
    --name my-kb-name \
    --license CC-BY-4.0
```

What lands by default:

- Human-navigable: `README.md`, `AGENTS.md`, `OVERVIEW.md`,
  `LICENSE`, `cards/`, `communities/`, `dashboard.md`,
  `index.md`, `log.md`, `.obsidian/{graph,app,appearance,community-plugins}.json`,
  `docs/graph.html` (latest D3 viz)
- Agent-readable: `.kg/graph/graph.json`, `.kg/communities*.json`,
  `.kg/community_centroids.npy`, `embeddings.tar.gz` (chroma
  archive at KB root for visibility),
  `CAPABILITIES.json`, `mcp_config.example.json`
- Provenance: `docs/eval-*.md` if present in the tool repo,
  `publish_manifest.jsonl` (per-file sha256 + role)

What is never copied: source PDFs (copyright), `.kg/extracted/`
body text (opt in via `--include-bodies`), `.kg/audit/`
(operator-private), `.kg/manifest.jsonl` (corpus-internal paths).

Useful flags:

- `--no-chroma`: skip the chroma archive. The KB is still
  queryable by the on-disk Obsidian surface; MCP retrieval
  tools (`corpus_search`, `community_search`) won't work
  without it.
- `--include-bodies`: ship the full body markdown under
  `.kg/extracted/`. Required only if downstream consumers
  need raw text beyond the cards.
- `--license <SPDX>`: defaults to `CC-BY-4.0` (right for
  bioRxiv / arXiv-derived corpora). Pass `Apache-2.0`, `MIT`,
  `CC-BY-SA-4.0`, `CC0-1.0`, or `PROPRIETARY` as appropriate.
- `--no-d3-html`: skip the D3 viz copy if your destination
  doesn't need it.

The destination is overwritten on each re-publish, and
`publish_manifest.jsonl` records per-file provenance so
re-publishes are diffable.

After publishing, the destination can be served by
`nuthatch serve --corpus ~/my-kb-name` (after the chroma
archive is unpacked: `mkdir -p .kg && tar xzf embeddings.tar.gz -C .kg/`).

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
- The per-agent skill files under `agent_skills/skill-*.md` for
  MCP registration details per host
