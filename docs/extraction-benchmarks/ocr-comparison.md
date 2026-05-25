---
title: "OCR extraction benchmark: Chandra-OCR-2 vs Docling backends"
date: 2026-05-25
---

## Context

Empirical comparison of the OCR backends nuthatch routes between in
the Sprint 2 extraction pipeline. nuthatch is an orchestrator that
calls user-installed OCR backends; it does not redistribute model
weights or run inference as a service. The benchmark exists to (a)
pin the routing decision in `docs/DECISIONS.md` with evidence and (b)
give users a clear map of trade-offs when they choose what to install.

## Configurations tested

| Run | Tool | Backend | Hardware |
| --- | --- | --- | --- |
| A | Chandra-OCR-2 v2 | HuggingFace transformers (Datalab) | RTX 5070 Ti (16 GB) |
| B | Docling 2.x | EasyOCR with `lang=['en']` | CPU |
| C | Docling 2.x VLM | Granite-Docling 258M (`ibm-granite/granite-docling-258M`) | RTX 5070 Ti |
| D | Docling 2.x VLM | SmolDocling 256M preview (`docling-project/SmolDocling-256M-preview`) | RTX 5070 Ti |

Three of the four (B, C, D) are Docling-based. A (Chandra-OCR-2) is a
separate standalone tool with its own CLI and multi-file output.

Earlier runs of Docling with RapidOCR (the bundled default, Chinese
mobile models) are not included: that configuration is appropriate
only for Chinese-language documents, not as a baseline for English
papers. To exercise English RapidOCR under Docling would require
downloading the English `det` / `rec` ONNX files explicitly; not
pursued because Docling+EasyOCR covers the same niche.

## Test corpus

Three papers chosen to cover different difficulty profiles:

| Paper | Pages | Why | Verified-scanned |
| --- | --- | --- | --- |
| McDonald and Kreitman 1991 | 3 | classic short scanned paper with one wide site-by-site table + one numeric result table; the original JSTOR PDF is in reverse page order, so this benchmark uses a page-reversed copy at `corpus/bioarxiv/McDonald_and_Kreitman_1991_fixed.pdf` | Yes (106 chars/page text yield) |
| Mendel 1866 (Bateson 1909 translation, 1925 Harvard reprint) | 52 | long body-text-heavy scanned book with embedded OCR layer of poor quality | Yes (page images are scans; embedded text is degraded JSTOR-style OCR) |
| Wright 1931 (Evolution in Mendelian Populations) | 63 | math-heavy population-genetics paper with equations and tables; 1931 Genetics journal scan | Yes (page images are scans; embedded text is degraded) |

The PDFs live under `corpus/bioarxiv/`; outputs land under
`pipeline_output/bench_*/` (gitignored). Re-running the benchmark
regenerates the outputs; this doc summarises them.

## Cross-paper summary

### Wall-clock timing (per-page)

| Tool | MK 1991 (3p) | Mendel (52p) | Wright (63p) | per-page mean |
| --- | --- | --- | --- | --- |
| EasyOCR | 8.0 s/p | 2.4 s/p | 1.6 s/p | ~4 s/p |
| SmolDocling | 22.0 s/p | 9.4 s/p | 10.7 s/p | ~14 s/p |
| Granite | 43.8 s/p | 14.7 s/p | 7.9 s/p | ~22 s/p |
| **Chandra** | **204.6 s/p** | **25.5 s/p** | **33.6 s/p** | **~88 s/p** |

EasyOCR is the throughput winner by a wide margin. Per-page Chandra
cost varies non-linearly with paper size: the 3-page MK run took
10 minutes because per-process model loading dominates short
documents. On longer papers the per-page rate stabilises at
~25-35 s/p.

### Key-facts recovery

Per-paper hand-curated checklists (see `scripts/bench/key_facts.py`),
scored after stripping HTML tags and markdown emphasis from each
tool's output:

| Paper | Chandra | EasyOCR | Granite | SmolDocling |
| --- | --- | --- | --- | --- |
| MK 1991 (10 facts, reversed-PDF rerun) | **10/10** | 9/10 | **10/10** | 5/10 |
| Mendel (10 facts) | **10/10** | **10/10** | 9/10 | **10/10** |
| Wright 1931 (11 facts) | 10/11 | 10/11 | **11/11** | **11/11** |
| **Aggregate** | **30/31** | 29/31 | **30/31** | 26/31 |

Chandra and Granite tied for top at 97 percent. EasyOCR close behind
at 94 percent. SmolDocling lags primarily because of MK 1991, where
it hallucinated `GLYPH` tokens and garbled author names. The
qualitative differentiators (math rendering on Wright, formatting
preservation on MK, sidecar bundle completeness) separate the tools
more sharply than the key-facts checklist on its own.

### Reference-list recovery

`scripts/bench/ref_recall.py` counts distinct numbered references at
line starts:

| Paper | Chandra | EasyOCR | Granite | SmolDocling |
| --- | --- | --- | --- | --- |
| MK 1991 | 18 | 16 | 25 | 10 |
| Mendel | 7 | 11 | 7 | 7 |
| Wright 1931 | 0 | 0 | 0 | 0 |

Wright 1931 returns zero across all four because the 1931 Genetics
journal style uses inline `(Author, year)` citations and an
unnumbered Literature Cited section. Not an OCR failure; the metric
does not generalise to all paper formats.

### Math-notation preservation (the real differentiator)

The key-facts checklist probes textual facts but misses what the
tools actually do with math. `scripts/bench/math_recall.py` counts
inline LaTeX math (`$...$`), block math (`$$...$$` or fenced),
and math symbols (Greek letters + LaTeX commands + operators).
Wright 1931 is the discriminator paper because it is
equation-heavy throughout.

The raw inline-math count overstates what the Docling backends do
with math. Many of their `$...$` spans are broken LaTeX fragments
(`\_{s}`, `^{6}`) where the host variable was dropped by the VLM.
The classifier in `math_recall.score` separates real equations
(have an operator or multiple variables) from broken fragments.

**Wright 1931 (63 pages, math-heavy):**

| Tool | Inline (real) | Inline (broken) | Block math | Math symbols | Equation recall |
| --- | --- | --- | --- | --- | --- |
| **Chandra** | **513** | 213 | 72 | **529** | 6/6 |
| Granite | 12 | 58 | 73 | 248 | 6/6 |
| SmolDocling | 12 | 31 | 66 | 222 | 6/6 |
| EasyOCR | 4 | 0 | 0 | 0 | 4/6 |

**Chandra dominates real equations 42× over Granite** on Wright. The
Docling VLMs (Granite and SmolDocling) emit lots of LaTeX-shaped
spans but the vast majority are mangled subscript/superscript
fragments with the host variable stripped. Examples from Granite
Wright: `\_{ab}`, `\_{a}`, `^{6}`, `^{s}` — un-renderable. Chandra's
spans on the same paper: `[(1-q)a+qA]`, `\Delta q = -uq + v(1-q)`,
`\Delta q = 0` — full equations the way the original paper laid them
out.

**MK 1991:**

| Tool | Inline (real) | Inline (broken) | Math symbols |
| --- | --- | --- | --- |
| **Chandra** | **16** | 2 | 8 |
| Granite | 1 | 17 | 0 |
| EasyOCR | 0 | 0 | 0 |
| SmolDocling | 0 | 0 | 0 |

Granite's 17 broken fragments on a 3-page paper: `\_{s}`, `\_{t}`,
`\_{r}`, `\_{b}`. Chandra's 16 real spans include `G=7.43`,
`P=0.006`, `M_r`, `T_b(\mu/3)M_r`. **Chandra is the only tool that
actually reads math.**

The metric that matters: real equations per page, on math-bearing
material. On Wright, Chandra is ~8 real equations per page; Granite
is 0.2; EasyOCR is 0.06. There is no middle option that closes that
gap.

### Output volume (chars)

| Paper | Chandra | EasyOCR | Granite | SmolDocling |
| --- | --- | --- | --- | --- |
| MK 1991 | 41,654 | 25,236 | 16,405 | 8,372 |
| Mendel | 106,550 | 91,716 | 90,361 | 91,666 |
| Wright 1931 | 161,215 | 144,977 | 148,162 | 127,597 |

Char count is not a reliable quality signal on its own. On MK 1991
Chandra emits 2.5× more text than Granite but the key facts land in
both. On Mendel and Wright (body-text-heavy) the tools converge.

### Sidecar outputs

The MK rerun explicitly requested all formats from every Docling
config to verify parity with Chandra's default bundle. Observed
output sizes (3-page MK paper):

| Tool | `.md` | `.html` | `.json` | Page-region images |
| --- | --- | --- | --- | --- |
| Chandra-OCR-2 | 41.6 KB | 42.7 KB | 0.7 KB (metadata) | Default bundle (WebP per region) |
| Docling + EasyOCR | 25.2 KB | 28.3 KB | 443.6 KB (full layout tree) | On-request via `generate_picture_images` |
| Docling + Granite VLM | 16.4 KB | 20.8 KB | 1711.5 KB (full layout tree) | On-request |
| Docling + SmolDocling VLM | 8.4 KB | 11.7 KB | 1676.1 KB (full layout tree) | On-request |

Docling configs can produce the same sidecar set as Chandra via
explicit `export_to_html()` / `export_to_dict()` calls plus the
`generate_picture_images` pipeline option. Docling's JSON contains
the full layout tree (bounding boxes, element types, hierarchy);
Chandra's metadata JSON is narrower (run-level metadata only). Both
are useful for different downstream needs.

## Per-paper highlights

### MK 1991

- Chandra preserved markdown formatting (italics for journals, bold
  for volumes), correct comma / period distinction, and the key
  Table 2 numerical result (`Replacement Fixed = 7`, `Synonymous = 17`,
  `G = 7.43`, `P = 0.006`).
- Chandra rendered the wide Table 1 as HTML with one row inflated by
  hundreds of `-` cells: a table-reconstruction artifact on very-sparse
  wide tables.
- Granite-Docling matched Chandra on key facts at much lower char
  count; preserved sequential reference order; emitted math subscripts
  as LaTeX.
- EasyOCR was right on character fidelity (`Wolfe`, not `Woife`) but
  jumbled reference order due to Docling's column-major layout pass.
- SmolDocling hallucinated `GLYPHy` / `GLYPHu` tokens and garbled
  author names. The `-preview` suffix in the model name is honest.

### Mendel-Bateson 1866 (1925 reprint)

- All four tools converged on body-text accuracy. The plainer
  typography levels the playing field.
- EasyOCR was the speed winner: 2 minutes vs Chandra's 22 minutes for
  the same 52 pages.
- Reference recall is misleading on this paper (only 7-11 numbered
  refs; Mendel uses inline citations more than a modern paper).

### Wright 1931

- Math notation is the discriminator and **Chandra wins decisively**
  on this paper. Equations and inline math read closest to the
  original typography. Granite is a respectable second: preserves
  most structure but with mild author-name and equation-token
  errors. EasyOCR ducks out of decoding many formulae entirely,
  emitting plain text where math should be. SmolDocling sits closer
  to EasyOCR than to Granite on math fidelity despite the
  key-facts score suggesting otherwise; the key-facts checklist
  doesn't probe equation rendering directly.
- Chandra missed `1931` in the year fact — likely a `1 → I` confusion
  on the journal-header typography. The body content survives.
- EasyOCR missed `University of Chicago` in the affiliation — title-page
  layout issue rather than an OCR character failure.
- For math-heavy papers the routing recommendation is unambiguous:
  **Chandra**. Granite is acceptable when Chandra is unavailable.
  EasyOCR and SmolDocling are not suitable for material with
  substantive equation content.

## Routing decision (pinned in DECISIONS.md)

For digital-text PDFs (embedded text layer): route to Docling
without OCR. No backend choice needed.

For scanned PDFs (image-based, low pdfminer text yield):

1. **Difficult source material** (handwriting, scanned tables that
   matter, math-heavy, multilingual, sparse-OCR-quality scans, or
   downstream needs the sidecar bundle): **Chandra-OCR-2**. Gold
   standard. Pay the ~14× wall-clock cost.
2. **Everything else** (most scanned papers in a corpus, plain
   body-text-heavy material, throughput-sensitive batches):
   **Docling + EasyOCR**. ~14× faster, ties Chandra on key facts on
   this benchmark, no GPU required. **Not suitable for math-heavy
   papers**: EasyOCR ducks out of equation rendering frequently.
3. **GPU present, smaller-model preferred over Chandra**: Docling +
   Granite-Docling 258M. Matches Chandra on key facts, preserves
   math subscripts and sequential reference order. Smaller output;
   skips wide sparse tables, which is arguably correct for
   graph-construction purposes.

Not recommended:

- Docling + RapidOCR with default Chinese mobile models (wrong
  language model for English papers).
- Docling + SmolDocling 256M preview (hallucinated `GLYPH` tokens,
  inconsistent across paper types).

Scanned-detection threshold lives at
`_SCANNED_TEXT_YIELD_PER_PAGE = 200.0` in
`src/nuthatch/ingest/extract.py`. The only verifiably image-only
paper in the test corpus yielded 106 chars/page; everything else
yielded > 1500 chars/page. The threshold sits well above the scanned
floor and well below any plausible digital-text content.

## Licensing posture (orchestrator vs deployer)

nuthatch is a routing layer that calls user-installed OCR backends.
nuthatch does not redistribute model weights and does not run
inference as a service. Backend licences attach to the end user's
runtime, not to nuthatch:

- **Chandra-OCR-2**: OpenRAIL (responsible-AI licence; permits
  commercial + OSS use, prohibits surveillance / weapons / illegal
  use). End user accepts these terms by installing and running
  Chandra. nuthatch's role is configuration and dispatch.
- **Docling + EasyOCR / Granite-Docling / SmolDocling**: all
  permissive (MIT / Apache / similar). End user accepts each by
  installing the corresponding package.

The README documents that users install backends themselves
(`sfw uv add chandra-ocr[hf]`, etc.) and choose their stack. nuthatch
ships Apache 2.0 and remains a permissive orchestrator regardless of
which backends a user runs.

## Reproduction

From the nuthatch repo root, with the venv set up via `uv sync`:

Each paper's full 4-tool run + sidecar exports is captured as a
single Python script in the conversation history; the script writes
to `pipeline_output/bench_{mk,mendel,wright}/{chandra,easyocr,granite,smol}/`
with timing recorded in `timing.json`. The MK paper uses the
page-reversed copy `corpus/bioarxiv/McDonald_and_Kreitman_1991_fixed.pdf`
because the original JSTOR scan is in reverse page order.

For a one-off Chandra run on a single file:

```bash
uv run chandra \
  corpus/bioarxiv/<paper>.pdf \
  ./pipeline_output/<dir> \
  --method hf
```

For a one-off Docling run with EasyOCR + all sidecars:

```bash
uv run python -c "
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
ocr = EasyOcrOptions(lang=['en'])
pipe = PdfPipelineOptions(ocr_options=ocr, do_ocr=True, generate_picture_images=True)
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)})
result = conv.convert('corpus/bioarxiv/<paper>.pdf')
doc = result.document
open('out.md', 'w').write(doc.export_to_markdown())
open('out.html', 'w').write(doc.export_to_html())
open('out.json', 'w').write(json.dumps(doc.export_to_dict(), indent=2, default=str))
"
```

Metric scripts:

```bash
uv run python scripts/bench/ref_recall.py <output.md> [<output.md> ...]
uv run python scripts/bench/key_facts.py --paper {mk1991,mendel,wright1931} <output.md>
```
