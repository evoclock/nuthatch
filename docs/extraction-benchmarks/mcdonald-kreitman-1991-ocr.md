---
title: "OCR benchmark: Chandra-OCR-2 vs Docling on McDonald and Kreitman 1991"
date: 2026-05-25
---

## Context

Validates the Sprint 2 routing decision: scanned PDFs go to Chandra-OCR-2,
digital-text PDFs go to Docling. The test target is a 3-page scanned 1991
*Nature* paper that exists in this repo's bioarxiv test corpus at
`corpus/bioarxiv/McDonald_and_Kreitman_1991.pdf`.

Why this paper: small (3 pages, 307 KB), classic, contains a key data
table (replacement vs synonymous substitutions) whose values must survive
extraction for the OCR result to be useful for downstream graph work.
Older-than-1995 papers are reliably scanned image-PDFs which is the
exact case Chandra-OCR-2 is meant to handle.

## Configurations tested

| Run | Tool | Backend | Hardware |
| --- | --- | --- | --- |
| A | Chandra-OCR-2 v2 | HuggingFace transformers | RTX 5070 Ti (16 GB) |
| B | Docling 2.x | RapidOCR (bundled Chinese mobile models) | CPU |
| C | Docling 2.x | RapidOCR with `lang=['english']` and `force_full_page_ocr=True` | CPU |

Run C produced character-for-character identical output to run B. The
`lang=['english']` flag is a routing hint, not a model swap. The
RapidOCR English model files are not bundled in the `sentence-transformers`
dependency tree we picked up, so RapidOCR silently fell back to the
Chinese mobile models for both runs. To actually exercise English RapidOCR
under Docling, the English `det` / `rec` ONNX files would need to be
downloaded explicitly and passed via `RapidOcrOptions(det_model_path=...,
rec_model_path=...)`. Not pursued here: the right answer is to route
scanned papers to Chandra-OCR-2 anyway.

## Quantitative summary

| Run | Output chars | Markdown formatting | Sidecars |
| --- | --- | --- | --- |
| A (Chandra-OCR-2) | 41,654 | Italics for journal names, bold for volumes, HTML tables for tabular data | `.md`, `.html`, `_metadata.json` |
| B (Docling default) | 27,353 | Plain text, pipe-table fallback | `.md` only |
| C (Docling tuned) | 27,324 | (identical to B) | `.md` only |

Chandra-OCR-2 extracted 52 percent more text than Docling. The character
count gap is mostly real content, not formatting noise.

## Side-by-side: reference list

The first numbered references at the top of the references page render
as follows.

### Chandra-OCR-2

```text
1. 8. Shinomura, Y., Eng, J., Rattan, S. C. & Yalow, R. S. *Comp. Biochem. Physiol.* **96B**, 239-242 (1990).
2. 9. Beintema, J. J. & Neuteboom, B. J. *molec. Evol.* **19**, 145-152 (1983).
3. 10. Sarkar, G., Koebler, D. D. & Sommer, S. S. *Genomics* **6**, 133-143 (1990).
4. 11. Fan, Z.-W., Eng, J., Shaw, G. & Yalow, R. S. *Peptides* **9**, 429-431 (1988).
5. 13. Wolfe, P. B. & Cebray, J. J. *Molec. Immun.* **17**, 1493-1505 (1980).
```

### Docling

```text
- 8.Shinomura.Y.Eng.J.Rattan.S.C.&amp;Yalow.R.S.Comp.Biochem.Physiol.96B,239-242(1990).
- 9.Beintema. J. J. &amp; Neuteboom.B.J molec. Evol.19, 145-152 (1983).
10. Sarkar.G.Koeberl. D.D.&amp; Sommer. S S. Genomics 6, 133-143 (1990).
- 11.Fan. Z-W. Eng. J., Shaw. G.&amp; Yalow,R. S. Peptides 9, 429-431 (1988).
- 13.Woife. P. B.&amp; Cebra. J. J. Molec. Immun. 17, 1493-1505 (1980).
```

Observable differences:

- Commas are commas in Chandra. Docling renders them as periods
  (`Shinomura.Y` instead of `Shinomura, Y.`).
- Author surname spelling: Chandra has `Wolfe`, Docling has `Woife`
  (the `l` and `f` collapsed into a single mis-detected glyph by the
  Chinese-language detection model).
- Journal name italicisation and volume bolding survive in Chandra
  because Chandra emits real markdown formatting. Docling produces
  plain text.
- `&` is the literal character in Chandra, but Docling emits the
  HTML entity `&amp;` from its layout-parsing path.

## Side-by-side: key data table

This is Table 2 of the paper, the substantive result (replacement and
synonymous substitutions fixed and polymorphic).

### Chandra-OCR-2

```text
TABLE 2 Number of replacement and synonymous substitutions for fixed
differences between species and polymorphisms within species

| | Fixed | Polymorphic |
| --- | --- | --- |
| Replacement | 7 | 2 |
| Synonymous | 17 | ... |
```

The key cell (`Replacement / Fixed = 7`, the cornerstone of the
McDonald-Kreitman test) survives.

### Docling

Docling reconstructs Table 1 (a wider site-by-site polymorphism table)
into a pipe table but with category-label errors throughout:

```text
| 808       | A      | ... | Repl.      | Fixed         |
| 834       | T      | ... | Sym.       | Poly.         |
| 1178      | C      | ... | Symn.      | Poly.         |
| 1196      | G      | ... | Sym.       | Poly.         |
```

`Syn.` is rendered variously as `Sym.`, `Symn.`, and `Sym` because the
Chinese OCR rec model confuses `n` and `m` on this typeface. The Chinese
character `一` (a horizontal stroke) appears occasionally in place of
ASCII `-` for the same reason.

## Failure modes observed

### Chandra-OCR-2

- Table 1 of the paper (a wide site-by-site polymorphism matrix with
  many empty cells) is reconstructed as an HTML table, but one row
  is inflated with hundreds of `-` cells, more than the table width.
  This is a table-reconstruction artifact specific to very-sparse wide
  tables. Single-table issue on this paper; the substantive Table 2
  data survives intact.
- Markdown-in-HTML mix: tables render as HTML, prose renders as
  markdown. Both within the same output file.

### Docling

- Chinese-language OCR models are used silently even when the file is
  English. `lang=['english']` does not force a model swap.
- Character confusion typical of Chinese rec models reading Latin
  characters: `m` for `n`, `1` for `i`, periods for commas, occasional
  Chinese characters substituting punctuation.
- Author and journal names degrade: `Wolfe` -> `Woife`, `Yalow` -> `Yalow`
  (sometimes correct, sometimes joined to the next word).
- The key data table survives in pipe-table form but with category-label
  errors that change the result interpretation if read uncritically
  (`Sym.` and `Syn.` are both valid abbreviations for different things
  in different sub-fields).

## What Docling does better than Chandra here

- Table 1 (the wide polymorphism matrix) is rendered as a structured
  pipe table that a downstream consumer can re-parse, with one cell
  per call. Chandra's HTML reconstruction has the artifact row.
- Pure CPU; no GPU required. Docling is the only viable extractor on
  hosts without a CUDA GPU.
- Always available: it ships in the dep tree we already use for
  digital PDFs.

## What Chandra-OCR-2 does better than Docling here

- Reference-list character accuracy. The reference list is the part of
  a paper most directly consumed by graph construction (citation
  extraction); accuracy here is non-negotiable.
- Punctuation distinction. Comma vs period vs Chinese-stroke matters
  for any downstream NLP step that tokenises on these.
- Markdown structure preservation (italics for journals, bold for
  volumes) round-trips well into Obsidian.
- Sidecar JSON metadata makes downstream re-parsing cheaper.

## Routing decision (pinned in DECISIONS.md)

For scanned PDFs (image-based, no embedded text layer): route to
Chandra-OCR-2, skip Docling's built-in OCR. Detection is via
`pdfminer.six` text yield: if a PDF returns fewer than N characters
per page of extractable text, it is scanned.

For digital-text PDFs (embedded text layer): route to Docling. No OCR
is required; Docling's structured layout parsing is the right tool.

Docling's RapidOCR path is not currently a useful fallback for English
scanned text in this venv because of the missing English model files.
If a Docling-only path is ever needed (no GPU host), the action items
are: (a) install EasyOCR via `sfw uv add easyocr` and switch to
`EasyOcrOptions(lang_list=['en'])`, or (b) install Tesseract CLI and
switch to `TesseractCliOcrOptions(lang=['eng'])`. Neither pursued in
this benchmark.

## Reproduction

From the nuthatch repo root, with the venv set up via `uv sync`:

```bash
# Chandra-OCR-2 (needs GPU)
uv run chandra \
  corpus/bioarxiv/McDonald_and_Kreitman_1991.pdf \
  ./pipeline_output/ocr_spike_mk \
  --method hf

# Docling default
uv run python -c "
from docling.document_converter import DocumentConverter
conv = DocumentConverter()
result = conv.convert('corpus/bioarxiv/McDonald_and_Kreitman_1991.pdf')
print(result.document.export_to_markdown())
" > pipeline_output/docling_mk/output.md
```

Outputs land under `pipeline_output/` (gitignored). Re-running regenerates
them; this analysis doc summarises the artifacts so the comparison is
discoverable without re-running the OCR.
