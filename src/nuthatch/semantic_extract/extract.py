#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Extract conceptual entities (topics, methods, genes) from already-extracted docs.

The current ingest pipeline only extracts bibliographic entities
(authors, citations) per document. For a mixed AI/ML/biology corpus
this means the knowledge graph captures the bibliographic network
("who cites whom") not the conceptual network ("which papers share
ideas"), which is exactly what GraphRAG-style community detection
needs to be useful.

This script is the bridge: it reads every `inputs/.kg/extracted/<doc_id>.md`
body, asks a fast LLM to extract a structured list of topics + methods
+ genes/models present in the text, and writes a sidecar
`inputs/.kg/extracted/<doc_id>.concepts.json` per doc.

The graph augmenter (`scripts/concepts/augment_graph.py`) then folds
those concepts in as topic/method/gene nodes with `mentioned_in`
edges to the source documents. Re-clustering on the augmented graph
yields communities by SHARED CONCEPTS rather than shared authorship.

Usage:
    python scripts/concepts/extract_concepts.py --corpus inputs \\
        --backend ollama --model gemini-3-flash-preview:cloud \\
        --max-docs 0

Output: one `<doc_id>.concepts.json` per processed doc, idempotent
(skips docs that already have a sidecar unless `--force` is passed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

from nuthatch.util import parse_llm_json, resolve_corpus
from nuthatch.util.llm_backends import build_llm

_CONCEPT_PROMPT = """\
You are building a wiki-style card for a research paper. Given the \
excerpt below, produce a structured JSON with both a semantic summary \
and the conceptual entities discussed.

Five fields:

- "summary": 2-4 sentences describing WHAT the paper does and WHAT it \
shows. Write in the present tense, neutral voice. Avoid title \
restatement; assume the reader already knows the title. This is the \
primary clustering primitive - papers with similar summaries should \
cluster together by topic.

- "topics": broad themes or research areas. Examples: "circadian \
rhythms", "reinforcement learning", "phylogenomics", "transformer \
architectures". 5-10 items, lowercase, multi-word.

- "methods": named techniques, algorithms, tools, or analytical \
pipelines. Examples: "BLAST", "PCR", "PPO", "transformer", "MAFFT", \
"Leiden community detection". 3-7 items, preserve canonical \
capitalisation.

- "named_entities": specific named entities of scientific significance \
(genes, proteins, model architectures, datasets, organisms). Examples: \
"Per1", "Bmal1", "GPT-4", "ImageNet", "Mus musculus", "Drosophila \
melanogaster", "TP53". 0-10 items; omit if none apply to this paper.

- "domain": one of {{"ai_ml", "biology", "popgen", "bioinformatics", \
"phylogenetics", "neuroscience", "chemistry", "physics", "other"}}. \
The single best-fit domain label. Use "other" if no fit.

Be liberal about extracting topics + methods (a paper usually has both). \
Be conservative about named_entities (only those the paper actually \
discusses, not those mentioned in passing or in citations).

Refusal is allowed: if the excerpt is too short, boilerplate, or has \
no extractable content, respond with:
{{"refusal": "<short reason>"}}

Otherwise respond with valid JSON in exactly this shape:
{{
  "summary": "...",
  "topics": ["...", "..."],
  "methods": ["...", "..."],
  "named_entities": ["...", "..."],
  "domain": "..."
}}

Do not include code fences, prefix, or any text outside the JSON.

PAPER EXCERPT:
\"\"\"
{excerpt}
\"\"\"
"""


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
    )
    p.add_argument("--corpus", required=True, help="corpus name or path")
    p.add_argument(
        "--backend", default="ollama", help="LLM backend for concept extraction (default: ollama)"
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview:cloud",
        help="model id; default is Gemini Flash cloud for speed",
    )
    p.add_argument(
        "--num-predict",
        type=int,
        default=3072,
        help="token budget for the LLM. Default 3072 accommodates "
        "a 2-4 sentence summary + 5-10 topics + 3-7 methods + "
        "named_entities without truncating the JSON.",
    )
    p.add_argument(
        "--max-docs", type=int, default=0, help="process only the first N docs (0 = all)"
    )
    p.add_argument(
        "--excerpt-chars",
        type=int,
        default=6000,
        help="how much of each body to send to the LLM. Bigger "
        "captures more concepts but slows extraction.",
    )
    p.add_argument(
        "--force", action="store_true", help="re-extract even if a concepts sidecar already exists"
    )
    args = p.parse_args(argv)

    from nuthatch.corpus.layout import CorpusLayout

    corpus_root = resolve_corpus(args.corpus)
    layout = CorpusLayout(root=corpus_root)
    extracted_dir = layout.extracted_dir
    print(f"[concepts] corpus: {layout.root}")
    print(f"[concepts] extracted dir: {extracted_dir}")

    doc_files = sorted(extracted_dir.glob("*.md"))
    if args.max_docs > 0:
        doc_files = doc_files[: args.max_docs]
    print(f"[concepts] candidates: {len(doc_files)} docs")

    print(f"[concepts] LLM: {args.backend}::{args.model}")
    llm = build_llm(
        backend=args.backend,
        model=args.model,
        num_predict=args.num_predict,
        temperature=0.0,
    )

    t0 = time.perf_counter()
    n_processed = 0
    n_skipped = 0
    n_failed = 0
    for i, md_path in enumerate(doc_files, start=1):
        doc_id = md_path.stem
        sidecar = extracted_dir / f"{doc_id}.concepts.json"
        if sidecar.exists() and not args.force:
            n_skipped += 1
            continue

        body = md_path.read_text(encoding="utf-8", errors="ignore")
        excerpt = body[: args.excerpt_chars]
        if len(excerpt.strip()) < 200:
            print(f"  [{i:>4d}/{len(doc_files)}] SKIP {doc_id} (body too short)")
            n_skipped += 1
            continue

        prompt = _CONCEPT_PROMPT.format(excerpt=excerpt)
        print(f"  [{i:>4d}/{len(doc_files)}] {doc_id[:60]}...", flush=True)
        try:
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", str(resp)).strip()
        except Exception as exc:
            print(f"    WARN: LLM call failed: {exc!s}")
            n_failed += 1
            continue

        parsed = parse_llm_json(content)
        if parsed is None:
            print("    WARN: malformed JSON from LLM")
            n_failed += 1
            continue
        # Refusal is a structured outcome, not a failure — honoured silently.
        if "refusal" in parsed and "summary" not in parsed:
            n_skipped += 1
            continue

        summary = str(parsed.get("summary") or "").strip()
        topics = _clean_list(parsed.get("topics"))
        methods = _clean_list(parsed.get("methods"))
        named = _clean_list(parsed.get("named_entities"))
        domain = str(parsed.get("domain") or "other").strip().lower()
        if not summary:
            print("    WARN: no summary returned, skipping")
            n_failed += 1
            continue
        out = {
            "doc_id": doc_id,
            "extracted_at": datetime.now(UTC).isoformat(),
            "extractor_model": f"{args.backend}::{args.model}",
            "extractor_version": "2",
            "summary": summary,
            "topics": topics,
            "methods": methods,
            "named_entities": named,
            "domain": domain,
        }
        sidecar.write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n_processed += 1

    dt = time.perf_counter() - t0
    print(
        f"[concepts] done: {n_processed} processed, {n_skipped} skipped, "
        f"{n_failed} failed ({dt:.1f}s)"
    )
    return 0 if n_failed == 0 else 4


def _clean_list(value: object) -> list[str]:
    """Normalise an LLM-returned list: strip, dedupe, drop empties."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
