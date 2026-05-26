#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""RAGAS-based quality evaluation for a nuthatch corpus.

Runs against any corpus that has completed `ingest` + `embed`. Builds
a synthetic test set, runs the live retrieve + answer pipeline on
each question, scores with RAGAS, writes a markdown report.

Usage:
    python scripts/eval/ragas_eval.py \\
        --corpus demo \\
        --n-questions 30 \\
        --llm-backend ollama \\
        --llm-model minimax-m2.5:cloud \\
        --k 5

Backends (see `llm_backends.py` for full list):
    --llm-backend ollama|openai|anthropic
    --llm-model   <any model id valid for the backend>

Output: `pipeline_output/ragas_report_<UTC>.md` and a side-car
`ragas_dataset_<UTC>.jsonl` capturing the test set used (so a
second eval can be apples-to-apples vs the first).

Metrics reported:
    - context_precision: did retrieved passages contain the GT answer?
    - context_recall:    did retrieved passages cover the GT answer?
    - faithfulness:      did the generated answer stay in-context?
    - answer_relevancy:  did the generated answer address the question?
    - answer_correctness: how close is the generated answer to GT?
    - hit@k:             did the source chunk appear in the top-k? (intrinsic)
    - source_doc_hit@k:  did the source DOC appear in the top-k? (intrinsic)
    - rerank_delta:      pre vs post rerank source-chunk rank shift (intrinsic)

The intrinsic metrics are cheap; the RAGAS metrics need LLM-as-judge
calls per row, so they dominate runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Make this script runnable both as a module and standalone.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from llm_backends import build_llm  # noqa: E402
from rag_pipeline import RAGResult, build_retriever, run_rag_turn  # noqa: E402
from testset_generator import TestExample, generate_test_set, write_testset_jsonl  # noqa: E402


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--corpus", required=True, help="corpus name or path")
    p.add_argument("--n-questions", type=int, default=30)
    p.add_argument("--per-doc-cap", type=int, default=3,
                   help="max questions sampled per source document")
    p.add_argument("--k", type=int, default=5, help="top-k retrieved per query")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm-backend", default="ollama",
                   help="ollama (default) | openai | anthropic")
    p.add_argument("--llm-model", default=None,
                   help="backend-specific model id; defaults per backend")
    p.add_argument("--num-predict", type=int, default=2048,
                   help="answer-token budget (Ollama num_predict / OpenAI max_tokens)")
    p.add_argument("--out-dir", default="pipeline_output",
                   help="where the report + dataset land")
    p.add_argument("--skip-ragas", action="store_true",
                   help="run only the intrinsic metrics (hit@k, rerank-delta) "
                        "and skip the LLM-as-judge RAGAS metrics; useful for "
                        "fast smoke tests")
    args = p.parse_args(argv)

    # Resolve corpus
    from nuthatch.corpus import (
        Registry,
        discover_corpus_root,
        init_corpus,
    )
    from nuthatch.corpus.layout import CorpusLayout

    corpus_root = _resolve_corpus(args.corpus)
    layout = CorpusLayout(root=corpus_root)
    print(f"[eval] corpus: {layout.root}")

    # Pull every chunk from Chroma for sampling
    import chromadb

    client = chromadb.PersistentClient(path=str(layout.embeddings_dir))
    coll = client.get_collection("chunks")
    n_chunks = coll.count()
    print(f"[eval] chunks in store: {n_chunks}")
    if n_chunks == 0:
        print("[eval] no chunks; run `nuthatch embed` first.")
        return 2

    got = coll.get(include=["documents", "metadatas"])
    chunk_records = list(zip(
        got["ids"], got["documents"], got["metadatas"], strict=True,
    ))

    # Build LLMs
    print(f"[eval] LLM backend: {args.llm_backend}; model: "
          f"{args.llm_model or '(default)'}")
    llm = build_llm(
        backend=args.llm_backend,
        model=args.llm_model,
        num_predict=args.num_predict,
        temperature=0.0,
    )

    # Generate test set
    print(f"[eval] generating {args.n_questions} synthetic QA pairs "
          f"(cap {args.per_doc_cap}/doc, seed {args.seed})...")
    t0 = time.perf_counter()

    def _testset_progress(i: int, total: int, cid: str) -> None:
        print(f"  [{i:>3d}/{total}] {cid}", flush=True)

    examples = generate_test_set(
        chunk_records,
        llm,
        n=args.n_questions,
        per_doc_cap=args.per_doc_cap,
        seed=args.seed,
        on_progress=_testset_progress,
    )
    print(f"[eval] generated {len(examples)} usable examples "
          f"({time.perf_counter() - t0:.1f}s)")
    if not examples:
        print("[eval] no test examples generated; aborting.")
        return 3

    tag = _now_tag()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / f"ragas_dataset_{tag}.jsonl"
    write_testset_jsonl(examples, dataset_path)
    print(f"[eval] testset persisted: {dataset_path}")

    # Run RAG pipeline
    print(f"[eval] running RAG pipeline (k={args.k})...")
    retriever = build_retriever(layout)
    t0 = time.perf_counter()
    rag_results: list[RAGResult] = []
    for i, ex in enumerate(examples, start=1):
        print(f"  [{i:>3d}/{len(examples)}] {ex.question[:60]}...", flush=True)
        rag_results.append(run_rag_turn(
            ex.question, retriever=retriever, answerer_llm=llm, k=args.k,
        ))
    print(f"[eval] RAG pipeline done ({time.perf_counter() - t0:.1f}s)")

    # Intrinsic metrics (cheap, always run)
    print("[eval] computing intrinsic metrics...")
    intrinsic = _compute_intrinsic(examples, rag_results)
    for k, v in intrinsic.items():
        print(f"  {k:24s} {v:.3f}")

    # RAGAS metrics (LLM-as-judge, optional)
    ragas_scores: dict[str, float] = {}
    if not args.skip_ragas:
        print("[eval] running RAGAS evaluators (this calls the LLM "
              "per-row per-metric)...")
        ragas_scores = _run_ragas(examples, rag_results, llm)
        for k, v in ragas_scores.items():
            print(f"  {k:24s} {v:.3f}")

    # Write report
    report_path = out_dir / f"ragas_report_{tag}.md"
    _write_report(
        report_path=report_path,
        layout=layout,
        args=args,
        examples=examples,
        rag_results=rag_results,
        intrinsic=intrinsic,
        ragas_scores=ragas_scores,
        dataset_path=dataset_path,
        n_chunks=n_chunks,
    )
    print(f"[eval] report: {report_path}")
    return 0


def _resolve_corpus(name_or_path: str) -> Path:
    """Mirror the CLI resolver: try registry first, then path."""
    from nuthatch.corpus import Registry

    try:
        registry = Registry.load()
        entry = registry.get(name_or_path)
        if entry is not None:
            return Path(entry).resolve()
    except Exception:
        pass
    p = Path(name_or_path).resolve()
    if (p / ".kg").is_dir():
        return p
    raise SystemExit(f"corpus not found: {name_or_path!r}")


def _compute_intrinsic(
    examples: list[TestExample], rag_results: list[RAGResult],
) -> dict[str, float]:
    """Cheap retrieval-quality metrics that don't need an LLM judge.

    Source-chunk hit@k: did the chunk we generated the question from
    appear in the top-k? This is a strict signal (chunk granularity).

    Source-doc hit@k: did at least one chunk from the SAME doc appear?
    Looser; closer to the real-world success criterion ("the agent
    found the right paper, even if a different chunk of it").
    """
    n = len(examples)
    chunk_hits = 0
    doc_hits = 0
    reciprocal_ranks: list[float] = []
    for ex, rag in zip(examples, rag_results, strict=True):
        chunk_rank = None
        if ex.source_chunk_id in rag.retrieved_chunk_ids:
            chunk_hits += 1
            chunk_rank = rag.retrieved_chunk_ids.index(ex.source_chunk_id) + 1
        if ex.source_doc_id in rag.retrieved_doc_ids:
            doc_hits += 1
            if chunk_rank is None:
                doc_rank = rag.retrieved_doc_ids.index(ex.source_doc_id) + 1
                chunk_rank = doc_rank
        reciprocal_ranks.append(1.0 / chunk_rank if chunk_rank else 0.0)
    return {
        "source_chunk_hit@k": chunk_hits / n if n else 0.0,
        "source_doc_hit@k": doc_hits / n if n else 0.0,
        "MRR": sum(reciprocal_ranks) / n if n else 0.0,
    }


def _run_ragas(
    examples: list[TestExample],
    rag_results: list[RAGResult],
    llm: object,
) -> dict[str, float]:
    """RAGAS LLM-as-judge metrics."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    data = {
        "question": [ex.question for ex in examples],
        "ground_truth": [ex.ground_truth for ex in examples],
        "answer": [r.answer for r in rag_results],
        "contexts": [r.retrieved_contexts for r in rag_results],
    }
    ds = Dataset.from_dict(data)
    ragas_llm = LangchainLLMWrapper(llm)
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]
    result = evaluate(ds, metrics=metrics, llm=ragas_llm)
    return {k: float(v) for k, v in result.items() if isinstance(v, int | float)}


def _write_report(
    *,
    report_path: Path,
    layout: object,
    args: argparse.Namespace,
    examples: list[TestExample],
    rag_results: list[RAGResult],
    intrinsic: dict[str, float],
    ragas_scores: dict[str, float],
    dataset_path: Path,
    n_chunks: int,
) -> None:
    """Write a self-contained markdown report."""
    lines: list[str] = []
    lines.append("---")
    lines.append(f'title: "RAGAS evaluation: {layout.root.name}"')
    lines.append(f"date: {datetime.now(UTC).isoformat()}")
    lines.append("---")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Corpus: `{layout.root}`")
    lines.append(f"- Chunks in store: {n_chunks}")
    lines.append(f"- Test examples generated: {len(examples)}")
    lines.append(f"- Retrieval top-k: {args.k}")
    lines.append(f"- LLM backend: `{args.llm_backend}`")
    lines.append(f"- LLM model: `{args.llm_model or '(backend default)'}`")
    lines.append(f"- Sampling seed: {args.seed}")
    lines.append(f"- Per-doc question cap: {args.per_doc_cap}")
    lines.append(f"- Test set (reproducibility): `{dataset_path.name}`")
    lines.append("")
    lines.append("## Intrinsic metrics (no LLM judge)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    for k, v in intrinsic.items():
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")
    if ragas_scores:
        lines.append("## RAGAS metrics (LLM-as-judge)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | ---: |")
        for k, v in ragas_scores.items():
            lines.append(f"| {k} | {v:.3f} |")
        lines.append("")
    else:
        lines.append("## RAGAS metrics")
        lines.append("")
        lines.append("_skipped via `--skip-ragas`._")
        lines.append("")
    lines.append("## Sample per-question outcomes (first 5)")
    lines.append("")
    for ex, rag in list(zip(examples, rag_results, strict=True))[:5]:
        lines.append(f"### Q: {ex.question}")
        lines.append("")
        lines.append(f"- **Ground truth**: {ex.ground_truth}")
        lines.append(f"- **Generated**: {rag.answer}")
        lines.append(f"- **Source chunk**: `{ex.source_chunk_id}` (doc `{ex.source_doc_id}`)")
        lines.append(f"- **Retrieved doc_ids (top-{args.k})**: " +
                     ", ".join(f"`{d}`" for d in rag.retrieved_doc_ids))
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
