#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""RAGAS-based quality evaluation for a nuthatch corpus.

Runs against any corpus that has completed `ingest` + `embed`. Builds
a synthetic test set, runs the live retrieve + answer pipeline on
each question, scores with RAGAS, writes a markdown report.

Usage (defaults: minimax-m2.5:cloud generates + answers; gpt-oss judges):
    python scripts/eval/ragas_eval.py \\
        --corpus inputs \\
        --n-questions 100 \\
        --k 5

Override roles (any langchain-compatible backend + model):
    --gen-backend  ollama  --gen-model  minimax-m2.5:cloud   # generator + answerer
    --judge-backend ollama --judge-model gpt-oss:120b-cloud  # RAGAS judge

The generator + judge are kept on DIFFERENT models on purpose: the
judge grading the same model that wrote the answer is self-preference
bias that inflates faithfulness / answer_correctness. Using a different
family for the judge is the cheapest path to a defensible eval.

Output: `pipeline_output/ragas_report_<UTC>.md` plus a side-car
`ragas_dataset_<UTC>.jsonl` so a future run can reproduce or compare.

Metric trust tiers (the report leads with the trustworthy ones):

Tier 1 - derived ground truth (no LLM judge):
    - source_chunk_hit@k:  did the seeding chunk land in top-k?
    - source_doc_hit@k:    did the seeding DOC land in top-k?
    - MRR:                 mean reciprocal rank of the seeding chunk.

Tier 2 - LLM-judged, two-model split (judge != generator):
    - context_precision:   were retrieved passages relevant?
    - context_recall:      did retrieved passages cover the GT?
    - faithfulness:        did the answer stay grounded in context?
    - answer_relevancy:    did the answer address the question?
    - answer_correctness:  how close is the answer to GT?

Caveat: ground_truth itself is LLM-synthesised, so answer_correctness
is partially circular even with a different judge. Use the intrinsic
tier for the headline "are the embeddings fit for purpose" number.
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--n-questions", type=int, default=100)
    p.add_argument("--per-doc-cap", type=int, default=3,
                   help="max questions sampled per source document")
    p.add_argument("--k", type=int, default=5, help="top-k retrieved per query")
    p.add_argument("--seed", type=int, default=0)

    # Generator + answerer role (default: Minimax M2.5 cloud)
    p.add_argument("--gen-backend", default="ollama",
                   help="generator+answerer backend: ollama | openai | anthropic")
    p.add_argument("--gen-model", default="minimax-m2.5:cloud",
                   help="generator+answerer model id")
    p.add_argument("--gen-num-predict", type=int, default=2048,
                   help="answer token budget for generator+answerer")

    # Judge role (default: gpt-oss 120b cloud, a different family from Minimax)
    p.add_argument("--judge-backend", default="ollama",
                   help="judge backend: ollama | openai | anthropic")
    p.add_argument("--judge-model", default="gpt-oss:120b-cloud",
                   help="judge model id; should differ from --gen-model")
    p.add_argument("--judge-num-predict", type=int, default=1024,
                   help="token budget for judge calls")

    p.add_argument("--out-dir", default="pipeline_output",
                   help="where the report + dataset land")
    p.add_argument("--skip-ragas", action="store_true",
                   help="run only the intrinsic metrics; skip LLM-as-judge "
                        "RAGAS metrics. Useful for fast smoke tests.")

    # Back-compat: keep old --llm-backend / --llm-model as aliases for the
    # generator. Any session that still passes them works without surprise.
    p.add_argument("--llm-backend", default=None,
                   help="(deprecated) alias for --gen-backend")
    p.add_argument("--llm-model", default=None,
                   help="(deprecated) alias for --gen-model")
    p.add_argument("--num-predict", type=int, default=None,
                   help="(deprecated) alias for --gen-num-predict")

    args = p.parse_args(argv)

    # Honour the deprecated aliases when set.
    if args.llm_backend is not None:
        args.gen_backend = args.llm_backend
    if args.llm_model is not None:
        args.gen_model = args.llm_model
    if args.num_predict is not None:
        args.gen_num_predict = args.num_predict

    # Resolve corpus
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

    # Build LLMs (generator + judge, different by default)
    print(f"[eval] generator: {args.gen_backend}::{args.gen_model}")
    gen_llm = build_llm(
        backend=args.gen_backend,
        model=args.gen_model,
        num_predict=args.gen_num_predict,
        temperature=0.0,
    )
    judge_llm = None
    if not args.skip_ragas:
        if (
            args.judge_backend == args.gen_backend
            and args.judge_model == args.gen_model
        ):
            print(
                "[eval] WARNING: judge model identical to generator; this "
                "is self-grading. Use a different --judge-model for "
                "calibrated RAGAS scores.",
            )
        print(f"[eval] judge:     {args.judge_backend}::{args.judge_model}")
        judge_llm = build_llm(
            backend=args.judge_backend,
            model=args.judge_model,
            num_predict=args.judge_num_predict,
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
        gen_llm,
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
            ex.question, retriever=retriever, answerer_llm=gen_llm, k=args.k,
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
        print("[eval] running RAGAS evaluators "
              "(judge LLM per row per metric)...")
        ragas_scores = _run_ragas(examples, rag_results, judge_llm)
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
    """Tier-1 retrieval-quality metrics that don't need an LLM judge.

    These compare retrieval against a derived ground truth: the chunk
    the question was generated from. No LLM is in the loop, so the
    numbers are calibrated by construction.

    source_chunk_hit@k - did the seeding chunk appear in the top-k?
                        Strict: chunk-id match.

    source_doc_hit@k -  did any chunk from the seeding DOC appear?
                        Looser: doc-id match. Closer to the real-world
                        success criterion ("found the right paper").

    MRR -               mean reciprocal rank of the seeding chunk
                        (falls back to the seeding doc's earliest
                        chunk if the exact chunk is missing).
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


def _retrieval_failures(
    examples: list[TestExample], rag_results: list[RAGResult],
) -> list[tuple[TestExample, RAGResult]]:
    """Questions where the seeding chunk did NOT make the top-k.

    These are the diagnostic cases: a human reading the report can
    inspect each one and decide whether retrieval is missing a real
    semantic signal, or whether the question itself is degenerate.
    """
    out: list[tuple[TestExample, RAGResult]] = []
    for ex, rag in zip(examples, rag_results, strict=True):
        if ex.source_chunk_id not in rag.retrieved_chunk_ids:
            out.append((ex, rag))
    return out


def _run_ragas(
    examples: list[TestExample],
    rag_results: list[RAGResult],
    judge_llm: object,
) -> dict[str, float]:
    """RAGAS LLM-as-judge metrics.

    Uses a nuthatch-backed embeddings adapter so RAGAS does not reach
    for OpenAI's `text-embedding-ada-002` (the default that crashes
    without `OPENAI_API_KEY`). The judge LLM is separate from the
    generator LLM by default to avoid self-preference bias.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from ragas_embeddings import NuthatchEmbeddings

    data = {
        "question": [ex.question for ex in examples],
        "ground_truth": [ex.ground_truth for ex in examples],
        "answer": [r.answer for r in rag_results],
        "contexts": [r.retrieved_contexts for r in rag_results],
    }
    ds = Dataset.from_dict(data)
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeds = LangchainEmbeddingsWrapper(NuthatchEmbeddings())
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]
    result = evaluate(
        ds, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeds,
    )
    # RAGAS 0.4.x returns an `EvaluationResult`, not a dict. The
    # per-metric means live in `_repr_dict` (private but stable since
    # 0.4) and the per-row scores in `_scores_dict`. We pick the
    # per-metric means; fall back to averaging the per-row lists if
    # the private attribute moves.
    repr_dict = getattr(result, "_repr_dict", None)
    if repr_dict is not None:
        return {
            str(k): float(v)
            for k, v in repr_dict.items()
            if isinstance(v, int | float)
        }
    scores_dict = getattr(result, "_scores_dict", {})
    out: dict[str, float] = {}
    for k, vals in scores_dict.items():
        clean = [float(v) for v in vals if isinstance(v, int | float)]
        if clean:
            out[str(k)] = sum(clean) / len(clean)
    return out


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
    """Write a self-contained markdown report.

    Layout, in trust order:
        1. How to read these numbers (the caveats, up front)
        2. Run configuration (reproducibility)
        3. Intrinsic metrics (Tier 1: derived ground truth)
        4. Retrieval-failure diagnostic (per-question, Tier 1)
        5. RAGAS metrics (Tier 2: LLM-judged, with caveats inline)
        6. Sample per-question outcomes (qualitative, first 5)
    """
    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(f'title: "RAGAS evaluation: {layout.root.name}"')
    lines.append(f"date: {datetime.now(UTC).isoformat()}")
    lines.append("---")
    lines.append("")

    # Section 1: How to read these numbers
    lines.append("## How to read these numbers")
    lines.append("")
    lines.append(
        "Metrics below are split into two trust tiers. Read the intrinsic "
        "tier as the headline; the RAGAS-judged tier is directional and "
        "carries the caveats noted inline.",
    )
    lines.append("")
    lines.append("- **Intrinsic (derived ground truth, no LLM judge)** -")
    lines.append(
        "  the question was generated from a known chunk; we measure "
        "whether retrieval surfaced that chunk (or at least the right "
        "paper). No LLM is in the scoring loop, so these numbers are "
        "calibrated by construction. **This is the headline for "
        '"are the embeddings fit for purpose."**',
    )
    lines.append("- **LLM-judged (RAGAS, two-model split)** -")
    lines.append(
        "  the answer + retrieved contexts are scored by a separate "
        "judge model (different family from the generator to reduce "
        "self-preference bias). Useful for hallucination and grounding "
        "signal; less reliable than the intrinsic tier.",
    )
    lines.append("")
    lines.append("Known caveats baked into this run:")
    lines.append("")
    lines.append(
        "1. The synthetic test set is LLM-generated, not human-curated. "
        "Questions skew toward lexical surface features of the seeding "
        "chunk, which inflates retrieval hit rates relative to a "
        "human-written set.",
    )
    lines.append(
        "2. Every question is single-chunk and single-document. The "
        "community-aware retrieval surface (community.search, "
        "community.brief) is NOT exercised by this eval; it needs a "
        "multi-hop test set built separately.",
    )
    lines.append(
        "3. `answer_correctness` compares the generated answer to an "
        "LLM-synthesised ground_truth. Even with a separate judge, this "
        "metric is partially circular. Treat as directional only.",
    )
    lines.append(
        f"4. n = {len(examples)} questions; standard error on a "
        "[0, 1] metric at this n is roughly +/- 0.05. Two runs with "
        "different seeds can disagree by that much and both be "
        "statistically consistent.",
    )
    lines.append("")

    # Section 2: Run configuration
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Corpus: `{layout.root}`")
    lines.append(f"- Chunks in store: {n_chunks}")
    lines.append(f"- Test examples generated: {len(examples)}")
    lines.append(f"- Retrieval top-k: {args.k}")
    lines.append(f"- Generator + answerer: `{args.gen_backend}::{args.gen_model}`")
    if ragas_scores:
        lines.append(f"- Judge: `{args.judge_backend}::{args.judge_model}`")
    else:
        lines.append("- Judge: _skipped via `--skip-ragas`_")
    lines.append(f"- Embeddings (retriever + RAGAS): nuthatch BGE-M3")
    lines.append(f"- Sampling seed: {args.seed}")
    lines.append(f"- Per-doc question cap: {args.per_doc_cap}")
    lines.append(f"- Test set (reproducibility): `{dataset_path.name}`")
    lines.append("")

    # Section 3: Intrinsic metrics
    lines.append("## Intrinsic metrics (Tier 1: derived ground truth)")
    lines.append("")
    lines.append("| metric | value | reads as |")
    lines.append("| --- | ---: | --- |")
    intrinsic_legend = {
        "source_chunk_hit@k": "fraction of questions where the seeding chunk appeared in top-k",
        "source_doc_hit@k": "fraction where any chunk from the seeding paper appeared in top-k",
        "MRR": "mean reciprocal rank of the seeding chunk; 1.0 = always first, 0.0 = never",
    }
    for k, v in intrinsic.items():
        legend = intrinsic_legend.get(k, "")
        lines.append(f"| `{k}` | {v:.3f} | {legend} |")
    lines.append("")

    # Section 4: Retrieval-failure diagnostic
    failures = _retrieval_failures(examples, rag_results)
    lines.append("## Retrieval-failure diagnostic (Tier 1)")
    lines.append("")
    if not failures:
        lines.append(
            "_No retrieval failures: every seeding chunk landed in top-k._",
        )
    else:
        lines.append(
            f"{len(failures)} of {len(examples)} questions did not "
            "retrieve the seeding chunk. Inspect each: retrieval gap, "
            "or degenerate question?",
        )
        lines.append("")
        for ex, rag in failures:
            lines.append(f"### Q: {ex.question}")
            lines.append("")
            lines.append(f"- **Seeding chunk**: `{ex.source_chunk_id}` "
                         f"(doc `{ex.source_doc_id}`)")
            seed_doc_in_topk = ex.source_doc_id in rag.retrieved_doc_ids
            if seed_doc_in_topk:
                lines.append(
                    "- **Seeding doc landed top-k**: yes (different chunk)",
                )
            else:
                lines.append(
                    "- **Seeding doc landed top-k**: no (whole-paper miss)",
                )
            lines.append(
                "- **Top-k doc_ids retrieved**: "
                + ", ".join(f"`{d}`" for d in rag.retrieved_doc_ids),
            )
            lines.append("")
    lines.append("")

    # Section 5: RAGAS metrics
    if ragas_scores:
        lines.append("## RAGAS metrics (Tier 2: LLM-judged)")
        lines.append("")
        lines.append(
            f"Judge: `{args.judge_backend}::{args.judge_model}` "
            f"(different from generator `{args.gen_model}`).",
        )
        lines.append("")
        lines.append("| metric | value | caveat |")
        lines.append("| --- | ---: | --- |")
        ragas_caveats = {
            "context_precision":
                "judge call per retrieved passage; sensitive to judge calibration",
            "context_recall":
                "uses LLM ground_truth as the gold; soft",
            "faithfulness":
                "stronger signal: did the answer cite only retrieved context?",
            "answer_relevancy":
                "embedding-cosine + judge; directional",
            "answer_correctness":
                "partially circular (judge grades against LLM-written GT); "
                "treat as directional only",
        }
        for k, v in ragas_scores.items():
            caveat = ragas_caveats.get(k, "")
            lines.append(f"| `{k}` | {v:.3f} | {caveat} |")
        lines.append("")
    else:
        lines.append("## RAGAS metrics (Tier 2)")
        lines.append("")
        lines.append("_skipped via `--skip-ragas`._")
        lines.append("")

    # Section 6: Sample per-question outcomes
    lines.append("## Sample per-question outcomes (first 5)")
    lines.append("")
    for ex, rag in list(zip(examples, rag_results, strict=True))[:5]:
        lines.append(f"### Q: {ex.question}")
        lines.append("")
        lines.append(f"- **Ground truth**: {ex.ground_truth}")
        lines.append(f"- **Generated**: {rag.answer}")
        lines.append(
            f"- **Source chunk**: `{ex.source_chunk_id}` "
            f"(doc `{ex.source_doc_id}`)",
        )
        lines.append(
            f"- **Retrieved doc_ids (top-{args.k})**: "
            + ", ".join(f"`{d}`" for d in rag.retrieved_doc_ids),
        )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
