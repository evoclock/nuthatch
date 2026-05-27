#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""RAGAS-based quality evaluation for a nuthatch corpus.

Runs against any corpus that has completed `ingest` + `embed`. Builds
a synthetic test set, runs the live retrieve + answer pipeline on
each question, scores with RAGAS, writes a markdown report.

Usage (canonical defaults: gemini-3-flash-preview:cloud generates +
answers; granite3-dense:8b local judges):
    nuthatch eval ragas --corpus inputs --n-questions 100 --k 5

Override roles (any backend + model the factory supports):
    --gen-backend  ollama  --gen-model  gemini-3-flash-preview:cloud  # generator + answerer
    --judge-backend ollama --judge-model granite3-dense:8b             # RAGAS judge

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
from typing import Any, cast

from nuthatch.eval.rag import (
    RAGResult,
    build_retriever,
    retrieve_only,
    run_rag_turn,
)
from nuthatch.eval.testset import (
    TestExample,
    generate_test_set,
    write_testset_jsonl,
)
from nuthatch.util import resolve_corpus, utc_tag
from nuthatch.util.llm_backends import build_llm


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--corpus", required=True, help="corpus name or path")
    p.add_argument("--n-questions", type=int, default=100)
    p.add_argument(
        "--per-doc-cap", type=int, default=3, help="max questions sampled per source document"
    )
    p.add_argument("--k", type=int, default=5, help="top-k retrieved per query")
    p.add_argument("--seed", type=int, default=0)

    # Generator + answerer role (default: Gemini Flash cloud — fast +
    # JSON-compliant for testset generation).
    p.add_argument(
        "--gen-backend",
        default="ollama",
        help="generator+answerer backend: ollama | openai | anthropic",
    )
    p.add_argument(
        "--gen-model", default="gemini-3-flash-preview:cloud", help="generator+answerer model id"
    )
    p.add_argument(
        "--gen-num-predict",
        type=int,
        default=2048,
        help="answer token budget for generator+answerer",
    )

    # Judge role (default: local granite3-dense:8b — IBM enterprise
    # instruct-tuned, reliable structured JSON output, non-reasoning so
    # it does not burn the token budget on a chain-of-thought trace
    # before emitting content. Different family from Gemini generator to
    # reduce self-preference bias on RAGAS metrics).
    p.add_argument(
        "--judge-backend", default="ollama", help="judge backend: ollama | openai | anthropic"
    )
    p.add_argument(
        "--judge-model",
        default="granite3-dense:8b",
        help="judge model id; should differ from --gen-model",
    )
    p.add_argument(
        "--judge-num-predict", type=int, default=1024, help="token budget for judge calls"
    )

    p.add_argument("--out-dir", default="pipeline_output", help="where the report + dataset land")
    p.add_argument(
        "--skip-ragas",
        action="store_true",
        help="run only the intrinsic metrics; skip LLM-as-judge "
        "RAGAS metrics. Useful for fast smoke tests.",
    )
    p.add_argument(
        "--measure-rerank",
        action="store_true",
        help="add a second retrieval pass per question with the "
        "cross-encoder reranker enabled, and report mean / "
        "median rerank-rank-delta of the seeding chunk. "
        "Tells you whether the reranker earns its inference "
        "cost on this corpus. Costs ~one extra retrieval "
        "per question (cheap, GPU-bound on the reranker).",
    )
    p.add_argument(
        "--reuse-testset",
        nargs="+",
        default=None,
        help="one or more paths to previously persisted "
        "ragas_dataset_*.jsonl files to reuse instead of "
        "regenerating. Multiple paths are concatenated and "
        "deduplicated by question text, expanding the test "
        "set for free. Skips the testset generation phase. "
        "Required for apples-to-apples comparison across "
        "runs that vary only the judge or retrieval config.",
    )

    # Back-compat: keep old --llm-backend / --llm-model as aliases for the
    # generator. Any session that still passes them works without surprise.
    p.add_argument("--llm-backend", default=None, help="(deprecated) alias for --gen-backend")
    p.add_argument("--llm-model", default=None, help="(deprecated) alias for --gen-model")
    p.add_argument(
        "--num-predict", type=int, default=None, help="(deprecated) alias for --gen-num-predict"
    )

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

    corpus_root = resolve_corpus(args.corpus)
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
    _docs = got["documents"] or []
    _metas = got["metadatas"] or []
    chunk_records: list[tuple[str, str, dict[str, Any]]] = [
        (cid, doc, cast(dict[str, Any], meta))
        for cid, doc, meta in zip(got["ids"], _docs, _metas, strict=True)
    ]

    # Build LLMs. The generator is only needed when we are about to
    # generate a testset; with --reuse-testset we skip generator entirely.
    gen_llm = None
    if args.reuse_testset is None:
        print(f"[eval] generator: {args.gen_backend}::{args.gen_model}")
        gen_llm = build_llm(
            backend=args.gen_backend,
            model=args.gen_model,
            num_predict=args.gen_num_predict,
            temperature=0.0,
        )
    else:
        print("[eval] generator: SKIPPED (reusing testset)")

    # Answerer always needs an LLM; default to the same backend/model
    # the user gave for --gen-* so a reused testset still gets answered.
    print(f"[eval] answerer:  {args.gen_backend}::{args.gen_model}")
    answerer_llm = (
        gen_llm
        if gen_llm is not None
        else build_llm(
            backend=args.gen_backend,
            model=args.gen_model,
            num_predict=args.gen_num_predict,
            temperature=0.0,
        )
    )

    if not args.skip_ragas:
        if args.judge_backend == args.gen_backend and args.judge_model == args.gen_model:
            print(
                "[eval] WARNING: judge model identical to generator; this "
                "is self-grading. Use a different --judge-model for "
                "calibrated RAGAS scores.",
            )
        print(f"[eval] judge:     {args.judge_backend}::{args.judge_model}")
        # Judge is built inside _run_ragas via the modern factory so we
        # can also pre-warm it; nothing to construct up here.

    # Test set: generate fresh or reuse persisted
    tag = utc_tag()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_testset is not None:
        reuse_paths = [Path(p) for p in args.reuse_testset]
        for rp in reuse_paths:
            if not rp.is_file():
                raise SystemExit(f"--reuse-testset path not found: {rp}")
        print(f"[eval] reusing {len(reuse_paths)} testset file(s):")
        for rp in reuse_paths:
            print(f"        {rp}")
        examples = _load_testset_jsonl(reuse_paths, chunk_records)
        print(f"[eval] loaded {len(examples)} examples (deduplicated by question text)")
        # Report references either the single reuse path or a combined
        # pointer-file we write next so a future session can reproduce.
        if len(reuse_paths) == 1:
            dataset_path = reuse_paths[0]
        else:
            dataset_path = out_dir / f"ragas_dataset_combined_{tag}.jsonl"
            write_testset_jsonl(examples, dataset_path)
            print(f"[eval] combined testset persisted: {dataset_path}")
    else:
        print(
            f"[eval] generating {args.n_questions} synthetic QA pairs "
            f"(cap {args.per_doc_cap}/doc, seed {args.seed})..."
        )
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
        print(f"[eval] generated {len(examples)} usable examples ({time.perf_counter() - t0:.1f}s)")
        if not examples:
            print("[eval] no test examples generated; aborting.")
            return 3

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
        rag_results.append(
            run_rag_turn(
                ex.question,
                retriever=retriever,
                answerer_llm=answerer_llm,
                k=args.k,
            )
        )
    print(f"[eval] RAG pipeline done ({time.perf_counter() - t0:.1f}s)")

    # Intrinsic metrics (cheap, always run)
    print("[eval] computing intrinsic metrics...")
    intrinsic = _compute_intrinsic(examples, rag_results)
    for k, v in intrinsic.items():
        print(f"  {k:24s} {v:.3f}")

    # Optional rerank-delta: per-question second retrieval with
    # cross-encoder rerank, compare ranks to the no-rerank pass.
    rerank_metrics: dict[str, float] = {}
    if args.measure_rerank:
        print("[eval] measuring rerank delta...")
        rerank_metrics = _compute_rerank_delta(
            examples,
            rag_results,
            retriever,
            k=args.k,
        )
        for k, v in rerank_metrics.items():
            print(f"  {k:24s} {v:.3f}")

    # RAGAS metrics (LLM-as-judge, optional)
    ragas_scores: dict[str, float] = {}
    if not args.skip_ragas:
        print("[eval] running RAGAS evaluators (judge LLM per row per metric)...")
        ragas_scores = _run_ragas(
            examples,
            rag_results,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            judge_num_predict=args.judge_num_predict,
        )
        for k, v in ragas_scores.items():
            print(f"  {k:24s} {v:.3f}")

    # Write report. Filename is self-describing: judge state inline
    # (skipragas = intrinsic-only smoke; judge = full RAGAS judge ran)
    # so smoke runs cannot masquerade as real ones on disk.
    judge_marker = "skipragas" if args.skip_ragas else "judge"
    report_path = out_dir / f"ragas_report_{judge_marker}_{tag}.md"
    _write_report(
        report_path=report_path,
        layout=layout,
        args=args,
        examples=examples,
        rag_results=rag_results,
        intrinsic=intrinsic,
        rerank_metrics=rerank_metrics,
        ragas_scores=ragas_scores,
        dataset_path=dataset_path,
        n_chunks=n_chunks,
    )
    print(f"[eval] report: {report_path}")
    return 0


def _load_testset_jsonl(
    paths: list[Path],
    chunk_records: list[tuple[str, str, dict[str, Any]]],
) -> list[TestExample]:
    """Load one or more persisted testsets and deduplicate by question.

    The persisted JSONL has 4 fields per row: question, ground_truth,
    source_chunk_id, source_doc_id. We don't persist `source_text`
    because it can be looked up from Chroma by chunk_id; we do so here
    so loaded examples have the same shape as freshly-generated ones.

    Dedup by exact question string. With multiple files this avoids
    double-counting if you (accidentally or intentionally) pass two
    testsets that share questions.
    """
    import json

    chunk_text_by_id = {cid: text for cid, text, _meta in chunk_records}
    seen_questions: set[str] = set()
    out: list[TestExample] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = d["question"]
                if q in seen_questions:
                    continue
                seen_questions.add(q)
                source_text = chunk_text_by_id.get(d["source_chunk_id"], "")
                out.append(
                    TestExample(
                        question=q,
                        ground_truth=d["ground_truth"],
                        source_chunk_id=d["source_chunk_id"],
                        source_doc_id=d["source_doc_id"],
                        source_text=source_text,
                    )
                )
    return out


def _compute_intrinsic(
    examples: list[TestExample],
    rag_results: list[RAGResult],
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


def _compute_rerank_delta(
    examples: list[TestExample],
    rag_results: list[RAGResult],
    retriever: object,
    *,
    k: int,
) -> dict[str, float]:
    """Measure how much the cross-encoder reranker moves the seeding chunk.

    For each question we already have the no-rerank top-k (in
    `rag_results`). We do a second retrieval with `rerank=True` and
    compute the rank of the seeding chunk in each pass. Delta is
    `pre_rank - post_rank` where positive = rerank helped (moved the
    chunk up). Chunks not in top-k get rank `k + 1` so a "not present
    -> present" outcome is a positive delta of finite size, not inf.
    """
    deltas: list[float] = []
    new_hits = 0  # rerank brought into top-k a chunk that wasn't there
    lost_hits = 0  # rerank evicted a chunk that was there
    for ex, rag in zip(examples, rag_results, strict=True):
        pre_ids = rag.retrieved_chunk_ids
        try:
            post_ids = retrieve_only(
                ex.question,
                retriever,
                k=k,
                rerank=True,
            )
        except Exception:
            continue
        pre_rank = pre_ids.index(ex.source_chunk_id) + 1 if ex.source_chunk_id in pre_ids else k + 1
        post_rank = (
            post_ids.index(ex.source_chunk_id) + 1 if ex.source_chunk_id in post_ids else k + 1
        )
        deltas.append(float(pre_rank - post_rank))
        if pre_rank > k and post_rank <= k:
            new_hits += 1
        elif pre_rank <= k and post_rank > k:
            lost_hits += 1

    if not deltas:
        return {}
    deltas_sorted = sorted(deltas)
    n = len(deltas)
    median = deltas_sorted[n // 2]
    return {
        "rerank_delta_mean": sum(deltas) / n,
        "rerank_delta_median": float(median),
        "rerank_new_hits": float(new_hits),
        "rerank_lost_hits": float(lost_hits),
    }


def _retrieval_failures(
    examples: list[TestExample],
    rag_results: list[RAGResult],
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


def _build_ragas_judge(
    *,
    backend: str,
    model: str,
    num_predict: int,
) -> Any:
    """Build a RAGAS-native judge LLM using the new factory API.

    Migrated off the deprecated `LangchainLLMWrapper`. Talks to Ollama
    (local OR cloud) via Ollama's OpenAI-compatible endpoint at
    `http://localhost:11434/v1`. For non-Ollama backends, falls through
    to direct provider clients (OpenAI, Anthropic).

    The Ollama path uses `provider="openai"` because Ollama exposes an
    OpenAI-shaped REST API; the `client` we pass overrides the base URL
    and tells RAGAS where to call. No actual OpenAI account is used.
    """
    from ragas.llms import llm_factory

    backend = backend.lower()
    if backend == "ollama":
        import os

        import openai

        base_url = os.environ.get(
            "OLLAMA_HOST",
            "http://localhost:11434",
        ).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        client = openai.OpenAI(
            base_url=base_url,
            api_key="ollama",  # any string; Ollama ignores auth
            timeout=600.0,  # generous: local models can be slow
        )
        return llm_factory(
            model=model,
            provider="openai",
            client=client,
        )
    if backend == "openai":
        import openai

        client = openai.OpenAI(timeout=600.0)
        return llm_factory(model=model, provider="openai", client=client)
    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic(timeout=600.0)
        return llm_factory(model=model, provider="anthropic", client=client)
    raise ValueError(f"unsupported judge backend for RAGAS: {backend!r}")


def _prewarm_local_judge(
    backend: str,
    model: str,
    num_predict: int,
) -> None:
    """Force a single tiny inference call so the model is hot before RAGAS.

    Local Ollama models pay a 60-200s cold-start cost on the first
    inference (weight load + CUDA init + JIT). RAGAS dispatches jobs
    in parallel; if the first 16+ jobs all hit the cold model, they
    queue behind the warmup and time out. This pre-warm absorbs the
    cold start in a single dummy call so every real call lands on a
    warm process.

    No-op for non-local backends (cloud APIs keep models warm).
    """
    if backend.lower() != "ollama":
        return
    # Cloud-served Ollama models (`:cloud` suffix) live in Ollama's
    # cloud infrastructure and are kept warm across users; no local
    # cold-start cost.
    if model.endswith(":cloud"):
        return
    print(f"[eval] pre-warming local judge {model}...")
    t0 = time.perf_counter()
    try:
        from nuthatch.util.llm_backends import build_llm

        warm_llm = build_llm(
            backend=backend,
            model=model,
            num_predict=16,
            temperature=0.0,
        )
        _ = warm_llm.invoke("Reply with just the word: ok")
        dt = time.perf_counter() - t0
        print(f"[eval] pre-warm done in {dt:.1f}s; judge model is hot")
    except Exception as exc:
        print(f"[eval] WARN: pre-warm failed: {exc!s}; first RAGAS call may time out")


def _run_ragas(
    examples: list[TestExample],
    rag_results: list[RAGResult],
    *,
    judge_backend: str,
    judge_model: str,
    judge_num_predict: int,
) -> dict[str, float]:
    """RAGAS LLM-as-judge metrics, modern API + tolerant of slow judges.

    Three robustness fixes vs the prior implementation:
      1. Pre-warm the judge if it's a local model (absorbs cold-start).
      2. RunConfig(max_workers=1, timeout=600) - serial dispatch with a
         generous per-call ceiling so head-of-line blocking from a slow
         call cannot starve the queue.
      3. Migrated off deprecated `LangchainLLMWrapper` /
         `LangchainEmbeddingsWrapper` to the post-langchain factory API.

    The embeddings adapter is `NuthatchEmbeddings` (BGE-M3, same as the
    retriever) so RAGAS scores in the index's vector space rather than
    reaching for OpenAI ada-002.
    """
    from datasets import Dataset
    from ragas import RunConfig, evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from nuthatch.eval.ragas_embeddings import NuthatchEmbeddings

    _prewarm_local_judge(judge_backend, judge_model, judge_num_predict)

    ragas_llm = _build_ragas_judge(
        backend=judge_backend,
        model=judge_model,
        num_predict=judge_num_predict,
    )
    ragas_embeds = NuthatchEmbeddings()

    data = {
        "question": [ex.question for ex in examples],
        "ground_truth": [ex.ground_truth for ex in examples],
        "answer": [r.answer for r in rag_results],
        "contexts": [r.retrieved_contexts for r in rag_results],
    }
    ds = Dataset.from_dict(data)
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]
    # Serial dispatch (max_workers=1) eliminates the head-of-line
    # blocking that killed the prior run: when one slow call holds the
    # pool, no other call sits in a timeout-eligible queue. timeout=600s
    # covers any single local-model call short of pathological. seed=42
    # is RAGAS's default; surfaced for reproducibility.
    run_config = RunConfig(
        timeout=600,
        max_workers=1,
        max_retries=3,
        max_wait=60,
        seed=42,
    )
    result = evaluate(
        ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeds,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )
    # RAGAS 0.4.x returns an `EvaluationResult`, not a dict. The
    # per-metric means live in `_repr_dict` (private but stable since
    # 0.4) and the per-row scores in `_scores_dict`. We pick the
    # per-metric means; fall back to averaging the per-row lists if
    # the private attribute moves.
    repr_dict = getattr(result, "_repr_dict", None)
    if repr_dict is not None:
        return {str(k): float(v) for k, v in repr_dict.items() if isinstance(v, int | float)}
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
    layout: Any,
    args: argparse.Namespace,
    examples: list[TestExample],
    rag_results: list[RAGResult],
    intrinsic: dict[str, float],
    rerank_metrics: dict[str, float],
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
    lines.append("- Embeddings (retriever + RAGAS): nuthatch BGE-M3")
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

    # Section 3b: Rerank-delta (Tier 1, only if --measure-rerank ran)
    if rerank_metrics:
        lines.append("## Rerank delta (Tier 1: derived ground truth)")
        lines.append("")
        lines.append(
            "Per-question delta in the seeding chunk's rank, comparing "
            "vector retrieval alone vs vector + cross-encoder rerank. "
            "**Positive = rerank helped** (moved the chunk up).",
        )
        lines.append("")
        lines.append("| metric | value | reads as |")
        lines.append("| --- | ---: | --- |")
        rerank_legend = {
            "rerank_delta_mean": "mean rank-points the seeding chunk moved (+ = rerank helped)",
            "rerank_delta_median": "median rank-points moved; less sensitive to outliers",
            "rerank_new_hits": "count of questions where rerank brought the seeding chunk INTO top-k",
            "rerank_lost_hits": "count of questions where rerank EVICTED the seeding chunk from top-k",
        }
        for k, v in rerank_metrics.items():
            legend = rerank_legend.get(k, "")
            if k in ("rerank_new_hits", "rerank_lost_hits"):
                lines.append(f"| `{k}` | {int(v)} | {legend} |")
            else:
                lines.append(f"| `{k}` | {v:+.3f} | {legend} |")
        lines.append("")
        net = rerank_metrics.get("rerank_new_hits", 0) - rerank_metrics.get(
            "rerank_lost_hits",
            0,
        )
        if net > 0:
            lines.append(
                f"**Net hit gain from reranker: +{int(net)}** "
                "questions. Reranker is a net win on this corpus."
            )
        elif net < 0:
            lines.append(
                f"**Net hit loss from reranker: {int(net)}** "
                "questions. Reranker is a net loss on this corpus; "
                "consider dropping it."
            )
        else:
            lines.append(
                "**Net hit change from reranker: 0** (movement "
                "within top-k only; check mean delta for "
                "ordering quality)."
            )
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
            lines.append(f"- **Seeding chunk**: `{ex.source_chunk_id}` (doc `{ex.source_doc_id}`)")
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
            "context_precision": "judge call per retrieved passage; sensitive to judge calibration",
            "context_recall": "uses LLM ground_truth as the gold; soft",
            "faithfulness": "stronger signal: did the answer cite only retrieved context?",
            "answer_relevancy": "embedding-cosine + judge; directional",
            "answer_correctness": "partially circular (judge grades against LLM-written GT); "
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
            f"- **Source chunk**: `{ex.source_chunk_id}` (doc `{ex.source_doc_id}`)",
        )
        lines.append(
            f"- **Retrieved doc_ids (top-{args.k})**: "
            + ", ".join(f"`{d}`" for d in rag.retrieved_doc_ids),
        )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
