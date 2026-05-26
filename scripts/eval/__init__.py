# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""RAGAS-based RAG evaluation harness for nuthatch corpora.

CLI entry point: `scripts/eval/ragas_eval.py`. Reusable; runs against
any nuthatch corpus that has completed `ingest` + `embed`.

Layout:
- `llm_backends`       Factory: build a langchain-compatible LLM from
                       a backend + model spec. Used twice per run, once
                       for the generator + answerer and once for the
                       judge. Backends:
                         * `ollama` (default) - any model in the local
                           Ollama registry; cloud or on-device.
                         * `openai` - OPENAI_API_KEY from env.
                         * `anthropic` - ANTHROPIC_API_KEY from env.
- `ragas_embeddings`   Langchain-style embeddings adapter backed by
                       nuthatch's Embedder (BGE-M3). RAGAS uses this
                       so it does not default to OpenAI ada-002 and
                       crash without `OPENAI_API_KEY`. Side-benefit:
                       the judge scores in the same vector space the
                       retriever was indexed in.
- `testset_generator`  Synthetic (question, ground_truth) pair
                       generator sampled from the corpus's chunks.
- `rag_pipeline`       Glue: nuthatch retriever to retrieved contexts
                       to LLM answer synthesis.
- `ragas_eval`         CLI: orchestrate testset gen + RAG run + RAGAS
                       scoring + markdown report. Splits the generator
                       and judge into different models by default
                       (minimax-m2.5:cloud generates + answers;
                       gpt-oss:120b-cloud judges) to reduce
                       self-preference bias on RAGAS metrics.

Output: `pipeline_output/ragas_report_<timestamp>.md` per run, with a
side-car `pipeline_output/ragas_dataset_<timestamp>.jsonl` of the
test set used so runs are reproducible.
"""
