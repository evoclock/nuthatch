# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""RAGAS-based RAG evaluation harness for nuthatch corpora.

CLI entry point: `scripts/eval/ragas_eval.py`. Reusable; runs against
any nuthatch corpus that has completed `ingest` + `embed`.

Layout:
- `llm_backends`       Factory: build a langchain-compatible LLM from
                       a `--llm-backend` + `--llm-model` spec.
                       Backends:
                         * `ollama` (default) — any model in the
                           local Ollama registry; cloud or
                           on-device. Defaults to `minimax-m2.5:cloud`.
                         * `openai` — OPENAI_API_KEY from env; user
                           passes any compatible model id.
                         * `anthropic` — ANTHROPIC_API_KEY from env;
                           user passes any compatible model id.
- `testset_generator`  Synthetic (question, ground_truth) pair
                       generator sampled from the corpus's chunks.
- `rag_pipeline`       Glue: nuthatch retriever -> retrieved
                       contexts -> LLM answer synthesis.
- `ragas_eval`         CLI: orchestrate testset gen + RAG run +
                       RAGAS scoring + markdown report.

Output: `pipeline_output/ragas_report_<timestamp>.md` per run, with
a side-car `pipeline_output/ragas_dataset_<timestamp>.jsonl` of
the test set used so runs are reproducible.
"""
