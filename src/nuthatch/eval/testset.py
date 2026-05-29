# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate a synthetic (question, ground_truth) test set from a corpus.

Sampling strategy:
    1. Stratified sample: pull N chunks at random, with a per-doc cap
       so questions are spread across many papers (not all from one).
    2. For each chunk, prompt the configured LLM to write a single
       question and a single ground-truth answer that can be answered
       FROM THAT CHUNK ALONE.
    3. Filter: drop chunks where the LLM declines, returns an empty
       answer, or returns a malformed JSON.

The output is a list of `TestExample(question, ground_truth, source_chunk_id, source_doc_id)`
records, ready to feed into the RAG pipeline + RAGAS scoring.

Determinism: passing the same `seed` reproduces the chunk sample.
The LLM responses themselves are non-deterministic unless the
backend supports `temperature=0` (Ollama + OpenAI + Anthropic all do).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from nuthatch.util import parse_llm_json


@dataclass(frozen=True, slots=True)
class TestExample:
    """One (question, ground_truth) pair sourced from a single chunk."""

    question: str
    ground_truth: str
    source_chunk_id: str
    source_doc_id: str
    source_text: str


_PROMPT = """\
You are evaluating a retrieval-augmented system. Given the passage \
below, produce exactly ONE evaluation pair:

- A specific, fact-grounded question that can be answered \
ONLY using the passage.
- A concise ground-truth answer (1-3 sentences) drawn directly \
from the passage.

Avoid yes/no questions, definition-only questions, or questions \
that can be answered from general knowledge. The question should \
test whether a retrieval system can surface this specific passage.

Refusal is a first-class option. If the passage is unsuitable for \
generating a meaningful evaluation question, respond with refusal \
instead of inventing one. A passage is unsuitable when:
- It is dominated by boilerplate (acknowledgements, funding, \
disclaimers, page headers).
- It is mostly tables or formulas with no expository prose.
- It is a fragment of a reference list with no semantic content.
- It does not contain any specific factual claim distinct enough \
to test retrieval against.

Respond with valid JSON in exactly ONE of these two shapes:
{{"question": "...", "answer": "..."}}
{{"refusal": "<short reason>"}}

Do not include code fences, prefix, or any text outside the JSON.

PASSAGE:
\"\"\"
{passage}
\"\"\"
"""


def sample_chunks(
    chunk_records: list[tuple[str, str, dict[str, Any]]],
    *,
    n: int = 30,
    per_doc_cap: int = 3,
    seed: int = 0,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Stratified sample: at most `per_doc_cap` chunks per doc.

    `chunk_records` is a list of `(chunk_id, text, metadata)` triples
    pulled from Chroma. We bucket by `metadata["doc_id"]`, cap each
    bucket, then sample uniformly across buckets.
    """
    rng = random.Random(seed)
    by_doc: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for cid, text, meta in chunk_records:
        doc_id = str(meta.get("doc_id") or "_unknown")
        by_doc[doc_id].append((cid, text, meta))

    pool: list[tuple[str, str, dict[str, Any]]] = []
    for items in by_doc.values():
        rng.shuffle(items)
        pool.extend(items[:per_doc_cap])

    rng.shuffle(pool)
    return pool[:n]


def generate_test_set(
    chunk_records: list[tuple[str, str, dict[str, Any]]],
    llm: Any,
    *,
    n: int = 30,
    per_doc_cap: int = 3,
    seed: int = 0,
    on_progress: Any = None,
) -> list[TestExample]:
    """Build the test set. Returns examples; drops LLM failures silently.

    `llm` is a langchain-compatible chat model (see `llm_backends`).
    The LLM is prompted once per sampled chunk; failures (malformed
    JSON, empty answer, refusal) are skipped rather than retried, so
    the output may have fewer than `n` examples.
    """
    sampled = sample_chunks(chunk_records, n=n, per_doc_cap=per_doc_cap, seed=seed)
    examples: list[TestExample] = []
    for i, (cid, text, meta) in enumerate(sampled, start=1):
        if on_progress is not None:
            on_progress(i, len(sampled), cid)
        passage = text[:4000]  # cap to keep prompt size bounded
        try:
            resp = llm.invoke(_PROMPT.format(passage=passage))
            content = getattr(resp, "content", str(resp)).strip()
        except Exception as exc:
            print(f"  WARN: LLM invoke failed on {cid}: {exc!s}")
            continue
        parsed = parse_llm_json(content)
        if parsed is None:
            continue
        # Refusal is a first-class output. The LLM is explicitly
        # permitted to decline on unsuitable chunks (boilerplate,
        # tables, reference fragments). Honoured silently to keep
        # log noise low; the dropped-count is implicit in the
        # final examples-generated tally.
        if "refusal" in parsed and "question" not in parsed:
            continue
        q = str(parsed.get("question", "")).strip()
        a = str(parsed.get("answer", "")).strip()
        if not q or not a:
            continue
        examples.append(
            TestExample(
                question=q,
                ground_truth=a,
                source_chunk_id=cid,
                source_doc_id=str(meta.get("doc_id") or "_unknown"),
                source_text=text,
            )
        )
    return examples


def write_testset_jsonl(examples: Iterable[TestExample], path: Any) -> int:
    """Persist the test set to JSONL for reproducibility."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(
                json.dumps(
                    {
                        "question": ex.question,
                        "ground_truth": ex.ground_truth,
                        "source_chunk_id": ex.source_chunk_id,
                        "source_doc_id": ex.source_doc_id,
                    }
                )
                + "\n"
            )
            count += 1
    return count
