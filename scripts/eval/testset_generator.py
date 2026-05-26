# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

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
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


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

Respond with valid JSON exactly in the shape:
{{"question": "...", "answer": "..."}}

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
    for doc_id, items in by_doc.items():
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
        parsed = _try_parse_json(content)
        if parsed is None:
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


_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Recover JSON from a raw LLM response.

    Some models wrap the JSON in code fences or add a preamble even
    when asked not to. We strip fences and grab the first {...} block.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove leading fence with optional language tag, then trailing fence
        stripped = re.sub(r"^```\w*\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    # Try direct parse first.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block.
    match = _JSON_BLOCK_RE.search(stripped)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def write_testset_jsonl(examples: Iterable[TestExample], path: Any) -> int:
    """Persist the test set to JSONL for reproducibility."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(
                json.dumps({
                    "question": ex.question,
                    "ground_truth": ex.ground_truth,
                    "source_chunk_id": ex.source_chunk_id,
                    "source_doc_id": ex.source_doc_id,
                }) + "\n"
            )
            count += 1
    return count
