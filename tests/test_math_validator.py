# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.math_validator` + the --skip-chandra
retry-marker writer in `IngestOrchestrator`."""

from __future__ import annotations

import json
from pathlib import Path

from nuthatch.corpus import init_corpus
from nuthatch.ingest.math_validator import classify_inline_math, validate_math
from nuthatch.ingest.state_machine import IngestOrchestrator
from nuthatch.schema.profile import FieldSpec, SchemaProfile


class _PassthroughProfile(SchemaProfile):
    profile_name = "math_test"
    fields = (FieldSpec("title", required=True, expected_type=str),)


class TestClassifyInlineMath:
    def test_real_equation_with_operator(self) -> None:
        real, broken, _ = classify_inline_math("The result is $x = y + 1$.")
        assert real == 1
        assert broken == 0

    def test_broken_subscript_fragment(self) -> None:
        real, broken, samples = classify_inline_math("Saw a $\\_{s}$ here.")
        assert real == 0
        assert broken == 1
        assert "\\_{s}" in samples

    def test_broken_superscript_fragment(self) -> None:
        real, broken, _ = classify_inline_math("And a $^{6}$ there.")
        assert broken == 1
        assert real == 0

    def test_single_variable_is_broken(self) -> None:
        # $x$ alone (no operator, single var) is treated as fragment.
        real, broken, _ = classify_inline_math("Just $x$ on its own.")
        assert broken == 1
        assert real == 0

    def test_two_variables_no_operator_is_real(self) -> None:
        real, broken, _ = classify_inline_math("The pair $xy$ here.")
        assert real == 1
        assert broken == 0

    def test_no_math_returns_zeros(self) -> None:
        real, broken, samples = classify_inline_math("Plain prose only.")
        assert real == 0
        assert broken == 0
        assert samples == []

    def test_mixed_paper(self) -> None:
        # 3 real, 4 broken.
        text = (
            "Recall $E = mc^2$ and $\\Delta q = -uq + v(1-q)$ and $f(x) = x$. "
            "Saw $\\_{s}$ and $^{6}$ and $_{ab}$ and $z$ in passing."
        )
        real, broken, _ = classify_inline_math(text)
        assert real == 3
        assert broken == 4


class TestValidateMath:
    def test_clean_paper_no_retry(self) -> None:
        # All-real spans.
        text = "$a = 1$ and $b + c$ and $d * e$ etc. " * 20
        result = validate_math(text)
        assert result.real_count >= 60
        assert result.broken_count == 0
        assert not result.needs_retry

    def test_few_broken_does_not_trigger_retry(self) -> None:
        # 2 broken spans, no real ones. Below the default broken_threshold=5.
        result = validate_math("Just $\\_{s}$ and $^{6}$ here.")
        assert result.broken_count == 2
        assert not result.needs_retry  # under threshold

    def test_many_broken_triggers_retry(self) -> None:
        # 10 broken, 0 real -> ratio=1.0, count=10. Both thresholds met.
        result = validate_math("$\\_{s}$ " * 10)
        assert result.broken_count == 10
        assert result.needs_retry

    def test_balanced_real_and_broken_does_not_trigger(self) -> None:
        # 8 broken + 100 real -> count over threshold but ratio (8/108)
        # under 0.5, so no retry.
        text = "$\\_{s}$ " * 8 + "$x = y$ " * 100
        result = validate_math(text)
        assert result.broken_count == 8
        assert not result.needs_retry  # 8 / 108 = 0.07, below 0.5

    def test_broken_majority_triggers_retry(self) -> None:
        # 20 broken + 5 real -> broken=20 (over threshold), ratio=0.8
        # (over 0.5).
        text = "$\\_{s}$ " * 20 + "$x = y$ " * 5
        result = validate_math(text)
        assert result.broken_count == 20
        assert result.needs_retry
        assert result.broken_ratio == 0.8


class TestRetryMarkerWriter:
    """End-to-end: skip_chandra=True + a math-broken markdown extractor
    writes the JSONL retry entry; clean docs don't."""

    def _build_orchestrator(self, layout, *, broken_text: str):
        def _extract(path: Path) -> str:
            return broken_text

        return IngestOrchestrator(
            layout,
            extractor=_extract,
            schema_profile=_PassthroughProfile,
            skip_chandra=True,
        )

    def test_broken_math_writes_retry_entry(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        # Drop one fixture under inputs/arxiv/.
        arxiv = layout.root / "arxiv"
        arxiv.mkdir(parents=True, exist_ok=True)
        (arxiv / "p1.pdf").write_bytes(b"fake-pdf-bytes")

        # Fake extractor returns broken-math-heavy markdown so the
        # validator triggers a retry write.
        broken_md = "# Paper title\n\n" + "$\\_{s}$ " * 20 + "body. " * 50
        orch = self._build_orchestrator(layout, broken_text=broken_md)
        results = orch.ingest_corpus()
        assert len(results) == 1
        assert results[0].status.value == "ingested"

        retry_path = layout.kg / "math_retry.jsonl"
        assert retry_path.is_file()
        lines = retry_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["doc_id"] == "p1"
        assert entry["broken_count"] >= 20
        assert entry["broken_ratio"] >= 0.5
        assert "deferred_at" in entry
        assert entry["extracted_md_path"].endswith("p1.md")

    def test_clean_math_writes_no_retry_entry(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "c")
        arxiv = layout.root / "arxiv"
        arxiv.mkdir(parents=True, exist_ok=True)
        (arxiv / "p2.pdf").write_bytes(b"fake-pdf-bytes")

        clean_md = "# Paper title\n\nbody text. " * 200  # no math at all
        orch = self._build_orchestrator(layout, broken_text=clean_md)
        orch.ingest_corpus()
        retry_path = layout.kg / "math_retry.jsonl"
        # Either the file doesn't exist (nothing flagged) or it's empty.
        assert not retry_path.is_file() or retry_path.stat().st_size == 0

    def test_skip_chandra_default_off_writes_nothing(self, tmp_path: Path) -> None:
        """Without skip_chandra=True, the orchestrator should NOT touch
        the math_retry.jsonl file even on broken-math input."""
        layout = init_corpus(tmp_path / "c")
        arxiv = layout.root / "arxiv"
        arxiv.mkdir(parents=True, exist_ok=True)
        (arxiv / "p3.pdf").write_bytes(b"fake-pdf-bytes")

        broken_md = "# T\n\n" + "$\\_{s}$ " * 20 + "body. " * 50

        def _extract(path: Path) -> str:
            return broken_md

        orch = IngestOrchestrator(
            layout,
            extractor=_extract,
            schema_profile=_PassthroughProfile,
            # skip_chandra default False
        )
        orch.ingest_corpus()
        retry_path = layout.kg / "math_retry.jsonl"
        assert not retry_path.is_file()
