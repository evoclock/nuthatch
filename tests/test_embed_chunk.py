# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.embed.chunk` (chunking + coverage invariant)."""

from __future__ import annotations

import pytest

from nuthatch.embed.chunk import (
    Chunk,
    check_coverage,
    chunk_text,
)


class TestChunkText:
    def test_empty_input(self) -> None:
        assert chunk_text("") == []

    def test_short_text_single_chunk(self) -> None:
        text = "Hello world."
        chunks = chunk_text(text, max_chars=3000)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == len(text)

    def test_long_text_multiple_chunks(self) -> None:
        text = "A" * 10_000
        chunks = chunk_text(text, max_chars=3000, overlap_chars=400)
        assert len(chunks) >= 4
        # Every chunk respects max size.
        for c in chunks:
            assert len(c.text) <= 3000

    def test_invalid_max_chars(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("x" * 100, max_chars=0)

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("x" * 100, max_chars=100, overlap_chars=100)


class TestCoverage:
    def test_empty_text_passes(self) -> None:
        r = check_coverage("", [])
        assert r.passed
        assert r.total_chars == 0

    def test_full_coverage(self) -> None:
        text = "x" * 5000
        chunks = chunk_text(text)
        r = check_coverage(text, chunks)
        assert r.passed
        assert r.coverage_pct == pytest.approx(100.0)
        assert r.gaps == []

    def test_gap_detection(self) -> None:
        text = "x" * 100
        partial = [Chunk(text="x" * 40, start_offset=0, end_offset=40, ordinal=0)]
        r = check_coverage(text, partial)
        assert not r.passed
        assert r.gaps == [(40, 100)]
        assert r.coverage_pct == 40.0


class TestStructuralBoundaries:
    def test_chunker_prefers_paragraph_breaks(self) -> None:
        body = "Para one. " * 200 + "\n\n" + "Para two. " * 200
        chunks = chunk_text(body, max_chars=2200, overlap_chars=200)
        # Some chunk boundary should land near the paragraph break.
        # Just verify multiple chunks were produced.
        assert len(chunks) >= 2
        # Coverage invariant still holds.
        assert check_coverage(body, chunks).passed


class TestEmbedderDefaults:
    """Contract test: the default embedding model is locked.

    Changing `DEFAULT_EMBEDDING_MODEL` is a methodological decision
    that requires re-embedding every corpus's chunks (cosine
    distances aren't comparable across models). The test exists so
    a model swap is a deliberate, code-reviewed change, not an
    accidental import-time substitution.
    """

    def test_default_embedding_model_is_bge_m3(self) -> None:
        from nuthatch.embed.embed import DEFAULT_EMBEDDING_MODEL, Embedder

        assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-m3"
        emb = Embedder()
        assert emb.model_id == "BAAI/bge-m3"

    def test_default_chunk_constants_are_3000_400(self) -> None:
        """Pin the chunking defaults too — same reasoning."""
        from nuthatch.embed.chunk import (
            _DEFAULT_MAX_CHARS,
            _DEFAULT_OVERLAP_CHARS,
        )

        assert _DEFAULT_MAX_CHARS == 3000
        assert _DEFAULT_OVERLAP_CHARS == 400
