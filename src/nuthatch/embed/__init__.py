# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Chunking + embedding + vector storage (Sprint 3)."""

from nuthatch.embed.chunk import Chunk, CoverageResult, check_coverage, chunk_text
from nuthatch.embed.store import ChromaVectorStore, Neighbour, VectorStore

__all__ = [
    "ChromaVectorStore",
    "Chunk",
    "CoverageResult",
    "Neighbour",
    "VectorStore",
    "check_coverage",
    "chunk_text",
]
