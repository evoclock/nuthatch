# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Semantic dedup pipeline (bi-encoder + optional reranker)."""

from nuthatch.dedup.semantic import (
    DedupConfig,
    DedupOutcome,
    DedupResult,
    SemanticDeduper,
)

__all__ = ["DedupConfig", "DedupOutcome", "DedupResult", "SemanticDeduper"]
