# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Corpus layout + registry: where a nuthatch corpus lives and how to find it."""

from nuthatch.corpus.config import (
    CorpusConfig,
    EmbeddingConfig,
    load_corpus_config,
)
from nuthatch.corpus.layout import (
    CORPUS_MARKER,
    CORPUS_RESERVED_DIRS,
    CorpusLayout,
    discover_corpus_root,
    init_corpus,
)
from nuthatch.corpus.registry import Registry, default_registry_path

__all__ = [
    "CORPUS_MARKER",
    "CORPUS_RESERVED_DIRS",
    "CorpusConfig",
    "CorpusLayout",
    "EmbeddingConfig",
    "Registry",
    "default_registry_path",
    "discover_corpus_root",
    "init_corpus",
    "load_corpus_config",
]
