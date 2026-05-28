# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Ingest pipeline: inbox -> hash dedup -> (placeholder extract) -> manifest -> route."""

from nuthatch.ingest.dedup import hash_file
from nuthatch.ingest.manifest import IngestStatus, ManifestEntry, ManifestStore
from nuthatch.ingest.state_machine import IngestOrchestrator, IngestResult

__all__ = [
    "IngestOrchestrator",
    "IngestResult",
    "IngestStatus",
    "ManifestEntry",
    "ManifestStore",
    "hash_file",
]
