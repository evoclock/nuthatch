# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Content-hash deduplication for inbox files.

Two files with the same SHA256 are treated as the same paper even
if filenames differ: common when the user re-downloads an arxiv
PDF or saves a preprint twice. Semantic dedup (preprint vs published,
multiple revisions of the same paper) is Sprint 3 work; this module
only handles the cheap byte-exact case.

Hashing reads the file in chunks so large PDFs don't fault the
process. SHA256 is overkill for collision avoidance in a 10k-paper
corpus (MD5 would suffice) but the marginal cost is invisible and
SHA256 is the standard. Use the same hash family everywhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 64 KiB read chunks. Tuned for "fits in CPU cache, doesn't fault on
# tiny files". Empirical sweet spot; not load-bearing.
_HASH_CHUNK_SIZE: int = 64 * 1024


def hash_file(path: Path) -> str:
    """Return the SHA256 hex digest of `path`'s contents.

    Streams the file rather than reading it all into memory. Raises
    `FileNotFoundError` if `path` does not exist (callers should
    handle this: typically by treating the file as gone and
    skipping it).
    """

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
