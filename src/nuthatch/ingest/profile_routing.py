# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Pick a `SchemaProfile` for a given source filename.

A bioRxiv preprint is structurally a different document than an arxiv
preprint (one carries a DOI, the other an arxiv_id), so validating
both against the same profile guarantees one of them fails on a field
it never had. The routing logic here picks the right validator for
the document at hand based on the filename pattern the user dropped
in their corpus.

When no pattern matches (internal memo, scanned legacy paper, etc.)
the caller's `fallback` is used; the orchestrator defaults this to
`InternalDocProfile`.
"""

from __future__ import annotations

from nuthatch.ingest.source_metadata import (
    extract_arxiv_id_from_filename,
    extract_biorxiv_doi_from_filename,
)
from nuthatch.schema.profile import SchemaProfile
from nuthatch.schema.profiles import (
    ArxivPaperProfile,
    BiorxivPaperProfile,
    InternalDocProfile,
)


def select_profile_for_filename(
    filename: str,
    *,
    fallback: type[SchemaProfile] = InternalDocProfile,
) -> type[SchemaProfile]:
    """Return the schema profile that matches the document class of `filename`.

    Order matters: arxiv ID and bioRxiv DOI patterns are disjoint, but
    we check arxiv first because it is the more common modern preprint
    on user corpora.
    """
    if extract_arxiv_id_from_filename(filename) is not None:
        return ArxivPaperProfile
    if extract_biorxiv_doi_from_filename(filename) is not None:
        return BiorxivPaperProfile
    return fallback
