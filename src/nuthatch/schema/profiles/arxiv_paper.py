# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Schema profile for arXiv-style preprints."""

from __future__ import annotations

from nuthatch.schema.profile import FieldSpec, SchemaProfile


class ArxivPaperProfile(SchemaProfile):
    profile_name = "arxiv_paper"
    fields = (
        FieldSpec("title", required=True, expected_type=str),
        FieldSpec("authors", required=True, expected_type=(list, tuple)),
        FieldSpec("abstract", required=True, expected_type=str),
        FieldSpec("arxiv_id", required=True, expected_type=str),
        FieldSpec("year", required=True, expected_type=int),
        FieldSpec("categories", required=False, expected_type=(list, tuple)),
        FieldSpec("doi", required=False, expected_type=str),
    )
