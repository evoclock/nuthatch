# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Schema profile for bioRxiv-style preprints."""

from __future__ import annotations

from nuthatch.schema.profile import FieldSpec, SchemaProfile


class BiorxivPaperProfile(SchemaProfile):
    profile_name = "biorxiv_paper"
    fields = (
        FieldSpec("title", required=True, expected_type=str),
        FieldSpec("authors", required=True, expected_type=(list, tuple)),
        FieldSpec("abstract", required=True, expected_type=str),
        FieldSpec("doi", required=True, expected_type=str),
        FieldSpec("year", required=True, expected_type=int),
        FieldSpec("affiliations", required=False, expected_type=(list, tuple)),
        FieldSpec("posted_date", required=False, expected_type=str),
    )
