# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Schema profile for patent documents (USPTO / EPO / WIPO)."""

from __future__ import annotations

from nuthatch.schema.profile import FieldSpec, SchemaProfile


class PatentProfile(SchemaProfile):
    profile_name = "patent"
    fields = (
        FieldSpec("title", required=True, expected_type=str),
        FieldSpec("patent_number", required=True, expected_type=str),
        FieldSpec("inventors", required=True, expected_type=(list, tuple)),
        FieldSpec("assignee", required=False, expected_type=str),
        FieldSpec("filing_date", required=True, expected_type=str),
        FieldSpec("publication_date", required=True, expected_type=str),
        FieldSpec("abstract", required=True, expected_type=str),
        FieldSpec("claims", required=False, expected_type=(list, tuple)),
        FieldSpec("classifications", required=False, expected_type=(list, tuple)),
    )
