# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Schema profile for internal lab / company documents (notes, memos, drafts)."""

from __future__ import annotations

from nuthatch.schema.profile import FieldSpec, SchemaProfile


class InternalDocProfile(SchemaProfile):
    profile_name = "internal_doc"
    fields = (
        FieldSpec("title", required=True, expected_type=str),
        FieldSpec("author", required=True, expected_type=str),
        FieldSpec("date", required=True, expected_type=str),
        FieldSpec("doc_type", required=False, expected_type=str),
        FieldSpec("project", required=False, expected_type=str),
        FieldSpec("confidentiality", required=False, expected_type=str),
    )
