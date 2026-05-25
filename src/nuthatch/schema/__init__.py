# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Schema-profile machinery for the Sprint 2 ingest gate.

A `SchemaProfile` declares the metadata fields a document must yield
to pass through the ingest pipeline into the graph. Files that cannot
fill the schema route to `quarantine/` with a reason.
"""
