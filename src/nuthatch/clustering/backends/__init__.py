# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Concrete clustering backends.

Each module exposes a `ClusteringBackend`-compliant class. Heavy
dependencies (graph-tool, igraph, sentence-transformers) are
imported lazily so importing this package does not require all
backends to be installed.
"""
