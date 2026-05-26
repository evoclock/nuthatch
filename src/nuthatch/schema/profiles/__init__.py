# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Built-in schema profiles for the four canonical document classes."""

from nuthatch.schema.profiles.arxiv_paper import ArxivPaperProfile
from nuthatch.schema.profiles.biorxiv_paper import BiorxivPaperProfile
from nuthatch.schema.profiles.internal_doc import InternalDocProfile
from nuthatch.schema.profiles.patent import PatentProfile

BUILTIN_PROFILES: dict[str, type] = {
    "arxiv_paper": ArxivPaperProfile,
    "biorxiv_paper": BiorxivPaperProfile,
    "patent": PatentProfile,
    "internal_doc": InternalDocProfile,
}

__all__ = [
    "BUILTIN_PROFILES",
    "ArxivPaperProfile",
    "BiorxivPaperProfile",
    "InternalDocProfile",
    "PatentProfile",
]
