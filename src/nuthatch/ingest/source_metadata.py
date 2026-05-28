# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Authoritative metadata fetchers for arxiv + bioRxiv source files.

PDF body extraction (Docling / Chandra) gives us the text but is
unreliable for title / authors / abstract because those fields live
in typeset cover sections that vary by template. The canonical
source for an arxiv or bioRxiv paper is the publisher's API, which
returns clean structured metadata keyed by ID.

This module:

1. Parses arxiv-ID / bioRxiv-DOI from filenames (`2605.15308v1.pdf`,
   `2026.05.14.725010v1.full.pdf`).
2. Fetches metadata from the publisher API.
3. Caches to `<corpus>/.kg/metadata_cache/<id>.json` so re-ingest
   is a free file read.
4. Returns a dict compatible with `nuthatch.ingest.metadata.extract_and_validate`
   so the schema-validation step uses these fields instead of (or
   merged on top of) the body-text heuristic.

Network safety: every fetch is gated by `nuthatch.ingest.security.validate_url`
(SSRF / loopback / metadata-IP / private-network protection from
Sprint 3). The two endpoints we hit are constants, but the safety
gate is consistent with the rest of the ingest layer.

Why this pattern: peer tools (graphify) fetch arxiv metadata from
the arxiv HTML page when the user passes a URL via their `add`
subcommand. The same pattern works for already-on-disk PDFs: the
arxiv ID is in the filename, so no URL is needed at all.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree as ET

from nuthatch.ingest.security import SecurityResult, validate_url

_LOG = logging.getLogger(__name__)

# arxiv preprint IDs since 2007: YYMM.NNNNN with an optional version
# suffix `vN`. Earlier IDs (cs/0507001 shape) are not handled here;
# they are rare in modern user-curated corpora.
_ARXIV_ID_FROM_FILENAME = re.compile(
    r"^(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)?(?:\.full)?(?:\.pdf)?$",
    re.IGNORECASE,
)

# bioRxiv preprints: YYYY.MM.DD.NNNNNN[vN][.full].pdf
_BIORXIV_DOI_TAIL_FROM_FILENAME = re.compile(
    r"^(?P<tail>\d{4}\.\d{2}\.\d{2}\.\d{6,7})(?P<version>v\d+)?(?:\.full)?(?:\.pdf)?$",
    re.IGNORECASE,
)

_ARXIV_API = "https://export.arxiv.org/api/query?id_list={id}"
_BIORXIV_API = "https://api.biorxiv.org/details/biorxiv/{doi}"

# Atom namespaces used by arxiv's API response.
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Network timeout. Empirically, arxiv's API responds in ~0.1-15s for
# uncached recent records (cached records: sub-second). The previous
# 10s cap was too aggressive: papers from the last few days regularly
# took 10-15s on the first hit, dropping into the filename-fallback
# path and losing authors / abstract / year.
_FETCH_TIMEOUT_SECONDS: float = 30.0

# Retry on HTTP 429 (rate-limit). arxiv's CDN periodically denies
# bursts; backoff is the polite recourse.
_MAX_RETRY_ON_429: int = 3
_RETRY_BASE_DELAY_SECONDS: float = 2.0

# Default User-Agent. arxiv asks for a non-default UA; this string
# identifies nuthatch's traffic so they can rate-limit / debug us.
_USER_AGENT: str = "nuthatch-ingest/0.1 (https://github.com/evoclock/nuthatch)"


@dataclass(frozen=True)
class SourceMetadata:
    """Structured metadata returned by a publisher API fetch."""

    title: str | None = None
    authors: tuple[str, ...] = ()
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source}
        if self.title:
            out["title"] = self.title
        if self.authors:
            out["authors"] = list(self.authors)
        if self.abstract:
            out["abstract"] = self.abstract
        if self.year is not None:
            out["year"] = self.year
        if self.doi:
            out["doi"] = self.doi
        if self.arxiv_id:
            out["arxiv_id"] = self.arxiv_id
        return out


# -- filename detectors ---------------------------------------------------


def extract_arxiv_id_from_filename(filename: str) -> str | None:
    """Return the arxiv ID from a filename, or None.

    Strips the version suffix so the API call is reliable; the
    publisher API resolves both `2605.15308` and `2605.15308v1` to
    the latest version of that paper.
    """
    stem = Path(filename).name
    # Strip any .pdf etc. ourselves; the regex also handles it.
    m = _ARXIV_ID_FROM_FILENAME.match(stem)
    if m is None:
        return None
    return m.group("id")


def extract_biorxiv_doi_from_filename(filename: str) -> str | None:
    """Return the bioRxiv DOI (`10.1101/<tail>`) from a filename, or None."""
    stem = Path(filename).name
    m = _BIORXIV_DOI_TAIL_FROM_FILENAME.match(stem)
    if m is None:
        return None
    return f"10.1101/{m.group('tail')}"


# -- low-level fetchers (with cache) --------------------------------------


def _cache_read(cache_dir: Path, key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return result
    except json.JSONDecodeError:
        return None


def _cache_write(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def _fetch_url(
    url: str,
    *,
    url_validator: Callable[[str], SecurityResult] = validate_url,
) -> str | None:
    """Validated HTTP GET. Returns body text on 200, None otherwise.

    Retries on HTTP 429 with exponential backoff (up to
    `_MAX_RETRY_ON_429` attempts). Other transient errors propagate
    as a None return.
    """
    import time

    check = url_validator(url)
    if not check.allowed:
        _LOG.warning("URL %s blocked by validator: %s", url, check.reason)
        return None
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    for attempt in range(_MAX_RETRY_ON_429 + 1):
        try:
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
                if resp.status != 200:
                    return None
                return cast(str, resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < _MAX_RETRY_ON_429:
                delay = _RETRY_BASE_DELAY_SECONDS * (2**attempt)
                _LOG.info(
                    "rate-limited (429) on %s; retry %d/%d in %.1fs",
                    url,
                    attempt + 1,
                    _MAX_RETRY_ON_429,
                    delay,
                )
                time.sleep(delay)
                continue
            _LOG.warning("fetch of %s failed: %s", url, exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            _LOG.warning("fetch of %s failed: %s", url, exc)
            return None
    return None


def fetch_arxiv_metadata(
    arxiv_id: str,
    *,
    cache_dir: Path | None = None,
    url_validator: Callable[[str], SecurityResult] = validate_url,
) -> SourceMetadata | None:
    """Fetch + parse arxiv metadata. Returns None on any error.

    Hits `http://export.arxiv.org/api/query?id_list=<id>` (Atom XML).
    Cached to `<cache_dir>/arxiv_<id>.json` on first success.
    """
    cache_key = f"arxiv_{arxiv_id}"
    if cache_dir is not None:
        cached = _cache_read(cache_dir, cache_key)
        if cached is not None:
            return _meta_from_cache(cached, source="arxiv_cache")

    url = _ARXIV_API.format(id=arxiv_id)
    body = _fetch_url(url, url_validator=url_validator)
    if body is None:
        return None
    parsed = _parse_arxiv_atom(body, arxiv_id=arxiv_id)
    if parsed is None:
        return None
    if cache_dir is not None:
        _cache_write(cache_dir, cache_key, parsed.to_dict())
    return parsed


def fetch_biorxiv_metadata(
    doi: str,
    *,
    cache_dir: Path | None = None,
    url_validator: Callable[[str], SecurityResult] = validate_url,
) -> SourceMetadata | None:
    """Fetch + parse bioRxiv metadata. Returns None on any error."""
    cache_key = "biorxiv_" + doi.replace("/", "_")
    if cache_dir is not None:
        cached = _cache_read(cache_dir, cache_key)
        if cached is not None:
            return _meta_from_cache(cached, source="biorxiv_cache")

    url = _BIORXIV_API.format(doi=doi)
    body = _fetch_url(url, url_validator=url_validator)
    if body is None:
        return None
    parsed = _parse_biorxiv_json(body, doi=doi)
    if parsed is None:
        return None
    if cache_dir is not None:
        _cache_write(cache_dir, cache_key, parsed.to_dict())
    return parsed


def enrich_from_source(
    source_filename: str,
    *,
    cache_dir: Path | None = None,
    url_validator: Callable[[str], SecurityResult] = validate_url,
) -> SourceMetadata | None:
    """Convenience: detect arxiv / bioRxiv from filename, fetch + return.

    The orchestrator-facing entrypoint. Returns None when the
    filename doesn't match any known publisher pattern (e.g. the
    user's internal PDFs, hand-titled scans, etc.).

    When the filename DOES match a publisher pattern but the publisher
    API has nothing (recent preprint not yet indexed, network failure),
    falls back to a stub `SourceMetadata` carrying only the identifier
    (`arxiv_id` or `doi`) and the year parseable from the filename.
    This lets the schema gate accept the identifier requirement on
    documents whose richer metadata will be filled by body extraction.
    """
    arxiv_id = extract_arxiv_id_from_filename(source_filename)
    if arxiv_id:
        meta = fetch_arxiv_metadata(arxiv_id, cache_dir=cache_dir, url_validator=url_validator)
        if meta is not None:
            return meta
        return SourceMetadata(arxiv_id=arxiv_id, source="arxiv_filename_fallback")
    biorxiv_doi = extract_biorxiv_doi_from_filename(source_filename)
    if biorxiv_doi:
        meta = fetch_biorxiv_metadata(biorxiv_doi, cache_dir=cache_dir, url_validator=url_validator)
        if meta is not None:
            return meta
        year = _year_from_biorxiv_doi(biorxiv_doi)
        return SourceMetadata(
            doi=biorxiv_doi,
            year=year,
            source="biorxiv_filename_fallback",
        )
    return None


def _year_from_biorxiv_doi(doi: str) -> int | None:
    """Extract the four-digit year from a bioRxiv DOI tail (`10.1101/YYYY.MM.DD.NNNNNN`)."""
    m = re.search(r"/(\d{4})\.\d{2}\.\d{2}\.", doi)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# -- parsers --------------------------------------------------------------


def _parse_arxiv_atom(xml_text: str, *, arxiv_id: str) -> SourceMetadata | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    # arxiv returns a feed with one entry per id.
    entry = root.find("atom:entry", _ATOM_NS)
    if entry is None:
        return None
    title_el = entry.find("atom:title", _ATOM_NS)
    summary_el = entry.find("atom:summary", _ATOM_NS)
    published_el = entry.find("atom:published", _ATOM_NS)
    authors_els = entry.findall("atom:author/atom:name", _ATOM_NS)
    doi_el = entry.find("arxiv:doi", _ATOM_NS)

    title = _clean_whitespace(title_el.text) if title_el is not None else None
    abstract = _clean_whitespace(summary_el.text) if summary_el is not None else None
    authors_list: list[str] = []
    for a in authors_els:
        if a.text is None:
            continue
        cleaned = _clean_whitespace(a.text)
        if cleaned:
            authors_list.append(cleaned)
    authors = tuple(authors_list)
    year = None
    if published_el is not None and published_el.text:
        try:
            year = int(published_el.text[:4])
        except ValueError:
            year = None
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

    return SourceMetadata(
        title=title,
        authors=authors,
        abstract=abstract,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        source="arxiv",
    )


def _parse_biorxiv_json(text: str, *, doi: str) -> SourceMetadata | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    collection = payload.get("collection") or []
    if not collection:
        return None
    # bioRxiv may return multiple entries for revisions; the last one
    # is the most recent.
    entry = collection[-1]
    title = entry.get("title")
    abstract = entry.get("abstract")
    # bioRxiv `authors` is a single string with `;`-separated names.
    authors_raw = entry.get("authors") or ""
    authors = tuple(a.strip() for a in str(authors_raw).split(";") if a.strip())
    year = None
    date_str = entry.get("date") or ""
    if isinstance(date_str, str) and len(date_str) >= 4:
        try:
            year = int(date_str[:4])
        except ValueError:
            year = None
    return SourceMetadata(
        title=_clean_whitespace(title) if title else None,
        authors=authors,
        abstract=_clean_whitespace(abstract) if abstract else None,
        year=year,
        doi=doi,
        arxiv_id=None,
        source="biorxiv",
    )


def _meta_from_cache(payload: dict[str, Any], *, source: str) -> SourceMetadata:
    return SourceMetadata(
        title=payload.get("title"),
        authors=tuple(payload.get("authors") or ()),
        abstract=payload.get("abstract"),
        year=payload.get("year"),
        doi=payload.get("doi"),
        arxiv_id=payload.get("arxiv_id"),
        source=source,
    )


def _clean_whitespace(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip() or None
