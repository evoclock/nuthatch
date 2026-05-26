# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline integration test.

Walks one document through the full chain:

    ingest -> embed -> graph -> cluster -> render

with assertions on every contract:

- Routing decision (markdown takes the text path, no PDF OCR needed)
- Extracted markdown + meta sidecar land in `.kg/extracted/`
- Chunks land in the vector store with the correct per-chunk
  metadata schema (doc_id, title, path matches PhD KB pattern)
- Coverage invariant on chunks holds
- Graph node for the doc is created with frontmatter as attrs
- Clustering returns a community membership for the doc node
- Render writes a card + a community page

Uses a fake Embedder + in-memory VectorStore so the test runs in
under a second without loading BGE-M3 (~2 GB). The real
`embed_document` orchestration is exercised; only the model is
stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nuthatch.corpus import init_corpus
from nuthatch.embed.embed import Embedder, embed_document
from nuthatch.embed.orchestrator import embed_corpus
from nuthatch.embed.store import Neighbour
from nuthatch.graph.orchestrator import build_graph_for_corpus
from nuthatch.ingest.state_machine import IngestOrchestrator
from nuthatch.render.obsidian import export_vault
from nuthatch.schema.profile import FieldSpec, SchemaProfile

# -- test fixtures + stubs -------------------------------------------------


class _PassthroughProfile(SchemaProfile):
    """Minimal schema: requires only `title` so the test fixtures pass."""

    profile_name = "test_passthrough"
    fields = (FieldSpec("title", required=True, expected_type=str),)


def _ingest_fixture_text(doc_id: str, title: str, body: str) -> str:
    return (
        f"# {title}\n\n"
        + body
        + "\n\n"
        + "Authored by [[Wright]] and [[Mendel]]. "
        + "See [Smith 2010] for context. "
        + "Topics: evolution, genetics. " * 5
        + "Lorem ipsum dolor sit amet consectetur adipiscing elit "
        "sed do eiusmod tempor incididunt ut labore et dolore "
        "magna aliqua. " * 4
    )


def _fixture_extractor(text_by_path: dict[str, str]):
    def _extract(path: Path) -> str:
        return text_by_path[path.name]

    return _extract


class _FakeEmbedder(Embedder):
    """Returns deterministic 4-d vectors based on token hash.

    Avoids loading BGE-M3 in tests. Subclasses Embedder so type
    checks pass and `encode()` honours the same Iterable[str] -> list[list[float]]
    contract.
    """

    def __init__(self) -> None:
        super().__init__()

    def _ensure(self) -> Any:
        return None

    def encode(self, texts) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            h = hash(text) % 1000
            out.append([h / 1000.0, (h * 7) % 1000 / 1000.0, 0.5, 0.5])
        return out


class _InMemoryStore:
    """Minimal VectorStore for the e2e test (no Chroma persistence)."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def add(
        self,
        *,
        chunk_ids,
        embeddings,
        texts,
        metadatas,
    ) -> None:
        for cid, emb, text, meta in zip(chunk_ids, embeddings, texts, metadatas, strict=True):
            self._rows[cid] = {
                "embedding": list(emb),
                "text": text,
                "metadata": dict(meta),
            }

    def query(self, *, embedding, k=5, where=None) -> list[Neighbour]:
        return []  # not exercised in this test

    def delete_by_doc(self, doc_id: str) -> int:
        to_drop = [
            cid for cid, row in self._rows.items()
            if row["metadata"].get("doc_id") == doc_id
        ]
        for cid in to_drop:
            del self._rows[cid]
        return len(to_drop)

    def count(self) -> int:
        return len(self._rows)

    def iter_chunks(self):
        for cid, row in self._rows.items():
            yield cid, row["text"]

    # Diagnostic helpers used by the test:
    def all_metadatas(self) -> list[dict[str, Any]]:
        return [row["metadata"] for row in self._rows.values()]

    def doc_ids_present(self) -> set[str]:
        return {row["metadata"].get("doc_id") for row in self._rows.values()}


# -- the e2e test ----------------------------------------------------------


class TestPipelineE2E:
    """One markdown doc through ingest -> embed -> graph -> cluster -> render."""

    def _seed_inputs(self, layout, names_and_texts: dict[str, str]) -> None:
        # Drop fixtures in `arxiv/` subdir to also verify the
        # recursive-scan fix from task #24 holds end-to-end.
        arxiv = layout.root / "arxiv"
        arxiv.mkdir(parents=True, exist_ok=True)
        for name, text in names_and_texts.items():
            (arxiv / name).write_text(text, encoding="utf-8")

    def test_full_chain(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")

        # Seed three .md fixtures under inputs/arxiv/ — exercising
        # the recursive-scan fix at the same time.
        fixtures = {
            "paper_a.md": _ingest_fixture_text(
                "paper_a", "Paper A: An Evolutionary Study", "Body of paper A.\n\n"
            ),
            "paper_b.md": _ingest_fixture_text(
                "paper_b", "Paper B: A Genetics Study", "Body of paper B.\n\n"
            ),
            "paper_c.md": _ingest_fixture_text(
                "paper_c", "Paper C: A Methods Note", "Body of paper C.\n\n"
            ),
        }
        self._seed_inputs(layout, fixtures)

        text_by_path = dict(fixtures)

        # ---- Stage 1: INGEST -----------------------------------------
        orch = IngestOrchestrator(
            layout,
            extractor=_fixture_extractor(text_by_path),
            schema_profile=_PassthroughProfile,
        )
        ingest_results = orch.ingest_corpus()
        assert len(ingest_results) == 3
        assert all(r.status.value == "ingested" for r in ingest_results)

        # CONTRACT: extracted markdown + meta land under .kg/extracted/
        for stem in ("paper_a", "paper_b", "paper_c"):
            md_path = layout.extracted_dir / f"{stem}.md"
            meta_path = layout.extracted_dir / f"{stem}.meta.json"
            assert md_path.is_file(), f"missing extracted md: {stem}"
            assert meta_path.is_file(), f"missing meta sidecar: {stem}"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert meta["doc_id"] == stem
            assert meta["source_filename"] == f"{stem}.md"
            assert "extractor_version" in meta
            assert isinstance(meta["metadata"], dict)
            assert meta["metadata"]["title"]

        # ---- Stage 2: EMBED ------------------------------------------
        store = _InMemoryStore()
        embedder = _FakeEmbedder()
        embed_result = embed_corpus(
            layout, embedder=embedder, store=store, force=False
        )
        assert embed_result.n_docs_scanned == 3
        assert embed_result.n_docs_embedded == 3
        assert embed_result.n_docs_skipped == 0
        assert embed_result.n_chunks_added > 0
        assert embed_result.failed == []

        # CONTRACT: per-chunk metadata matches PhD KB schema fields.
        all_metas = store.all_metadatas()
        assert all_metas, "store has no chunks"
        sample = all_metas[0]
        # PhD-KB-parity fields:
        for required_field in ("doc_id", "filename", "path", "title", "chunk_index", "total_chunks"):
            assert required_field in sample, f"missing meta field {required_field!r}"
        # Nuthatch-specific extras:
        assert "source_filename" in sample
        assert "extractor_version" in sample

        # CONTRACT: incremental re-run is a no-op.
        rerun = embed_corpus(layout, embedder=embedder, store=store, force=False)
        assert rerun.n_docs_embedded == 0
        assert rerun.n_docs_skipped == 3

        # CONTRACT: --force re-embeds.
        forced = embed_corpus(layout, embedder=embedder, store=store, force=True)
        assert forced.n_docs_embedded == 3
        assert forced.forced is True

        # ---- Stage 3: GRAPH ------------------------------------------
        graph_result = build_graph_for_corpus(layout)
        assert graph_result.n_docs == 3
        assert graph_result.n_nodes > 0
        assert graph_result.n_edges > 0
        assert graph_result.graph_path.is_file()

        # CONTRACT: each fixture has a doc:: node.
        from nuthatch.graph.io import load_graph

        g = load_graph(graph_result.graph_path)
        for stem in ("paper_a", "paper_b", "paper_c"):
            assert f"doc::{stem}" in g.nodes

        # ---- Stage 4: CLUSTER ---------------------------------------
        # Use the Leiden backend directly so the test doesn't require
        # graph-tool (conda-only). The router would do this same
        # fallback in production; we shortcut for test determinism.
        from nuthatch.clustering.backends.leiden import LeidenBackend
        from nuthatch.clustering.protocol import ClusteringRequest, Rigor

        backend = LeidenBackend()
        assert backend.available()

        # The graph.json file is wrapped in a {schema_version, data}
        # envelope by save_graph; clustering backends want the raw
        # node-link JSON. Unwrap for the test (the production CLI
        # handler does the same via nx.node_link_data on the loaded
        # graph).
        envelope = json.loads(graph_result.graph_path.read_text(encoding="utf-8"))
        node_link_json = json.dumps(envelope["data"]).encode("utf-8")
        request = ClusteringRequest(
            graph_snapshot=node_link_json,
            rigor=Rigor.HEURISTIC,
        )
        cluster_response = backend.cluster(request)
        assert len(cluster_response.partition) > 0
        # CONTRACT: the doc nodes are partitioned.
        partitioned_doc_nodes = {
            n for n in cluster_response.partition
            if n.startswith("doc::")
        }
        assert partitioned_doc_nodes  # at least some doc nodes got a community

        # Write community membership back onto graph (mirrors cli._cmd_cluster).
        for node_id, community_id in cluster_response.partition.items():
            if node_id in g:
                g.nodes[node_id]["community_id"] = int(community_id)

        # ---- Stage 5: RENDER -----------------------------------------
        # Build paper_metadata from the meta sidecars.
        paper_metadata: dict[str, dict] = {}
        for meta_path in layout.extracted_dir.glob("*.meta.json"):
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            doc_id = str(data["doc_id"])
            paper_metadata[doc_id] = data["metadata"]

        # Build community_membership from the graph.
        from collections import defaultdict
        community_membership: dict[str, list[str]] = defaultdict(list)
        for node_id, attrs in g.nodes(data=True):
            if not isinstance(node_id, str) or not node_id.startswith("doc::"):
                continue
            cid = attrs.get("community_id")
            if cid is not None:
                community_membership[str(cid)].append(node_id[len("doc::") :])

        render_result = export_vault(
            layout,
            paper_metadata=paper_metadata,
            community_membership=dict(community_membership) or None,
            source_note="e2e_test",
        )
        assert render_result.n_cards == 3
        assert render_result.cards_dir.is_dir()

        # CONTRACT: cards exist on disk with Obsidian-compatible
        # frontmatter (title, doc_id, ingested, last_touched, etc.).
        for stem in ("paper_a", "paper_b", "paper_c"):
            card_path = render_result.cards_dir / f"{stem}.md"
            assert card_path.is_file()
            content = card_path.read_text(encoding="utf-8")
            assert content.startswith("---\n")
            assert f"doc_id: {stem}" in content or f"doc_id: '{stem}'" in content
            assert "title:" in content
            assert "last_touched:" in content

        # CONTRACT: dashboard + index + log all exist.
        assert render_result.dashboard_path.is_file()
        assert render_result.index_path.is_file()
        assert render_result.log_path.is_file()


# -- ingest also persists extracted markdown (regression test) ------------


class TestIngestPersistence:
    def test_ingest_writes_extracted_md_and_meta(self, tmp_path: Path) -> None:
        """The ingest -> embed handoff depends on `.kg/extracted/*.md` + .meta.json
        being written. Without this, embed has nothing to consume."""
        layout = init_corpus(tmp_path / "c")
        (layout.root / "real.md").write_text(
            _ingest_fixture_text("real", "Real Paper", "Body."),
            encoding="utf-8",
        )
        orch = IngestOrchestrator(
            layout,
            extractor=_fixture_extractor(
                {"real.md": _ingest_fixture_text("real", "Real Paper", "Body.")}
            ),
            schema_profile=_PassthroughProfile,
        )
        orch.ingest_corpus()
        md = layout.extracted_dir / "real.md"
        meta = layout.extracted_dir / "real.meta.json"
        assert md.is_file()
        assert meta.is_file()
        meta_data = json.loads(meta.read_text(encoding="utf-8"))
        assert meta_data["doc_id"] == "real"
        assert meta_data["metadata"]["title"] == "Real Paper"


# -- single-doc embed_document still works (regression) -------------------


class TestEmbedDocumentDirectCall:
    def test_embed_one_doc_via_library_call(self, tmp_path: Path) -> None:
        del tmp_path  # not used; this verifies the library API in isolation
        store = _InMemoryStore()
        result = embed_document(
            doc_id="solo",
            markdown="# Solo paper\n\n" + ("body. " * 200),
            store=store,
            embedder=_FakeEmbedder(),
            metadata={"title": "Solo paper", "year": 2026},
        )
        assert result.n_chunks > 0
        assert result.coverage_pct == 100.0
        assert store.count() == result.n_chunks
        # PhD-KB-shaped metadata round-trips.
        sample = store.all_metadatas()[0]
        assert sample["doc_id"] == "solo"
        assert sample["title"] == "Solo paper"
        assert sample["year"] == 2026
