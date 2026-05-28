# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Tests for `_cmd_serve`'s TokenLog binding.

The MCP `token_econ_report` tool returns an error when no TokenLog
is bound to the server. `nuthatch serve` must wire one up so the
tool is functional out of the box; this pins that behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from nuthatch.corpus.layout import init_corpus


@pytest.fixture
def fake_corpus(tmp_path: Path) -> Path:
    """Minimal layout that `_cmd_serve` accepts.

    Needs `.kg/embeddings/` and `.kg/graph/graph.json` to satisfy the
    early-fail checks; the actual chroma + graph contents are
    irrelevant because we never hand control to `server.run()` in this
    test (we stop at construction via a monkeypatch).
    """
    layout = init_corpus(tmp_path / "corpus")
    embeddings = layout.embeddings_dir
    embeddings.mkdir(parents=True, exist_ok=True)
    # Marker file so is_dir + any(iterdir()) consider the dir populated.
    (embeddings / "chroma.sqlite3").write_bytes(b"")
    graph_dir = layout.kg / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "graph.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "graph_type": "MultiDiGraph",
                "data": {
                    "directed": True,
                    "multigraph": True,
                    "graph": {},
                    "nodes": [],
                    "edges": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return layout.root


class TestServeTokenLogWiring:
    def test_passes_token_log_to_server(
        self,
        fake_corpus: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`_cmd_serve` must construct a TokenLog and pass it to
        NuthatchMCPServer. We intercept the server constructor and
        capture the kwargs."""
        import nuthatch.cli as cli_mod

        captured: dict[str, Any] = {}

        class _SpyServer:
            def __init__(self, layout: Any, **kwargs: Any) -> None:
                captured["layout"] = layout
                captured["kwargs"] = kwargs

            def run(self) -> None:
                # Never block in tests.
                pass

        # Patch the NAME `_cmd_serve` resolves; serve imports
        # NuthatchMCPServer locally so we patch in the mcp.server module.
        import nuthatch.mcp.server as server_mod

        monkeypatch.setattr(server_mod, "NuthatchMCPServer", _SpyServer)

        # Also patch the embedder / store / retriever / graph loader so
        # the serve command doesn't try to talk to a real chroma.
        class _NullStore:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def iter_chunks(self) -> list:
                return []

        class _NullEmbedder:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

        class _NullRetriever:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

        import nuthatch.embed.embed as embed_mod
        import nuthatch.embed.store as store_mod
        import nuthatch.retrieve.vector as vector_mod

        monkeypatch.setattr(store_mod, "ChromaVectorStore", _NullStore)
        monkeypatch.setattr(embed_mod, "Embedder", _NullEmbedder)
        monkeypatch.setattr(vector_mod, "VectorRetriever", _NullRetriever)

        args = argparse.Namespace(
            corpus=str(fake_corpus),
            subcommand="serve",
        )
        rc = cli_mod._cmd_serve(args)
        assert rc == 0

        # Spy fired.
        assert "kwargs" in captured, "NuthatchMCPServer was not constructed"
        kwargs = captured["kwargs"]
        assert "token_log" in kwargs, "serve must pass token_log= to the MCP server"
        assert kwargs["token_log"] is not None, "serve must bind a real TokenLog (not None)"

        # And the log path is canonical: <layout.kg>/token_log.jsonl
        token_log = kwargs["token_log"]
        # TokenLog dataclass / class exposes its path; assert it points
        # under .kg/ so cross-restart aggregation works.
        path = getattr(token_log, "path", None)
        assert path is not None, "TokenLog should expose a .path attr"
        assert path.name == "token_log.jsonl"
        assert path.parent.name == ".kg"
