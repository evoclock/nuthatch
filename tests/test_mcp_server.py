# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.mcp.server` (JSON-RPC protocol + 5-tool dispatch)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from nuthatch.corpus.layout import init_corpus
from nuthatch.mcp.server import NuthatchMCPServer


@dataclass
class _FakeHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: dict


class _FakeRetriever:
    def search(self, query: str, *, k: int = 5):
        return [
            _FakeHit(
                chunk_id="paper::p1::chunk_0",
                doc_id="paper::p1",
                score=0.92,
                text="Lorem ipsum body content for the relevant chunk.",
                metadata={"title": "Paper One", "doc_id": "paper::p1", "ordinal": 0},
            )
        ]


def _make_server(tmp_path: Path, **overrides) -> NuthatchMCPServer:
    layout = init_corpus(tmp_path / "c")
    defaults = {
        "retriever": _FakeRetriever(),
        "graph_loader": lambda: _make_graph(),
        "card_reader": lambda doc_id: f"# Card {doc_id}" if doc_id == "p1" else None,
        "community_reader": lambda cid: f"# Community {cid}" if cid == "0" else None,
    }
    defaults.update(overrides)
    return NuthatchMCPServer(layout, **defaults)


def _make_graph() -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node("paper::p1", node_type="paper", title="Paper One")
    g.add_node("author::wright", node_type="entity", entity_type="author")
    g.add_edge("paper::p1", "author::wright", relation="authored_by")
    return g


class TestInitialize:
    def test_returns_protocol_version(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.initialize()
        assert "protocolVersion" in result
        assert result["serverInfo"]["name"] == "nuthatch-mcp"

    def test_handle_request_wraps_in_jsonrpc(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        response = server.handle_request(
            {"jsonrpc": "2.0", "id": 7, "method": "initialize"}
        )
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 7
        assert "result" in response


class TestListTools:
    def test_returns_full_tool_set(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.list_tools()
        names = {t["name"] for t in result["tools"]}
        assert names == {
            "corpus_search",
            "subgraph_extract",
            "card_get",
            "community_get",
            "community_brief",
            "community_search",
            "community_core_nodes",
            "community_hierarchy",
            "token_econ_report",
        }

    def test_each_tool_has_input_schema(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        for tool in server.list_tools()["tools"]:
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"


class TestCorpusSearch:
    def test_returns_hits(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool(
            "corpus_search", {"query": "anything", "k": 1}
        )
        assert "content" in result
        text = result["content"][0]["text"]
        assert "paper::p1" in text

    def test_missing_query_errors(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool("corpus_search", {})
        assert result.get("isError") is True


class TestSubgraphExtract:
    def test_bfs_from_seed(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool(
            "subgraph_extract", {"seed_nodes": ["paper::p1"], "depth": 1}
        )
        assert "isError" not in result or not result.get("isError")
        text = result["content"][0]["text"]
        assert "author::wright" in text

    def test_missing_seeds_errors(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool("subgraph_extract", {})
        assert result.get("isError") is True


class TestCardAndCommunityGet:
    def test_card_get_success(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool("card_get", {"doc_id": "p1"})
        assert "# Card p1" in result["content"][0]["text"]

    def test_card_get_missing(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool("card_get", {"doc_id": "missing"})
        assert result.get("isError") is True

    def test_community_get_success(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool("community_get", {"community_id": "0"})
        assert "# Community 0" in result["content"][0]["text"]


class TestTokenEconReportUnconfigured:
    def test_reports_not_configured(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, token_econ_reporter=None)
        result = server.call_tool("token_econ_report", {})
        assert result.get("isError") is True
        assert "not configured" in result["content"][0]["text"]


class TestUnknownTool:
    def test_unknown_tool_errors(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        result = server.call_tool("does_not_exist", {})
        assert result.get("isError") is True


class TestHandleRequestErrorHandling:
    def test_unknown_method_errors(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        response = server.handle_request({"id": 1, "method": "totally_made_up"})
        assert response["result"].get("isError") is True

    def test_handler_exception_returns_error(self, tmp_path: Path) -> None:
        class BoomRetriever:
            def search(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        server = _make_server(tmp_path, retriever=BoomRetriever())
        response = server.handle_request(
            {
                "id": 1,
                "method": "tools/call",
                "params": {"name": "corpus_search", "arguments": {"query": "x"}},
            }
        )
        result = response["result"]
        assert result.get("isError") is True
        assert "boom" in result["content"][0]["text"]
