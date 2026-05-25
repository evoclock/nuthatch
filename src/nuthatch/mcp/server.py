# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""MCP stdio server for nuthatch.

Purpose: expose nuthatch's corpus query surface (search, subgraph
    extraction, card retrieval, community retrieval, token-economy
    reports) to MCP-aware agents (Claude Code, Codex, Hermes,
    Obsidian plugin, any future MCP client) via the standard
    JSON-RPC 2.0 protocol over stdin/stdout.

Inputs at construction: a `CorpusLayout` for the active corpus +
    optional injected components (`VectorRetriever`, graph loader,
    card reader). All injectable so tests substitute fakes.

Outputs: a long-running process that reads JSON-RPC requests from
    stdin and writes responses to stdout. Logs to
    `<corpus>/.kg/mcp/mcp.log`.

Pattern reused from `~/agentic-orchestrator/mcp-server/base_mcp_server.py`
(your own Hillstar base): the `tools` dict registry shape, the
`initialize` / `list_tools` / `call_tool` / `handle_request` /
`run` method names and contracts, the JSON-RPC 2.0 wrapping, the
raw-STDIO event loop. nuthatch's implementation is a fresh write
in this module — no import dependency on hillstar (nuthatch is a
separate project) — but the protocol surface mirrors that base so
the same MCP clients that consume Hillstar servers consume
nuthatch's without re-implementation.

The 5 tool registry is:

- `corpus_search(query, k)` — dense vector retrieval over the
  corpus chunks (Sprint 3 `VectorRetriever`)
- `subgraph_extract(seed_nodes, depth)` — BFS subgraph around the
  given seeds (uses Sprint 4's saved `.kg/graph/graph.json`)
- `card_get(doc_id)` — read the per-paper card markdown
- `community_get(community_id)` — read the per-community page
- `token_econ_report(time_range, group_by)` — Sprint 7 stub
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from nuthatch.corpus.layout import CorpusLayout


class MCPServer:
    """Generic MCP stdio server base.

    Subclass to register tools in `__init__` and dispatch in
    `call_tool`. Pattern from Hillstar's `BaseMCPServer`.
    """

    PROTOCOL_VERSION: str = "2025-11-25"
    VERSION: str = "1.0.0"

    def __init__(self, server_name: str, *, log_path: Path | None = None) -> None:
        self.server_name = server_name
        self.tools: dict[str, dict[str, Any]] = {}
        self._log_path = log_path
        self._log = self._make_logger()

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"nuthatch.mcp.{self.server_name}")
        logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if instantiated multiple times.
        if not logger.handlers and self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self._log_path)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
        return logger

    def initialize(self) -> dict[str, Any]:
        self._log.info("initialize: %s", self.server_name)
        return {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": f"{self.server_name}-mcp",
                "version": self.VERSION,
            },
        }

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": name,
                    "description": info["description"],
                    "inputSchema": info["inputSchema"],
                }
                for name, info in self.tools.items()
            ]
        }

    def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError("subclasses implement call_tool")

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method")
        params = request.get("params", {}) or {}
        request_id = request.get("id")

        try:
            if method == "initialize":
                result: dict[str, Any] = self.initialize()
            elif method == "tools/list":
                result = self.list_tools()
            elif method == "tools/call":
                result = self.call_tool(
                    str(params.get("name", "")),
                    params.get("arguments", {}) or {},
                )
            else:
                result = {
                    "isError": True,
                    "content": [
                        {"type": "text", "text": f"unknown method: {method}"}
                    ],
                }
        except Exception as exc:  # noqa: BLE001
            self._log.exception("handler error on %s", method)
            result = {
                "isError": True,
                "content": [{"type": "text", "text": f"error: {exc!s}"}],
            }

        if request_id is not None:
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        return result

    def run(self) -> None:
        """Read JSON-RPC requests from stdin, write responses to stdout."""
        self._log.info("%s starting", self.server_name)
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._log.error("malformed request: %s", exc)
                    continue
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            self._log.info("interrupt, shutting down")
        finally:
            self._log.info("%s stopped", self.server_name)


class NuthatchMCPServer(MCPServer):
    """Per-corpus MCP server exposing nuthatch's 5 query tools.

    Construction parameters are dependency-injected so tests can
    substitute fakes for the retriever, graph loader, and card
    reader. In production the CLI wires them from the active
    `CorpusLayout`.
    """

    def __init__(
        self,
        layout: CorpusLayout,
        *,
        retriever: Any = None,
        graph_loader: Any = None,
        card_reader: Any = None,
        community_reader: Any = None,
        token_econ_reporter: Any = None,
    ) -> None:
        log_path = layout.kg / "mcp" / "mcp.log"
        super().__init__("nuthatch", log_path=log_path)
        self._layout = layout
        self._retriever = retriever
        self._graph_loader = graph_loader
        self._card_reader = card_reader or _default_card_reader(layout)
        self._community_reader = community_reader or _default_community_reader(layout)
        self._token_econ_reporter = token_econ_reporter
        self.tools = _tool_registry()

    def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name == "corpus_search":
            return self._corpus_search(arguments)
        if tool_name == "subgraph_extract":
            return self._subgraph_extract(arguments)
        if tool_name == "card_get":
            return self._card_get(arguments)
        if tool_name == "community_get":
            return self._community_get(arguments)
        if tool_name == "token_econ_report":
            return self._token_econ_report(arguments)
        return _error(f"unknown tool: {tool_name}")

    # -- tool handlers ----------------------------------------------------

    def _corpus_search(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._retriever is None:
            return _error("retriever not configured")
        query = str(args.get("query", "")).strip()
        if not query:
            return _error("query is required")
        k = int(args.get("k", 5))
        hits = self._retriever.search(query, k=k)
        return _text(
            json.dumps(
                [
                    {
                        "chunk_id": h.chunk_id,
                        "doc_id": h.doc_id,
                        "score": h.score,
                        "text": h.text[:500],
                        "metadata": {
                            k_: v
                            for k_, v in h.metadata.items()
                            if k_ in ("title", "doc_id", "ordinal")
                        },
                    }
                    for h in hits
                ],
                indent=2,
                default=str,
            )
        )

    def _subgraph_extract(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._graph_loader is None:
            return _error("graph_loader not configured")
        seeds = args.get("seed_nodes") or []
        if not isinstance(seeds, list) or not seeds:
            return _error("seed_nodes (non-empty list) is required")
        depth = int(args.get("depth", 2))
        g = self._graph_loader()
        visited: set[str] = set(seeds)
        frontier: set[str] = set(seeds)
        edges: list[tuple[str, str, str]] = []
        for _ in range(max(depth, 0)):
            next_frontier: set[str] = set()
            for n in frontier:
                if n not in g:
                    continue
                for neighbour in g.successors(n) if g.is_directed() else g.neighbors(n):
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
                    edges.append((n, neighbour, str(
                        g.get_edge_data(n, neighbour, default={}) or {}
                    )))
            visited.update(next_frontier)
            frontier = next_frontier
        return _text(
            json.dumps(
                {
                    "nodes": sorted(visited),
                    "edges": edges,
                },
                indent=2,
                default=str,
            )
        )

    def _card_get(self, args: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(args.get("doc_id", "")).strip()
        if not doc_id:
            return _error("doc_id is required")
        text = self._card_reader(doc_id)
        if text is None:
            return _error(f"card not found: {doc_id}")
        return _text(text)

    def _community_get(self, args: dict[str, Any]) -> dict[str, Any]:
        community_id = args.get("community_id")
        if community_id is None:
            return _error("community_id is required")
        text = self._community_reader(str(community_id))
        if text is None:
            return _error(f"community not found: {community_id}")
        return _text(text)

    def _token_econ_report(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._token_econ_reporter is None:
            return _error("token_econ_reporter not configured (Sprint 7)")
        report = self._token_econ_reporter(args)
        return _text(json.dumps(report, indent=2, default=str))


# -- helpers --------------------------------------------------------------


def _error(message: str) -> dict[str, Any]:
    return {"isError": True, "content": [{"type": "text", "text": message}]}


def _text(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}]}


def _default_card_reader(layout: CorpusLayout):
    cards_dir = layout.root / "cards"

    def _read(doc_id: str) -> str | None:
        path = cards_dir / f"{doc_id}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    return _read


def _default_community_reader(layout: CorpusLayout):
    communities_dir = layout.root / "communities"

    def _read(community_id: str) -> str | None:
        path = communities_dir / f"{community_id}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    return _read


def _tool_registry() -> dict[str, dict[str, Any]]:
    return {
        "corpus_search": {
            "description": "Dense vector search over the corpus chunks. Returns top-k hits.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query"},
                    "k": {"type": "integer", "default": 5, "description": "Number of hits"},
                },
                "required": ["query"],
            },
        },
        "subgraph_extract": {
            "description": "BFS subgraph around seed node IDs. Returns nodes + edges.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seed_nodes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Node IDs to seed the BFS from",
                    },
                    "depth": {"type": "integer", "default": 2, "description": "BFS hops"},
                },
                "required": ["seed_nodes"],
            },
        },
        "card_get": {
            "description": "Fetch the per-paper card markdown for `doc_id`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Paper doc_id"},
                },
                "required": ["doc_id"],
            },
        },
        "community_get": {
            "description": "Fetch the per-community page markdown for `community_id`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "community_id": {"type": "string", "description": "Community ID (string or int)"},
                },
                "required": ["community_id"],
            },
        },
        "token_econ_report": {
            "description": "Aggregate token-economy stats over a time range (Sprint 7).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "time_range": {"type": "string", "description": "e.g. '7d', '30d', 'all'"},
                    "group_by": {"type": "string", "description": "e.g. 'tool', 'day', 'surface'"},
                },
                "required": [],
            },
        },
    }
