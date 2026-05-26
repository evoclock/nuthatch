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
from nuthatch.token_econ.counterfactual import (
    CounterfactualEstimator,
    PerToolEstimator,
)
from nuthatch.token_econ.log import TokenLog, TokenRecord
from nuthatch.token_econ.measure import measure_query
from nuthatch.token_econ.report import (
    GroupBy,
    aggregate,
    summary_as_dict,
)


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
        except Exception as exc:
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
        token_log: TokenLog | None = None,
        counterfactual_estimator: CounterfactualEstimator | None = None,
        surface_id: str = "mcp",
    ) -> None:
        log_path = layout.kg / "mcp" / "mcp.log"
        super().__init__("nuthatch", log_path=log_path)
        self._layout = layout
        self._retriever = retriever
        self._graph_loader = graph_loader
        self._card_reader = card_reader or _default_card_reader(layout)
        self._community_reader = community_reader or _default_community_reader(layout)
        self._token_log = token_log
        self._estimator = counterfactual_estimator
        self._surface_id = surface_id
        # Default reporter reads the bound token_log if no override given.
        self._token_econ_reporter = token_econ_reporter or (
            _default_token_econ_reporter(token_log) if token_log is not None else None
        )
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

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        response = super().handle_request(request)
        # Instrument only successful tool calls; do not log token_econ_report
        # itself (its own served bytes are not corpus-derived).
        if (
            self._token_log is None
            or request.get("method") != "tools/call"
        ):
            return response
        params = request.get("params", {}) or {}
        tool_name = str(params.get("name", ""))
        if tool_name in ("", "token_econ_report"):
            return response
        result = response.get("result") if "result" in response else response
        if not isinstance(result, dict) or result.get("isError"):
            return response
        try:
            arguments = params.get("arguments", {}) or {}
            served_text = _extract_text(result)
            counterfactual = self._compute_counterfactual(
                tool_name, arguments, served_text, result
            )
            if counterfactual is None:
                # Tool has no honest counterfactual (e.g. card_get is pure
                # delivery). Skip logging entirely rather than recording a
                # 1.0 ratio that inflates the per-tool report noise.
                return response
            record_dict = measure_query(
                tool=tool_name,
                served_text=served_text,
                counterfactual_tokens=counterfactual,
                surface_id=self._surface_id,
            )
            self._token_log.append(TokenRecord.from_dict(record_dict))
        except Exception:
            self._log.exception("token-econ logging failed for %s", tool_name)
        return response

    def _compute_counterfactual(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        served_text: str,
        result: dict[str, Any],
    ) -> int | None:
        if self._estimator is None:
            return None
        # PerToolEstimator exposes estimate_from_served for tools whose
        # honest counterfactual depends on the served payload (subgraph
        # nodes, community member list).
        if isinstance(self._estimator, PerToolEstimator):
            return self._estimator.estimate_from_served(
                tool=tool_name,
                arguments=arguments,
                served_text=served_text,
                result=result,
            )
        return self._estimator.estimate(
            tool=tool_name,
            arguments=arguments,
            served_text=served_text,
            result=result,
        )

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
            return _error("token_econ_reporter not configured (no token_log bound)")
        report = self._token_econ_reporter(args)
        return _text(json.dumps(report, indent=2, default=str))


# -- helpers --------------------------------------------------------------


def _error(message: str) -> dict[str, Any]:
    return {"isError": True, "content": [{"type": "text", "text": message}]}


def _text(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}]}


def _extract_text(result: dict[str, Any]) -> str:
    """Concatenate text payloads from an MCP success response."""
    parts: list[str] = []
    for item in result.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "text":
            text_val = item.get("text", "")
            if isinstance(text_val, str):
                parts.append(text_val)
    return "\n".join(parts)


def _default_token_econ_reporter(token_log: TokenLog):
    """Build a reporter closure that aggregates the bound `token_log`."""

    def _report(args: dict[str, Any]) -> dict[str, Any]:
        since = args.get("since")
        until = args.get("until")
        group_by_raw = str(args.get("group_by", "tool"))
        group_by: GroupBy = (
            group_by_raw if group_by_raw in ("tool", "day", "surface") else "tool"
        )  # type: ignore[assignment]
        tool_filter = args.get("tool")
        surface_filter = args.get("surface_id")
        records = token_log.iter_records(
            since=since,
            until=until,
            tool=tool_filter,
            surface_id=surface_filter,
        )
        summary = aggregate(records, group_by=group_by)
        return summary_as_dict(summary)

    return _report


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
