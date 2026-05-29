# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Sprint 7 token-economy module + MCP wire-up.

Covers `measure.count_tokens` (tiktoken vs chars/4 fallback),
`measure.measure_query` shape, `TokenLog` append + filter
round-trip, `report.aggregate` per-group totals, and the
`NuthatchMCPServer` integration: every successful tool call
must land a record in the bound `TokenLog`, and the
`token_econ_report` MCP tool must read it back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nuthatch.corpus import init_corpus
from nuthatch.mcp.server import NuthatchMCPServer
from nuthatch.token_econ.counterfactual import (
    BM25Counterfactual,
    CardTokenIndex,
    build_default_estimator,
)
from nuthatch.token_econ.log import TokenLog, TokenRecord
from nuthatch.token_econ.measure import (
    DEFAULT_ENCODING,
    count_tokens,
    measure_query,
)
from nuthatch.token_econ.report import (
    aggregate,
    render_markdown_report,
    summary_as_dict,
)

# -- measure ---------------------------------------------------------------


class TestCountTokens:
    def test_returns_positive_count_and_encoding_name(self) -> None:
        n, enc = count_tokens("Hello world, this is a token-econ test.")
        assert n > 0
        # When tiktoken is installed we get cl100k; otherwise "chars_per_4".
        assert enc in (DEFAULT_ENCODING, "chars_per_4")

    def test_empty_string_is_zero(self) -> None:
        n, _ = count_tokens("")
        assert n == 0

    def test_fallback_path_uses_chars_per_4(self) -> None:
        # Force the chars/4 path with an invalid encoding name.
        n, enc = count_tokens("a" * 40, encoding_name="not-a-real-encoding")
        assert enc == "chars_per_4"
        assert n == 10


class TestMeasureQuery:
    def test_required_keys_present(self) -> None:
        rec = measure_query(
            tool="corpus_search",
            served_text="some served text",
            counterfactual_tokens=1000,
            surface_id="mcp",
        )
        for key in (
            "timestamp_utc",
            "tool",
            "surface_id",
            "tokens_served",
            "tokens_counterfactual",
            "reduction_ratio",
            "tokenizer",
        ):
            assert key in rec
        assert rec["tool"] == "corpus_search"
        assert rec["tokens_counterfactual"] == 1000
        assert rec["surface_id"] == "mcp"
        assert rec["tokens_served"] > 0
        # Reduction ratio = counterfactual / served, so >> 1 when served is tiny.
        assert rec["reduction_ratio"] > 1.0

    def test_counterfactual_text_path(self) -> None:
        rec = measure_query(
            tool="card_get",
            served_text="short",
            counterfactual_text="a much longer counterfactual text " * 50,
            surface_id="mcp",
        )
        assert rec["tokens_counterfactual"] > rec["tokens_served"]


# -- log -------------------------------------------------------------------


class TestTokenLog:
    def test_append_then_iter_round_trip(self, tmp_path: Path) -> None:
        log = TokenLog(tmp_path / "token_log.jsonl")
        rec = measure_query(
            tool="corpus_search",
            served_text="hello",
            counterfactual_tokens=100,
            surface_id="mcp",
        )
        log.append(TokenRecord.from_dict(rec))
        records = list(log.iter_records())
        assert len(records) == 1
        assert records[0].tool == "corpus_search"
        assert records[0].tokens_counterfactual == 100

    def test_iter_records_on_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        log = TokenLog(tmp_path / "does_not_exist.jsonl")
        assert list(log.iter_records()) == []
        assert log.count() == 0

    def test_filter_by_tool_and_surface(self, tmp_path: Path) -> None:
        log = TokenLog(tmp_path / "token_log.jsonl")
        for tool, surface in [
            ("corpus_search", "mcp"),
            ("card_get", "mcp"),
            ("corpus_search", "cli"),
        ]:
            log.append(
                TokenRecord.from_dict(
                    measure_query(
                        tool=tool,
                        served_text="x",
                        counterfactual_tokens=10,
                        surface_id=surface,
                    )
                )
            )
        assert len(list(log.iter_records(tool="corpus_search"))) == 2
        assert len(list(log.iter_records(surface_id="mcp"))) == 2
        assert len(list(log.iter_records(tool="corpus_search", surface_id="cli"))) == 1

    def test_filter_by_date_string(self, tmp_path: Path) -> None:
        log_path = tmp_path / "token_log.jsonl"
        log = TokenLog(log_path)
        # Hand-write so timestamps are deterministic.
        log_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "timestamp_utc": f"2026-05-{day:02d}T12:00:00+00:00",
                        "tool": "corpus_search",
                        "surface_id": "mcp",
                        "tokens_served": 10,
                        "tokens_counterfactual": 100,
                        "reduction_ratio": 10.0,
                        "tokenizer": "cl100k_base",
                    }
                )
                for day in (1, 10, 20)
            )
            + "\n",
            encoding="utf-8",
        )
        # since 2026-05-05 inclusive -> two records (10, 20).
        assert len(list(log.iter_records(since="2026-05-05"))) == 2
        # until 2026-05-15 inclusive -> two records (1, 10).
        assert len(list(log.iter_records(until="2026-05-15"))) == 2


# -- report ----------------------------------------------------------------


class TestAggregate:
    def _records(self) -> list[TokenRecord]:
        records: list[TokenRecord] = []
        for tool, served, counter in [
            ("corpus_search", 50, 1000),
            ("corpus_search", 100, 1000),
            ("card_get", 200, 1000),
        ]:
            records.append(
                TokenRecord.from_dict(
                    {
                        "timestamp_utc": "2026-05-25T12:00:00+00:00",
                        "tool": tool,
                        "surface_id": "mcp",
                        "tokens_served": served,
                        "tokens_counterfactual": counter,
                        "reduction_ratio": counter / served,
                        "tokenizer": "cl100k_base",
                    }
                )
            )
        return records

    def test_totals_by_tool(self) -> None:
        summary = aggregate(self._records(), group_by="tool")
        assert summary.n_queries == 3
        assert summary.tokens_served_total == 350
        assert summary.tokens_counterfactual_total == 3000
        assert summary.tokens_saved_total == 2650
        assert summary.group_by == "tool"
        by_key = {g.key: g for g in summary.groups}
        assert by_key["corpus_search"].n_queries == 2
        assert by_key["corpus_search"].tokens_served_total == 150
        assert by_key["card_get"].n_queries == 1

    def test_group_by_day(self) -> None:
        summary = aggregate(self._records(), group_by="day")
        assert len(summary.groups) == 1
        assert summary.groups[0].key == "2026-05-25"

    def test_render_markdown_emits_frontmatter_and_table(self) -> None:
        summary = aggregate(self._records(), group_by="tool")
        md = render_markdown_report(summary, corpus_name="test_corpus")
        assert md.startswith("---\n")
        assert 'corpus: "test_corpus"' in md
        assert "| corpus_search |" in md
        assert "| card_get |" in md

    def test_summary_as_dict_is_json_serialisable(self) -> None:
        summary = aggregate(self._records(), group_by="tool")
        payload = summary_as_dict(summary)
        # Must round-trip through json without TypeError.
        json.dumps(payload)
        assert payload["n_queries"] == 3
        assert payload["group_by"] == "tool"

    def test_empty_records_safe(self) -> None:
        summary = aggregate([], group_by="tool")
        assert summary.n_queries == 0
        assert summary.tokens_served_total == 0
        assert summary.tokens_counterfactual_total == 0
        md = render_markdown_report(summary)
        assert "_no records in range_" in md


# -- MCP wire-up -----------------------------------------------------------


class _StubHit:
    def __init__(self, doc_id: str, text: str) -> None:
        self.chunk_id = f"{doc_id}::0"
        self.doc_id = doc_id
        self.score = 0.9
        self.text = text
        self.metadata = {"title": doc_id, "doc_id": doc_id, "ordinal": 0}


class _StubRetriever:
    def search(self, query: str, k: int = 5) -> list[_StubHit]:
        return [_StubHit("doc-a", f"answer for {query}")][:k]


def _chunks_provider_factory(chunks: list[tuple[str, str]]):
    def _provider():
        yield from chunks

    return _provider


@pytest.fixture
def mcp_server_with_log(tmp_path: Path):
    layout = init_corpus(tmp_path / "corpus")
    token_log = TokenLog(layout.kg / "token_log.jsonl")
    # Seed a few chunks for BM25 so corpus_search's counterfactual is non-zero.
    chunks = [
        ("doc-a::0", "the answer for anything is forty two everywhere"),
        ("doc-b::0", "unrelated text about migratory birds and lakes"),
        ("doc-c::0", "more filler so the BM25 index has variety to score against"),
    ]
    estimator = build_default_estimator(layout, chunks_provider=_chunks_provider_factory(chunks))
    server = NuthatchMCPServer(
        layout,
        retriever=_StubRetriever(),
        token_log=token_log,
        counterfactual_estimator=estimator,
        surface_id="mcp",
    )
    return server, token_log


class TestMCPInstrumentation:
    def test_successful_tool_call_appends_record(self, mcp_server_with_log) -> None:
        server, token_log = mcp_server_with_log
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "corpus_search",
                "arguments": {"query": "anything", "k": 1},
            },
        }
        response = server.handle_request(request)
        assert "result" in response
        assert not response["result"].get("isError", False)
        records = list(token_log.iter_records())
        assert len(records) == 1
        assert records[0].tool == "corpus_search"
        assert records[0].surface_id == "mcp"
        # BM25 counterfactual for "anything" against the seeded chunks
        # picks the matching "the answer for anything is forty two..."
        # chunk. Exact token count varies by tokenizer; require >0 and
        # bounded above by the joined chunk text.
        assert records[0].tokens_counterfactual > 0

    def test_token_econ_report_tool_round_trip(self, mcp_server_with_log) -> None:
        server, _ = mcp_server_with_log
        # Generate a record first.
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "corpus_search",
                    "arguments": {"query": "anything"},
                },
            }
        )
        response = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "token_econ_report",
                    "arguments": {"group_by": "tool"},
                },
            }
        )
        text = response["result"]["content"][0]["text"]
        payload = json.loads(text)
        assert payload["n_queries"] == 1
        assert payload["group_by"] == "tool"
        assert payload["groups"][0]["key"] == "corpus_search"

    def test_error_responses_do_not_log(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        token_log = TokenLog(layout.kg / "token_log.jsonl")
        estimator = build_default_estimator(layout, chunks_provider=_chunks_provider_factory([]))
        server = NuthatchMCPServer(
            layout,
            retriever=None,
            token_log=token_log,
            counterfactual_estimator=estimator,
        )
        # No retriever bound -> tool returns isError; no record should land.
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "corpus_search",
                    "arguments": {"query": "x"},
                },
            }
        )
        assert list(token_log.iter_records()) == []

    def test_card_get_does_not_log(self, tmp_path: Path) -> None:
        layout = init_corpus(tmp_path / "corpus")
        # Write a card so card_get succeeds.
        cards_dir = layout.root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        (cards_dir / "doc-a.md").write_text("# doc a\nbody text", encoding="utf-8")
        token_log = TokenLog(layout.kg / "token_log.jsonl")
        estimator = build_default_estimator(layout)
        server = NuthatchMCPServer(
            layout,
            token_log=token_log,
            counterfactual_estimator=estimator,
        )
        response = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "card_get",
                    "arguments": {"doc_id": "doc-a"},
                },
            }
        )
        assert not response["result"].get("isError")
        # card_get is pure delivery; estimator returns None -> no log entry.
        assert list(token_log.iter_records()) == []


# -- counterfactual unit tests --------------------------------------------


class TestCardTokenIndex:
    def test_caches_and_sums(self, tmp_path: Path) -> None:
        cards = tmp_path / "cards"
        cards.mkdir()
        (cards / "a.md").write_text("aaa " * 100, encoding="utf-8")
        (cards / "b.md").write_text("bbb " * 50, encoding="utf-8")
        idx = CardTokenIndex(cards)
        assert idx.tokens_for("a") > 0
        # Missing card returns 0, not error.
        assert idx.tokens_for("missing") == 0
        total = idx.total_for(["a", "b", "missing"])
        assert total == idx.tokens_for("a") + idx.tokens_for("b")


class TestBM25Counterfactual:
    def test_top_k_text_token_count(self) -> None:
        chunks = [
            ("a", "the answer about authentication tokens lives here"),
            ("b", "totally unrelated paragraph about migratory birds"),
            ("c", "more authentication context for the same answer"),
        ]
        bm25 = BM25Counterfactual(lambda: iter(chunks))
        n = bm25.tokens_for_query("authentication answer", k=2)
        assert n > 0

    def test_empty_corpus_returns_zero(self) -> None:
        bm25 = BM25Counterfactual(lambda: iter([]))
        assert bm25.tokens_for_query("anything", k=5) == 0


class TestPerToolEstimatorSubgraphCommunity:
    def _layout_with_cards(self, tmp_path: Path):
        layout = init_corpus(tmp_path / "corpus")
        cards = layout.root / "cards"
        cards.mkdir(parents=True, exist_ok=True)
        (cards / "p1.md").write_text("content of paper one " * 30, encoding="utf-8")
        (cards / "p2.md").write_text("content of paper two " * 30, encoding="utf-8")
        return layout

    def test_subgraph_counterfactual_sums_doc_cards(self, tmp_path: Path) -> None:
        layout = self._layout_with_cards(tmp_path)
        estimator = build_default_estimator(layout)
        served = json.dumps(
            {
                "nodes": ["doc::p1", "doc::p2", "author::someone"],
                "edges": [],
            }
        )
        n = estimator.estimate_from_served(
            tool="subgraph_extract",
            arguments={"seed_nodes": ["doc::p1"]},
            served_text=served,
            result={"content": [{"type": "text", "text": served}]},
        )
        cards_idx = CardTokenIndex(layout.root / "cards")
        expected = cards_idx.tokens_for("p1") + cards_idx.tokens_for("p2")
        assert n == expected

    def test_community_counterfactual_sums_member_cards(self, tmp_path: Path) -> None:
        layout = self._layout_with_cards(tmp_path)
        estimator = build_default_estimator(layout)
        served = "## Members\n- [[p1|Paper One (2024)]]\n- [[p2|Paper Two (2025)]]\n"
        n = estimator.estimate_from_served(
            tool="community_get",
            arguments={"community_id": "0"},
            served_text=served,
            result={"content": [{"type": "text", "text": served}]},
        )
        cards_idx = CardTokenIndex(layout.root / "cards")
        expected = cards_idx.tokens_for("p1") + cards_idx.tokens_for("p2")
        assert n == expected

    def test_card_get_estimator_returns_none(self, tmp_path: Path) -> None:
        layout = self._layout_with_cards(tmp_path)
        estimator = build_default_estimator(layout)
        n = estimator.estimate(
            tool="card_get",
            arguments={"doc_id": "p1"},
            served_text="anything",
            result={},
        )
        assert n is None
