# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Tests for `nuthatch.ingest.hooks` (pre/post pipeline hooks)."""

from __future__ import annotations

import pytest

from nuthatch.ingest.hooks import HookRegistry, HookSpec, HookStage


def _add_marker(value: str):
    def fn(ctx: dict) -> dict:
        ctx.setdefault("markers", []).append(value)
        return ctx

    return fn


class TestHookRegistry:
    def test_no_hooks_returns_context(self) -> None:
        reg = HookRegistry()
        out = reg.run(HookStage.POST_EXTRACT, {"x": 1})
        assert out == {"x": 1}

    def test_single_hook_runs(self) -> None:
        reg = HookRegistry()
        reg.register(HookSpec(stage=HookStage.POST_EXTRACT, fn=_add_marker("a")))
        out = reg.run(HookStage.POST_EXTRACT, {})
        assert out["markers"] == ["a"]

    def test_priority_order(self) -> None:
        reg = HookRegistry()
        reg.register(HookSpec(stage=HookStage.POST_EXTRACT, fn=_add_marker("c"), priority=300))
        reg.register(HookSpec(stage=HookStage.POST_EXTRACT, fn=_add_marker("a"), priority=100))
        reg.register(HookSpec(stage=HookStage.POST_EXTRACT, fn=_add_marker("b"), priority=200))
        out = reg.run(HookStage.POST_EXTRACT, {})
        assert out["markers"] == ["a", "b", "c"]

    def test_stage_isolation(self) -> None:
        reg = HookRegistry()
        reg.register(HookSpec(stage=HookStage.POST_EXTRACT, fn=_add_marker("e")))
        reg.register(HookSpec(stage=HookStage.POST_METADATA, fn=_add_marker("m")))
        out_e = reg.run(HookStage.POST_EXTRACT, {})
        out_m = reg.run(HookStage.POST_METADATA, {})
        assert out_e["markers"] == ["e"]
        assert out_m["markers"] == ["m"]

    def test_hook_can_replace_context(self) -> None:
        reg = HookRegistry()

        def replace(_ctx: dict) -> dict:
            return {"replaced": True}

        reg.register(HookSpec(stage=HookStage.POST_INGEST, fn=replace))
        out = reg.run(HookStage.POST_INGEST, {"before": "yes"})
        assert out == {"replaced": True}

    def test_hook_can_raise_to_fail_fast(self) -> None:
        reg = HookRegistry()

        def boom(_ctx: dict) -> dict:
            raise ValueError("intentional")

        reg.register(HookSpec(stage=HookStage.PRE_QUARANTINE, fn=boom))
        with pytest.raises(ValueError):
            reg.run(HookStage.PRE_QUARANTINE, {})

    def test_clear_all_stages(self) -> None:
        reg = HookRegistry()
        reg.register(HookSpec(stage=HookStage.POST_EXTRACT, fn=_add_marker("x")))
        reg.clear()
        assert reg.run(HookStage.POST_EXTRACT, {}) == {}

    def test_clear_one_stage(self) -> None:
        reg = HookRegistry()
        reg.register(HookSpec(stage=HookStage.POST_EXTRACT, fn=_add_marker("e")))
        reg.register(HookSpec(stage=HookStage.POST_METADATA, fn=_add_marker("m")))
        reg.clear(HookStage.POST_EXTRACT)
        assert reg.run(HookStage.POST_EXTRACT, {}) == {}
        assert reg.run(HookStage.POST_METADATA, {}).get("markers") == ["m"]
