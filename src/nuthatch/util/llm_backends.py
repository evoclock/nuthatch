# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""LLM backend factory for the RAGAS eval harness.

Build a langchain-compatible LLM from a backend name + model id.
Three backends supported out of the box:

- `ollama`:   any model in the local Ollama registry, cloud or
              on-device. Honours OLLAMA_HOST env var (defaults to
              http://localhost:11434). Reasoning models (Minimax,
              Magistral) get a generous `num_predict` so the
              `content` field is actually written after the
              `thinking` field consumes its budget.
- `openai`:   OPENAI_API_KEY from env; pass any compatible model id
              (e.g. `gpt-4o`, `gpt-4-turbo`, `o3-mini`).
- `anthropic`: ANTHROPIC_API_KEY from env; pass any compatible model
              id (e.g. `claude-opus-4-7`, `claude-sonnet-4-6`).

Returns a tuple of `(generator_llm, judge_llm)` since RAGAS
distinguishes the two. Both default to the same model unless an
explicit judge spec is given.

Extension: add a new backend by adding a builder function below and
wiring it into `build_llm`. The function must return a
langchain-core `BaseLanguageModel` subclass instance.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# Default per backend so a bare `--llm-backend ollama` works without
# also having to specify a model. Override on the CLI.
_DEFAULT_MODEL_PER_BACKEND: dict[str, str] = {
    "ollama": "minimax-m2.5:cloud",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}


def build_llm(
    backend: str = "ollama",
    model: str | None = None,
    *,
    num_predict: int = 2048,
    temperature: float = 0.0,
) -> Any:
    """Return a langchain-compatible LLM for the given backend / model.

    `num_predict` is honoured by Ollama-backed models. For OpenAI /
    Anthropic the equivalent is `max_tokens`; we map it through. The
    default of 2048 leaves room for reasoning-model `thinking`
    budgets without truncating the actual answer.
    """
    backend = backend.lower()
    builder = _BACKEND_BUILDERS.get(backend)
    if builder is None:
        raise ValueError(
            f"unknown LLM backend: {backend!r}; supported: "
            f"{sorted(_BACKEND_BUILDERS)}"
        )
    return builder(
        model=model or _DEFAULT_MODEL_PER_BACKEND[backend],
        num_predict=num_predict,
        temperature=temperature,
    )


def _build_ollama(*, model: str, num_predict: int, temperature: float) -> Any:
    from langchain_ollama import ChatOllama

    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return ChatOllama(
        model=model,
        base_url=base_url,
        num_predict=num_predict,
        temperature=temperature,
    )


def _build_openai(*, model: str, num_predict: int, temperature: float) -> Any:
    from langchain_openai import ChatOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set; required for the openai backend"
        )
    return ChatOpenAI(
        model=model,
        max_tokens=num_predict,
        temperature=temperature,
    )


def _build_anthropic(*, model: str, num_predict: int, temperature: float) -> Any:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic backend requires `langchain-anthropic`; install via "
            "`sfw uv pip install --python .venv/bin/python langchain-anthropic`"
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set; required for the anthropic backend"
        )
    return ChatAnthropic(
        model=model,
        max_tokens=num_predict,
        temperature=temperature,
    )


_BACKEND_BUILDERS: dict[str, Callable[..., Any]] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
}


def list_backends() -> list[str]:
    """Names of supported LLM backends."""
    return sorted(_BACKEND_BUILDERS)
