# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.cli` (init / ingest / status / corpus list)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nuthatch.cli import main


class TestInitSubcommand:
    def test_creates_corpus_tree(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        rv = main(["init", str(tmp_path / "c")])
        assert rv == 0
        captured = capsys.readouterr()
        assert "initialised corpus at" in captured.out
        assert (tmp_path / "c" / ".kg").is_dir()

    def test_register_as_writes_registry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        rv = main(["init", str(tmp_path / "c"), "--register-as", "ml", "--set-default"])
        assert rv == 0
        registry_path = tmp_path / "config" / "nuthatch" / "registry.toml"
        assert registry_path.is_file()
        content = registry_path.read_text()
        assert 'default_corpus = "ml"' in content
        assert "[corpora.ml]" in content


class TestIngestSubcommand:
    def test_ingest_with_corpus_path_arg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        # init first
        main(["init", str(tmp_path / "c")])
        # drop a fake PDF in the inbox
        (tmp_path / "c" / "inbox" / "p.pdf").write_bytes(b"data")
        # ingest by path arg
        rv = main(["ingest", "--corpus", str(tmp_path / "c")])
        assert rv == 0
        captured = capsys.readouterr()
        assert "ingested" in captured.out
        assert (tmp_path / "c" / "papers" / "p.pdf").exists()

    def test_ingest_resolves_via_registry_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        main(["init", str(tmp_path / "c"), "--register-as", "ml", "--set-default"])
        (tmp_path / "c" / "inbox" / "p.pdf").write_bytes(b"data")
        # No --corpus passed; should use the registered default.
        # Run from a directory without a .kg/ ancestor.
        monkeypatch.chdir(tmp_path)
        rv = main(["ingest"])
        assert rv == 0
        assert (tmp_path / "c" / "papers" / "p.pdf").exists()


class TestStatusSubcommand:
    def test_status_reports_counts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        main(["init", str(tmp_path / "c")])
        (tmp_path / "c" / "inbox" / "a.pdf").write_bytes(b"a")
        (tmp_path / "c" / "inbox" / "b.pdf").write_bytes(b"b")
        (tmp_path / "c" / "inbox" / "weird.xyz").write_bytes(b"x")
        main(["ingest", "--corpus", str(tmp_path / "c")])

        capsys.readouterr()  # clear ingest output
        rv = main(["status", "--corpus", str(tmp_path / "c")])
        assert rv == 0
        out = capsys.readouterr().out
        assert "total:    3" in out
        assert "ingested" in out
        assert "quarantined" in out


class TestCorpusListSubcommand:
    def test_lists_registered_corpora(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        main(["init", str(tmp_path / "a"), "--register-as", "alpha", "--set-default"])
        main(["init", str(tmp_path / "b"), "--register-as", "beta"])
        capsys.readouterr()
        rv = main(["corpus", "list"])
        assert rv == 0
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out
        assert "*" in out  # default marker

    def test_empty_registry_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        rv = main(["corpus", "list"])
        assert rv == 0
        assert "registry is empty" in capsys.readouterr().out


class TestNoSubcommandShowsHelp:
    def test_prints_help(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        rv = main([])
        assert rv == 0
        assert "usage" in capsys.readouterr().out.lower()


class TestResolutionFailure:
    def test_no_corpus_anywhere_returns_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        # Sandbox into a directory with no .kg/ ancestor and no registry.
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.chdir(sandbox)
        with pytest.raises(SystemExit) as exc:
            main(["ingest"])
        assert exc.value.code == 2
        assert "could not resolve a corpus" in capsys.readouterr().err
