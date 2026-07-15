from __future__ import annotations

from pathlib import Path

import pytest


_TMP_DIR = "tests/_tmp_write"


@pytest.fixture
def tools(load_module):
    return load_module("tools", "tools.py")


@pytest.fixture
def tmp_dir(tools):
    directory = tools.WORKDIR / _TMP_DIR
    directory.mkdir(parents=True, exist_ok=True)
    yield directory
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


def _rel(*parts: str) -> str:
    return f"{_TMP_DIR}/{Path(*parts).as_posix()}"


def _file(tools, *parts: str) -> Path:
    return tools.WORKDIR / _TMP_DIR / Path(*parts)


def _seed(tools, path: str, content: str) -> Path:
    fp = tools.WORKDIR / path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8", newline="")
    return fp


def test_overwrite_new_and_existing_files(tools, tmp_dir):
    rel = _rel("target.txt")
    assert tools.run_write(rel, "hello").startswith("Wrote ")
    assert _file(tools, "target.txt").read_text(encoding="utf-8") == "hello"

    result = tools.run_write(rel, "after", mode="overwrite")
    assert result.startswith("Wrote ")
    assert _file(tools, "target.txt").read_text(encoding="utf-8") == "after"


def test_append_existing_and_missing_file_without_newline_insertion(tools, tmp_dir):
    existing = _rel("existing.txt")
    _seed(tools, existing, "abc")
    assert tools.run_write(existing, "def", mode="append").startswith("Appended ")
    assert _file(tools, "existing.txt").read_text(encoding="utf-8") == "abcdef"

    missing = _rel("nested/new.txt")
    assert tools.run_write(missing, "first", mode="append").startswith("Appended ")
    assert _file(tools, "nested", "new.txt").read_text(encoding="utf-8") == "first"


def test_write_creates_nested_parent_directories(tools, tmp_dir):
    rel = _rel("a/b/c/deep.txt")
    assert tools.run_write(rel, "deep").startswith("Wrote ")
    assert _file(tools, "a", "b", "c", "deep.txt").read_text(encoding="utf-8") == "deep"


def test_write_reports_utf8_byte_count_and_current_size(tools, tmp_dir):
    result = tools.run_write(_rel("unicode.txt"), "héllo")
    assert "6 bytes" in result
    assert "current size: 6 bytes" in result


def test_invalid_mode_and_workspace_escape_do_not_write(tools, tmp_dir):
    target = _file(tools, "bad.txt")
    assert tools.run_write(_rel("bad.txt"), "x", mode="bogus").startswith("Error: mode")
    assert not target.exists()
    assert tools.run_write("../outside.txt", "blocked").startswith("Error: Path escapes workspace")
