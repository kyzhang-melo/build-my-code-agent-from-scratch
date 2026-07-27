from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tools(load_module):
    return load_module("tools", "tools.py")


def _file(workspace, *parts: str) -> Path:
    return workspace.root / Path(*parts)


def _seed(workspace, path: str, content: str) -> Path:
    fp = workspace.root / path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8", newline="")
    return fp


def test_overwrite_new_and_existing_files(tools, workspace):
    assert tools.run_write(workspace, "target.txt", "hello").startswith("Wrote ")
    assert _file(workspace, "target.txt").read_text(encoding="utf-8") == "hello"

    result = tools.run_write(workspace, "target.txt", "after", mode="overwrite")
    assert result.startswith("Wrote ")
    assert _file(workspace, "target.txt").read_text(encoding="utf-8") == "after"


def test_append_existing_and_missing_file_without_newline_insertion(tools, workspace):
    _seed(workspace, "existing.txt", "abc")
    assert tools.run_write(workspace, "existing.txt", "def", mode="append").startswith("Appended ")
    assert _file(workspace, "existing.txt").read_text(encoding="utf-8") == "abcdef"

    missing = "nested/new.txt"
    assert tools.run_write(workspace, missing, "first", mode="append").startswith("Appended ")
    assert _file(workspace, "nested", "new.txt").read_text(encoding="utf-8") == "first"


def test_write_creates_nested_parent_directories(tools, workspace):
    rel = "a/b/c/deep.txt"
    assert tools.run_write(workspace, rel, "deep").startswith("Wrote ")
    assert _file(workspace, "a", "b", "c", "deep.txt").read_text(encoding="utf-8") == "deep"


def test_write_reports_utf8_byte_count_and_current_size(tools, workspace):
    result = tools.run_write(workspace, "unicode.txt", "héllo")
    assert "6 bytes" in result
    assert "current size: 6 bytes" in result


def test_invalid_mode_and_workspace_escape_do_not_write(tools, workspace):
    target = _file(workspace, "bad.txt")
    assert tools.run_write(workspace, "bad.txt", "x", mode="bogus").startswith("Error: mode")
    assert not target.exists()
    assert tools.run_write(workspace, "../outside.txt", "blocked").startswith(
        "Error: Path escapes workspace"
    )
