from __future__ import annotations

from pathlib import Path


def test_safe_path_rejects_workspace_escape(load_module) -> None:
    tools = load_module("tools", "tools.py")
    try:
        tools.safe_path("../outside.txt")
        raise AssertionError("Expected ValueError for path escape")
    except ValueError:
        pass


def test_file_tools_stay_inside_workspace(load_module) -> None:
    tools = load_module("tools", "tools.py")

    escaped_write = tools.run_write("../outside.txt", "blocked")
    assert escaped_write.startswith("Error: Path escapes workspace")

    rel = "tests/_tmp_boundary.txt"
    write_out = tools.run_write(rel, "hello world")
    assert write_out.startswith("Wrote ")

    assert "hello world" in tools.run_read(rel, limit=1)
    assert tools.run_edit(rel, "hello", "HELLO") == f"Edited {rel}"
    assert "HELLO world" in tools.run_read(rel)

    Path(tools.WORKDIR / rel).unlink(missing_ok=True)
