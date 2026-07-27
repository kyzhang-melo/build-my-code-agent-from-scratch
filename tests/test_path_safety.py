from __future__ import annotations


def test_resolve_rejects_workspace_escape(workspace) -> None:
    try:
        workspace.resolve("../outside.txt")
        raise AssertionError("Expected ValueError for path escape")
    except ValueError:
        pass


def test_file_tools_stay_inside_workspace(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    escaped_write = tools.run_write(workspace, "../outside.txt", "blocked")
    assert escaped_write.startswith("Error: Path escapes workspace")

    rel = "boundary.txt"
    write_out = tools.run_write(workspace, rel, "hello world")
    assert write_out.startswith("Wrote ")

    assert "hello world" in tools.run_read(workspace, rel, limit=1)
    edit_out = tools.run_edit(workspace, rel, "hello", "HELLO")
    assert edit_out.startswith(f"Edited {rel}")
    assert "1 replacement" in edit_out
    assert "HELLO world" in tools.run_read(workspace, rel)
