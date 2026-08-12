from __future__ import annotations


def test_resolve_allows_explicit_external_absolute_path(workspace, tmp_path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external.txt"

    assert workspace.resolve(
        str(external),
        allow_external_absolute=True,
    ) == external.resolve()


def test_resolve_rejects_workspace_escape(workspace) -> None:
    try:
        workspace.resolve("../outside.txt")
        raise AssertionError("Expected ValueError for path escape")
    except ValueError:
        pass


def test_resolve_rejects_relative_escape_even_when_external_absolute_is_allowed(
    workspace,
) -> None:
    try:
        workspace.resolve("../outside.txt", allow_external_absolute=True)
        raise AssertionError("Expected ValueError for relative path escape")
    except ValueError:
        pass


def test_resolve_rejects_relative_symlink_escape(
    workspace,
    tmp_path,
) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external-dir"
    external.mkdir()
    link = workspace.root / "external-link"
    link.symlink_to(external, target_is_directory=True)

    try:
        workspace.resolve(
            "external-link/file.txt",
            allow_external_absolute=True,
        )
        raise AssertionError("Expected ValueError for symlink escape")
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


def test_read_file_allows_external_absolute_text_path(
    load_module,
    workspace,
    tmp_path,
) -> None:
    tools = load_module("tools", "tools.py")
    external = tmp_path.parent / f"{tmp_path.name}-read-external.txt"
    external.write_text("outside workspace", encoding="utf-8")

    assert "outside workspace" in tools.run_read(workspace, str(external))


def test_read_file_blocks_external_sensitive_path(
    load_module,
    workspace,
    tmp_path,
) -> None:
    tools = load_module("tools", "tools.py")
    ssh_dir = tmp_path.parent / f"{tmp_path.name}-external-home" / ".ssh"
    ssh_dir.mkdir(parents=True)
    private_key = ssh_dir / "custom-name"
    private_key.write_text("secret", encoding="utf-8")

    output = tools.run_read(workspace, str(private_key))

    assert "sensitive path" in output.lower()
