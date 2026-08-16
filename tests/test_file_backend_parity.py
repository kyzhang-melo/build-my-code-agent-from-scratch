from __future__ import annotations

import shutil

import file_bridge


def _pair(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    return local, remote


def test_read_and_glob_match_container_bridge(load_module, tmp_path) -> None:
    tools = load_module("tools", "tools.py")
    from workspace import Workspace

    local, remote = _pair(tmp_path)
    (local / "pkg").mkdir()
    (local / "pkg" / "sample.py").write_text(
        "first\n" + "x" * 2100 + "\nthird\n", encoding="utf-8",
    )
    shutil.copytree(local, remote, dirs_exist_ok=True)
    workspace = Workspace(local)

    local_read = tools.run_read(workspace, "pkg/sample.py", 1, 2)
    remote_read = file_bridge.dispatch(
        "read", {"path": "pkg/sample.py", "offset": 1, "limit": 2}, root=remote,
    )
    local_glob = tools.run_glob(workspace, "**/*.py", ".", False, 100)
    remote_glob = file_bridge.dispatch(
        "glob", {"pattern": "**/*.py", "directory": ".", "include_dirs": False, "limit": 100},
        root=remote,
    )

    assert remote_read == local_read
    assert remote_glob == local_glob


def test_write_and_fuzzy_crlf_edit_match_container_bridge(load_module, tmp_path) -> None:
    tools = load_module("tools", "tools.py")
    from workspace import Workspace

    local, remote = _pair(tmp_path)
    original = "title = \u201chello\u201d  \r\nunchanged  \r\n"
    (local / "config.txt").write_bytes(original.encode("utf-8"))
    shutil.copytree(local, remote, dirs_exist_ok=True)
    workspace = Workspace(local)
    edits = [{"old_text": 'title = "hello"', "new_text": 'title = "updated"'}]
    expected = 'title = "updated"\r\nunchanged  \r\n'.encode("utf-8")

    local_edit = tools.run_edit(workspace, "config.txt", edits)
    remote_edit = file_bridge.dispatch(
        "edit", {"path": "config.txt", "edits": edits}, root=remote,
    )
    local_write = tools.run_write(workspace, "new.txt", "one\ntwo\n")
    remote_write = file_bridge.dispatch(
        "write", {"path": "new.txt", "content": "one\ntwo\n", "mode": "overwrite"},
        root=remote,
    )

    assert remote_edit == local_edit
    assert (local / "config.txt").read_bytes() == expected
    assert (remote / "config.txt").read_bytes() == expected
    assert remote_write == local_write
    assert (remote / "new.txt").read_bytes() == (local / "new.txt").read_bytes()


def test_read_rejects_known_binary_without_nul_in_both_backends(
    load_module, tmp_path,
) -> None:
    tools = load_module("tools", "tools.py")
    from workspace import Workspace

    local, remote = _pair(tmp_path)
    pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n" + b"A" * 5000
    (local / "doc.PDF").write_bytes(pdf)
    shutil.copytree(local, remote, dirs_exist_ok=True)

    local_read = tools.run_read(Workspace(local), "doc.PDF")
    remote_read = file_bridge.dispatch(
        "read", {"path": "doc.PDF", "offset": 1, "limit": 1000}, root=remote,
    )

    expected = (
        "Error: 'doc.PDF' appears to be a binary or non-text file. "
        "Use appropriate tools to inspect it."
    )
    assert local_read == expected
    assert remote_read == expected
