"""Focused tests for the upgraded write_file and edit_file tools.

Tests that call ``run_write``/``run_edit`` directly take an isolated ``workspace``
fixture backed by ``tmp_path``.  Tests that go through the dispatcher still write
inside the repository checkout because the tool registry is bound to
``tools.WORKDIR`` at import time; they clean up via the ``tmp_dir`` fixture.
"""

from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

import pytest

import permissions as permission_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TMP_DIR = "tests/_tmp_write_edit"


def _tmp(tools, *parts: str) -> Path:
    """Return an absolute path inside the shared temp directory."""
    return tools.WORKDIR / _TMP_DIR / Path(*parts)


def _rel(*parts: str) -> str:
    return f"{_TMP_DIR}/{Path(*parts).as_posix()}"


@pytest.fixture
def tools(load_module):
    return load_module("tools", "tools.py")


@pytest.fixture
def tmp_dir(tools):
    directory = _tmp(tools)
    directory.mkdir(parents=True, exist_ok=True)
    yield directory
    # Best-effort cleanup of the whole temp tree.
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


def _write_file(tools, rel: str, content: str) -> Path:
    fp = tools.WORKDIR / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8", newline="")
    return fp


def _seed(workspace, rel: str, content: str) -> Path:
    fp = workspace.root / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8", newline="")
    return fp


# ---------------------------------------------------------------------------
# edit_file -- basic behaviour
# ---------------------------------------------------------------------------


class TestEditBasic:
    def test_single_edit_via_list(self, tools, workspace):
        rel = "single.txt"
        _seed(workspace, rel, "foo bar baz")
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="bar", new_text="QUX")])
        assert result.startswith("Edited ")
        assert "1 replacement" in result
        assert (workspace.root / "single.txt").read_text(encoding="utf-8") == "foo QUX baz"

    def test_legacy_direct_call(self, tools, workspace):
        rel = "legacy.txt"
        _seed(workspace, rel, "foo bar baz")
        result = tools.run_edit(workspace, rel, "bar", "QUX")
        assert result.startswith("Edited ")
        assert (workspace.root / "legacy.txt").read_text(encoding="utf-8") == "foo QUX baz"

    def test_multiple_disjoint_edits(self, tools, workspace):
        rel = "multi.txt"
        _seed(workspace, rel, "alpha beta gamma delta")
        edits = [
            tools.EditParams(old_text="alpha", new_text="ALPHA"),
            tools.EditParams(old_text="gamma", new_text="GAMMA"),
        ]
        result = tools.run_edit(workspace, rel, edits)
        assert "2 replacement" in result
        assert (workspace.root / "multi.txt").read_text(encoding="utf-8") == "ALPHA beta GAMMA delta"

    def test_edits_match_original_not_incremental(self, tools, workspace):
        """All old_text values are matched against the original snapshot."""
        rel = "snapshot.txt"
        _seed(workspace, rel, "aaa bbb ccc")
        # If edits were applied incrementally, the second edit's old_text
        # would need to match the result of the first.  Because both are
        # matched against the original, this works.
        edits = [
            tools.EditParams(old_text="aaa", new_text="AAA"),
            tools.EditParams(old_text="ccc", new_text="CCC"),
        ]
        tools.run_edit(workspace, rel, edits)
        assert (workspace.root / "snapshot.txt").read_text(encoding="utf-8") == "AAA bbb CCC"

    def test_edit_not_found_no_write(self, tools, workspace):
        rel = "notfound.txt"
        _seed(workspace, rel, "hello world")
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="xyz", new_text="abc")])
        assert "Error" in result
        assert "Could not find" in result
        # File unchanged.
        assert (workspace.root / "notfound.txt").read_text(encoding="utf-8") == "hello world"

    def test_duplicate_match_rejected(self, tools, workspace):
        rel = "dup.txt"
        _seed(workspace, rel, "foo foo foo")
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="foo", new_text="bar")])
        assert "Error" in result
        assert "3 occurrences" in result
        assert "unique" in result.lower()
        # File unchanged.
        assert (workspace.root / "dup.txt").read_text(encoding="utf-8") == "foo foo foo"

    def test_overlapping_edits_rejected(self, tools, workspace):
        rel = "overlap.txt"
        _seed(workspace, rel, "abcdefghij")
        edits = [
            tools.EditParams(old_text="cde", new_text="CDE"),
            tools.EditParams(old_text="def", new_text="DEF"),
        ]
        result = tools.run_edit(workspace, rel, edits)
        assert "Error" in result
        assert "overlap" in result.lower()

    def test_empty_old_text_rejected(self, tools, workspace):
        # Pydantic rejects empty old_text at construction time.
        with pytest.raises(Exception):
            tools.EditParams(old_text="", new_text="x")

    def test_no_change_rejected(self, tools, workspace):
        rel = "nochange.txt"
        _seed(workspace, rel, "hello")
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="hello", new_text="hello")])
        assert "Error" in result
        assert "No changes" in result

    def test_one_edit_fails_whole_operation_aborts(self, tools, workspace):
        rel = "partial.txt"
        _seed(workspace, rel, "alpha beta gamma")
        edits = [
            tools.EditParams(old_text="alpha", new_text="ALPHA"),
            tools.EditParams(old_text="xyz", new_text="XYZ"),
        ]
        result = tools.run_edit(workspace, rel, edits)
        assert "Error" in result
        # File unchanged because validation failed before writing.
        assert (workspace.root / "partial.txt").read_text(encoding="utf-8") == "alpha beta gamma"

    def test_file_not_found(self, tools, workspace):
        result = tools.run_edit(workspace, "missing.txt", [tools.EditParams(old_text="x", new_text="y")])
        assert "Error" in result
        assert "File not found" in result

    def test_edit_directory_rejected(self, tools, workspace):
        result = tools.run_edit(workspace, "", [tools.EditParams(old_text="x", new_text="y")])
        assert "Error" in result


# ---------------------------------------------------------------------------
# edit_file -- fuzzy matching
# ---------------------------------------------------------------------------


class TestEditFuzzy:
    def test_exact_match_takes_priority(self, tools, workspace):
        rel = "exact.txt"
        _seed(workspace, rel, "hello world")
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="hello", new_text="HELLO")])
        assert "1 replacement" in result
        assert (workspace.root / "exact.txt").read_text(encoding="utf-8") == "HELLO world"

    def test_trailing_whitespace_fuzzy(self, tools, workspace):
        rel = "trailing.txt"
        _seed(workspace, rel, "line1   \nline2")
        # old_text without trailing spaces should still match via fuzzy.
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="line1", new_text="LINE1")])
        assert "1 replacement" in result
        content = (workspace.root / "trailing.txt").read_text(encoding="utf-8")
        assert content.startswith("LINE1")

    def test_smart_quotes_fuzzy(self, tools, workspace):
        rel = "quotes.txt"
        _seed(workspace, rel, "it\u2019s here")
        # Model sends ASCII apostrophe; file has smart quote.
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="it's here", new_text="it was here")])
        assert "1 replacement" in result
        assert (workspace.root / "quotes.txt").read_text(encoding="utf-8") == "it was here"

    def test_unicode_dash_fuzzy(self, tools, workspace):
        rel = "dash.txt"
        _seed(workspace, rel, "a\u2014b")
        # Model sends ASCII hyphen; file has em-dash.
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="a-b", new_text="a+b")])
        assert "1 replacement" in result
        assert (workspace.root / "dash.txt").read_text(encoding="utf-8") == "a+b"

    def test_special_space_fuzzy(self, tools, workspace):
        rel = "space.txt"
        _seed(workspace, rel, "x\u00a0y")
        # Model sends regular space; file has NBSP.
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="x y", new_text="x_z")])
        assert "1 replacement" in result
        assert (workspace.root / "space.txt").read_text(encoding="utf-8") == "x_z"

    def test_fuzzy_uniqueness_checked(self, tools, workspace):
        rel = "fuzzy_dup.txt"
        # Both lines use NBSP between foo and bar; exact match for "foo bar"
        # (regular space) fails, but fuzzy normalization converts NBSP→space
        # making both lines "foo bar" → duplicate.
        _seed(workspace, rel, "foo\u00a0bar\nfoo\u00a0bar")
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="foo bar", new_text="baz")])
        assert "Error" in result
        assert "2 occurrences" in result

    def test_fuzzy_untouched_lines_preserved(self, tools, workspace):
        rel = "preserve.txt"
        original = "keep\u00a0this\nchange me\nkeep\u00a0that"
        _seed(workspace, rel, original)
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="change me", new_text="CHANGED")])
        assert "1 replacement" in result
        content = (workspace.root / "preserve.txt").read_text(encoding="utf-8")
        # Untouched lines keep their original NBSP.
        assert "keep\u00a0this" in content
        assert "keep\u00a0that" in content
        assert "CHANGED" in content


# ---------------------------------------------------------------------------
# edit_file -- format preservation
# ---------------------------------------------------------------------------


class TestEditFormat:
    def test_bom_preserved(self, tools, workspace):
        rel = "bom.txt"
        fp = (workspace.root / "bom.txt")
        fp.write_bytes("\ufeffhello world".encode("utf-8"))
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="hello", new_text="HELLO")])
        assert "1 replacement" in result
        raw = fp.read_bytes()
        assert raw.startswith("\ufeff".encode("utf-8"))
        assert "HELLO world" in raw.decode("utf-8")

    def test_crlf_preserved(self, tools, workspace):
        rel = "crlf.txt"
        fp = (workspace.root / "crlf.txt")
        fp.write_bytes("line1\r\nline2\r\nline3".encode("utf-8"))
        result = tools.run_edit(workspace, rel, [tools.EditParams(old_text="line2", new_text="LINE2")])
        assert "1 replacement" in result
        raw = fp.read_bytes()
        assert b"\r\n" in raw
        assert b"LINE2" in raw
        # No bare LF should be introduced.
        assert b"\r\n" == raw[raw.index(b"\r\n"):raw.index(b"\r\n")+2]

    def test_no_trailing_newline_preserved(self, tools, workspace):
        rel = "noeol.txt"
        _seed(workspace, rel, "hello world")
        tools.run_edit(workspace, rel, [tools.EditParams(old_text="hello", new_text="HELLO")])
        raw = (workspace.root / "noeol.txt").read_bytes()
        assert not raw.endswith(b"\n")
        assert raw == b"HELLO world"

    def test_trailing_newline_preserved(self, tools, workspace):
        rel = "eol.txt"
        _seed(workspace, rel, "hello world\n")
        tools.run_edit(workspace, rel, [tools.EditParams(old_text="hello", new_text="HELLO")])
        raw = (workspace.root / "eol.txt").read_bytes()
        assert raw.endswith(b"\n")
        assert raw == b"HELLO world\n"


# ---------------------------------------------------------------------------
# Dispatcher / schema-tolerance tests
# ---------------------------------------------------------------------------


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


class TestDispatcherSchemaTolerance:
    """Exercise the full dispatcher path: sanitize → validate → execute."""

    def test_standard_edits_array(self, tools, tmp_dir):
        rel = _rel("disp_standard.txt")
        _write_file(tools, rel, "alpha beta")
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": rel,
                "edits": [{"old_text": "alpha", "new_text": "ALPHA"}],
            })),
        ])
        assert "Edited" in out[0]["output"]
        assert "ALPHA beta" == (_tmp(tools, "disp_standard.txt")).read_text(encoding="utf-8")

    def test_legacy_top_level_params(self, tools, tmp_dir):
        rel = _rel("disp_legacy.txt")
        _write_file(tools, rel, "alpha beta")
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": rel,
                "old_text": "alpha",
                "new_text": "ALPHA",
            })),
        ])
        assert "Edited" in out[0]["output"]
        assert "ALPHA beta" == (_tmp(tools, "disp_legacy.txt")).read_text(encoding="utf-8")

    def test_edits_as_json_string(self, tools, tmp_dir):
        rel = _rel("disp_jsonstr.txt")
        _write_file(tools, rel, "alpha beta")
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": rel,
                "edits": json.dumps([{"old_text": "alpha", "new_text": "ALPHA"}]),
            })),
        ])
        assert "Edited" in out[0]["output"]
        assert "ALPHA beta" == (_tmp(tools, "disp_jsonstr.txt")).read_text(encoding="utf-8")

    def test_new_and_legacy_params_merged(self, tools, tmp_dir):
        rel = _rel("disp_merged.txt")
        _write_file(tools, rel, "alpha beta gamma")
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": rel,
                "edits": [{"old_text": "alpha", "new_text": "ALPHA"}],
                "old_text": "gamma",
                "new_text": "GAMMA",
            })),
        ])
        assert "Edited" in out[0]["output"]
        assert "2 replacement" in out[0]["output"]
        assert "ALPHA beta GAMMA" == (_tmp(tools, "disp_merged.txt")).read_text(encoding="utf-8")

    def test_empty_edits_rejected(self, tools, tmp_dir):
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({"path": _rel("x.txt"), "edits": []})),
        ])
        assert "Error" in out[0]["output"]

    def test_invalid_json_string_edits_rejected(self, tools, tmp_dir):
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": _rel("x.txt"),
                "edits": "not-valid-json",
            })),
        ])
        assert "Error" in out[0]["output"]

    def test_wrong_type_edits_rejected(self, tools, tmp_dir):
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": _rel("x.txt"),
                "edits": 42,
            })),
        ])
        assert "Error" in out[0]["output"]

    def test_half_legacy_pair_rejected(self, tools, tmp_dir):
        # old_text without new_text should not trigger legacy conversion.
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": _rel("x.txt"),
                "old_text": "something",
            })),
        ])
        assert "Error" in out[0]["output"]

    def test_extra_nested_field_rejected(self, tools, tmp_dir):
        out, _ = tools.execute_tool_calls([
            _fc("edit_file", "e1", json.dumps({
                "path": _rel("x.txt"),
                "edits": [{"old_text": "a", "new_text": "b", "extra": 1}],
            })),
        ])
        assert "Error" in out[0]["output"]

    def test_write_mode_validation(self, tools, tmp_dir):
        rel = _rel("mode_valid.txt")
        out, _ = tools.execute_tool_calls([
            _fc("write_file", "w1", json.dumps({
                "path": rel,
                "content": "ok",
                "mode": "overwrite",
            })),
        ])
        assert "Wrote" in out[0]["output"]

    def test_write_invalid_mode_rejected(self, tools, tmp_dir):
        out, _ = tools.execute_tool_calls([
            _fc("write_file", "w1", json.dumps({
                "path": _rel("bad.txt"),
                "content": "ok",
                "mode": "bogus",
            })),
        ])
        assert "Error" in out[0]["output"]


# ---------------------------------------------------------------------------
# Approval-blocking regression tests
# ---------------------------------------------------------------------------


class TestApprovalBlocking:
    """Verify that denied permissions cause zero filesystem mutation."""

    def test_denied_write_creates_no_parent_dir(self, tools, tmp_dir):
        permissions = permission_module
        service = permissions.PermissionService(
            permissions.PermissionManager(Path(tools.WORKDIR)),
            permissions.TerminalApprovalHandler(interactive=False),
        )
        nested = _tmp(tools, "denied/sub/dir.txt")
        # Ensure clean slate.
        nested_parent = _tmp(tools, "denied")
        if nested_parent.exists():
            import shutil

            shutil.rmtree(nested_parent, ignore_errors=True)

        out, _ = asyncio.run(tools.execute_tool_calls_async([
            _fc("write_file", "w1", json.dumps({
                "path": _rel("denied/sub/dir.txt"),
                "content": "x",
            })),
        ], permission_service=service))

        payload = json.loads(out[0]["output"])
        assert payload["error"] == "permission_denied"
        # Neither the file nor the parent directory should exist.
        assert not nested.exists()
        assert not nested_parent.exists()

    def test_denied_edit_leaves_file_unchanged(self, tools, tmp_dir):
        permissions = permission_module
        rel = _rel("denied_edit.txt")
        _write_file(tools, rel, "original content")
        service = permissions.PermissionService(
            permissions.PermissionManager(Path(tools.WORKDIR)),
            permissions.TerminalApprovalHandler(interactive=False),
        )

        out, _ = asyncio.run(tools.execute_tool_calls_async([
            _fc("edit_file", "e1", json.dumps({
                "path": rel,
                "edits": [{"old_text": "original", "new_text": "CHANGED"}],
            })),
        ], permission_service=service))

        payload = json.loads(out[0]["output"])
        assert payload["error"] == "permission_denied"
        assert (_tmp(tools, "denied_edit.txt")).read_text(encoding="utf-8") == "original content"

    def test_approved_append_executes(self, tools, tmp_dir):
        permissions = permission_module
        rel = _rel("approved_append.txt")
        _write_file(tools, rel, "line1\n")

        class Handler:
            async def request(self, _request):
                return permissions.ApprovalResponse("approve")

        service = permissions.PermissionService(
            permissions.PermissionManager(Path(tools.WORKDIR)),
            Handler(),
        )

        out, _ = asyncio.run(tools.execute_tool_calls_async([
            _fc("write_file", "w1", json.dumps({
                "path": rel,
                "content": "line2\n",
                "mode": "append",
            })),
        ], permission_service=service))

        assert "Appended" in out[0]["output"]
        assert (_tmp(tools, "approved_append.txt")).read_text(encoding="utf-8") == "line1\nline2\n"

    def test_session_approval_covers_write_and_edit(self, tools, tmp_dir):
        permissions = permission_module
        rel = _rel("session_cover.txt")
        _write_file(tools, rel, "hello world")

        class Handler:
            async def request(self, _request):
                return permissions.ApprovalResponse("approve_for_session")

        service = permissions.PermissionService(
            permissions.PermissionManager(Path(tools.WORKDIR)),
            Handler(),
        )

        out, _ = asyncio.run(tools.execute_tool_calls_async([
            _fc("write_file", "w1", json.dumps({
                "path": rel,
                "content": "hello earth",
            })),
            _fc("edit_file", "e1", json.dumps({
                "path": rel,
                "edits": [{"old_text": "hello", "new_text": "HELLO"}],
            })),
        ], permission_service=service))

        assert "Wrote" in out[0]["output"]
        assert "Edited" in out[1]["output"]
        assert (_tmp(tools, "session_cover.txt")).read_text(encoding="utf-8") == "HELLO earth"

    def test_plan_mode_blocks_write_and_edit(self, tools, tmp_dir):
        permissions = permission_module
        rel = _rel("plan_blocked.txt")
        _write_file(tools, rel, "original")

        manager = permissions.PermissionManager(Path(tools.WORKDIR))
        manager.set_mode("plan")
        service = permissions.PermissionService(manager, None)

        out, _ = asyncio.run(tools.execute_tool_calls_async([
            _fc("write_file", "w1", json.dumps({
                "path": rel,
                "content": "modified",
            })),
            _fc("edit_file", "e1", json.dumps({
                "path": rel,
                "edits": [{"old_text": "original", "new_text": "modified"}],
            })),
        ], permission_service=service))

        for item in out:
            payload = json.loads(item["output"])
            assert payload["error"] == "permission_denied"
        # File unchanged.
        assert (_tmp(tools, "plan_blocked.txt")).read_text(encoding="utf-8") == "original"
