from __future__ import annotations

import contextlib
from pathlib import Path


@contextlib.contextmanager
def _tmp_file(tools, name: str, content: str):
    """Create a file under the workspace, yield its workspace-relative path, clean up."""
    rel = f"tests/{name}"
    fp = Path(tools.WORKDIR / rel)
    fp.write_text(content)
    try:
        yield rel
    finally:
        fp.unlink(missing_ok=True)


def test_offset_and_limit_window(load_module) -> None:
    tools = load_module("tools", "tools.py")
    content = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"  # 10 lines
    with _tmp_file(tools, "_read_window.txt", content) as rel:
        out = tools.run_read(rel, offset=3, limit=4)

    assert "3\tline3" in out
    assert "6\tline6" in out
    assert "2\tline2" not in out   # before the window
    assert "7\tline7" not in out   # after the window
    assert "lines 3-6" in out


def test_line_number_format(load_module) -> None:
    tools = load_module("tools", "tools.py")
    with _tmp_file(tools, "_read_fmt.txt", "alpha\nbeta\n") as rel:
        out = tools.run_read(rel)
    assert "1\talpha" in out
    assert "2\tbeta" in out


def test_default_cap_truncates_with_continue_hint(load_module) -> None:
    tools = load_module("tools", "tools.py")
    content = "\n".join(f"L{i}" for i in range(1, 1201)) + "\n"  # 1200 short lines
    with _tmp_file(tools, "_read_cap.txt", content) as rel:
        out = tools.run_read(rel)  # default offset=1, limit=1000

    assert "1000\tL1000" in out
    assert "1001\tL1001" not in out          # capped at MAX_READ_LINES
    assert f"Stopped at the {tools.MAX_READ_LINES}-line limit" in out
    assert "use offset=1001 to continue" in out.lower()
    assert "Total lines: 1200" in out         # small file -> exact total


def test_byte_cap_stops_collection(load_module) -> None:
    tools = load_module("tools", "tools.py")
    # 100 lines * ~1000 chars each ~= 100KB, well over MAX_READ_BYTES (40KB),
    # so the byte cap trips before the 1000-line cap. Each line < MAX_LINE_CHARS.
    content = "\n".join("x" * 1000 for _ in range(100)) + "\n"
    with _tmp_file(tools, "_read_bytes.txt", content) as rel:
        out = tools.run_read(rel)

    assert f"Stopped at the {tools.MAX_READ_BYTES}-byte limit" in out
    numbered = [ln for ln in out.splitlines() if "\t" in ln]
    assert 0 < len(numbered) < 100


def test_eof_footer_and_exact_total(load_module) -> None:
    tools = load_module("tools", "tools.py")
    with _tmp_file(tools, "_read_eof.txt", "a\nb\nc\nd\ne\n") as rel:
        out = tools.run_read(rel)
    assert "End of file" in out
    assert "Total lines: 5" in out
    assert "offset=" not in out  # nothing more to read -> no continue hint


def test_per_line_truncation(load_module) -> None:
    tools = load_module("tools", "tools.py")
    long_line = "z" * (tools.MAX_LINE_CHARS + 500)
    with _tmp_file(tools, "_read_longline.txt", long_line + "\n") as rel:
        out = tools.run_read(rel)
    assert "[...line truncated]" in out
    assert "truncated to" in out  # footer note
    body = out.splitlines()[0]
    assert len(body) < tools.MAX_LINE_CHARS + 100  # line was clipped


def test_missing_and_nonfile_paths(load_module) -> None:
    tools = load_module("tools", "tools.py")
    assert tools.run_read("tests/_does_not_exist.txt").startswith("Error: File not found")
    assert tools.run_read("tests").startswith("Error: Not a file")


def test_backward_compat_small_file_no_args(load_module) -> None:
    tools = load_module("tools", "tools.py")
    with _tmp_file(tools, "_read_compat.txt", "hello world\n") as rel:
        out = tools.run_read(rel)
    assert "hello world" in out


# --- Stage 2 ---


def test_offset_beyond_eof_is_system_reminder(load_module) -> None:
    tools = load_module("tools", "tools.py")
    with _tmp_file(tools, "_read_oob.txt", "a\nb\nc\n") as rel:
        out = tools.run_read(rel, offset=100)
    assert "<system-reminder>" in out
    assert "past the end of the file" in out
    assert "3 lines" in out


def test_empty_file_is_system_reminder(load_module) -> None:
    tools = load_module("tools", "tools.py")
    with _tmp_file(tools, "_read_empty.txt", "") as rel:
        out = tools.run_read(rel)
    assert out == "<system-reminder>File exists but is empty.</system-reminder>"


def test_binary_file_rejected(load_module) -> None:
    tools = load_module("tools", "tools.py")
    rel = "tests/_read_binary.bin"
    fp = Path(tools.WORKDIR / rel)
    fp.write_bytes(b"\x89PNG\x00\x01\x02binary\x00data")
    try:
        out = tools.run_read(rel)
    finally:
        fp.unlink(missing_ok=True)
    assert out.startswith("Error:")
    assert "binary" in out.lower()


def test_large_file_guidance(load_module) -> None:
    tools = load_module("tools", "tools.py")
    content = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
    original = tools.MAX_READ_FILE_BYTES
    tools.MAX_READ_FILE_BYTES = 1  # force the "large file" branch on a small file
    try:
        with _tmp_file(tools, "_read_large.txt", content) as rel:
            out = tools.run_read(rel)
    finally:
        tools.MAX_READ_FILE_BYTES = original
    assert "large file" in out.lower()
    assert "grep" in out.lower()


def test_concurrency_safe_flags(load_module) -> None:
    tools = load_module("tools", "tools.py")
    registry = tools.TOOL_REGISTRY
    for name in ("read_file", "glob", "grep", "task"):
        assert registry[name].concurrency_safe is True, name
    for name in ("bash", "write_file", "edit_file", "todo"):
        assert registry[name].concurrency_safe is False, name
