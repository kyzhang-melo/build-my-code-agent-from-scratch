from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def test_search_tools_are_registered(load_module) -> None:
    tools = load_module("tools", "tools.py")

    tool_names = {tool["name"] for tool in tools.TOOLS}
    assert "glob" in tool_names
    assert "grep" in tool_names
    assert "glob" in tools.TOOL_REGISTRY
    assert "grep" in tools.TOOL_REGISTRY
    glob_schema = next(tool for tool in tools.TOOLS if tool["name"] == "glob")
    grep_schema = next(tool for tool in tools.TOOLS if tool["name"] == "grep")
    assert "pass it as directory" in glob_schema["description"]
    assert "Put the search location in path" in grep_schema["description"]


def test_run_glob_finds_sorted_relative_matches(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR / "tests" / "_tmp_glob")
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / "b.py").write_text("b")
        (base / "a.py").write_text("a")
        (base / "notes.txt").write_text("notes")

        output = tools.run_glob("*.py", "tests/_tmp_glob")

        assert "Found 2 matches" in output
        assert output.splitlines()[1:] == ["a.py", "b.py"]
    finally:
        for child in base.glob("*"):
            child.unlink()
        base.rmdir()


def test_run_glob_can_exclude_directories(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR / "tests" / "_tmp_glob_dirs")
    try:
        (base / "pkg").mkdir(parents=True, exist_ok=True)
        (base / "file.py").write_text("file")

        output = tools.run_glob("*", "tests/_tmp_glob_dirs", include_dirs=False)

        assert "file.py" in output
        assert "pkg" not in output.splitlines()[1:]
    finally:
        (base / "file.py").unlink()
        (base / "pkg").rmdir()
        base.rmdir()


def test_run_glob_rejects_escape(load_module) -> None:
    tools = load_module("tools", "tools.py")

    assert tools.run_glob("*.py", "../").startswith("Error: Path escapes workspace")


def test_run_glob_broad_pattern_returns_top_level_listing(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR / "tests" / "_tmp_glob_broad")
    try:
        (base / "pkg").mkdir(parents=True, exist_ok=True)
        (base / "__pycache__").mkdir(parents=True, exist_ok=True)
        (base / "root.py").write_text("root")

        output = tools.run_glob("**/*", "tests/_tmp_glob_broad")

        assert output.startswith("Error: pattern `**/*` matches everything")
        assert "pkg/" in output  # directories marked with a trailing slash
        assert "root.py" in output
        assert "__pycache__" not in output  # noisy dir omitted from the listing
    finally:
        (base / "root.py").unlink()
        (base / "pkg").rmdir()
        (base / "__pycache__").rmdir()
        base.rmdir()


def test_run_glob_recursive_finds_nested_and_root_level(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR / "tests" / "_tmp_glob_deep")
    try:
        (base / "a" / "b").mkdir(parents=True, exist_ok=True)
        (base / "config.json").write_text("root")  # depth 0
        (base / "a" / "b" / "config.json").write_text("nested")  # depth 2

        output = tools.run_glob("**/config.json", "tests/_tmp_glob_deep")

        assert "Found 2 matches" in output
        assert "config.json" in output
        assert "a/b/config.json" in output
    finally:
        (base / "config.json").unlink()
        (base / "a" / "b" / "config.json").unlink()
        (base / "a" / "b").rmdir()
        (base / "a").rmdir()
        base.rmdir()


def test_run_glob_recursive_prunes_excluded_dirs(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR / "tests" / "_tmp_glob_prune")
    try:
        (base / "src").mkdir(parents=True, exist_ok=True)
        (base / "node_modules" / "dep").mkdir(parents=True, exist_ok=True)
        (base / "src" / "keep.py").write_text("keep")
        (base / "node_modules" / "dep" / "skip.py").write_text("skip")

        output = tools.run_glob("**/*.py", "tests/_tmp_glob_prune", include_dirs=False)

        assert "Found 1 matches" in output
        assert "src/keep.py" in output
        assert "node_modules" not in output
    finally:
        (base / "src" / "keep.py").unlink()
        (base / "node_modules" / "dep" / "skip.py").unlink()
        (base / "node_modules" / "dep").rmdir()
        (base / "node_modules").rmdir()
        (base / "src").rmdir()
        base.rmdir()


def test_run_glob_allows_recursive_pattern_in_specific_directory(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR / "tests" / "_tmp_glob_recursive")
    try:
        (base / "pkg").mkdir(parents=True, exist_ok=True)
        (base / "pkg" / "nested.py").write_text("nested")

        output = tools.run_glob("**/*.py", "tests/_tmp_glob_recursive", include_dirs=False)

        assert "Found 1 matches" in output
        assert "pkg/nested.py" in output
    finally:
        (base / "pkg" / "nested.py").unlink()
        (base / "pkg").rmdir()
        base.rmdir()


def test_run_glob_limits_results(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR / "tests" / "_tmp_glob_limit")
    try:
        base.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            (base / f"{index}.txt").write_text(str(index))

        output = tools.run_glob("*.txt", "tests/_tmp_glob_limit", limit=2)

        assert "Found 3 matches" in output
        assert "Showing first 2" in output
        assert len(output.splitlines()) == 3
    finally:
        for child in base.glob("*"):
            child.unlink()
        base.rmdir()


def test_run_grep_reports_missing_ripgrep(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)

    output = tools.run_grep("needle")

    assert output.startswith("Error: ripgrep (`rg`) is not installed")


@pytest.mark.parametrize(
    "output_mode, expected_flag",
    [
        ("files_with_matches", "--files-with-matches"),
        ("count_matches", "--count-matches"),
        ("content", "--line-number"),
    ],
)
def test_run_grep_builds_safe_rg_command(
    load_module,
    monkeypatch,
    output_mode,
    expected_flag,
) -> None:
    tools = load_module("tools", "tools.py")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout=f"{tools.WORKDIR}/tools.py\n", stderr="")

    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    output = tools.run_grep(
        "needle",
        path=".",
        glob="*.py",
        output_mode=output_mode,
        ignore_case=True,
    )

    assert output == "tools.py"
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == str(tools.WORKDIR)
    assert captured["kwargs"]["timeout"] == 20
    assert expected_flag in captured["args"]
    assert "--ignore-case" in captured["args"]
    assert "--glob" in captured["args"]
    assert "*.py" in captured["args"]
    assert "!.env" in captured["args"]
    assert captured["args"][-3:] == ["--", "needle", str(tools.WORKDIR)]


def test_run_grep_handles_no_matches_and_limits_output(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")

    def no_match(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr(tools.subprocess, "run", no_match)
    assert tools.run_grep("missing") == "No matches found."

    def many_matches(args, **kwargs):
        stdout = "\n".join(f"{tools.WORKDIR}/file_{index}.py" for index in range(3))
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(tools.subprocess, "run", many_matches)
    output = tools.run_grep("needle", head_limit=2)

    assert output.splitlines() == [
        "file_0.py",
        "file_1.py",
        "... (1 more lines)",
    ]


def test_run_grep_rejects_workspace_escape(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/rg")

    output = tools.run_grep("needle", path="../")

    assert output.startswith("Error: Path escapes workspace")


@pytest.mark.parametrize(
    "item, expected_substring",
    [
        (_fc("grep", "g1", '{"pattern":"needle","output_mode":"bad"}'), "Input should be"),
        (_fc("grep", "g2", '{"pattern":"needle","head_limit":true}'), "Input should be a valid integer"),
        (_fc("grep", "g3", '{"pattern":"needle","extra":1}'), "Extra inputs are not permitted"),
        (_fc("glob", "gb1", '{"pattern":"*.py","limit":"1"}'), "Input should be a valid integer"),
        (_fc("glob", "gb2", '{"pattern":"*.py","limit":0}'), "Input should be greater than or equal to 1"),
    ],
)
def test_search_tools_validate_with_pydantic(load_module, item, expected_substring) -> None:
    tools = load_module("tools", "tools.py")
    out, used_todo = tools.execute_tool_calls([item])

    assert used_todo is False
    assert f"Error: invalid arguments for tool '{item.name}':" in out[0]["output"]
    assert expected_substring in out[0]["output"]


def test_execute_tool_calls_sanitizes_search_arguments(load_module) -> None:
    tools = load_module("tools", "tools.py")
    tools.run_glob = lambda pattern, directory, include_dirs, limit: f"{pattern}:{directory}"
    tools.TOOL_REGISTRY["glob"].execute = lambda params: tools.run_glob(
        params.pattern,
        params.directory,
        params.include_dirs,
        params.limit,
    )

    out, used_todo = tools.execute_tool_calls([
        _fc("glob", "gb1", '{"pattern":" >  *.py","directory":" $#  tests"}'),
    ])

    assert used_todo is False
    assert out[0]["output"] == "*.py:tests"
