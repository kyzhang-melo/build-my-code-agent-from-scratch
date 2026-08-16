from __future__ import annotations

import subprocess
import types
import pytest


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def _runtime(tools, workspace):
    todo = tools.TodoManager()
    return tools.build_tool_registry(workspace, todo), todo


def test_search_tools_are_registered(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    registry, _ = _runtime(tools, workspace)

    tool_names = {tool["name"] for tool in tools.TOOLS}
    assert "glob" in tool_names
    assert "grep" in tool_names
    assert "glob" in registry
    assert "grep" in registry
    glob_schema = next(tool for tool in tools.TOOLS if tool["name"] == "glob")
    grep_schema = next(tool for tool in tools.TOOLS if tool["name"] == "grep")
    assert "pass it as directory" in glob_schema["description"]
    assert "CRITICAL GLOB RULE" in glob_schema["description"]
    assert "never use broad recursive patterns" in glob_schema["description"]
    assert "Put the search location in path" in grep_schema["description"]
    assert (
        "pattern, path, glob, output_mode, ignore_case, line_number, and head_limit"
        in grep_schema["description"]
    )
    assert "Do not pass command-line flags such as -n, -A, -B, or -C" in grep_schema["description"]


def test_run_glob_finds_sorted_relative_matches(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "b.py").write_text("b")
    (workspace.root / "a.py").write_text("a")
    (workspace.root / "notes.txt").write_text("notes")
    output = tools.run_glob(workspace, "*.py")
    assert "Found 2 matches" in output
    assert output.splitlines()[1:] == ["a.py", "b.py"]


def test_run_glob_can_exclude_directories(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "pkg").mkdir()
    (workspace.root / "file.py").write_text("file")
    output = tools.run_glob(workspace, "*", include_dirs=False)
    assert "file.py" in output
    assert "pkg" not in output.splitlines()[1:]


def test_run_glob_rejects_escape(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    assert tools.run_glob(workspace, "*.py", "../").startswith("Error: Path escapes workspace")


def test_run_glob_allows_external_absolute_directory(
    load_module,
    workspace,
    tmp_path,
) -> None:
    tools = load_module("tools", "tools.py")
    external = tmp_path.parent / f"{tmp_path.name}-external-glob"
    external.mkdir()
    (external / "found.py").write_text("pass", encoding="utf-8")

    output = tools.run_glob(workspace, "*.py", str(external))

    assert "found.py" in output


def test_run_glob_prunes_external_sensitive_paths(
    load_module,
    workspace,
    tmp_path,
) -> None:
    tools = load_module("tools", "tools.py")
    external = tmp_path.parent / f"{tmp_path.name}-external-sensitive-glob"
    ssh_dir = external / ".ssh"
    ssh_dir.mkdir(parents=True)
    (ssh_dir / "secret.txt").write_text("secret", encoding="utf-8")
    (external / "visible.txt").write_text("ok", encoding="utf-8")

    output = tools.run_glob(workspace, "**/*.txt", str(external))

    assert "visible.txt" in output
    assert ".ssh" not in output


def test_run_glob_broad_pattern_returns_top_level_listing(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "pkg").mkdir()
    (workspace.root / "__pycache__").mkdir()
    (workspace.root / "root.py").write_text("root")
    output = tools.run_glob(workspace, "**/*")
    assert output.startswith("Error: pattern `**/*` matches everything")
    assert "pkg/" in output
    assert "root.py" in output
    assert "__pycache__" not in output


@pytest.mark.parametrize("pattern", ["**", "**/", "**/**"])
def test_run_glob_rejects_broad_recursive_variants(load_module, workspace, pattern) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "root.py").write_text("root")
    output = tools.run_glob(workspace, pattern)
    assert output.startswith(f"Error: pattern `{pattern}` matches everything")
    assert "`**/*.py`" in output
    assert "pattern `*`" in output
    assert "root.py" in output


def test_run_glob_allows_shallow_star_listing(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "pkg").mkdir()
    (workspace.root / "root.py").write_text("root")
    output = tools.run_glob(workspace, "*")
    assert "Found 2 matches" in output
    assert "pkg" in output
    assert "root.py" in output


def test_run_glob_recursive_finds_nested_and_root_level(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "a" / "b").mkdir(parents=True)
    (workspace.root / "config.json").write_text("root")
    (workspace.root / "a" / "b" / "config.json").write_text("nested")
    output = tools.run_glob(workspace, "**/config.json")
    assert "Found 2 matches" in output
    assert "config.json" in output
    assert "a/b/config.json" in output


def test_run_glob_recursive_prunes_excluded_dirs(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "src").mkdir()
    (workspace.root / "node_modules" / "dep").mkdir(parents=True)
    (workspace.root / "src" / "keep.py").write_text("keep")
    (workspace.root / "node_modules" / "dep" / "skip.py").write_text("skip")
    output = tools.run_glob(workspace, "**/*.py", include_dirs=False)
    assert "Found 1 matches" in output
    assert "src/keep.py" in output
    assert "node_modules" not in output


def test_run_glob_allows_recursive_pattern_in_specific_directory(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    (workspace.root / "pkg").mkdir()
    (workspace.root / "pkg" / "nested.py").write_text("nested")
    output = tools.run_glob(workspace, "**/*.py", include_dirs=False)
    assert "Found 1 matches" in output
    assert "pkg/nested.py" in output


def test_run_glob_limits_results(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    for index in range(3):
        (workspace.root / f"{index}.txt").write_text(str(index))
    output = tools.run_glob(workspace, "*.txt", limit=2)
    assert "Found 3 matches" in output
    assert "Showing first 2" in output
    assert len(output.splitlines()) == 3


def test_run_grep_falls_back_when_ripgrep_is_missing(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    import grep_engine
    (workspace.root / "target.txt").write_text("a needle here\n", encoding="utf-8")
    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)

    assert tools.run_grep(workspace, "needle") == "target.txt"


@pytest.mark.parametrize(
    "output_mode, expected_output",
    [
        ("files_with_matches", "tools.py"),
        ("count_matches", "tools.py:1"),
        ("content", "tools.py:2:needle"),
    ],
)
def test_run_grep_builds_safe_rg_command(
    load_module,
    monkeypatch,
    workspace,
    output_mode,
    expected_output,
) -> None:
    tools = load_module("tools", "tools.py")
    import grep_engine
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        event = {
            "type": "match",
            "data": {
                "path": {"text": str(workspace.root / "tools.py")},
                "line_number": 2,
                "lines": {"text": "needle\n"},
                "submatches": [{"start": 0, "end": 6}],
            },
        }
        import json
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(event) + "\n", stderr="")

    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(grep_engine.subprocess, "run", fake_run)

    output = tools.run_grep(
        workspace, "needle",
        path=".",
        glob="*.py",
        output_mode=output_mode,
        ignore_case=True,
    )

    assert output == expected_output
    assert captured["kwargs"]["cwd"] == workspace.root
    assert captured["kwargs"]["timeout"] == 20
    assert "--json" in captured["args"]
    assert "--ignore-case" in captured["args"]
    assert "--glob" in captured["args"]
    assert "*.py" in captured["args"]
    assert "!.env" in captured["args"]
    assert captured["args"][-3:] == ["--", "needle", str(workspace.root)]


def test_run_grep_handles_no_matches_and_limits_output(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    import grep_engine
    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)
    assert tools.run_grep(workspace, "missing") == "No matches found."

    for index in range(3):
        (workspace.root / f"file_{index}.py").write_text("needle\n", encoding="utf-8")
    output = tools.run_grep(workspace, "needle", head_limit=2)

    assert output.splitlines() == [
        "file_0.py",
        "file_1.py",
        "... (1 more lines)",
    ]


def test_run_grep_rejects_workspace_escape(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    import grep_engine
    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)

    output = tools.run_grep(workspace, "needle", path="../")

    assert output.startswith("Error: Path escapes workspace")


def test_run_grep_allows_external_absolute_path(
    load_module,
    monkeypatch,
    workspace,
    tmp_path,
) -> None:
    tools = load_module("tools", "tools.py")
    import grep_engine
    external = tmp_path.parent / f"{tmp_path.name}-external-grep"
    external.mkdir()
    (external / "target.txt").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)

    output = tools.run_grep(workspace, "needle", path=str(external))

    assert output == "target.txt"


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
def test_search_tools_validate_with_pydantic(load_module, workspace, item, expected_substring) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    out, used_todo = tools.execute_tool_calls([item], registry, todo)

    assert used_todo is False
    assert f"Error: invalid arguments for tool '{item.name}':" in out[0]["output"]
    assert expected_substring in out[0]["output"]


def test_grep_rejects_cli_flags_with_corrective_guidance(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    item = _fc(
        "grep",
        "g-flags",
        '{"pattern":"needle","-n":"true","-A":"20"}',
    )

    out, used_todo = tools.execute_tool_calls([item], registry, todo)

    assert used_todo is False
    output = out[0]["output"]
    assert "Valid grep arguments:" in output
    assert "pattern, path, glob, output_mode, ignore_case, line_number, head_limit" in output
    assert "CLI flags such as -n, -A, -B, and -C are not supported" in output
    assert "Use line_number=true for line numbers" in output


def test_execute_tool_calls_sanitizes_search_arguments(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    registry["glob"].execute = tools.async_tool(lambda params: (
        f"{params.pattern}:{params.directory}"
    ))

    out, used_todo = tools.execute_tool_calls([
        _fc("glob", "gb1", '{"pattern":" >  *.py","directory":" $#  tests"}'),
    ], registry, todo)

    assert used_todo is False
    assert out[0]["output"] == "*.py:tests"
