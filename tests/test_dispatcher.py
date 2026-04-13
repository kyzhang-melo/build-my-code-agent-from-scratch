from __future__ import annotations

import types

import pytest


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


@pytest.mark.parametrize(
    "item, expected_substring",
    [
        (_fc("no_such_tool", "u1", "{}"), "unknown tool 'no_such_tool'"),
        (_fc("write_file", "u2", '{"path":"tmp/x.txt"}'), "invalid arguments for tool 'write_file'"),
        (_fc("bash", "u3", "{not-valid-json"), "invalid arguments for tool 'bash'"),
    ],
)
def test_execute_tool_calls_failure_paths(load_module, item, expected_substring) -> None:
    tools = load_module("tools", "tools.py")
    out, used_todo = tools.execute_tool_calls([item])
    assert len(out) == 1
    assert used_todo is False
    assert out[0]["type"] == "function_call_output"
    assert expected_substring in out[0]["output"]


def test_execute_tool_calls_known_tools(load_module) -> None:
    tools = load_module("tools", "tools.py")
    tools.run_bash = lambda command: f"ran:{command}"
    tools.TOOL_REGISTRY["bash"].execute = lambda args: tools.run_bash(args["command"])

    out, used_todo = tools.execute_tool_calls([
        _fc("bash", "c1", '{"command":"echo hi"}'),
        _fc("read_file", "c2", '{"path":"README.md","limit":1}'),
    ])

    assert len(out) == 2
    assert used_todo is False
    assert out[0]["call_id"] == "c1"
    assert out[0]["output"] == "ran:echo hi"
    assert out[1]["call_id"] == "c2"
    assert isinstance(out[1]["output"], str)
    assert out[1]["output"]


def test_execute_tool_calls_sanitizes_prompt_prefix_for_bash(load_module) -> None:
    tools = load_module("tools", "tools.py")
    tools.run_bash = lambda command: f"ran:{command}"
    tools.TOOL_REGISTRY["bash"].execute = lambda args: tools.run_bash(args["command"])

    out, used_todo = tools.execute_tool_calls([
        _fc("bash", "c1", '{"command":"   >$#   echo hi"}'),
    ])

    assert used_todo is False
    assert out[0]["output"] == "ran:echo hi"


def test_execute_tool_calls_sanitizes_prompt_prefix_for_path(load_module) -> None:
    tools = load_module("tools", "tools.py")

    out, used_todo = tools.execute_tool_calls([
        _fc("read_file", "c1", '{"path":" >  README.md","limit":1}'),
    ])

    assert used_todo is False
    assert isinstance(out[0]["output"], str)
    assert out[0]["output"]


def test_execute_tool_calls_todo_sets_used_flag(load_module) -> None:
    tools = load_module("tools", "tools.py")
    out, used_todo = tools.execute_tool_calls([
        _fc("todo", "t1", '{"items":[{"content":"step 1","status":"in_progress"}]}'),
    ])

    assert len(out) == 1
    assert used_todo is True
    assert "[>] step 1" in out[0]["output"]
