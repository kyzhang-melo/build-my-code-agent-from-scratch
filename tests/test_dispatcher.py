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
    tools.TOOL_REGISTRY["bash"].execute = lambda params: tools.run_bash(params.command)

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
    tools.TOOL_REGISTRY["bash"].execute = lambda params: tools.run_bash(params.command)

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


def test_task_tool_uses_configured_runner(load_module) -> None:
    tools = load_module("tools", "tools.py")
    tools.configure_task_runner(lambda prompt, description: f"{description}:{prompt}")

    out, used_todo = tools.execute_tool_calls([
        _fc("task", "task1", '{"prompt":"inspect auth","description":"auth scan"}'),
    ])

    assert used_todo is False
    assert out[0]["output"] == "auth scan:inspect auth"


@pytest.mark.parametrize(
    "arguments, expected_substring",
    [
        ('{"items":[{"content":"one","status":"in_progress"},{"content":"two","status":"in_progress"}]}',
         "Only one plan item can be in_progress"),
        ('{"items":[{"content":"step","status":"blocked"}]}', "Input should be"),
        ('{"items":[{"content":"   ","status":"pending"}]}', "String should have at least 1 character"),
    ],
)
def test_execute_tool_calls_todo_validates_with_pydantic(
    load_module,
    arguments,
    expected_substring,
) -> None:
    tools = load_module("tools", "tools.py")
    out, used_todo = tools.execute_tool_calls([_fc("todo", "t1", arguments)])

    assert used_todo is True
    assert "Error: invalid arguments for tool 'todo':" in out[0]["output"]
    assert expected_substring in out[0]["output"]


@pytest.mark.parametrize(
    "item, expected_substring",
    [
        (_fc("bash", "b1", '{"command":""}'), "String should have at least 1 character"),
        (_fc("read_file", "r1", '{"path":"README.md","limit":true}'), "Input should be a valid integer"),
        (_fc("read_file", "r2", '{"path":"README.md","limit":"1"}'), "Input should be a valid integer"),
        (_fc("write_file", "w1", '{"path":"tmp/x.txt","content":"","extra":1}'), "Extra inputs are not permitted"),
        (_fc("edit_file", "e1", '{"path":"tmp/x.txt","new_text":"new"}'), "Field required"),
    ],
)
def test_execute_tool_calls_basic_tools_validate_with_pydantic(
    load_module,
    item,
    expected_substring,
) -> None:
    tools = load_module("tools", "tools.py")
    out, used_todo = tools.execute_tool_calls([item])

    assert used_todo is False
    assert f"Error: invalid arguments for tool '{item.name}':" in out[0]["output"]
    assert expected_substring in out[0]["output"]
