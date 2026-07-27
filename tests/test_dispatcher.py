from __future__ import annotations

import asyncio
import types

import pytest


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def _runtime(tools, workspace, task_runner=None):
    todo = tools.TodoManager()
    return tools.build_tool_registry(
        workspace,
        todo,
        task_runner=task_runner,
    ), todo


@pytest.mark.parametrize(
    "item, expected_substring",
    [
        (_fc("no_such_tool", "u1", "{}"), "unknown tool 'no_such_tool'"),
        (_fc("write_file", "u2", '{"path":"tmp/x.txt"}'), "invalid arguments for tool 'write_file'"),
        (_fc("bash", "u3", "{not-valid-json"), "invalid arguments for tool 'bash'"),
    ],
)
def test_execute_tool_calls_failure_paths(load_module, workspace, item, expected_substring) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    out, used_todo = tools.execute_tool_calls([item], registry, todo)
    assert len(out) == 1
    assert used_todo is False
    assert out[0]["type"] == "function_call_output"
    assert expected_substring in out[0]["output"]


def test_execute_tool_calls_known_tools(load_module, repo_workspace) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, repo_workspace)
    registry["bash"].execute = tools.async_tool(lambda params: f"ran:{params.command}")

    out, used_todo = tools.execute_tool_calls([
        _fc("bash", "c1", '{"command":"echo hi"}'),
        _fc("read_file", "c2", '{"path":"README.md","limit":1}'),
    ], registry, todo)

    assert len(out) == 2
    assert used_todo is False
    assert out[0]["call_id"] == "c1"
    assert out[0]["output"] == "ran:echo hi"
    assert out[1]["call_id"] == "c2"
    assert isinstance(out[1]["output"], str)
    assert out[1]["output"]


def test_execute_tool_calls_sanitizes_prompt_prefix_for_bash(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    registry["bash"].execute = tools.async_tool(lambda params: f"ran:{params.command}")

    out, used_todo = tools.execute_tool_calls([
        _fc("bash", "c1", '{"command":"   >$#   echo hi"}'),
    ], registry, todo)

    assert used_todo is False
    assert out[0]["output"] == "ran:echo hi"


def test_execute_tool_calls_sanitizes_prompt_prefix_for_path(load_module, repo_workspace) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, repo_workspace)

    out, used_todo = tools.execute_tool_calls([
        _fc("read_file", "c1", '{"path":" >  README.md","limit":1}'),
    ], registry, todo)

    assert used_todo is False
    assert isinstance(out[0]["output"], str)
    assert out[0]["output"]


def test_execute_tool_calls_todo_sets_used_flag(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    out, used_todo = tools.execute_tool_calls([
        _fc("todo", "t1", '{"items":[{"content":"step 1","status":"in_progress"}]}'),
    ], registry, todo)

    assert len(out) == 1
    assert used_todo is True
    assert "[>] step 1" in out[0]["output"]


def test_task_tool_uses_configured_runner(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    async def task_runner(prompt, description):
        return f"{description}:{prompt}"

    registry, todo = _runtime(tools, workspace, task_runner)

    out, used_todo = tools.execute_tool_calls([
        _fc("task", "task1", '{"prompt":"inspect auth","description":"auth scan"}'),
    ], registry, todo)

    assert used_todo is False
    assert out[0]["output"] == "auth scan:inspect auth"


def test_task_tool_runs_configured_runner_concurrently(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    active = 0
    max_active = 0

    async def task_runner(prompt, description):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return f"{description}:{prompt}"

    registry, todo = _runtime(tools, workspace, task_runner)

    out, used_todo = asyncio.run(tools.execute_tool_calls_async([
        _fc("task", "task1", '{"prompt":"inspect auth","description":"auth"}'),
        _fc("task", "task2", '{"prompt":"inspect billing","description":"billing"}'),
    ], registry, todo))

    assert used_todo is False
    assert max_active == 2
    assert [item["output"] for item in out] == [
        "auth:inspect auth",
        "billing:inspect billing",
    ]


def test_execute_tool_calls_async_runs_safe_tools_concurrently(load_module) -> None:
    tools = load_module("tools", "tools.py")
    active = 0
    max_active = 0

    class _Model:
        @staticmethod
        def model_validate(data):
            return types.SimpleNamespace(value=data["value"])

    async def execute(params):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return params.value

    spec = types.SimpleNamespace(
        sanitize_args=lambda args: args,
        params_model=_Model,
        execute=execute,
        concurrency_safe=True,
    )

    out, used_todo = asyncio.run(tools.execute_tool_calls_async([
        _fc("safe", "c1", '{"value":"first"}'),
        _fc("safe", "c2", '{"value":"second"}'),
    ], {"safe": spec}, tools.TodoManager()))

    assert used_todo is False
    assert max_active == 2
    assert [item["output"] for item in out] == ["first", "second"]


def test_execute_tool_calls_async_preserves_order_for_safe_tools(load_module) -> None:
    tools = load_module("tools", "tools.py")

    class _Model:
        @staticmethod
        def model_validate(data):
            return types.SimpleNamespace(value=data["value"], delay=data["delay"])

    async def execute(params):
        await asyncio.sleep(params.delay)
        return params.value

    spec = types.SimpleNamespace(
        sanitize_args=lambda args: args,
        params_model=_Model,
        execute=execute,
        concurrency_safe=True,
    )

    out, _ = asyncio.run(tools.execute_tool_calls_async([
        _fc("safe", "c1", '{"value":"slow","delay":0.02}'),
        _fc("safe", "c2", '{"value":"fast","delay":0}'),
    ], {"safe": spec}, tools.TodoManager()))

    assert [item["call_id"] for item in out] == ["c1", "c2"]
    assert [item["output"] for item in out] == ["slow", "fast"]


def test_execute_tool_calls_async_runs_unsafe_tools_sequentially(load_module) -> None:
    tools = load_module("tools", "tools.py")
    active = 0
    max_active = 0

    class _Model:
        @staticmethod
        def model_validate(data):
            return types.SimpleNamespace(value=data["value"])

    async def execute(params):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return params.value

    spec = types.SimpleNamespace(
        sanitize_args=lambda args: args,
        params_model=_Model,
        execute=execute,
        concurrency_safe=False,
    )

    out, _ = asyncio.run(tools.execute_tool_calls_async([
        _fc("unsafe", "c1", '{"value":"first"}'),
        _fc("unsafe", "c2", '{"value":"second"}'),
    ], {"unsafe": spec}, tools.TodoManager()))

    assert max_active == 1
    assert [item["output"] for item in out] == ["first", "second"]


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
    workspace,
    arguments,
    expected_substring,
) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    out, used_todo = tools.execute_tool_calls(
        [_fc("todo", "t1", arguments)],
        registry,
        todo,
    )

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
    workspace,
    item,
    expected_substring,
) -> None:
    tools = load_module("tools", "tools.py")
    registry, todo = _runtime(tools, workspace)
    out, used_todo = tools.execute_tool_calls([item], registry, todo)

    assert used_todo is False
    assert f"Error: invalid arguments for tool '{item.name}':" in out[0]["output"]
    assert expected_substring in out[0]["output"]
