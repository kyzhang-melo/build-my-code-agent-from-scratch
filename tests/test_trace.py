from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import threading
import types

import permissions as runtime_permissions
import trace as runtime_trace


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def test_memory_sink_assigns_sequence_and_filters() -> None:
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink, run_id="run-1", agent_id="parent")

    runtime_trace.emit_trace(context, "tool.requested", call_id="c1", tool_name="read_file")
    runtime_trace.emit_trace(context, "tool.completed", call_id="c1", tool_name="read_file")

    assert [event["sequence"] for event in sink.events] == [1, 2]
    assert sink.by_type("tool.completed")[0]["call_id"] == "c1"


def test_memory_sink_sequence_is_safe_across_threads() -> None:
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    threads = [
        threading.Thread(
            target=runtime_trace.emit_trace,
            args=(context, "tool.completed"),
            kwargs={"call_id": f"c{index}"},
        )
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [event["sequence"] for event in sink.events] == list(range(1, 21))


def test_jsonl_sink_writes_parseable_redacted_records(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    sink = runtime_trace.JsonlTraceSink(path)
    context = runtime_trace.TraceContext(sink=sink)

    runtime_trace.emit_trace(
        context,
        "debug",
        command="TOP_SECRET_COMMAND",
        note="x" * (runtime_trace.MAX_TRACE_STRING_CHARS + 20),
    )

    record = json.loads(path.read_text().strip())
    assert record["sequence"] == 1
    assert "TOP_SECRET_COMMAND" not in record["command"]
    assert "omitted" in record["note"]


def test_emit_trace_isolates_broken_sink() -> None:
    class BrokenSink:
        def emit(self, _event):
            raise RuntimeError("broken")

    context = runtime_trace.TraceContext(sink=BrokenSink())
    runtime_trace.emit_trace(context, "tool.completed", status="success")


def test_dispatcher_traces_success_and_safe_write_metadata(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink, run_id="test")
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["write_file"].execute = lambda _params: asyncio.sleep(
        0, result="wrote"
    )

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("write_file", "w1", '{"path":"x.txt","content":"SECRET","mode":"append"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    requested, completed = sink.events
    assert requested["event"] == "tool.requested"
    assert requested["arguments"] == {
        "path": "x.txt",
        "mode": "append",
        "content_chars": 6,
    }
    assert "SECRET" not in json.dumps(sink.events)
    assert completed["status"] == "success"
    assert completed["success"] is True


def test_dispatcher_traces_failure_statuses(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    asyncio.run(tools.execute_tool_calls_async([
        _fc("missing", "c1", "{}"),
        _fc("bash", "c2", "{bad-json"),
    ], registry, tools.TodoManager(), trace_context=context))

    completed = sink.by_type("tool.completed")
    assert [(event["call_id"], event["status"]) for event in completed] == [
        ("c1", "unknown_tool"),
        ("c2", "invalid_arguments"),
    ]


def test_dispatcher_traces_permission_denial(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    permissions = runtime_permissions
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    service = permissions.PermissionService(
        permissions.PermissionManager(workspace.root),
        permissions.TerminalApprovalHandler(interactive=False),
    )
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        registry,
        tools.TodoManager(),
        permission_service=service,
        trace_context=context,
    ))

    assert sink.by_type("permission.decided")[0]["decision"] == "deny"
    assert sink.by_type("tool.completed")[0]["status"] == "permission_denied"


def test_dispatcher_traces_cancellation_and_reraises(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)

    async def cancel(_params):
        raise asyncio.CancelledError

    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["read_file"].execute = cancel

    async def scenario():
        await tools.execute_tool_calls_async(
            [_fc("read_file", "r1", '{"path":"README.md"}')],
            registry,
            tools.TodoManager(),
            trace_context=context,
        )

    try:
        asyncio.run(scenario())
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation should propagate")

    assert sink.by_type("tool.completed")[0]["status"] == "cancelled"


def test_dispatcher_traces_todo_transitions(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    todo = tools.TodoManager()
    registry = tools.build_tool_registry(workspace, todo)

    tools.execute_tool_calls([
        _fc("todo", "t1", '{"items":[{"content":"step","status":"in_progress"}]}'),
        _fc("todo", "t2", '{"items":[{"content":"step","status":"completed"}]}'),
    ], registry, todo, trace_context=context)

    changed = sink.by_type("todo.changed")
    assert changed[0]["transitions"] == [
        {"content": "step", "from": None, "to": "in_progress"}
    ]
    assert changed[1]["transitions"] == [
        {"content": "step", "from": "in_progress", "to": "completed"}
    ]


def test_permission_trace_distinguishes_policy_and_approval() -> None:
    permissions = runtime_permissions
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)

    class Handler:
        async def request(self, _request):
            return permissions.ApprovalResponse("approve_for_session")

    service = permissions.PermissionService(
        permissions.PermissionManager(Path.cwd()),
        Handler(),
    )
    decision = asyncio.run(service.authorize(
        "write_file",
        {"path": "tmp/x.txt"},
        trace_context=context,
        call_id="w1",
    ))

    assert decision.behavior.value == "allow"
    event = sink.by_type("permission.decided")[0]
    assert event["policy_behavior"] == "ask"
    assert event["approval_kind"] == "approve_for_session"
    assert event["decision"] == "allow"


def test_stop_gate_trace_records_block_and_give_up(load_module, monkeypatch, workspace) -> None:
    main_module = load_module("main", "main.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    tools_module = sys.modules["tools"]

    class DenyHandler:
        async def request(self, _request):
            return runtime_permissions.ApprovalResponse("deny")

    session = main_module.create_parent_session(
        workspace.root,
        approval_handler=DenyHandler(),
        trace_context=context,
        on_text=None,
    )
    session.todo.update(tools_module.TodoParams.model_validate({
        "items": [{"content": "unfinished", "status": "in_progress"}],
    }))
    response = types.SimpleNamespace(output=[], output_text="Done.", usage=None)

    async def create(**_kwargs):
        return response

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=create)),
    )
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])
    assert asyncio.run(main_module.run_one_turn(state, session)) is None
    state.nudges = main_module.TODO_CONTRACT_MAX_NUDGES
    assert asyncio.run(main_module.run_one_turn(state, session)) is not None

    assert [event["decision"] for event in sink.by_type("stop_gate.checked")] == [
        "block",
        "give_up",
    ]


def test_dispatcher_traces_validation_issues(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("grep", "g1", '{"pattern":"x","-n":true}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    requested = sink.by_type("tool.requested")[0]
    completed = sink.by_type("tool.completed")[0]
    assert completed["status"] == "invalid_arguments"
    assert completed["error_type"] == "validation"
    assert requested["validation_issues"] == [
        {"path": "-n", "type": "extra_forbidden"},
    ]
    # No raw input value leaked into the trace.
    assert "true" not in json.dumps(requested["validation_issues"])


def test_dispatcher_traces_empty_validation_issues_on_success(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result="ok")

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", '{"path":"x.txt"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    requested = sink.by_type("tool.requested")[0]
    assert requested["validation_issues"] == []


def test_dispatcher_traces_empty_validation_issues_on_json_parse_error(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{bad-json')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    requested = sink.by_type("tool.requested")[0]
    # JSON parse errors have no Pydantic validation issues.
    assert requested["validation_issues"] == []
    assert requested["argument_error"] is not None


def test_dispatcher_traces_raw_arguments_fingerprint(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result="ok")

    raw = '{"path":"x.txt"}'
    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", raw)],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    requested = sink.by_type("tool.requested")[0]
    assert requested["raw_arguments_chars"] == len(raw)
    assert len(requested["raw_arguments_sha256"]) == 16
    # The fingerprint must not contain the raw JSON string itself.
    assert requested["raw_arguments_sha256"] != raw


def test_dispatcher_raw_fingerprint_detects_exact_repeats(load_module, workspace) -> None:
    import hashlib
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result="ok")

    raw = '{"path":"x.txt"}'
    expected_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", raw), _fc("read_file", "r2", raw)],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    events = sink.by_type("tool.requested")
    assert events[0]["raw_arguments_sha256"] == expected_hash
    assert events[1]["raw_arguments_sha256"] == expected_hash
    # Different arguments produce a different hash.
    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r3", '{"path":"y.txt"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))
    third = sink.by_type("tool.requested")[2]
    assert third["raw_arguments_sha256"] != expected_hash


def test_dispatcher_raw_fingerprint_handles_non_string_arguments(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    item = types.SimpleNamespace(
        type="function_call", name="bash", call_id="b1", arguments=None,
    )
    asyncio.run(tools.run_tool_call_async(
        item, registry, tools.TodoManager(), trace_context=context,
    ))

    requested = sink.by_type("tool.requested")[0]
    assert requested["raw_arguments_chars"] == 0
    assert requested["raw_arguments_sha256"] == ""


def test_dispatcher_traces_api_call_and_step_index(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result="ok")
    registry["glob"].execute = lambda _params: asyncio.sleep(0, result="ok")

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", '{"path":"a.txt"}'),
         _fc("glob", "g1", '{"pattern":"*.py"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
        api_call=7,
    ))

    requested = sink.by_type("tool.requested")
    completed = sink.by_type("tool.completed")
    # Both tools share the same api_call; step_index is sequential.
    assert requested[0]["api_call"] == 7
    assert requested[0]["step_index"] == 0
    assert requested[1]["api_call"] == 7
    assert requested[1]["step_index"] == 1
    assert completed[0]["api_call"] == 7
    assert completed[0]["step_index"] == 0
    assert completed[1]["api_call"] == 7
    assert completed[1]["step_index"] == 1


def test_dispatcher_traces_null_api_call_when_not_provided(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result="ok")

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", '{"path":"a.txt"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    requested = sink.by_type("tool.requested")[0]
    assert requested["api_call"] is None
    assert requested["step_index"] == 0


def test_dispatcher_traces_runtime_truncation(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    long_output = "A" * (tools.TOOL_OUTPUT_MAX_CHARS + 5000)
    registry["grep"].execute = lambda _params: asyncio.sleep(0, result=long_output)

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("grep", "g1", '{"pattern":"x"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    completed = sink.by_type("tool.completed")[0]
    assert completed["runtime_output_truncated"] is True
    assert completed["output_truncated"] is True  # backward-compat alias
    assert completed["tool_internal_truncated"] is False
    assert completed["truncated_chars"] > 0
    # output_chars + truncated_chars should equal the original output length
    assert completed["output_chars"] + completed["truncated_chars"] == tools.TOOL_OUTPUT_MAX_CHARS + 5000


def test_dispatcher_traces_no_truncation_on_small_output(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result="short")

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", '{"path":"x.txt"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    completed = sink.by_type("tool.completed")[0]
    assert completed["runtime_output_truncated"] is False
    assert completed["tool_internal_truncated"] is False
    assert completed["truncated_chars"] == 0


def test_dispatcher_detects_bash_internal_truncation(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    # Simulate a bash result that hit the internal output limit.
    bash_output = (
        "[status] completed\n"
        "[exit_code] 0\n"
        "[timed_out] false\n"
        "[post_exit_cleanup] false\n"
        "[truncated] true\n"
        "[duration_ms] 100\n\n"
        "some output"
    )
    registry["bash"].execute = lambda _params: asyncio.sleep(0, result=bash_output)

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    completed = sink.by_type("tool.completed")[0]
    assert completed["tool_internal_truncated"] is True
    assert completed["runtime_output_truncated"] is False  # bash skips truncate_middle


def test_dispatcher_detects_bash_no_internal_truncation(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    bash_output = (
        "[status] completed\n"
        "[exit_code] 0\n"
        "[timed_out] false\n"
        "[post_exit_cleanup] false\n"
        "[truncated] false\n"
        "[duration_ms] 100\n\n"
        "some output"
    )
    registry["bash"].execute = lambda _params: asyncio.sleep(0, result=bash_output)

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    completed = sink.by_type("tool.completed")[0]
    assert completed["tool_internal_truncated"] is False


def test_dispatcher_detects_read_file_internal_truncation(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    read_output = "line1\nline2\n\n[Read 2 lines (lines 1-2). Total lines: 100+. Stopped at the 1000-line limit. Use offset=3 to continue.]"
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result=read_output)

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", '{"path":"x.txt"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    completed = sink.by_type("tool.completed")[0]
    assert completed["tool_internal_truncated"] is True
    assert completed["runtime_output_truncated"] is False  # read_file skips truncate_middle


def test_dispatcher_detects_read_file_no_internal_truncation(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    registry = tools.build_tool_registry(workspace, tools.TodoManager())

    read_output = "line1\nline2\n\n[Read 2 lines (lines 1-2). Total lines: 2. End of file.]"
    registry["read_file"].execute = lambda _params: asyncio.sleep(0, result=read_output)

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("read_file", "r1", '{"path":"x.txt"}')],
        registry,
        tools.TodoManager(),
        trace_context=context,
    ))

    completed = sink.by_type("tool.completed")[0]
    assert completed["tool_internal_truncated"] is False
