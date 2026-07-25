from __future__ import annotations

import asyncio
import dataclasses
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


def test_dispatcher_traces_success_and_safe_write_metadata(load_module) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink, run_id="test")
    tools.TOOL_REGISTRY["write_file"].execute = lambda _params: asyncio.sleep(
        0, result="wrote"
    )

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("write_file", "w1", '{"path":"x.txt","content":"SECRET","mode":"append"}')],
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


def test_dispatcher_traces_failure_statuses(load_module) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)

    asyncio.run(tools.execute_tool_calls_async([
        _fc("missing", "c1", "{}"),
        _fc("bash", "c2", "{bad-json"),
    ], trace_context=context))

    completed = sink.by_type("tool.completed")
    assert [(event["call_id"], event["status"]) for event in completed] == [
        ("c1", "unknown_tool"),
        ("c2", "invalid_arguments"),
    ]


def test_dispatcher_traces_permission_denial(load_module) -> None:
    tools = load_module("tools", "tools.py")
    permissions = runtime_permissions
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    service = permissions.PermissionService(
        permissions.PermissionManager(Path(tools.WORKDIR)),
        permissions.TerminalApprovalHandler(interactive=False),
    )

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        permission_service=service,
        trace_context=context,
    ))

    assert sink.by_type("permission.decided")[0]["decision"] == "deny"
    assert sink.by_type("tool.completed")[0]["status"] == "permission_denied"


def test_dispatcher_traces_cancellation_and_reraises(load_module) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)

    async def cancel(_params):
        raise asyncio.CancelledError

    tools.TOOL_REGISTRY["read_file"].execute = cancel

    async def scenario():
        await tools.execute_tool_calls_async(
            [_fc("read_file", "r1", '{"path":"README.md"}')],
            trace_context=context,
        )

    try:
        asyncio.run(scenario())
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation should propagate")

    assert sink.by_type("tool.completed")[0]["status"] == "cancelled"


def test_dispatcher_traces_todo_transitions(load_module) -> None:
    tools = load_module("tools", "tools.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)

    tools.execute_tool_calls([
        _fc("todo", "t1", '{"items":[{"content":"step","status":"in_progress"}]}'),
        _fc("todo", "t2", '{"items":[{"content":"step","status":"completed"}]}'),
    ], trace_context=context)

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


def test_stop_gate_trace_records_block_and_give_up(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)
    tools_module = sys.modules["tools"]
    main_module.TODO.update(tools_module.TodoParams.model_validate({
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
    config = dataclasses.replace(
        main_module.PARENT_CONFIG,
        trace_context=context,
        on_text=None,
    )
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])
    assert asyncio.run(main_module.run_one_turn(state, config)) is None
    state.nudges = main_module.TODO_CONTRACT_MAX_NUDGES
    assert asyncio.run(main_module.run_one_turn(state, config)) is not None

    assert [event["decision"] for event in sink.by_type("stop_gate.checked")] == [
        "block",
        "give_up",
    ]
