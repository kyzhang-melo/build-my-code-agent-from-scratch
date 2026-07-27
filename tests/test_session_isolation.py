"""Verify that two AgentSession instances share no mutable state.

These tests are the regression guard for the eliminate-globals refactor:
the whole point of AgentSession is that one session's workspace, todo,
tool registry, permission service, and trace context can never leak into
another session. If a future change reintroduces a module-level singleton
captured by create_parent_session, one of these tests should fail.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
from pathlib import Path

import permissions as runtime_permissions
import trace as runtime_trace


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


class _StubHandler:
    """Approval handler that approves everything, for session construction."""

    async def request(self, _request):
        return runtime_permissions.ApprovalResponse("approve")


def test_two_parent_sessions_have_distinct_workspaces(load_module) -> None:
    main = load_module("main", "main.py")

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = main.create_parent_session(Path(a), approval_handler=_StubHandler())
        sb = main.create_parent_session(Path(b), approval_handler=_StubHandler())

    assert sa.workspace.root != sb.workspace.root
    assert sa.workspace is not sb.workspace


def test_two_parent_sessions_have_distinct_todos(load_module) -> None:
    main = load_module("main", "main.py")
    tools = sys.modules["tools"]

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = main.create_parent_session(Path(a), approval_handler=_StubHandler())
        sb = main.create_parent_session(Path(b), approval_handler=_StubHandler())

    assert sa.todo is not sb.todo
    # Mutating one todo must not affect the other.
    sa.todo.update(tools.TodoParams.model_validate({
        "items": [{"content": "only-in-a", "status": "in_progress"}],
    }))
    assert sa.todo.has_active_plan() is True
    assert sb.todo.has_active_plan() is False
    assert "only-in-a" in sa.todo.render()
    assert "only-in-a" not in sb.todo.render()


def test_two_parent_sessions_have_distinct_registries(load_module, workspace) -> None:
    """Registries are built per session; patching one tool must not leak."""
    main = load_module("main", "main.py")

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = main.create_parent_session(Path(a), approval_handler=_StubHandler())
        sb = main.create_parent_session(Path(b), approval_handler=_StubHandler())

    assert sa.registry is not sb.registry
    assert sa.registry["bash"] is not sb.registry["bash"]

    # Patching session A's bash execute must not affect session B.
    sa.registry["bash"].execute = lambda _params: asyncio.sleep(0, result="from-a")
    # Re-resolve B's bash execute through the workspace-bound closure; since
    # the registry was built independently, B's bash still runs the real
    # run_bash against B's workspace, not A's patched lambda.
    assert sb.registry["bash"].execute != sa.registry["bash"].execute


def test_two_parent_sessions_have_distinct_permission_services(load_module) -> None:
    main = load_module("main", "main.py")

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = main.create_parent_session(Path(a), approval_handler=_StubHandler())
        sb = main.create_parent_session(Path(b), approval_handler=_StubHandler())

    assert sa.permission_service is not sb.permission_service
    assert sa.permission_service.manager is not sb.permission_service.manager
    # Session-scoped approvals must not cross over.
    decision = sa.permission_service.manager.check(
        "write_file", {"path": "tmp/x.txt"},
    )
    if decision.action is not None:
        sa.permission_service.manager.remember(decision.action)
    # A's session approval should not grant B's write to the same relative path.
    other = sb.permission_service.manager.check(
        "write_file", {"path": "tmp/x.txt"},
    )
    assert other.behavior.value == "ask"


def test_two_parent_sessions_have_distinct_trace_contexts(load_module) -> None:
    main = load_module("main", "main.py")

    sink_a = runtime_trace.MemoryTraceSink()
    ctx_a = runtime_trace.TraceContext(sink=sink_a, run_id="run-a", agent_id="parent")
    sink_b = runtime_trace.MemoryTraceSink()
    ctx_b = runtime_trace.TraceContext(sink=sink_b, run_id="run-b", agent_id="parent")

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = main.create_parent_session(
            Path(a), approval_handler=_StubHandler(), trace_context=ctx_a,
        )
        sb = main.create_parent_session(
            Path(b), approval_handler=_StubHandler(), trace_context=ctx_b,
        )

    assert sa.trace_context is not sb.trace_context
    assert sa.trace_context.sink is not sb.trace_context.sink

    # Emitting to A's context must not appear in B's sink.
    runtime_trace.emit_trace(sa.trace_context, "tool.completed", call_id="c1")
    assert len(sink_a.events) == 1
    assert len(sink_b.events) == 0


def test_two_parent_sessions_have_distinct_stop_gates(load_module) -> None:
    main = load_module("main", "main.py")

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = main.create_parent_session(Path(a), approval_handler=_StubHandler())
        sb = main.create_parent_session(Path(b), approval_handler=_StubHandler())

    assert sa.stop_gate is not sb.stop_gate
    assert sa.stop_gate.todo is not sb.stop_gate.todo
    # The stop gate's todo is the session's todo (same object), not a copy.
    assert sa.stop_gate.todo is sa.todo
    assert sb.stop_gate.todo is sb.todo


def test_tool_call_in_one_session_does_not_mutate_anothers_todo(
    load_module, workspace, tmp_path,
) -> None:
    """End-to-end: a todo call routed through session A's registry must not
    change session B's todo state."""
    main = load_module("main", "main.py")
    tools = sys.modules["tools"]

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        sa = main.create_parent_session(Path(a), approval_handler=_StubHandler())
        sb = main.create_parent_session(Path(b), approval_handler=_StubHandler())

    asyncio.run(tools.execute_tool_calls_async(
        [_fc("todo", "t1", '{"items":[{"content":"step-a","status":"in_progress"}]}')],
        sa.registry,
        sa.todo,
    ))

    assert sa.todo.has_active_plan() is True
    assert "step-a" in sa.todo.render()
    # B's todo is untouched.
    assert sb.todo.has_active_plan() is False
    assert "step-a" not in sb.todo.render()


def test_create_explore_session_is_isolated_from_parent(load_module, workspace) -> None:
    """An explore subagent session shares the workspace and permission service
    with its parent (by design) but has its own todo, registry, trace context,
    and stop gate."""
    main = load_module("main", "main.py")
    parent = main.create_parent_session(
        workspace.root, approval_handler=_StubHandler(),
    )
    child = main.create_explore_session(
        parent.workspace, parent.permission_service, parent.trace_context,
    )

    # Shared by design (read-only subagent inherits parent's safety boundary).
    assert child.workspace is parent.workspace
    assert child.permission_service is parent.permission_service
    # Isolated state.
    assert child.todo is not parent.todo
    assert child.registry is not parent.registry
    assert child.stop_gate is not parent.stop_gate
    # Trace context is a new instance with a different agent_id.
    assert child.trace_context is not parent.trace_context
    assert child.trace_context.agent_id == "subagent:explore"
    # The explore registry is restricted to read-only tools.
    assert set(child.registry.keys()) <= set(main.READ_ONLY_TOOL_NAMES)
