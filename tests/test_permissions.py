from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import types
from pathlib import Path

import pytest

import permissions as permission_module


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def test_default_and_plan_mode_matrix(load_module) -> None:
    permissions = permission_module
    manager = permissions.PermissionManager(Path.cwd())

    assert manager.check("read_file", {"path": "README.md"}).behavior.value == "allow"
    assert manager.check("glob", {"pattern": "*.py"}).behavior.value == "allow"
    assert manager.check("grep", {"pattern": "needle"}).behavior.value == "allow"
    assert manager.check("task", {"prompt": "inspect"}).behavior.value == "allow"
    assert manager.check("todo", {"items": []}).behavior.value == "allow"
    assert manager.check("write_file", {"path": "tmp/x.txt"}).behavior.value == "ask"
    assert manager.check("edit_file", {"path": "tmp/x.txt"}).behavior.value == "ask"
    assert manager.check("bash", {"command": "echo hi"}).behavior.value == "ask"

    manager.set_mode("plan")

    assert manager.check("read_file", {"path": "README.md"}).behavior.value == "allow"
    assert manager.check("write_file", {"path": "tmp/x.txt"}).behavior.value == "deny"
    assert manager.check("edit_file", {"path": "tmp/x.txt"}).behavior.value == "deny"
    assert manager.check("bash", {"command": "echo hi"}).behavior.value == "deny"


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", "secret.pem", "id_rsa", ".git/config"],
)
def test_sensitive_reads_are_denied(load_module, path) -> None:
    permissions = permission_module
    manager = permissions.PermissionManager(Path.cwd())

    decision = manager.check("read_file", {"path": path})

    assert decision.behavior.value == "deny"
    assert "sensitive" in decision.reason.lower()


@pytest.mark.parametrize(
    "path",
    [".env", ".git/config", ".ssh/config", "secret.key", "../outside.txt"],
)
def test_protected_writes_are_hard_denied(load_module, path) -> None:
    permissions = permission_module
    manager = permissions.PermissionManager(Path.cwd())

    decision = manager.check("write_file", {"path": path})

    assert decision.behavior.value == "deny"


def test_symlink_write_escape_is_denied(load_module, tmp_path) -> None:
    permissions = permission_module
    link = Path.cwd() / "tests" / "_permission_escape_link"
    link.unlink(missing_ok=True)
    link.symlink_to(tmp_path, target_is_directory=True)
    try:
        manager = permissions.PermissionManager(Path.cwd())
        decision = manager.check("write_file", {"path": f"{link.relative_to(Path.cwd())}/x.txt"})
    finally:
        link.unlink(missing_ok=True)

    assert decision.behavior.value == "deny"
    assert "escapes workspace" in decision.reason.lower()


@pytest.mark.parametrize(
    "command, expected",
    [
        ("sudo echo hi", "sudo"),
        ("shutdown -h now", "shutdown"),
        ("echo x > /dev/null", "device"),
        ("cd .. && pwd", "outside"),
        ("echo x > ../outside.txt", "outside"),
        ("rm -rf ../outside", "outside"),
    ],
)
def test_bash_hard_deny_checks(load_module, command, expected) -> None:
    permissions = permission_module

    reason = permissions.bash_hard_deny_reason(command, Path.cwd())

    assert reason is not None
    assert expected in reason.lower()


def test_file_session_approval_is_path_scoped(load_module) -> None:
    permissions = permission_module
    manager = permissions.PermissionManager(Path.cwd())
    first = manager.check("write_file", {"path": "tmp/x.txt"})
    assert first.action is not None
    manager.remember(first.action)

    same_path = manager.check("edit_file", {"path": "tmp/x.txt"})
    other_path = manager.check("write_file", {"path": "tmp/y.txt"})

    assert same_path.behavior.value == "allow"
    assert other_path.behavior.value == "ask"


def test_headless_ask_is_denied(load_module) -> None:
    permissions = permission_module
    service = permissions.PermissionService(
        permissions.PermissionManager(Path.cwd()),
        permissions.TerminalApprovalHandler(interactive=False),
    )

    decision = asyncio.run(service.authorize("bash", {"command": "echo hi"}))

    assert decision.behavior.value == "deny"
    assert "headless" in decision.reason.lower()


def test_terminal_handler_does_not_block_event_loop(load_module) -> None:
    permissions = permission_module
    release = threading.Event()

    def blocking_input(_prompt: str) -> str:
        release.wait(timeout=1)
        return "y"

    handler = permissions.TerminalApprovalHandler(
        interactive=True,
        input_fn=blocking_input,
    )
    request = permissions.ApprovalRequest(
        tool_name="bash",
        action="bash:echo hi",
        description="echo hi",
        allow_for_session=False,
    )

    async def scenario():
        task = asyncio.create_task(handler.request(request))
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        return await task

    response = asyncio.run(scenario())

    assert response.kind == "approve"


def test_permission_denial_short_circuits_tool_execution(load_module) -> None:
    tools = load_module("tools", "tools.py")
    permissions = permission_module
    executed = False

    async def execute(_params):
        nonlocal executed
        executed = True
        return "should not run"

    tools.TOOL_REGISTRY["bash"].execute = execute
    service = permissions.PermissionService(
        permissions.PermissionManager(Path(tools.WORKDIR)),
        permissions.TerminalApprovalHandler(interactive=False),
    )

    output, _ = asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        permission_service=service,
    ))

    payload = json.loads(output[0]["output"])
    assert executed is False
    assert payload["error"] == "permission_denied"
    assert payload["tool"] == "bash"
    assert payload["retryable"] is False


def test_permission_failure_is_fail_closed(load_module) -> None:
    tools = load_module("tools", "tools.py")
    executed = False

    class BrokenService:
        async def authorize(self, *_args, **_kwargs):
            raise RuntimeError("broken policy")

    async def execute(_params):
        nonlocal executed
        executed = True
        return "should not run"

    tools.TOOL_REGISTRY["bash"].execute = execute
    output, _ = asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        permission_service=BrokenService(),
    ))

    payload = json.loads(output[0]["output"])
    assert executed is False
    assert payload["error"] == "permission_denied"
    assert "failed" in payload["reason"].lower()


def test_write_preview_does_not_print_content(load_module, capsys) -> None:
    tools = load_module("tools", "tools.py")
    tools.TOOL_REGISTRY["write_file"].execute = lambda _params: asyncio.sleep(
        0, result="wrote"
    )
    tools.execute_tool_calls([
        _fc("write_file", "w1", '{"path":"tmp/x.txt","content":"TOP_SECRET_VALUE"}'),
    ])

    printed = capsys.readouterr().out
    assert "path='tmp/x.txt'" in printed
    assert "chars=16" in printed
    assert "TOP_SECRET_VALUE" not in printed


def test_glob_hides_sensitive_paths(load_module) -> None:
    tools = load_module("tools", "tools.py")
    base = Path(tools.WORKDIR) / "tests" / "_tmp_permission_glob"
    base.mkdir(parents=True, exist_ok=True)
    (base / ".env").write_text("SECRET=value")
    (base / "visible.txt").write_text("ok")
    try:
        output = tools.run_glob("*", "tests/_tmp_permission_glob")
    finally:
        (base / ".env").unlink()
        (base / "visible.txt").unlink()
        base.rmdir()

    assert "visible.txt" in output
    assert ".env" not in output


def test_grep_sensitive_excludes_override_caller_glob(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    monkeypatch.setattr(tools.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools.run_grep("SECRET", glob=".env")

    args = captured["args"]
    include_index = args.index(".env")
    final_exclude_index = len(args) - 1 - args[::-1].index("!.env")
    assert final_exclude_index > include_index


def test_approve_for_session_executes_and_covers_edit(load_module) -> None:
    tools = load_module("tools", "tools.py")
    permissions = permission_module
    calls = 0

    class Handler:
        async def request(self, _request):
            nonlocal calls
            calls += 1
            return permissions.ApprovalResponse("approve_for_session")

    service = permissions.PermissionService(
        permissions.PermissionManager(Path(tools.WORKDIR)),
        Handler(),
    )
    tools.TOOL_REGISTRY["write_file"].execute = lambda _params: asyncio.sleep(0, result="wrote")
    tools.TOOL_REGISTRY["edit_file"].execute = lambda _params: asyncio.sleep(0, result="edited")

    output, _ = asyncio.run(tools.execute_tool_calls_async([
        _fc("write_file", "w1", '{"path":"tmp/x.txt","content":"x"}'),
        _fc("edit_file", "e1", '{"path":"tmp/x.txt","old_text":"x","new_text":"y"}'),
    ], permission_service=service))

    assert [item["output"] for item in output] == ["wrote", "edited"]
    assert calls == 1


def test_bash_approval_never_becomes_session_allow(load_module) -> None:
    permissions = permission_module
    calls = 0

    class Handler:
        async def request(self, request):
            nonlocal calls
            calls += 1
            assert request.allow_for_session is False
            return permissions.ApprovalResponse("approve_for_session")

    service = permissions.PermissionService(
        permissions.PermissionManager(Path.cwd()),
        Handler(),
    )

    first = asyncio.run(service.authorize("bash", {"command": "echo hi"}))
    second = asyncio.run(service.authorize("bash", {"command": "echo hi"}))

    assert first.behavior.value == "deny"
    assert second.behavior.value == "deny"
    assert calls == 2
