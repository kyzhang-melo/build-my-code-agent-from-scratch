from __future__ import annotations

import asyncio
import json
import subprocess
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
    assert manager.check("git_diff", {}).behavior.value == "allow"
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
    [
        ".env",
        ".env.local",
        "secret.pem",
        "id_rsa",
        ".git/config",
        ".sessions/run.jsonl",
        ".transcripts/precompact.jsonl",
    ],
)
def test_sensitive_reads_are_denied(load_module, path) -> None:
    permissions = permission_module
    manager = permissions.PermissionManager(Path.cwd())

    decision = manager.check("read_file", {"path": path})

    assert decision.behavior.value == "deny"
    assert "sensitive" in decision.reason.lower()


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".git/config",
        ".ssh/config",
        "secret.key",
        ".sessions/run.jsonl",
        ".transcripts/precompact.jsonl",
        "../outside.txt",
    ],
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
        ("echo x > /dev/tty", "device"),
        ("echo x > /dev/sda1", "device"),
        ("echo x > /dev/random", "device"),
        ("cd .. && pwd", "outside"),
        ("echo x > ../outside.txt", "outside"),
        ("rm -rf ../outside", "outside"),
        ("cat .sessions/run.jsonl", "session"),
        ("echo x > .transcripts/precompact.jsonl", "session"),
    ],
)
def test_bash_hard_deny_checks(load_module, command, expected) -> None:
    permissions = permission_module

    reason = permissions.bash_hard_deny_reason(command, Path.cwd())

    assert reason is not None
    assert expected in reason.lower()


@pytest.mark.parametrize(
    "command",
    [
        "find . 2>/dev/null",
        "echo x >/dev/null",
        "echo x >>/dev/null",
        "echo x 1>/dev/null",
        "cmd 2>>/dev/null",
        "echo x > '/dev/null'",
        "echo x > /dev/../dev/null",
        "grep -r foo . 2>/dev/null | head -10",
    ],
)
def test_devnull_redirects_are_allowed(load_module, command) -> None:
    permissions = permission_module

    reason = permissions.bash_hard_deny_reason(command, Path.cwd())

    assert reason is None


@pytest.mark.parametrize(
    "command",
    [
        "device_target=zero; printf 'x' > /dev/$device_target",
        "printf 'x' > /dev/zer*",
        "printf 'x' > /dev/{zero,null}",
    ],
)
def test_dynamic_dev_redirects_are_hard_denied(load_module, command) -> None:
    permissions = permission_module
    manager = permissions.PermissionManager(Path.cwd())

    reason = permissions.bash_hard_deny_reason(command, Path.cwd())
    decision = manager.check("bash", {"command": command})

    assert reason is not None
    assert "device" in reason.lower()
    assert decision.behavior.value == "deny"
    assert "device" in decision.reason.lower()


def test_devnull_bash_returns_ask_not_allow(load_module) -> None:
    permissions = permission_module
    manager = permissions.PermissionManager(Path.cwd())

    decision = manager.check("bash", {"command": "find . 2>/dev/null"})

    assert decision.behavior.value == "ask"


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
    request = permissions.ApprovalRequest(
        tool_name="bash",
        action="bash:echo hi",
        description="echo hi",
        allow_for_session=False,
    )

    async def scenario():
        release = asyncio.Event()

        async def delayed_prompt(_prompt: str) -> str:
            await release.wait()
            return "y"

        handler = permissions.TerminalApprovalHandler(
            interactive=True,
            prompt_fn=delayed_prompt,
        )
        task = asyncio.create_task(handler.request(request))
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        return await task

    response = asyncio.run(scenario())

    assert response.kind == "approve"


@pytest.mark.parametrize(
    ("answers", "expected_kind"),
    [
        (["y"], "approve"),
        (["n"], "reject"),
        (["a"], "approve_for_session"),
        (["invalid", "still invalid", "nope"], "reject"),
    ],
)
def test_terminal_handler_async_prompt_choices(answers, expected_kind) -> None:
    iterator = iter(answers)

    async def prompt(_message: str) -> str:
        return next(iterator)

    handler = permission_module.TerminalApprovalHandler(
        interactive=True,
        prompt_fn=prompt,
    )
    request = permission_module.ApprovalRequest(
        tool_name="write_file",
        action="write:/tmp/x",
        description="write x",
        allow_for_session=True,
    )

    response = asyncio.run(handler.request(request))

    assert response.kind == expected_kind


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt(), asyncio.CancelledError()])
def test_terminal_handler_async_prompt_cancellation(error) -> None:
    async def cancelled_prompt(_message: str) -> str:
        raise error

    handler = permission_module.TerminalApprovalHandler(
        interactive=True,
        prompt_fn=cancelled_prompt,
    )
    request = permission_module.ApprovalRequest(
        tool_name="bash",
        action="bash:echo hi",
        description="echo hi",
        allow_for_session=False,
    )

    response = asyncio.run(handler.request(request))

    assert response.kind == "reject"
    assert "cancelled" in response.feedback.lower()


def test_terminal_handler_without_prompt_provider_fails_closed() -> None:
    handler = permission_module.TerminalApprovalHandler(interactive=True)
    request = permission_module.ApprovalRequest(
        tool_name="bash",
        action="bash:echo hi",
        description="echo hi",
        allow_for_session=False,
    )

    response = asyncio.run(handler.request(request))

    assert response.kind == "reject"
    assert "not configured" in response.feedback.lower()


def test_permission_denial_short_circuits_tool_execution(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    permissions = permission_module
    executed = False

    async def execute(_params):
        nonlocal executed
        executed = True
        return "should not run"

    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["bash"].execute = execute
    service = permissions.PermissionService(
        permissions.PermissionManager(workspace.root),
        permissions.TerminalApprovalHandler(interactive=False),
    )

    output, _ = asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        registry,
        tools.TodoManager(),
        permission_service=service,
    ))

    payload = json.loads(output[0]["output"])
    assert executed is False
    assert payload["error"] == "permission_denied"
    assert payload["tool"] == "bash"
    assert payload["retryable"] is False


def test_permission_failure_is_fail_closed(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    executed = False

    class BrokenService:
        async def authorize(self, *_args, **_kwargs):
            raise RuntimeError("broken policy")

    async def execute(_params):
        nonlocal executed
        executed = True
        return "should not run"

    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["bash"].execute = execute
    output, _ = asyncio.run(tools.execute_tool_calls_async(
        [_fc("bash", "b1", '{"command":"echo hi"}')],
        registry,
        tools.TodoManager(),
        permission_service=BrokenService(),
    ))

    payload = json.loads(output[0]["output"])
    assert executed is False
    assert payload["error"] == "permission_denied"
    assert "failed" in payload["reason"].lower()


def test_write_preview_does_not_print_content(load_module, workspace, capsys) -> None:
    tools = load_module("tools", "tools.py")
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["write_file"].execute = lambda _params: asyncio.sleep(
        0, result="wrote"
    )
    tools.execute_tool_calls([
        _fc("write_file", "w1", '{"path":"tmp/x.txt","content":"TOP_SECRET_VALUE"}'),
    ], registry, tools.TodoManager())

    printed = capsys.readouterr().out
    assert "path='tmp/x.txt'" in printed
    assert "chars=16" in printed
    assert "TOP_SECRET_VALUE" not in printed


def test_glob_hides_sensitive_paths(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    base = workspace.root / "tests" / "_tmp_permission_glob"
    base.mkdir(parents=True, exist_ok=True)
    (base / ".env").write_text("SECRET=value")
    (base / "visible.txt").write_text("ok")
    try:
        output = tools.run_glob(workspace, "*", "tests/_tmp_permission_glob")
    finally:
        (base / ".env").unlink()
        (base / "visible.txt").unlink()
        base.rmdir()

    assert "visible.txt" in output
    assert ".env" not in output


def test_grep_sensitive_excludes_override_caller_glob(load_module, workspace, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    monkeypatch.setattr(tools.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    tools.run_grep(workspace, "SECRET", glob=".env")

    args = captured["args"]
    include_index = args.index(".env")
    final_exclude_index = len(args) - 1 - args[::-1].index("!.env")
    assert final_exclude_index > include_index


def test_approve_for_session_executes_and_covers_edit(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    permissions = permission_module
    calls = 0

    class Handler:
        async def request(self, _request):
            nonlocal calls
            calls += 1
            return permissions.ApprovalResponse("approve_for_session")

    service = permissions.PermissionService(
        permissions.PermissionManager(workspace.root),
        Handler(),
    )
    registry = tools.build_tool_registry(workspace, tools.TodoManager())
    registry["write_file"].execute = lambda _params: asyncio.sleep(0, result="wrote")
    registry["edit_file"].execute = lambda _params: asyncio.sleep(0, result="edited")

    output, _ = asyncio.run(tools.execute_tool_calls_async([
        _fc("write_file", "w1", '{"path":"tmp/x.txt","content":"x"}'),
        _fc("edit_file", "e1", '{"path":"tmp/x.txt","old_text":"x","new_text":"y"}'),
    ], registry, tools.TodoManager(), permission_service=service))

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
