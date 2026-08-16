from __future__ import annotations

import asyncio
import os
import re
import time

import pytest


def _run(tools, workspace, command: str) -> str:
    return asyncio.run(tools.run_bash(workspace, command))


def _metadata(output: str) -> tuple[dict[str, str], str]:
    header, body = output.split("\n\n", 1)
    values = {}
    for line in header.splitlines():
        key, value = line.split("] ", 1)
        values[key.removeprefix("[")] = value
    return values, body


def test_bash_returns_structured_success_with_output(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    metadata, body = _metadata(_run(tools, workspace, "printf hello"))

    assert metadata["status"] == "completed"
    assert metadata["exit_code"] == "0"
    assert metadata["timed_out"] == "false"
    assert metadata["truncated"] == "false"
    assert int(metadata["duration_ms"]) >= 0
    assert body == "hello"


def test_registry_bash_uses_local_workspace(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    from sandbox import LocalSandbox

    runtime = LocalSandbox(workspace)
    registry = tools.build_tool_registry(
        workspace,
        tools.TodoManager(),
        {"bash"},
        file_backend=runtime.file_backend,
        command_runner=runtime.command_runner,
    )

    output = asyncio.run(registry["bash"].execute(
        tools.BashParams(command="printf hello; pwd"),
    ))
    metadata, body = _metadata(output)

    assert metadata["status"] == "completed"
    assert body.splitlines() == ["hello" + str(workspace.root)]


@pytest.mark.parametrize(
    "result, expected_status, expected_exit_code, expected_timed_out, expected_body",
    [
        ((7, "", "failure", False), "failed", "7", "false", "failure"),
        ((124, "", "timeout", True), "timed_out", "null", "true", "timeout"),
    ],
)
def test_remote_bash_uses_structured_result(
    load_module,
    workspace,
    result,
    expected_status,
    expected_exit_code,
    expected_timed_out,
    expected_body,
) -> None:
    tools = load_module("tools", "tools.py")
    from sandbox import CommandResult

    del workspace
    rendered = tools.render_remote_bash_result(
        CommandResult(*result[:3], timed_out=result[3]), duration_ms=12,
    )
    metadata, body = _metadata(rendered)

    assert metadata["status"] == expected_status
    assert metadata["exit_code"] == expected_exit_code
    assert metadata["timed_out"] == expected_timed_out
    assert body == expected_body


def test_registry_uses_explicit_local_backend_location(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")
    from sandbox import LocalSandbox

    runtime = LocalSandbox(workspace)

    class WrappedLocalBackend:
        execution_location = "local"

        def grep(self, request):
            return runtime.file_backend.grep(request)

    registry = tools.build_tool_registry(
        workspace,
        tools.TodoManager(),
        {"bash"},
        file_backend=WrappedLocalBackend(),
        command_runner=runtime.command_runner,
    )

    output = asyncio.run(registry["bash"].execute(
        tools.BashParams(command="printf wrapped-local"),
    ))

    assert "wrapped-local" in output
    assert not hasattr(runtime.file_backend, "call")


def test_registry_rejects_remote_backend_without_remote_capability(
    load_module, workspace,
) -> None:
    tools = load_module("tools", "tools.py")

    class InvalidRemoteBackend:
        execution_location = "remote"

        def grep(self, _request):
            return ""

    with pytest.raises(TypeError, match="must implement call"):
        tools.build_tool_registry(
            workspace,
            tools.TodoManager(),
            {"grep"},
            file_backend=InvalidRemoteBackend(),
        )


def test_bash_uses_bash_syntax(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    metadata, body = _metadata(_run(tools, workspace, 'items=(zero one); printf "%s" "${items[1]}"'))

    assert metadata["status"] == "completed"
    assert body == "one"


def test_bash_returns_structured_error_when_bash_is_unavailable(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_CANDIDATE_PATHS", ("/definitely/missing/bash",))

    metadata, body = _metadata(_run(tools, workspace, "printf should-not-run"))

    assert metadata["status"] == "execution_error"
    assert metadata["exit_code"] == "null"
    assert "Bash executable not found" in body


def test_bash_reports_nonzero_exit_code(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    metadata, body = _metadata(_run(tools, workspace, "printf failure; exit 7"))

    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == "7"
    assert body == "failure"


def test_bash_merges_stdout_and_stderr_as_chunks_arrive(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    _, body = _metadata(
        _run(
            tools,
            workspace,
            "printf out1; sleep 0.05; printf err1 >&2; sleep 0.05; printf out2",
        )
    )

    assert "out1" in body
    assert "err1" in body
    assert "out2" in body
    # The delays make this a deterministic integration check for the observed
    # stream arrival order without promising a cross-pipe kernel ordering API.
    assert body.index("out1") < body.index("err1") < body.index("out2")


def test_bash_timeout_keeps_already_received_output(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_TIMEOUT_SECONDS", 0.05)

    metadata, body = _metadata(_run(tools, workspace, "printf before; sleep 1"))

    assert metadata["status"] == "timed_out"
    assert metadata["exit_code"] == "null"
    assert metadata["timed_out"] == "true"
    assert "before" in body
    assert "Command killed by timeout (0.05s)" in body


def test_bash_waits_for_process_not_pipe_eof(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_TIMEOUT_SECONDS", 0.05)

    metadata, _ = _metadata(_run(tools, workspace, "exec sleep 1 1>&- 2>&-"))

    assert metadata["status"] == "timed_out"
    assert metadata["exit_code"] == "null"


def test_bash_post_exit_cleanup_preserves_known_exit_code(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_POST_EXIT_DRAIN_SECONDS", 0.05)

    metadata, body = _metadata(_run(tools, workspace, "sleep 1 >&1 &"))

    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == "0"
    assert metadata["post_exit_cleanup"] == "true"
    assert "descendants were terminated" in body


def test_bash_timeout_kills_process_group_children(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_TIMEOUT_SECONDS", 0.05)
    child_pid: int | None = None

    try:
        _, body = _metadata(_run(tools, workspace, "sleep 10 & child=$!; echo $child; wait"))
        match = re.search(r"\b(\d+)\b", body)
        assert match is not None
        child_pid = int(match.group(1))

        for _ in range(50):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"background child {child_pid} survived bash timeout")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


def test_bash_keeps_tail_and_marks_truncated_output(load_module, monkeypatch, workspace) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_OUTPUT_MAX_CHARS", 120)
    command = (
        "i=1; while [ $i -le 100 ]; do "
        "printf 'build-noise-%03d\\n' \"$i\"; "
        "i=$((i + 1)); "
        "done; printf 'FINAL-RESULT\\n'"
    )

    metadata, body = _metadata(_run(tools, workspace, command))

    assert metadata["truncated"] == "true"
    assert "earlier command output discarded" in body
    assert "FINAL-RESULT" in body
    assert "build-noise-001" not in body


def test_bash_reports_silent_success_and_reuses_permission_hard_deny(load_module, workspace) -> None:
    tools = load_module("tools", "tools.py")

    success_metadata, success_body = _metadata(_run(tools, workspace, "true"))
    blocked_metadata, blocked_body = _metadata(_run(tools, workspace, "sudo echo should-not-run"))

    assert success_metadata["status"] == "completed"
    assert success_body == "(no output)"
    assert blocked_metadata["status"] == "blocked"
    assert blocked_metadata["exit_code"] == "null"
    assert "privilege escalation" in blocked_body.lower()
