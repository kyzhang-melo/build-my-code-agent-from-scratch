from __future__ import annotations

import asyncio
import os
import re
import time


def _run(tools, command: str) -> str:
    return asyncio.run(tools.run_bash(command))


def _metadata(output: str) -> tuple[dict[str, str], str]:
    header, body = output.split("\n\n", 1)
    values = {}
    for line in header.splitlines():
        key, value = line.split("] ", 1)
        values[key.removeprefix("[")] = value
    return values, body


def test_bash_returns_structured_success_with_output(load_module) -> None:
    tools = load_module("tools", "tools.py")

    metadata, body = _metadata(_run(tools, "printf hello"))

    assert metadata["status"] == "completed"
    assert metadata["exit_code"] == "0"
    assert metadata["timed_out"] == "false"
    assert metadata["truncated"] == "false"
    assert int(metadata["duration_ms"]) >= 0
    assert body == "hello"


def test_bash_uses_bash_syntax(load_module) -> None:
    tools = load_module("tools", "tools.py")

    metadata, body = _metadata(_run(tools, 'items=(zero one); printf "%s" "${items[1]}"'))

    assert metadata["status"] == "completed"
    assert body == "one"


def test_bash_returns_structured_error_when_bash_is_unavailable(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_CANDIDATE_PATHS", ("/definitely/missing/bash",))

    metadata, body = _metadata(_run(tools, "printf should-not-run"))

    assert metadata["status"] == "execution_error"
    assert metadata["exit_code"] == "null"
    assert "Bash executable not found" in body


def test_bash_reports_nonzero_exit_code(load_module) -> None:
    tools = load_module("tools", "tools.py")

    metadata, body = _metadata(_run(tools, "printf failure; exit 7"))

    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == "7"
    assert body == "failure"


def test_bash_merges_stdout_and_stderr_as_chunks_arrive(load_module) -> None:
    tools = load_module("tools", "tools.py")

    _, body = _metadata(
        _run(
            tools,
            "printf out1; sleep 0.05; printf err1 >&2; sleep 0.05; printf out2",
        )
    )

    assert "out1" in body
    assert "err1" in body
    assert "out2" in body
    # The delays make this a deterministic integration check for the observed
    # stream arrival order without promising a cross-pipe kernel ordering API.
    assert body.index("out1") < body.index("err1") < body.index("out2")


def test_bash_timeout_keeps_already_received_output(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_TIMEOUT_SECONDS", 0.05)

    metadata, body = _metadata(_run(tools, "printf before; sleep 1"))

    assert metadata["status"] == "timed_out"
    assert metadata["exit_code"] == "null"
    assert metadata["timed_out"] == "true"
    assert "before" in body
    assert "Command killed by timeout (0.05s)" in body


def test_bash_waits_for_process_not_pipe_eof(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_TIMEOUT_SECONDS", 0.05)

    metadata, _ = _metadata(_run(tools, "exec sleep 1 1>&- 2>&-"))

    assert metadata["status"] == "timed_out"
    assert metadata["exit_code"] == "null"


def test_bash_post_exit_cleanup_preserves_known_exit_code(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_POST_EXIT_DRAIN_SECONDS", 0.05)

    metadata, body = _metadata(_run(tools, "sleep 1 >&1 &"))

    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == "0"
    assert metadata["post_exit_cleanup"] == "true"
    assert "descendants were terminated" in body


def test_bash_timeout_kills_process_group_children(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_TIMEOUT_SECONDS", 0.05)
    child_pid: int | None = None

    try:
        _, body = _metadata(_run(tools, "sleep 10 & child=$!; echo $child; wait"))
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


def test_bash_keeps_tail_and_marks_truncated_output(load_module, monkeypatch) -> None:
    tools = load_module("tools", "tools.py")
    monkeypatch.setattr(tools, "BASH_OUTPUT_MAX_CHARS", 120)
    command = (
        "i=1; while [ $i -le 100 ]; do "
        "printf 'build-noise-%03d\\n' \"$i\"; "
        "i=$((i + 1)); "
        "done; printf 'FINAL-RESULT\\n'"
    )

    metadata, body = _metadata(_run(tools, command))

    assert metadata["truncated"] == "true"
    assert "earlier command output discarded" in body
    assert "FINAL-RESULT" in body
    assert "build-noise-001" not in body


def test_bash_reports_silent_success_and_reuses_permission_hard_deny(load_module) -> None:
    tools = load_module("tools", "tools.py")

    success_metadata, success_body = _metadata(_run(tools, "true"))
    blocked_metadata, blocked_body = _metadata(_run(tools, "sudo echo should-not-run"))

    assert success_metadata["status"] == "completed"
    assert success_body == "(no output)"
    assert blocked_metadata["status"] == "blocked"
    assert blocked_metadata["exit_code"] == "null"
    assert "privilege escalation" in blocked_body.lower()
