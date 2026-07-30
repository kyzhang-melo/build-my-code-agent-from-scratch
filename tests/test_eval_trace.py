from __future__ import annotations

import sys


def test_eval_matches_legacy_and_precise_trace_expectations(load_module, tmp_path) -> None:
    runner = load_module("eval_runner", "evals/run_evals.py")
    events = [
        {
            "event": "tool.requested",
            "tool_name": "write_file",
            "arguments": {"path": "hello.txt", "content_chars": 5},
        },
        {
            "event": "permission.decided",
            "tool_name": "write_file",
            "policy_behavior": "ask",
            "approval_kind": "approve",
            "decision": "allow",
        },
        {
            "event": "tool.completed",
            "tool_name": "write_file",
            "arguments": {"path": "hello.txt", "content_chars": 5},
            "status": "success",
        },
        {
            "event": "todo.changed",
            "transitions": [
                {"content": "write file", "from": "in_progress", "to": "completed"},
            ],
        },
        {
            "event": "stop_gate.checked",
            "gate": "todo",
            "decision": "allow",
            "reason": "requirements_satisfied",
        },
    ]
    expect = {
        "tools_used": [
            {"name": "write_file", "args_contains": {"path": "hello.txt"}},
        ],
        "tools_not_used": ["edit_file"],
        "tool_completed": [{"name": "write_file", "status": "success"}],
        "permission_decisions": [
            {"tool": "write_file", "approval_kind": "approve", "decision": "allow"},
        ],
        "todo_transitions": [{"content": "write file", "to": "completed"}],
        "stop_gate_decisions": [{"gate": "todo", "decision": "allow"}],
    }

    checks = runner.evaluate(expect, tmp_path, events, "done")

    assert checks
    assert all(check.passed for check in checks)


def test_eval_does_not_infer_tool_completion_from_messages(load_module, tmp_path) -> None:
    runner = load_module("eval_runner", "evals/run_evals.py")

    checks = runner.evaluate(
        {"tools_used": [{"name": "write_file"}]},
        tmp_path,
        [{"event": "tool.requested", "tool_name": "write_file"}],
        "",
    )

    assert checks[0].passed is False


def test_behavioral_eval_output_limit_is_opt_in(load_module, monkeypatch) -> None:
    runner = load_module("eval_runner", "evals/run_evals.py")
    monkeypatch.setattr(sys, "argv", ["run_evals.py"])
    assert runner.parse_args().max_output_tokens is None

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_evals.py", "--max-output-tokens", "8000"],
    )
    assert runner.parse_args().max_output_tokens == 8000
