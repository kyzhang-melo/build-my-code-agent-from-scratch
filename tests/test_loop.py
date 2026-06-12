from __future__ import annotations

import sys
import types

import pytest


def test_input_prompt_marks_ansi_sequences_as_nonprinting(load_module) -> None:
    main_module = load_module("main", "main.py")

    assert main_module.INPUT_PROMPT == "\001\033[36m\002s01 >> \001\033[0m\002"


def _todo_params(items: list[dict]):
    return sys.modules["tools"].TodoParams.model_validate({"items": items})


def test_run_one_turn_full_iteration(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    captured = {}

    function_call = types.SimpleNamespace(
        type="function_call",
        call_id="c1",
        name="bash",
        arguments='{"command":"echo hi"}',
    )
    fake_response = types.SimpleNamespace(
        output=[function_call],
        output_text="Running command...",
    )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return fake_response

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=fake_create)),
    )
    monkeypatch.setattr(
        main_module,
        "execute_tool_calls",
        lambda _output: ([{"type": "function_call_output", "call_id": "c1", "output": "ok"}], False),
    )

    state = main_module.LoopState(
        messages=[
            {"role": "user", "content": "task part 1"},
            {"role": "user", "content": "task part 2"},
        ]
    )

    outcome = main_module.run_one_turn(state)

    assert outcome is None
    assert state.api_call_count == 1
    assert captured["input"][0]["role"] == "user"
    assert captured["input"][0]["content"] == "task part 1\ntask part 2"
    assert any(m.get("type") == "function_call" and m.get("call_id") == "c1" for m in state.messages)
    assert any(m.get("role") == "assistant" and m.get("content") == "Running command..." for m in state.messages)
    assert state.messages[-1]["type"] == "function_call_output"
    assert state.messages[-1]["output"] == "ok"


def _no_tool_response(output_text: str):
    return types.SimpleNamespace(output=[], output_text=output_text)


def test_run_one_turn_sets_final_text_from_current_turn(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update(_todo_params([]))

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **_: _no_tool_response("Fresh answer."))
        ),
    )

    state = main_module.LoopState(messages=[{"role": "user", "content": "new query"}])
    outcome = main_module.run_one_turn(state)

    assert outcome is not None
    assert outcome.stop_reason == "completed"
    assert outcome.final_text == "Fresh answer."


def test_run_one_turn_does_not_surface_stale_assistant_text(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update(_todo_params([]))

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **_: _no_tool_response(""))
        ),
    )

    # History carries a previous turn's answer; the current turn produces no text.
    state = main_module.LoopState(messages=[
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "STALE ANSWER"},
        {"role": "user", "content": "new query"},
    ])
    outcome = main_module.run_one_turn(state)

    assert outcome is not None
    assert outcome.final_text == ""


def test_run_one_turn_contract_nudge_when_unresolved_todo(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update(_todo_params([{"content": "unfinished", "status": "in_progress"}]))

    fake_response = types.SimpleNamespace(output=[], output_text="All done.")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **_: fake_response)),
    )

    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])
    outcome = main_module.run_one_turn(state)

    assert outcome is None
    assert state.nudges == 1
    assert state.messages[-1]["role"] == "user"
    assert "Before ending, either complete all todo items" in state.messages[-1]["content"]


def test_run_one_turn_contract_warns_after_max_nudges(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update(_todo_params([{"content": "unfinished", "status": "pending"}]))

    fake_response = types.SimpleNamespace(output=[], output_text="Done.")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **_: fake_response)),
    )

    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        nudges=main_module.TODO_CONTRACT_MAX_NUDGES,
    )
    outcome = main_module.run_one_turn(state)

    assert outcome is not None
    assert outcome.stop_reason == "completed"
    assert state.messages[-1]["role"] == "assistant"
    assert "Ending with unresolved todo items" in state.messages[-1]["content"]


def test_todo_update_echoes_system_reminder(load_module) -> None:
    main_module = load_module("main", "main.py")

    output = main_module.TODO.update(_todo_params([{"content": "one", "status": "in_progress"}]))

    assert "<system-reminder>" in output
    assert "Your todo list has changed" in output
    assert "[>] one" in output


def test_todo_update_empty_echoes_cleared_reminder(load_module) -> None:
    main_module = load_module("main", "main.py")

    output = main_module.TODO.update(_todo_params([]))

    assert "<system-reminder>" in output
    assert "Your todo list is now empty." in output


def test_todo_stop_gate_nudges_on_unresolved_todo(load_module) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update(_todo_params([{"content": "unfinished", "status": "in_progress"}]))

    nudge = main_module.PARENT_CONFIG.stop_gate.check("All done.")

    assert nudge is not None
    assert "Before ending, either complete all todo items" in nudge


def test_todo_stop_gate_accepts_when_plan_completed(load_module) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update(_todo_params([{"content": "done", "status": "completed"}]))

    assert main_module.PARENT_CONFIG.stop_gate.check("All done.") is None


def test_run_one_turn_tool_calls_extend_messages(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update(_todo_params([]))
    function_call = types.SimpleNamespace(type="function_call", call_id="c1", name="bash", arguments="{}")
    fake_response = types.SimpleNamespace(output=[function_call], output_text="")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **_: fake_response)),
    )
    monkeypatch.setattr(
        main_module,
        "execute_tool_calls",
        lambda _output: ([{"type": "function_call_output", "call_id": "c1", "output": "ok"}], False),
    )
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = main_module.run_one_turn(state)

    assert outcome is None
    assert state.api_call_count == 1
    assert state.messages[-1]["type"] == "function_call_output"


def test_explore_subagent_tools_are_read_only(load_module) -> None:
    main_module = load_module("main", "main.py")

    tool_names = {tool["name"] for tool in main_module.EXPLORE_SUBAGENT_CONFIG.tools}

    assert tool_names == {"read_file", "glob", "grep"}
    assert set(main_module.EXPLORE_SUBAGENT_CONFIG.registry) == {"read_file", "glob", "grep"}


def test_parent_tools_include_task(load_module) -> None:
    main_module = load_module("main", "main.py")

    tool_names = {tool["name"] for tool in main_module.PARENT_CONFIG.tools}

    assert "task" in tool_names


def test_summary_stop_gate_nudges_short_summary(load_module) -> None:
    main_module = load_module("main", "main.py")

    nudge = main_module.EXPLORE_SUBAGENT_CONFIG.stop_gate.check("Short.")

    assert nudge is not None
    assert "too brief" in nudge


def test_summary_stop_gate_accepts_long_summary(load_module) -> None:
    main_module = load_module("main", "main.py")

    assert main_module.EXPLORE_SUBAGENT_CONFIG.stop_gate.check("x" * 200) is None


def test_run_one_turn_subagent_nudges_short_summary(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **_: _no_tool_response("Short."))
        ),
    )
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = main_module.run_one_turn(state, main_module.EXPLORE_SUBAGENT_CONFIG)

    assert outcome is None
    assert state.nudges == 1
    assert state.messages[-1]["role"] == "user"
    assert "too brief" in state.messages[-1]["content"]


def test_run_one_turn_subagent_accepts_after_max_attempts(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **_: _no_tool_response("Short."))
        ),
    )
    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        nudges=main_module.SUMMARY_CONTINUATION_ATTEMPTS,
    )

    outcome = main_module.run_one_turn(state, main_module.EXPLORE_SUBAGENT_CONFIG)

    assert outcome is not None
    assert outcome.final_text == "Short."
    # SummaryStopGate has no give-up note: no nudge or warning is appended.
    assert state.messages[-1] == {"role": "assistant", "content": "Short."}


def test_run_one_turn_subagent_evaluates_only_current_text(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(
            responses=types.SimpleNamespace(create=lambda **_: _no_tool_response(""))
        ),
    )
    # A long prior assistant message must NOT be mistaken for this turn's summary.
    state = main_module.LoopState(
        messages=[{"role": "assistant", "content": "x" * 200}],
    )

    outcome = main_module.run_one_turn(state, main_module.EXPLORE_SUBAGENT_CONFIG)

    assert outcome is None
    assert state.nudges == 1
    assert "too brief" in state.messages[-1]["content"]


def test_run_one_turn_budget_exhausted_returns_max_api_calls_outcome(load_module) -> None:
    main_module = load_module("main", "main.py")
    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        api_call_count=main_module.PARENT_CONFIG.max_api_calls,
    )

    outcome = main_module.run_one_turn(state)

    assert outcome is not None
    assert outcome.stop_reason == "max_api_calls"
    assert "stopped after max_api_calls" in outcome.final_text
    assert state.messages[-1]["content"] == outcome.final_text
