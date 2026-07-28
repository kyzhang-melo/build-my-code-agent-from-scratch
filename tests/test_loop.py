from __future__ import annotations

import asyncio
import sys
import types

import pytest


def test_input_prompt_marks_ansi_sequences_as_nonprinting(load_module) -> None:
    main_module = load_module("main", "main.py")

    assert main_module.INPUT_PROMPT == "\001\033[36m\002s01 >> \001\033[0m\002"


def test_permission_mode_commands(load_module, capsys, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    assert _run(main_module.handle_command("/mode plan", [], session)) is True
    assert session.permission_service.manager.mode.value == "plan"
    assert _run(main_module.handle_command("/permissions", [], session)) is True

    output = capsys.readouterr().out
    assert "mode=plan" in output
    assert "session-approved paths: none" in output


def _todo_params(items: list[dict]):
    return sys.modules["tools"].TodoParams.model_validate({"items": items})


def _run(awaitable):
    return asyncio.run(awaitable)


def _parent(main_module, tmp_path, on_text=None):
    return main_module.create_parent_session(
        tmp_path,
        approval_handler=None,
        on_text=on_text,
    )


def _explore(main_module, parent):
    return main_module.create_explore_session(
        parent.workspace,
        parent.permission_service,
        parent.trace_context,
        parent.session_id,
    )


def _client_for_response(response, captured: dict | None = None):
    async def fake_create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return response

    return types.SimpleNamespace(responses=types.SimpleNamespace(create=fake_create))


def test_run_one_turn_full_iteration(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
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

    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response, captured))

    async def fake_execute(_output, *_args, **_kwargs):
        return ([{"type": "function_call_output", "call_id": "c1", "output": "ok"}], False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)

    state = main_module.LoopState(
        messages=[
            {"role": "user", "content": "task part 1"},
            {"role": "user", "content": "task part 2"},
        ]
    )

    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is None
    assert state.api_call_count == 1
    assert captured["input"][0]["role"] == "user"
    assert captured["input"][0]["content"] == "task part 1"
    assert captured["input"][1]["role"] == "user"
    assert captured["input"][1]["content"] == "task part 2"
    assert any(m.get("type") == "function_call" and m.get("call_id") == "c1" for m in state.messages)
    assert any(m.get("role") == "assistant" and m.get("content") == "Running command..." for m in state.messages)
    assert state.messages[-1]["type"] == "function_call_output"
    assert state.messages[-1]["output"] == "ok"


def _no_tool_response(output_text: str):
    return types.SimpleNamespace(output=[], output_text=output_text)


def _reasoning_item():
    return types.SimpleNamespace(
        type="reasoning",
        id="rs_1",
        status="completed",
        summary=[types.SimpleNamespace(type="summary_text", text="thought")],
    )


def test_run_one_turn_surfaces_intermediate_text_with_tool_calls(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")

    function_call = types.SimpleNamespace(type="function_call", call_id="c1", name="bash", arguments="{}")
    fake_response = types.SimpleNamespace(output=[function_call], output_text="INTERIM SUMMARY")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))

    async def fake_execute(_output, *_args, **_kwargs):
        return ([{"type": "function_call_output", "call_id": "c1", "output": "ok"}], False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)

    surfaced: list[str] = []
    session = _parent(main_module, tmp_path, surfaced.append)
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = _run(main_module.run_one_turn(state, session))

    # Text that rides along with a tool call is shown, not swallowed.
    assert outcome is None
    assert surfaced == ["INTERIM SUMMARY"]


def test_run_one_turn_no_tool_calls_does_not_surface_via_on_text(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.setattr(main_module, "client", _client_for_response(_no_tool_response("Final answer.")))

    surfaced: list[str] = []
    session = _parent(main_module, tmp_path, surfaced.append)
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = _run(main_module.run_one_turn(state, session))

    # The final no-tool answer is delivered via the return value (and __main__),
    # not on_text -- so it is not surfaced twice.
    assert outcome is not None
    assert outcome.final_text == "Final answer."
    assert surfaced == []


def test_run_one_turn_surfaces_text_when_nudged(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    # An unresolved todo makes TodoStopGate.check() reject the stop with a nudge,
    # so the model's no-tool answer rides a *continuing* turn (Door 2).
    monkeypatch.setattr(main_module, "client", _client_for_response(_no_tool_response("MY SUMMARY")))

    surfaced: list[str] = []
    session = _parent(main_module, tmp_path, surfaced.append)
    session.todo.update(_todo_params([{"content": "unfinished", "status": "in_progress"}]))
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = _run(main_module.run_one_turn(state, session))

    # The rejected answer is shown before the contract nudge, not swallowed.
    assert outcome is None
    assert surfaced == ["MY SUMMARY"]
    assert state.messages[-1]["role"] == "user"
    assert "Before ending, either complete all todo items" in state.messages[-1]["content"]


def test_run_one_turn_sets_final_text_from_current_turn(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    monkeypatch.setattr(main_module, "client", _client_for_response(_no_tool_response("Fresh answer.")))

    state = main_module.LoopState(messages=[{"role": "user", "content": "new query"}])
    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is not None
    assert outcome.stop_reason == "completed"
    assert outcome.final_text == "Fresh answer."


def test_run_one_turn_does_not_surface_stale_assistant_text(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    monkeypatch.setattr(main_module, "client", _client_for_response(_no_tool_response("")))

    # History carries a previous turn's answer; the current turn produces no text.
    state = main_module.LoopState(messages=[
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "STALE ANSWER"},
        {"role": "user", "content": "new query"},
    ])
    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is None
    assert state.messages[-1]["role"] == "user"
    assert "assistant-visible text" in state.messages[-1]["content"]


def test_run_one_turn_contract_nudge_when_unresolved_todo(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    session.todo.update(_todo_params([{"content": "unfinished", "status": "in_progress"}]))

    fake_response = types.SimpleNamespace(output=[], output_text="All done.")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))

    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])
    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is None
    assert state.nudges == 1
    assert state.messages[-1]["role"] == "user"
    assert "Before ending, either complete all todo items" in state.messages[-1]["content"]


def test_run_one_turn_contract_warns_after_max_nudges(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")

    fake_response = types.SimpleNamespace(output=[], output_text="Done.")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))

    surfaced: list[str] = []
    session = _parent(main_module, tmp_path, surfaced.append)
    session.todo.update(_todo_params([{"content": "unfinished", "status": "pending"}]))
    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        nudges=main_module.TODO_CONTRACT_MAX_NUDGES,
    )
    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is not None
    assert outcome.stop_reason == "completed"
    assert state.messages[-1]["role"] == "assistant"
    assert "Ending with unresolved todo items" in state.messages[-1]["content"]
    # Give-up path stops: the text is delivered as final_text, not surfaced twice.
    assert outcome.final_text == "Done."
    assert surfaced == []


def test_todo_update_echoes_system_reminder(load_module) -> None:
    load_module("main", "main.py")
    todo = sys.modules["tools"].TodoManager()
    output = todo.update(_todo_params([{"content": "one", "status": "in_progress"}]))

    assert "<system-reminder>" in output
    assert "Your todo list has changed" in output
    assert "[>] one" in output


def test_todo_update_empty_echoes_cleared_reminder(load_module) -> None:
    load_module("main", "main.py")
    todo = sys.modules["tools"].TodoManager()
    output = todo.update(_todo_params([]))

    assert "<system-reminder>" in output
    assert "Your todo list is now empty." in output


def test_todo_update_completed_reminder_steers_to_final_answer(load_module) -> None:
    load_module("main", "main.py")
    todo = sys.modules["tools"].TodoManager()
    output = todo.update(_todo_params([{"content": "done", "status": "completed"}]))

    # A fully-complete plan should stop nudging for more todo calls and steer
    # the model to deliver its result instead.
    assert "Provide the final result" in output
    assert "Do not call the todo tool again" in output
    assert "Keep using the todo tool" not in output


def test_todo_stop_gate_nudges_on_unresolved_todo(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    session.todo.update(_todo_params([{"content": "unfinished", "status": "in_progress"}]))

    nudge = session.stop_gate.check("All done.")

    assert nudge is not None
    assert "Before ending, either complete all todo items" in nudge


def test_todo_stop_gate_accepts_when_plan_completed(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    session.todo.update(_todo_params([{"content": "done", "status": "completed"}]))

    assert session.stop_gate.check("All done.") is None


def test_run_one_turn_tool_calls_extend_messages(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    function_call = types.SimpleNamespace(type="function_call", call_id="c1", name="bash", arguments="{}")
    fake_response = types.SimpleNamespace(output=[function_call], output_text="")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))

    async def fake_execute(_output, *_args, **_kwargs):
        return ([{"type": "function_call_output", "call_id": "c1", "output": "ok"}], False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is None
    assert state.api_call_count == 1
    assert state.empty_response_nudges == 0
    assert state.messages[-1]["type"] == "function_call_output"


def test_run_one_turn_reasoning_only_response_is_nudged(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    fake_response = types.SimpleNamespace(output=[_reasoning_item()], output_text="")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is None
    assert state.empty_response_nudges == 1
    assert state.messages[-2]["type"] == "reasoning"
    assert state.messages[-2]["summary"][0]["text"] == "thought"
    assert state.messages[-1]["role"] == "user"
    assert "assistant-visible text" in state.messages[-1]["content"]


def test_agent_loop_replays_reasoning_item_after_empty_response(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    responses = [
        types.SimpleNamespace(output=[_reasoning_item()], output_text=""),
        _no_tool_response("Final answer."),
    ]
    captured_inputs: list[list[dict]] = []

    async def fake_create(**kwargs):
        captured_inputs.append(kwargs["input"])
        return responses.pop(0)

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=fake_create)),
    )
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = _run(main_module.agent_loop(state, session))

    assert outcome.final_text == "Final answer."
    assert outcome.api_calls == 2
    assert any(msg.get("type") == "reasoning" for msg in captured_inputs[1])
    assert any(
        msg.get("role") == "user" and "assistant-visible text" in msg.get("content", "")
        for msg in captured_inputs[1]
    )


def test_run_one_turn_empty_response_after_retry_returns_warning(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    fake_response = types.SimpleNamespace(output=[_reasoning_item()], output_text="")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))
    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        empty_response_nudges=main_module.EMPTY_RESPONSE_MAX_NUDGES,
    )

    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is not None
    assert outcome.stop_reason == "completed"
    assert "empty response with no tool calls" in outcome.final_text
    assert state.empty_response_nudges == 0
    assert state.messages[-1]["role"] == "assistant"
    assert state.messages[-1]["content"] == outcome.final_text


def test_run_one_turn_debugs_empty_output_text_shape(load_module, monkeypatch, capsys, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    message_item = types.SimpleNamespace(
        type="message",
        status="completed",
        content=[types.SimpleNamespace(type="output_text", text="hidden")],
    )
    fake_response = types.SimpleNamespace(output=[message_item], output_text="")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    _run(main_module.run_one_turn(state, session))

    output = capsys.readouterr().out
    assert "output_text empty; response.output has 1 item(s)" in output
    assert "output[0] type='message' status='completed'" in output
    assert "content_types=['output_text']" in output


def test_explore_subagent_tools_are_read_only(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    parent = _parent(main_module, tmp_path)
    explore = _explore(main_module, parent)

    tool_names = {tool["name"] for tool in explore.tools}

    assert tool_names == {"read_file", "glob", "grep"}
    assert set(explore.registry) == {"read_file", "glob", "grep"}
    assert explore.todo is not parent.todo


def test_explore_subagent_system_includes_glob_rule(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    system = _explore(main_module, _parent(main_module, tmp_path)).system

    assert "CRITICAL GLOB RULE" in system
    assert "**/*" in system
    assert "Prefer grep for content discovery" in system


def test_build_subagent_prompt_includes_glob_rule_and_task(load_module) -> None:
    main_module = load_module("main", "main.py")
    task = "Inspect src and report findings."

    prompt = main_module.build_subagent_prompt(task)

    assert "Mode: explore" in prompt
    assert "CRITICAL GLOB RULE" in prompt
    assert "Use pattern='*' only for a shallow top-level listing" in prompt
    assert prompt.endswith(f"Task:\n{task}")


def test_parent_tools_include_task(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    tool_names = {tool["name"] for tool in _parent(main_module, tmp_path).tools}

    assert "task" in tool_names


def test_summary_stop_gate_nudges_short_summary(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    explore = _explore(main_module, _parent(main_module, tmp_path))
    nudge = explore.stop_gate.check("Short.")

    assert nudge is not None
    assert "too brief" in nudge


def test_summary_stop_gate_accepts_long_summary(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    explore = _explore(main_module, _parent(main_module, tmp_path))
    assert explore.stop_gate.check("x" * 200) is None


def test_run_one_turn_subagent_nudges_short_summary(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    explore = _explore(main_module, _parent(main_module, tmp_path))
    monkeypatch.setattr(main_module, "client", _client_for_response(_no_tool_response("Short.")))
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    outcome = _run(main_module.run_one_turn(state, explore))

    assert outcome is None
    assert state.nudges == 1
    assert state.messages[-1]["role"] == "user"
    assert "too brief" in state.messages[-1]["content"]


def test_run_one_turn_subagent_accepts_after_max_attempts(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    explore = _explore(main_module, _parent(main_module, tmp_path))
    monkeypatch.setattr(main_module, "client", _client_for_response(_no_tool_response("Short.")))
    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        nudges=main_module.SUMMARY_CONTINUATION_ATTEMPTS,
    )

    outcome = _run(main_module.run_one_turn(state, explore))

    assert outcome is not None
    assert outcome.final_text == "Short."
    # ReportStopGate has no give-up note: no nudge or warning is appended.
    assert state.messages[-1] == {"role": "assistant", "content": "Short."}


def test_run_one_turn_subagent_evaluates_only_current_text(load_module, monkeypatch, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    explore = _explore(main_module, _parent(main_module, tmp_path))
    monkeypatch.setattr(main_module, "client", _client_for_response(_no_tool_response("")))
    # A long prior assistant message must NOT be mistaken for this turn's summary.
    state = main_module.LoopState(
        messages=[{"role": "assistant", "content": "x" * 200}],
    )

    outcome = _run(main_module.run_one_turn(state, explore))

    assert outcome is None
    assert state.empty_response_nudges == 1
    assert "assistant-visible text" in state.messages[-1]["content"]


def test_run_one_turn_budget_exhausted_returns_max_api_calls_outcome(load_module, tmp_path) -> None:
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)
    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        api_call_count=session.max_api_calls,
    )

    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is not None
    assert outcome.stop_reason == "max_api_calls"
    assert "stopped after max_api_calls" in outcome.final_text
    assert state.messages[-1]["content"] == outcome.final_text


def test_malformed_json_arguments_normalized_in_history(load_module, monkeypatch, tmp_path) -> None:
    """A malformed function_call must not be replayed to the Provider as-is.

    The replayed history gets arguments="{}"; the tool layer still receives
    the original SDK item so it can report the exact parse error; the
    Provider-assigned id is dropped to avoid pairing conflicts.
    """
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    function_call = types.SimpleNamespace(
        type="function_call",
        call_id="c1",
        id="fc_abc",
        name="edit_file",
        arguments='{bad json',
    )
    fake_response = types.SimpleNamespace(output=[function_call], output_text="")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))

    captured_tool_calls: list = []

    async def fake_execute(tool_calls, *_args, **_kwargs):
        captured_tool_calls.extend(tool_calls)
        return ([{
            "type": "function_call_output",
            "call_id": "c1",
            "output": "Error: invalid arguments for tool 'edit_file': Expecting property name enclosed in double quotes",
        }], False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)

    state = main_module.LoopState(messages=[{"role": "user", "content": "edit the file"}])
    outcome = _run(main_module.run_one_turn(state, session))

    assert outcome is None  # loop continues
    # Tool layer received the original SDK item with malformed arguments.
    assert len(captured_tool_calls) == 1
    assert captured_tool_calls[0].arguments == '{bad json'
    # Replayed history has sanitized arguments.
    fc_records = [m for m in state.messages if m.get("type") == "function_call"]
    assert len(fc_records) == 1
    assert fc_records[0]["arguments"] == "{}"
    assert "id" not in fc_records[0]
    assert fc_records[0]["call_id"] == "c1"
    # Error output preserved with parse error detail.
    fco_records = [m for m in state.messages if m.get("type") == "function_call_output"]
    assert len(fco_records) == 1
    assert "Expecting" in fco_records[0]["output"]


def test_valid_json_arguments_preserved_with_id(load_module, monkeypatch, tmp_path) -> None:
    """Valid JSON arguments are normalized (re-dumped) but id is retained."""
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    function_call = types.SimpleNamespace(
        type="function_call",
        call_id="c1",
        id="fc_abc",
        name="bash",
        arguments='{"command":"echo hi"}',
    )
    fake_response = types.SimpleNamespace(output=[function_call], output_text="")
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))

    async def fake_execute(_tool_calls, *_args, **_kwargs):
        return ([{"type": "function_call_output", "call_id": "c1", "output": "ok"}], False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)

    state = main_module.LoopState(messages=[{"role": "user", "content": "run"}])
    _run(main_module.run_one_turn(state, session))

    fc = [m for m in state.messages if m.get("type") == "function_call"][0]
    import json
    assert json.loads(fc["arguments"]) == {"command": "echo hi"}
    assert fc["id"] == "fc_abc"
    assert fc["call_id"] == "c1"


def test_mixed_batch_valid_and_malformed(load_module, monkeypatch, tmp_path) -> None:
    """A batch with one valid and one malformed call: valid keeps id, malformed drops id."""
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    valid_call = types.SimpleNamespace(
        type="function_call", call_id="c1", id="fc_valid",
        name="bash", arguments='{"command":"echo hi"}',
    )
    malformed_call = types.SimpleNamespace(
        type="function_call", call_id="c2", id="fc_bad",
        name="edit_file", arguments='{broken',
    )
    fake_response = types.SimpleNamespace(
        output=[valid_call, malformed_call], output_text="",
    )
    monkeypatch.setattr(main_module, "client", _client_for_response(fake_response))

    async def fake_execute(_tool_calls, *_args, **_kwargs):
        return ([
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "function_call_output", "call_id": "c2",
             "output": "Error: invalid arguments for tool 'edit_file': Expecting property name"},
        ], False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)

    state = main_module.LoopState(messages=[{"role": "user", "content": "do both"}])
    _run(main_module.run_one_turn(state, session))

    fc_records = {m["call_id"]: m for m in state.messages if m.get("type") == "function_call"}
    assert fc_records["c1"]["id"] == "fc_valid"
    import json
    json.loads(fc_records["c1"]["arguments"])  # valid JSON
    assert "id" not in fc_records["c2"]
    assert fc_records["c2"]["arguments"] == "{}"

    fco_call_ids = {m["call_id"] for m in state.messages if m.get("type") == "function_call_output"}
    assert fco_call_ids == {"c1", "c2"}


def test_normalize_messages_preserves_sanitized_arguments(load_module) -> None:
    """normalize_messages must not break already-sanitized arguments."""
    import json
    main_module = load_module("main", "main.py")
    from message_utils import normalize_messages

    messages = [
        {"role": "user", "content": "task"},
        {"type": "function_call", "call_id": "c1", "name": "edit_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1",
         "output": "Error: invalid arguments for tool 'edit_file': Expecting ',' delimiter"},
    ]
    cleaned = normalize_messages(messages)
    fc = [m for m in cleaned if m.get("type") == "function_call"][0]
    assert fc["arguments"] == "{}"
    assert json.loads(fc["arguments"]) == {}  # valid JSON
    fco = [m for m in cleaned if m.get("type") == "function_call_output"][0]
    assert "Expecting" in fco["output"]


def test_agent_loop_recovers_from_malformed_then_succeeds(load_module, monkeypatch, tmp_path) -> None:
    """Two-turn regression: malformed call -> normalized history -> corrected call -> done.

    Turn 1: model emits edit_file with invalid JSON arguments.
    Turn 2: model sees the sanitized function_call + error output, emits a
            corrected edit_file call, which succeeds; then a no-tool final answer.

    This directly verifies:
    - agent_loop completes the full recovery (does not crash or exit early);
    - the second API input contains no raw malformed arguments;
    - the second API input's function_call has arguments="{}" and no id;
    - the error function_call_output reaches the second turn.
    """
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    malformed_call = types.SimpleNamespace(
        type="function_call",
        call_id="c1",
        id="fc_bad",
        name="edit_file",
        arguments='{broken',
    )
    corrected_call = types.SimpleNamespace(
        type="function_call",
        call_id="c2",
        id="fc_good",
        name="edit_file",
        arguments='{"path":"tmp/x.txt","old_text":"a","new_text":"b"}',
    )
    responses = [
        types.SimpleNamespace(output=[malformed_call], output_text=""),
        types.SimpleNamespace(output=[corrected_call], output_text=""),
        _no_tool_response("Done."),
    ]
    captured_inputs: list[list[dict]] = []

    async def fake_create(**kwargs):
        captured_inputs.append(kwargs["input"])
        return responses.pop(0)

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=fake_create)),
    )

    async def fake_execute(tool_calls, *_args, **_kwargs):
        outputs = []
        for item in tool_calls:
            if item.arguments == '{broken':
                outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": "Error: invalid arguments for tool 'edit_file': Expecting property name",
                })
            else:
                outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": "Edited tmp/x.txt",
                })
        return (outputs, False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)

    state = main_module.LoopState(messages=[{"role": "user", "content": "edit the file"}])
    outcome = _run(main_module.agent_loop(state, session))

    # agent_loop completed the full recovery and returned a final answer.
    assert outcome.final_text == "Done."
    assert outcome.api_calls == 3

    # The second API input (turn 2) must not contain raw malformed arguments.
    second_input = captured_inputs[1]
    fc_items = [m for m in second_input if m.get("type") == "function_call"]
    assert len(fc_items) == 1
    assert fc_items[0]["arguments"] == "{}"  # sanitized, not '{broken'
    assert "id" not in fc_items[0]  # Provider id dropped
    assert fc_items[0]["call_id"] == "c1"
    # The error output reached the second turn.
    fco_items = [m for m in second_input if m.get("type") == "function_call_output"]
    assert len(fco_items) == 1
    assert "Expecting" in fco_items[0]["output"]
    assert fco_items[0]["call_id"] == "c1"

    # No raw malformed string anywhere in the second API input.
    import json
    assert "{broken" not in json.dumps(second_input)


def test_agent_loop_repeated_malformed_stops_at_max_api_calls(load_module, monkeypatch, tmp_path) -> None:
    """If the model keeps emitting malformed JSON, the loop ends via max_api_calls.

    This verifies the recovery path is bounded and does not spin forever.
    """
    main_module = load_module("main", "main.py")
    session = _parent(main_module, tmp_path)

    def malformed_call(call_id: str):
        return types.SimpleNamespace(
            type="function_call",
            call_id=call_id,
            id=f"fc_{call_id}",
            name="edit_file",
            arguments='{always broken',
        )

    # Enough malformed responses to exceed max_api_calls.
    responses = [
        types.SimpleNamespace(output=[malformed_call(f"c{i}")], output_text="")
        for i in range(session.max_api_calls + 2)
    ]
    captured_inputs: list[list[dict]] = []

    async def fake_create(**kwargs):
        captured_inputs.append(kwargs["input"])
        return responses.pop(0)

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=fake_create)),
    )

    async def fake_execute(tool_calls, *_args, **_kwargs):
        return ([{
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": "Error: invalid arguments for tool 'edit_file': Expecting property name",
        } for item in tool_calls], False)

    monkeypatch.setattr(main_module, "execute_tool_calls_async", fake_execute)

    state = main_module.LoopState(messages=[{"role": "user", "content": "edit the file"}])
    outcome = _run(main_module.agent_loop(state, session))

    assert outcome.stop_reason == "max_api_calls"
    assert outcome.api_calls == session.max_api_calls
    # Every replayed input had sanitized arguments (no raw malformed string).
    import json
    for api_input in captured_inputs:
        assert "{always broken" not in json.dumps(api_input)


def test_repl_discards_interrupted_turn_and_todo_mutation(
    load_module, monkeypatch, tmp_path,
) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "do work")

    async def interrupted_loop(state, session):
        state.messages.append({
            "type": "function_call",
            "call_id": "unfinished",
            "name": "write_file",
            "arguments": "{}",
        })
        session.todo.update(_todo_params([
            {"content": "unfinished", "status": "in_progress"},
        ]))
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "agent_loop", interrupted_loop)
    _run(main_module.repl())

    session_path = main_module.find_most_recent_session(
        tmp_path / ".sessions", tmp_path.resolve(),
    )
    assert session_path is not None
    reopened = main_module.SessionStore.open(
        session_path,
        main_module.Workspace(tmp_path),
        main_module.MODEL_ID,
        main_module.OPENROUTER_PROVIDER or "",
    )
    assert reopened.messages() == []
    assert reopened.last_todo_items() == []


def test_repl_no_session_does_not_claim_it_saved(
    load_module, monkeypatch, tmp_path, capsys,
) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")

    _run(main_module.repl(no_session=True))

    output = capsys.readouterr().out
    assert "persistence disabled" in output
    assert "[session] saved" not in output
    assert not (tmp_path / ".sessions").exists()


def test_repl_prints_complete_resume_sanitization_diagnostics(
    load_module, monkeypatch, tmp_path, capsys,
) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.chdir(tmp_path)
    workspace = main_module.Workspace(tmp_path)
    store = main_module.SessionStore.create(
        workspace,
        "resume-diagnostics",
        "old-model",
        "old-provider",
    )
    store.sync([
        {"role": "user", "content": "continue"},
        {"type": "reasoning", "id": "r1", "summary": "provider state"},
        {
            "type": "function_call",
            "id": "fc-paired",
            "call_id": "paired",
            "name": "bash",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "paired",
            "output": "done",
        },
        {
            "type": "function_call",
            "id": "fc-orphan",
            "call_id": "orphan-call",
            "name": "read_file",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "orphan-output",
            "output": "stray",
        },
    ])
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("{invalid-json}\n")

    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")
    _run(main_module.repl(resume=str(store.path)))

    output = capsys.readouterr().out
    assert "[session] resume sanitized:" in output
    assert "reasoning=1" in output
    assert "function_call ids=2" in output
    assert "orphan calls=1" in output
    assert "orphan outputs=1" in output
    assert "invalid JSONL lines=1" in output


def test_repl_warns_and_clears_invalid_restored_todo_state(
    load_module, monkeypatch, tmp_path, capsys,
) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.chdir(tmp_path)
    workspace = main_module.Workspace(tmp_path)
    store = main_module.SessionStore.create(
        workspace,
        "invalid-todo",
        main_module.MODEL_ID,
        main_module.OPENROUTER_PROVIDER or "",
    )
    store.sync([{"role": "user", "content": "continue"}])
    store.sync_todo([{
        "content": "broken item",
        "status": "not-a-valid-status",
    }])

    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")
    _run(main_module.repl(resume=str(store.path)))

    output = capsys.readouterr().out
    assert "[session] warning: could not restore todo state:" in output

    reopened = main_module.SessionStore.open(
        store.path,
        workspace,
        main_module.MODEL_ID,
        main_module.OPENROUTER_PROVIDER or "",
    )
    assert reopened.last_todo_items() == []


def test_repl_reports_session_lock_conflict_without_claiming_save(
    load_module, monkeypatch, tmp_path, capsys,
) -> None:
    main_module = load_module("main", "main.py")
    monkeypatch.chdir(tmp_path)
    workspace = main_module.Workspace(tmp_path)
    owner = main_module.SessionStore.create(
        workspace,
        "locked-repl",
        main_module.MODEL_ID,
        main_module.OPENROUTER_PROVIDER or "",
        acquire_lock=True,
    )
    owner.sync([{"role": "user", "content": "hello"}])
    try:
        _run(main_module.repl(resume=str(owner.path)))
    finally:
        owner.close()

    output = capsys.readouterr().out
    assert "Error: Session is already open by process" in output
    assert "[session] saved" not in output
