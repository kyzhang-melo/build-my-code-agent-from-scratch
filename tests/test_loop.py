from __future__ import annotations

import types


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

    should_continue = main_module.run_one_turn(state)

    assert should_continue is True
    assert state.turn_count == 2
    assert state.transition_reason == "function_call_output"
    assert captured["input"][0]["role"] == "user"
    assert captured["input"][0]["content"] == "task part 1\ntask part 2"
    assert any(m.get("type") == "function_call" and m.get("call_id") == "c1" for m in state.messages)
    assert any(m.get("role") == "assistant" and m.get("content") == "Running command..." for m in state.messages)
    assert state.messages[-1]["type"] == "function_call_output"
    assert state.messages[-1]["output"] == "ok"


def test_run_one_turn_contract_nudge_when_unresolved_todo(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update([{"content": "unfinished", "status": "in_progress"}])

    fake_response = types.SimpleNamespace(output=[], output_text="All done.")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **_: fake_response)),
    )

    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])
    should_continue = main_module.run_one_turn(state)

    assert should_continue is True
    assert state.transition_reason == "todo_contract_nudge"
    assert state.contract_nudges == 1
    assert state.messages[-1]["role"] == "user"
    assert "Before ending, either complete all todo items" in state.messages[-1]["content"]


def test_run_one_turn_contract_allows_finish_after_rewrite_ack(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update([{"content": "unfinished", "status": "pending"}])

    fake_response = types.SimpleNamespace(output=[], output_text="Done.")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **_: fake_response)),
    )

    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        todo_rewrite_ack_pending=True,
    )
    should_continue = main_module.run_one_turn(state)

    assert should_continue is False
    assert state.transition_reason is None
    assert state.todo_rewrite_ack_pending is False


def test_run_one_turn_contract_warns_after_max_nudges(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update([{"content": "unfinished", "status": "pending"}])

    fake_response = types.SimpleNamespace(output=[], output_text="Done.")
    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **_: fake_response)),
    )

    state = main_module.LoopState(
        messages=[{"role": "user", "content": "task"}],
        contract_nudges=main_module.TODO_CONTRACT_MAX_NUDGES,
    )
    should_continue = main_module.run_one_turn(state)

    assert should_continue is False
    assert state.messages[-1]["role"] == "assistant"
    assert "Ending with unresolved todo items" in state.messages[-1]["content"]


def test_run_one_turn_adds_todo_reminder_after_interval(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update([{"content": "one", "status": "pending"}])
    main_module.TODO.state.rounds_since_update = 2

    function_call = types.SimpleNamespace(
        type="function_call",
        call_id="c1",
        name="bash",
        arguments='{"command":"echo hi"}',
    )
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
    should_continue = main_module.run_one_turn(state)

    assert should_continue is True
    assert state.messages[-1]["role"] == "user"
    assert "Refresh your current plan before continuing." in state.messages[-1]["content"]


def test_handle_no_tool_calls_unresolved_todo_nudges(load_module) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update([{"content": "unfinished", "status": "in_progress"}])
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    should_continue = main_module.handle_no_tool_calls(state)

    assert should_continue is True
    assert state.transition_reason == "todo_contract_nudge"
    assert state.messages[-1]["role"] == "user"
    assert "Before ending, either complete all todo items" in state.messages[-1]["content"]


def test_handle_tool_calls_updates_transition_and_turn(load_module, monkeypatch) -> None:
    main_module = load_module("main", "main.py")
    main_module.TODO.update([])
    monkeypatch.setattr(
        main_module,
        "execute_tool_calls",
        lambda _output: ([{"type": "function_call_output", "call_id": "c1", "output": "ok"}], False),
    )
    response_output = [types.SimpleNamespace(type="function_call", call_id="c1", name="bash", arguments="{}")]
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])

    should_continue = main_module.handle_tool_calls(state, response_output)

    assert should_continue is True
    assert state.turn_count == 2
    assert state.transition_reason == "function_call_output"
    assert state.messages[-1]["type"] == "function_call_output"
