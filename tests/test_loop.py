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
        lambda _output: [{"type": "function_call_output", "call_id": "c1", "output": "ok"}],
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
