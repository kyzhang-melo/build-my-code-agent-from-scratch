from __future__ import annotations

import asyncio
import types

import pytest


def _fc(name: str, call_id: str, arguments: str):
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


class _AsyncFakeClient:
    def __init__(self, output_text: str = "SUMMARY: editing foo.py:10; next run tests."):
        captured: dict = {}

        class _Responses:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(output_text=output_text)

        self.responses = _Responses()
        self.captured = captured


def _run(awaitable):
    return asyncio.run(awaitable)


# --------------------------------------------------------------------------- #
# Tier 1: persist_large_output (truncate_middle at the chokepoint)
# --------------------------------------------------------------------------- #


def test_truncate_middle_keeps_both_ends(load_module) -> None:
    tools = load_module("tools", "tools.py")
    assert tools.truncate_middle("small") == "small"  # no-op under budget

    big = "\n".join(f"L{i:05d}" + "x" * 100 for i in range(1000))
    out = tools.truncate_middle(big)
    assert out.startswith("Total output lines: 1000")
    assert "L00000" in out, "head must survive"
    assert "L00999" in out, "tail must survive (the old [:50000] bug dropped it)"
    assert "elided from the middle" in out
    assert len(out) < len(big)


def test_truncate_middle_per_line_cap(load_module) -> None:
    tools = load_module("tools", "tools.py")
    out = tools.truncate_middle("a\n" + "z" * 5000 + "\nb")
    assert "a" in out and "b" in out
    assert "[...line truncated]" in out
    assert max(len(line) for line in out.split("\n")) <= 2000 + 40


def test_run_tool_call_truncates_large_output(load_module) -> None:
    tools = load_module("tools", "tools.py")
    big = "\n".join(
        "FIRSTLINE" if i == 0 else ("LASTLINE" if i == 999 else "y" * 200)
        for i in range(1000)
    )

    class _Model:
        @staticmethod
        def model_validate(data):
            return data

    spec = types.SimpleNamespace(
        sanitize_args=lambda args: args,
        params_model=_Model,
        execute=tools.async_tool(lambda params: big),
    )
    result, used_todo = tools.run_tool_call(
        _fc("fake", "c1", "{}"),
        {"fake": spec},
        tools.TodoManager(),
    )
    out = result["output"]
    assert used_todo is False
    assert "FIRSTLINE" in out and "LASTLINE" in out, "both ends preserved at chokepoint"
    assert "elided from the middle" in out
    assert len(out) < len(big)


def test_run_tool_call_leaves_todo_verbatim(load_module) -> None:
    tools = load_module("tools", "tools.py")
    huge = "x" * 60000

    class _Model:
        @staticmethod
        def model_validate(data):
            return data

    spec = types.SimpleNamespace(
        sanitize_args=lambda args: args,
        params_model=_Model,
        execute=tools.async_tool(lambda params: huge),
    )
    result, used_todo = tools.run_tool_call(
        _fc("todo", "c1", "{}"),
        {"todo": spec},
        tools.TodoManager(),
    )
    assert used_todo is True
    assert result["output"] == huge, "todo output (control-plane state) is not truncated"


# --------------------------------------------------------------------------- #
# Tier 2: compact_history building blocks
# --------------------------------------------------------------------------- #


def test_render_prompt_fills_focus_slot(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    no_focus = cc.render_prompt(None)
    with_focus = cc.render_prompt("keep the auth refactor")
    with_previous = cc.render_prompt(None, "old summary")
    assert "{{ focus }}" not in no_focus and "{{ focus }}" not in with_focus
    assert "{{ previous_summary }}" not in with_previous
    assert "keep the auth refactor" in with_focus
    assert "<previous-summary>\nold summary\n</previous-summary>" in with_previous
    assert "Focus for this compaction" in with_focus
    assert "Focus for this compaction" not in no_focus


def test_extract_previous_summary_unwraps_leading_summary(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    summary = cc.build_summary_message("old summary")
    previous, start_index = cc.extract_previous_summary([summary, {"role": "user", "content": "new"}])

    assert previous == "old summary"
    assert start_index == 1


def test_build_compacted_history_keeps_summary_then_tail(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    tail = [
        {"role": "user", "content": "recent request"},
        {"role": "assistant", "content": "recent answer"},
    ]
    history = cc.build_compacted_history("THE SUMMARY", tail)

    assert history[0]["content"].startswith(cc.SUMMARY_PREFIX)
    assert "THE SUMMARY" in history[0]["content"]
    assert history[1:] == tail


def test_find_cut_index_keeps_complete_recent_turn(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    messages = [
        {"role": "user", "content": "old request " + "x" * 200},
        {"role": "assistant", "content": "old answer " + "x" * 200},
        {"role": "user", "content": "recent request"},
        {"role": "assistant", "content": [{
            "type": "tool_call", "name": "bash", "arguments": "{}",
            "pairing": {"call_id": "c1"},
        }], "runtime": {"model_id": "m", "provider": "p", "protocol": "responses"}},
        {"role": "tool", "call_id": "c1", "content": "recent output", "is_error": False},
        {"role": "assistant", "content": [{
            "type": "text", "text": "recent answer", "source": "test",
        }], "runtime": {"model_id": "m", "provider": "p", "protocol": "responses"}},
    ]

    cut = cc.find_cut_index(messages, keep_recent_tokens=20)
    assert cut == 2
    assert messages[cut]["role"] == "user"


def test_find_cut_index_summarizes_short_history(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    messages = [
        {"role": "user", "content": "small"},
        {"role": "assistant", "content": "done"},
    ]

    assert cc.find_cut_index(messages, keep_recent_tokens=1000) == len(messages)


def test_summarize_async_omits_tools_and_raises_on_empty(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    client = _AsyncFakeClient(output_text="  the summary  ")
    out = _run(cc.summarize_async([{"role": "user", "content": "hi"}], "focus", client=client, model="m"))
    assert out == "the summary"
    assert "tools" not in client.captured, "side-call must not expose tools"
    assert client.captured["max_output_tokens"] == cc.SUMMARY_MAX_OUTPUT_TOKENS

    empty_client = _AsyncFakeClient(output_text="   ")
    with pytest.raises(ValueError):
        _run(cc.summarize_async([{"role": "user", "content": "hi"}], None, client=empty_client, model="m"))


def test_summarize_async_uses_previous_summary_fold_forward(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    client = _AsyncFakeClient(output_text="updated")

    _run(cc.summarize_async(
        [{"role": "user", "content": "new work"}],
        None,
        client=client,
        model="m",
        previous_summary="old summary",
    ))

    prompt = client.captured["input"][-1]["content"]
    assert "<previous-summary>\nold summary\n</previous-summary>" in prompt
    assert "update it" in prompt


def test_provider_extra_body_threads_to_side_call(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path
    routing = {"provider": {"only": ["moonshotai"], "allow_fallbacks": False}}

    # summarize_async forwards extra_body verbatim to responses.create
    client = _AsyncFakeClient()
    _run(cc.summarize_async([{"role": "user", "content": "hi"}], None, client=client, model="m",
                 extra_body=routing))
    assert client.captured["extra_body"] == routing

    # compact_history_async threads it down to the same side-call
    client2 = _AsyncFakeClient()
    state = types.SimpleNamespace(messages=list(_droppable_history(cc)), last_input_tokens=0)
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")
    _run(cc.compact_history_async(
        state, todo=todo, source="auto", client=client2, model="m", extra_body=routing,
    ))
    assert client2.captured["extra_body"] == routing

    # default (no pinning) leaves routing unset -> SDK no-op
    client3 = _AsyncFakeClient()
    _run(cc.summarize_async([{"role": "user", "content": "hi"}], None, client=client3, model="m"))
    assert client3.captured["extra_body"] is None


def test_provider_extra_body_disabled_by_default(load_module, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_PROVIDER", raising=False)
    main = load_module("main", "main.py")
    # OPENROUTER_PROVIDER is unset in the test env -> no routing override.
    assert main.PROVIDER_EXTRA_BODY is None


def test_provider_extra_body_supports_minimax_fp8(load_module, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_PROVIDER", "minimax/fp8")
    main = load_module("main", "main.py")

    assert main.PROVIDER_EXTRA_BODY == {
        "provider": {"only": ["minimax/fp8"], "allow_fallbacks": False}
    }


def test_reinject_todo_gated_on_active_plan(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    inactive = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "X")
    active = types.SimpleNamespace(has_active_plan=lambda: True, render=lambda: "[ ] do thing")
    assert cc.reinject_todo("S", inactive) == "S"
    out = cc.reinject_todo("S", active)
    assert "S" in out and "[ ] do thing" in out and "Current TODO state" in out


# --------------------------------------------------------------------------- #
# Tier 2: compact_history orchestration
# --------------------------------------------------------------------------- #


def _droppable_history(cc):
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "ok"},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "tool output"},
    ]


def test_compact_history_async_happy_path(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")

    messages = _droppable_history(cc)
    state = types.SimpleNamespace(messages=list(messages), last_input_tokens=5000)
    original_list_id = id(state.messages)
    client = _AsyncFakeClient()

    result = _run(cc.compact_history_async(
        state,
        todo=todo,
        source="manual",
        focus="foo",
        client=client,
        model="m",
    ))

    assert result is not None and result.source == "manual"
    assert id(state.messages) == original_list_id, "history mutated in place (alias-safe)"
    assert len(state.messages) == 1, "short compacted history becomes summary-only"
    assert state.messages[0]["content"].startswith(cc.SUMMARY_PREFIX)
    assert "do the task" not in state.messages[0]["content"], "old request is not replayed verbatim"
    assert state.last_input_tokens == result.tokens_after
    assert (tmp_path / "transcripts").exists()
    assert result.transcript_path.endswith(".jsonl")
    assert client.captured["max_output_tokens"] == cc.SUMMARY_MAX_OUTPUT_TOKENS


def test_compact_history_async_preserves_recent_tail(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")
    cc.KEEP_RECENT_TOKENS = 10

    old_request = "please read old.py and summarize " + "x" * 400
    recent_request = "answer this recent question"
    state = types.SimpleNamespace(messages=[
        {"role": "user", "content": old_request},
        {"role": "assistant", "content": "old answer " + "x" * 400},
        {"role": "user", "content": recent_request},
        {"type": "function_call", "call_id": "c2", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "recent output"},
        {"role": "assistant", "content": "recent answer"},
    ], last_input_tokens=5000)

    result = _run(cc.compact_history_async(
        state, todo=todo, source="manual", client=_AsyncFakeClient(), model="m",
    ))

    assert result is not None
    assert state.messages[0]["content"].startswith(cc.SUMMARY_PREFIX)
    assert all(old_request not in str(msg.get("content", "")) for msg in state.messages[1:])
    assert state.messages[1]["content"] == recent_request
    assert state.messages[3]["type"] == "function_call_output"


def test_compact_history_async_folds_forward_previous_summary(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")
    client = _AsyncFakeClient(output_text="updated summary")
    state = types.SimpleNamespace(messages=[
        cc.build_summary_message("old summary"),
        {"role": "user", "content": "new completed request"},
        {"role": "assistant", "content": "new completed answer"},
    ], last_input_tokens=5000)

    result = _run(cc.compact_history_async(
        state, todo=todo, source="manual", client=client, model="m",
    ))

    assert result is not None
    assert len(state.messages) == 1
    assert state.messages[0]["content"].count(cc.SUMMARY_PREFIX) == 1
    assert "updated summary" in state.messages[0]["content"]
    assert "<previous-summary>\nold summary\n</previous-summary>" in client.captured["input"][-1]["content"]


def test_compact_history_async_aborts_on_summary_failure(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"

    class _AsyncBoom:
        class responses:
            @staticmethod
            async def create(**kwargs):
                raise RuntimeError("api down")

    messages = _droppable_history(cc)
    state = types.SimpleNamespace(messages=list(messages), last_input_tokens=5000)
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")
    result = _run(cc.compact_history_async(
        state, todo=todo, source="auto", client=_AsyncBoom(), model="m",
    ))
    assert result is None, "failure returns None"
    assert state.messages == messages, "history is left untouched on failure"


def test_compact_history_async_noop_when_nothing_to_compact(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    state = types.SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], last_input_tokens=10
    )
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")
    assert _run(cc.compact_history_async(
        state, todo=todo, source="manual", client=_AsyncFakeClient(), model="m",
    )) is None


# --------------------------------------------------------------------------- #
# main.py: budget, dispatcher, escalation
# --------------------------------------------------------------------------- #


def test_input_budget_and_threshold(load_module) -> None:
    main = load_module("main", "main.py")
    main.MODEL_LIMITS_OVERRIDE = ""
    main.MODEL_ID = "moonshotai/kimi-k2.5"
    assert main.context_window() == 262144
    assert main.input_budget() == 262144 - main.AUTO_MAX_OUTPUT_TOKEN_RESERVATION - main.RESERVED_OVERHEAD_TOKENS
    assert main.input_budget(16000) == 262144 - 16000 - main.RESERVED_OVERHEAD_TOKENS
    assert main.input_budget(None) == 262144 - main.AUTO_MAX_OUTPUT_TOKEN_RESERVATION - main.RESERVED_OVERHEAD_TOKENS

    main.MODEL_ID = "nope/unknown"
    assert main.context_window() == main.DEFAULT_CONTEXT_WINDOW
    assert main.input_budget() == 32000 - 16000 - main.RESERVED_OVERHEAD_TOKENS

    # Provider-default output mode reserves half of a small context window.
    assert main.COMPACT_TRIGGER_RATIO * main.input_budget() + main.output_token_reservation(None) < main.context_window()

    state = main.LoopState(messages=[])
    state.last_input_tokens = 10200  # 0.85 * 12000
    assert main.should_auto_compact(state) is True
    state.last_input_tokens = 10199
    assert main.should_auto_compact(state) is False

    main.MODEL_ID = "tencent/hy3"
    assert main.model_limits() == main.ModelLimits(262_144, 192_000, 128_000)
    assert main.input_budget() == 192_000 - main.RESERVED_OVERHEAD_TOKENS
    with pytest.raises(ValueError, match="maximum of 128000"):
        main.validate_generation_config(None, 128_001)


def test_session_rejects_output_budget_that_consumes_context(
    load_module, tmp_path
) -> None:
    main = load_module("main", "main.py")
    main.MODEL_LIMITS_OVERRIDE = (
        '{"context_window_tokens":20000,"max_input_tokens":20000,'
        '"max_output_tokens":null}'
    )
    try:
        with pytest.raises(ValueError, match="reserved overhead"):
            main.create_parent_session(
                tmp_path,
                approval_handler=None,
                max_output_tokens=16000,
            )
    finally:
        main.MODEL_LIMITS_OVERRIDE = ""


def test_context_window_normalizes_routing_and_quant_suffixes(load_module) -> None:
    main = load_module("main", "main.py")
    main.MODEL_LIMITS_OVERRIDE = ""

    assert main.normalize_model_id("moonshotai/kimi-k2.5:exacto") == "kimi-k2.5"
    assert main.normalize_model_id("moonshotai/kimi-k2.5") == "kimi-k2.5"
    assert main.normalize_model_id("kimi-k2.5") == "kimi-k2.5"

    # The routing/quant suffix must NOT defeat the lookup (the original bug:
    # an exact-key dict missed "...:exacto" and fell back to 32000).
    for model in (
        "moonshotai/kimi-k2.5:exacto",
        "moonshotai/kimi-k2.5:nitro",
        "kimi-k2-0905",
    ):
        main.MODEL_ID = model
        assert main.context_window() == 262144, model

    main.MODEL_ID = "deepseek/deepseek-v4-pro:floor"
    assert main.context_window() == 1_000_000

    main.MODEL_ID = "minimax/minimax-m3:exacto"
    assert main.context_window() == 524_288

    main.MODEL_ID = "z-ai/glm-5:exacto"
    assert main.context_window() == 202_800

    main.MODEL_ID = "nope/unknown:exacto"
    assert main.context_window() == main.DEFAULT_CONTEXT_WINDOW


def test_model_limits_override_wins_and_is_validated(load_module) -> None:
    main = load_module("main", "main.py")
    main.MODEL_ID = "moonshotai/kimi-k2.5"  # would resolve to 262144
    main.MODEL_LIMITS_OVERRIDE = (
        '{"context_window_tokens":20000,"max_input_tokens":18000,'
        '"max_output_tokens":4000}'
    )
    try:
        assert main.context_window() == 20000
        assert main.model_limits().max_input_tokens == 18000
    finally:
        main.MODEL_LIMITS_OVERRIDE = ""

    main.MODEL_LIMITS_OVERRIDE = '{"context_window_tokens":20000}'
    with pytest.raises(ValueError, match="exactly"):
        main.model_limits()


def test_handle_command_dispatch(load_module, tmp_path) -> None:
    main = load_module("main", "main.py")
    session = main.create_parent_session(tmp_path, approval_handler=None, on_text=None)
    calls: list = []

    async def fake_compact(
        state, *, todo, source, focus=None, client, model, extra_body=None,
        trace_context=None,
    ):
        assert todo is session.todo
        calls.append((source, focus))
        state.messages[:] = [{"role": "user", "content": main.SUMMARY_PREFIX + "\ns"}]
        return types.SimpleNamespace(tokens_before=1, tokens_after=0, transcript_path="x.jsonl")

    main.compact_history_async = fake_compact

    assert _run(main.handle_command("build a feature", [], session)) is False
    assert _run(main.handle_command("/help", [], session)) is True
    assert _run(main.handle_command("/bogus", [], session)) is True

    history = [{"type": "function_call_output", "call_id": "c", "output": "o"}]
    assert _run(main.handle_command("/compact keep auth", history, session)) is True
    assert calls == [("manual", "keep auth")]
    assert history[-1]["content"].startswith(main.SUMMARY_PREFIX)

    calls.clear()
    _run(main.handle_command(
        "/compact",
        [{"type": "function_call_output", "call_id": "c", "output": "o"}],
        session,
    ))
    assert calls[0] == ("manual", None), "no focus -> None"


def test_drop_oldest_user_message_protects_summary(load_module) -> None:
    main = load_module("main", "main.py")
    state = main.LoopState(messages=[
        {"role": "user", "content": main.SUMMARY_PREFIX + "\nsummary"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"type": "function_call_output", "call_id": "c", "output": "first output"},
        {"role": "user", "content": "second"},
    ])
    assert main._drop_oldest_user_message(state) is True
    assert [m["content"] for m in state.messages if "content" in m] == [
        main.SUMMARY_PREFIX + "\nsummary",
        "second",
    ]
    assert main._drop_oldest_user_message(state) is True
    assert main._drop_oldest_user_message(state) is False, "summary is never dropped"
    assert state.messages[0]["content"].startswith(main.SUMMARY_PREFIX)


def test_auto_compaction_fires_and_kill_switch(load_module, tmp_path) -> None:
    main = load_module("main", "main.py")
    session = main.create_parent_session(tmp_path, approval_handler=None, on_text=None)
    main.MODEL_ID = "nope/unknown"  # budget 20000, threshold 17000
    compact_calls: list = []

    async def fake_compact(
        state, *, todo, source, focus=None, client, model, extra_body=None,
        trace_context=None,
    ):
        assert todo is session.todo
        compact_calls.append(source)
        state.messages[:] = [{"role": "user", "content": main.SUMMARY_PREFIX + "\ns"}]
        state.last_input_tokens = 50
        return types.SimpleNamespace(tokens_before=18000, tokens_after=50, transcript_path=".t/x")

    main.compact_history_async = fake_compact

    turn_count = {"n": 0}

    async def fake_turn(state, received_session):
        assert received_session is session
        index = turn_count["n"]
        turn_count["n"] += 1
        if index == 0:
            state.last_input_tokens = 18000  # over threshold -> should compact
            return None
        return main.StepOutcome(stop_reason="completed", final_text="done")

    main.run_one_turn = fake_turn

    main.AUTO_COMPACT_ENABLED = True
    state = main.LoopState(messages=[{"type": "function_call_output", "call_id": "c", "output": "o"}])
    outcome = _run(main.agent_loop(state, session))
    assert compact_calls == ["auto"]
    assert outcome.final_text == "done"

    # Kill switch: disabled -> never fires.
    compact_calls.clear()
    turn_count["n"] = 0
    main.AUTO_COMPACT_ENABLED = False
    _run(main.agent_loop(
        main.LoopState(messages=[{"type": "function_call_output", "call_id": "c", "output": "o"}]),
        session,
    ))
    assert compact_calls == []


def test_auto_compaction_preflights_tool_output_and_steering(load_module, tmp_path) -> None:
    main = load_module("main", "main.py")
    main.MODEL_LIMITS_OVERRIDE = (
        '{"context_window_tokens":10000,"max_input_tokens":10000,'
        '"max_output_tokens":100}'
    )
    steering_policy = types.SimpleNamespace(
        name="test",
        after_turn=lambda _api_call: types.SimpleNamespace(
            content="s" * 12_000, reason="test"
        ),
    )
    session = main.create_parent_session(
        tmp_path, approval_handler=None, on_text=None,
        steering_policy=steering_policy,
    )
    compact_calls: list[str] = []

    async def fake_compact(state, **_kwargs):
        compact_calls.append("auto")
        state.messages[:] = [{"role": "user", "content": main.SUMMARY_PREFIX + "\ns"}]
        state.last_input_tokens = 0
        return types.SimpleNamespace(tokens_before=500, tokens_after=10, transcript_path=".t/x")

    turns = iter([None, main.StepOutcome(stop_reason="completed", final_text="done")])

    async def fake_turn(state, _session):
        state.last_input_tokens = 1
        return next(turns)

    main.compact_history_async = fake_compact
    main.run_one_turn = fake_turn
    main.AUTO_COMPACT_ENABLED = True
    try:
        outcome = _run(main.agent_loop(
            main.LoopState(messages=[
                {"role": "user", "content": "start"},
                {"type": "function_call_output", "call_id": "c", "output": "o" * 12_000},
            ]),
            session,
        ))
        assert outcome.final_text == "done"
        assert compact_calls == ["auto"]
    finally:
        main.MODEL_LIMITS_OVERRIDE = ""
