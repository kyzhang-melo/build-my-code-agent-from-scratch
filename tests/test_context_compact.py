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


class _FakeClient:
    """Minimal client whose responses.create returns a fixed summary and records
    the kwargs it was called with."""

    def __init__(self, output_text: str = "SUMMARY: editing foo.py:10; next run tests."):
        captured: dict = {}

        class _Responses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(output_text=output_text)

        self.responses = _Responses()
        self.captured = captured


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
    result, used_todo = tools.run_tool_call(_fc("fake", "c1", "{}"), {"fake": spec})
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
    result, used_todo = tools.run_tool_call(_fc("todo", "c1", "{}"), {"todo": spec})
    assert used_todo is True
    assert result["output"] == huge, "todo output (control-plane state) is not truncated"


# --------------------------------------------------------------------------- #
# Tier 2: compact_history building blocks
# --------------------------------------------------------------------------- #


def test_render_prompt_fills_focus_slot(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    no_focus = cc.render_prompt(None)
    with_focus = cc.render_prompt("keep the auth refactor")
    assert "{{ focus }}" not in no_focus and "{{ focus }}" not in with_focus
    assert "keep the auth refactor" in with_focus
    assert "Focus for this compaction" in with_focus
    assert "Focus for this compaction" not in no_focus


def test_collect_user_messages_skips_prior_summaries(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    messages = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "working"},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "out"},
        {"role": "user", "content": cc.SUMMARY_PREFIX + "\nold summary"},
        {"role": "user", "content": "second request"},
    ]
    assert cc.collect_user_messages(messages) == ["first request", "second request"]


def test_build_compacted_history_is_start_fresh(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    history = cc.build_compacted_history(["first", "second"], "THE SUMMARY")
    assert all(msg["role"] == "user" for msg in history), "no assistant/tool items survive"
    assert [m["content"] for m in history[:2]] == ["first", "second"], "user order preserved"
    assert history[-1]["content"].startswith(cc.SUMMARY_PREFIX)
    assert "THE SUMMARY" in history[-1]["content"]


def test_cap_user_messages_newest_first(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    budget = cc.USER_MESSAGE_MAX_CHARS
    big = ["a" * (budget // 2 + 10), "b" * (budget // 2 + 10), "c" * (budget // 2 + 10)]
    capped = cc._cap_user_messages(big)
    assert capped[-1].startswith("c"), "newest message is kept"
    assert "a" * (budget // 2 + 10) not in capped, "oldest dropped when over budget"


def test_summarize_omits_tools_and_raises_on_empty(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    client = _FakeClient(output_text="  the summary  ")
    out = cc.summarize([{"role": "user", "content": "hi"}], "focus", client=client, model="m")
    assert out == "the summary"
    assert "tools" not in client.captured, "side-call must not expose tools"
    assert client.captured["max_output_tokens"] == cc.SUMMARY_MAX_OUTPUT_TOKENS

    empty_client = _FakeClient(output_text="   ")
    with pytest.raises(ValueError):
        cc.summarize([{"role": "user", "content": "hi"}], None, client=empty_client, model="m")


def test_provider_extra_body_threads_to_side_call(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path
    routing = {"provider": {"only": ["moonshotai"], "allow_fallbacks": False}}

    # summarize forwards extra_body verbatim to responses.create
    client = _FakeClient()
    cc.summarize([{"role": "user", "content": "hi"}], None, client=client, model="m",
                 extra_body=routing)
    assert client.captured["extra_body"] == routing

    # compact_history threads it down to the same side-call
    client2 = _FakeClient()
    state = types.SimpleNamespace(messages=list(_droppable_history(cc)), last_input_tokens=0)
    cc.compact_history(state, source="auto", client=client2, model="m", extra_body=routing)
    assert client2.captured["extra_body"] == routing

    # default (no pinning) leaves routing unset -> SDK no-op
    client3 = _FakeClient()
    cc.summarize([{"role": "user", "content": "hi"}], None, client=client3, model="m")
    assert client3.captured["extra_body"] is None


def test_provider_extra_body_disabled_by_default(load_module) -> None:
    main = load_module("main", "main.py")
    # OPENROUTER_PROVIDER is unset in the test env -> no routing override.
    assert main.PROVIDER_EXTRA_BODY is None


def test_reinject_todo_gated_on_active_plan(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TODO = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "X")
    assert cc.reinject_todo("S") == "S"
    cc.TODO = types.SimpleNamespace(has_active_plan=lambda: True, render=lambda: "[ ] do thing")
    out = cc.reinject_todo("S")
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


def test_compact_history_happy_path(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"
    cc.TODO = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")

    messages = _droppable_history(cc)
    state = types.SimpleNamespace(messages=list(messages), last_input_tokens=5000)
    original_list_id = id(state.messages)

    result = cc.compact_history(
        state, source="manual", focus="foo", client=_FakeClient(), model="m"
    )
    assert result is not None and result.source == "manual"
    assert id(state.messages) == original_list_id, "history mutated in place (alias-safe)"
    assert all(m.get("role") == "user" for m in state.messages), "assistant/tool dropped"
    assert state.messages[-1]["content"].startswith(cc.SUMMARY_PREFIX)
    assert state.last_input_tokens == result.tokens_after
    assert (tmp_path / "transcripts").exists()
    assert result.transcript_path.endswith(".jsonl")


def test_compact_history_async_happy_path(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"
    cc.TODO = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")

    state = types.SimpleNamespace(messages=list(_droppable_history(cc)), last_input_tokens=5000)
    client = _AsyncFakeClient()

    result = _run(cc.compact_history_async(
        state,
        source="auto",
        focus="async",
        client=client,
        model="m",
    ))

    assert result is not None and result.source == "auto"
    assert state.messages[-1]["content"].startswith(cc.SUMMARY_PREFIX)
    assert state.last_input_tokens == result.tokens_after
    assert client.captured["max_output_tokens"] == cc.SUMMARY_MAX_OUTPUT_TOKENS


def test_compact_history_aborts_on_summary_failure(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"

    class _Boom:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("api down")

    messages = _droppable_history(cc)
    state = types.SimpleNamespace(messages=list(messages), last_input_tokens=5000)
    result = cc.compact_history(state, source="auto", client=_Boom(), model="m")
    assert result is None, "failure returns None"
    assert state.messages == messages, "history is left untouched on failure"


def test_compact_history_noop_when_nothing_to_compact(load_module) -> None:
    cc = load_module("context_compact", "context_compact.py")
    state = types.SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], last_input_tokens=10
    )
    assert cc.compact_history(state, source="manual", client=_FakeClient(), model="m") is None


# --------------------------------------------------------------------------- #
# main.py: budget, dispatcher, escalation
# --------------------------------------------------------------------------- #


def test_input_budget_and_threshold(load_module) -> None:
    main = load_module("main", "main.py")
    main.CONTEXT_WINDOW_OVERRIDE = 0
    main.MODEL_ID = "moonshotai/kimi-k2.5"
    assert main.context_window() == 262144
    assert main.input_budget() == 262144 - main.RESERVED_OUTPUT_TOKENS - main.RESERVED_OVERHEAD_TOKENS

    main.MODEL_ID = "nope/unknown"
    assert main.context_window() == main.DEFAULT_CONTEXT_WINDOW
    assert main.input_budget() == 32000 - 12000  # = 20000

    # The 8k response reservation must keep the trigger safe on a small window.
    assert main.COMPACT_TRIGGER_RATIO * main.input_budget() + main.RESERVED_OUTPUT_TOKENS < main.context_window()

    state = main.LoopState(messages=[])
    state.last_input_tokens = 17000  # 0.85 * 20000
    assert main.should_auto_compact(state) is True
    state.last_input_tokens = 16999
    assert main.should_auto_compact(state) is False


def test_context_window_normalizes_routing_and_quant_suffixes(load_module) -> None:
    main = load_module("main", "main.py")
    main.CONTEXT_WINDOW_OVERRIDE = 0

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
    assert main.context_window() == 131072

    main.MODEL_ID = "nope/unknown:exacto"
    assert main.context_window() == main.DEFAULT_CONTEXT_WINDOW


def test_context_window_override_wins(load_module) -> None:
    main = load_module("main", "main.py")
    main.MODEL_ID = "moonshotai/kimi-k2.5"  # would resolve to 262144
    main.CONTEXT_WINDOW_OVERRIDE = 20000
    try:
        assert main.context_window() == 20000
    finally:
        main.CONTEXT_WINDOW_OVERRIDE = 0


def test_handle_command_dispatch(load_module) -> None:
    main = load_module("main", "main.py")
    calls: list = []

    async def fake_compact(state, *, source, focus=None, client, model, extra_body=None):
        calls.append((source, focus))
        state.messages[:] = [{"role": "user", "content": main.SUMMARY_PREFIX + "\ns"}]
        return types.SimpleNamespace(tokens_before=1, tokens_after=0, transcript_path="x.jsonl")

    main.compact_history_async = fake_compact

    assert _run(main.handle_command("build a feature", [])) is False, "ordinary input is forwarded"
    assert _run(main.handle_command("/help", [])) is True
    assert _run(main.handle_command("/bogus", [])) is True, "unknown command handled, not forwarded"

    history = [{"type": "function_call_output", "call_id": "c", "output": "o"}]
    assert _run(main.handle_command("/compact keep auth", history)) is True
    assert calls == [("manual", "keep auth")]
    assert history[-1]["content"].startswith(main.SUMMARY_PREFIX)

    calls.clear()
    _run(main.handle_command("/compact", [{"type": "function_call_output", "call_id": "c", "output": "o"}]))
    assert calls[0] == ("manual", None), "no focus -> None"


def test_drop_oldest_user_message_protects_summary(load_module) -> None:
    main = load_module("main", "main.py")
    state = main.LoopState(messages=[
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": main.SUMMARY_PREFIX + "\nsummary"},
    ])
    assert main._drop_oldest_user_message(state) is True
    assert [m["content"] for m in state.messages] == ["second", main.SUMMARY_PREFIX + "\nsummary"]
    assert main._drop_oldest_user_message(state) is True
    assert main._drop_oldest_user_message(state) is False, "summary is never dropped"
    assert state.messages[0]["content"].startswith(main.SUMMARY_PREFIX)


def test_auto_compaction_fires_and_kill_switch(load_module) -> None:
    main = load_module("main", "main.py")
    main.MODEL_ID = "nope/unknown"  # budget 20000, threshold 17000
    compact_calls: list = []

    async def fake_compact(state, *, source, focus=None, client, model, extra_body=None):
        compact_calls.append(source)
        state.messages[:] = [{"role": "user", "content": main.SUMMARY_PREFIX + "\ns"}]
        state.last_input_tokens = 50
        return types.SimpleNamespace(tokens_before=18000, tokens_after=50, transcript_path=".t/x")

    main.compact_history_async = fake_compact

    turn_count = {"n": 0}

    async def fake_turn(state, config=None):
        index = turn_count["n"]
        turn_count["n"] += 1
        if index == 0:
            state.last_input_tokens = 18000  # over threshold -> should compact
            return None
        return main.StepOutcome(stop_reason="completed", final_text="done")

    main.run_one_turn = fake_turn

    main.AUTO_COMPACT_ENABLED = True
    state = main.LoopState(messages=[{"type": "function_call_output", "call_id": "c", "output": "o"}])
    outcome = _run(main.agent_loop(state))
    assert compact_calls == ["auto"]
    assert outcome.final_text == "done"

    # Kill switch: disabled -> never fires.
    compact_calls.clear()
    turn_count["n"] = 0
    main.AUTO_COMPACT_ENABLED = False
    _run(main.agent_loop(main.LoopState(messages=[{"type": "function_call_output", "call_id": "c", "output": "o"}])))
    assert compact_calls == []
