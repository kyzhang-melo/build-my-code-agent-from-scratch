from __future__ import annotations

import asyncio
import types

import trace as runtime_trace
from message_utils import extract_usage


def _response(usage):
    return types.SimpleNamespace(output=[], output_text="done", usage=usage)


def test_missing_usage_yields_all_none() -> None:
    fields = extract_usage(_response(None))

    assert set(fields) == {
        "cost",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "provider",
    }
    assert all(value is None for value in fields.values())


def test_absent_field_is_none_not_zero() -> None:
    # A provider that reports token counts but no cache counters must not be
    # recorded as a confirmed cache miss.
    usage = types.SimpleNamespace(input_tokens=100, output_tokens=10)
    fields = extract_usage(_response(usage))

    assert fields["input_tokens"] == 100
    assert fields["cached_tokens"] is None
    assert fields["cost"] is None


def test_reported_zero_is_preserved_as_zero() -> None:
    usage = {
        "input_tokens": 100,
        "cost": 0,
        "input_tokens_details": {"cached_tokens": 0},
    }
    fields = extract_usage(_response(usage))

    assert fields["cached_tokens"] == 0
    assert fields["cost"] == 0


def test_responses_shape() -> None:
    usage = {
        "cost": 0.0123,
        "input_tokens": 10339,
        "output_tokens": 60,
        "total_tokens": 10399,
        "input_tokens_details": {"cached_tokens": 10318, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 42},
    }
    fields = extract_usage(_response(usage))

    assert fields["cached_tokens"] == 10318
    assert fields["cache_write_tokens"] == 0
    assert fields["reasoning_tokens"] == 42
    assert fields["total_tokens"] == 10399


def test_chat_completions_shape_is_also_read() -> None:
    usage = {
        "prompt_tokens": 194,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 7},
    }
    fields = extract_usage(_response(usage))

    assert fields["input_tokens"] == 194
    assert fields["output_tokens"] == 2
    assert fields["cached_tokens"] == 100
    assert fields["reasoning_tokens"] == 7


def test_actual_provider_is_read_from_response() -> None:
    response = types.SimpleNamespace(usage={"input_tokens": 1}, provider="deepinfra")
    assert extract_usage(response)["provider"] == "deepinfra"


def test_turn_emits_usage_event_with_call_attribution(
    load_module, workspace, monkeypatch
) -> None:
    main_module = load_module("main", "main.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink, run_id="run-1")

    class AutoApprove:
        async def request(self, _request):
            import permissions

            return permissions.ApprovalResponse("allow")

    session = main_module.create_parent_session(
        workspace.root,
        approval_handler=AutoApprove(),
        trace_context=context,
        on_text=None,
    )
    response = _response({
        "cost": 0.5,
        "input_tokens": 1000,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 900},
    })

    async def create(**_kwargs):
        return response

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=create)),
    )
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])
    asyncio.run(main_module.run_one_turn(state, session))

    events = sink.by_type("llm.usage")
    assert len(events) == 1
    assert events[0]["kind"] == "turn"
    assert events[0]["api_call"] == 1
    assert events[0]["cost"] == 0.5
    assert events[0]["cached_tokens"] == 900


def test_subagent_usage_is_attributed_separately_from_parent(
    load_module, workspace, monkeypatch
) -> None:
    main_module = load_module("main", "main.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink, run_id="run-1")

    class AutoApprove:
        async def request(self, _request):
            import permissions

            return permissions.ApprovalResponse("allow")

    parent = main_module.create_parent_session(
        workspace.root,
        approval_handler=AutoApprove(),
        trace_context=context,
        on_text=None,
    )
    subagent = main_module.create_explore_session(
        parent.workspace,
        parent.permission_service,
        parent.trace_context,
        parent.session_id,
    )
    responses = iter([
        _response({"cost": 0.1, "input_tokens": 100, "output_tokens": 10}),
        _response({"cost": 0.2, "input_tokens": 200, "output_tokens": 20}),
    ])

    async def create(**_kwargs):
        return next(responses)

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=create)),
    )
    asyncio.run(main_module.run_one_turn(
        main_module.LoopState(messages=[{"role": "user", "content": "parent"}]),
        parent,
    ))
    asyncio.run(main_module.run_one_turn(
        main_module.LoopState(messages=[{"role": "user", "content": "explore"}]),
        subagent,
    ))

    events = sink.by_type("llm.usage")
    assert [(event["agent_id"], event["cost"]) for event in events] == [
        ("parent", 0.1),
        ("subagent:explore", 0.2),
    ]
    assert {event["run_id"] for event in events} == {"run-1"}


def test_estimated_tokens_never_reported_as_usage(
    load_module, workspace, monkeypatch
) -> None:
    # With no provider usage, the compaction trigger still gets a char-based
    # estimate, but the traced usage must stay null rather than inherit it.
    main_module = load_module("main", "main.py")
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)

    class AutoApprove:
        async def request(self, _request):
            import permissions

            return permissions.ApprovalResponse("allow")

    session = main_module.create_parent_session(
        workspace.root,
        approval_handler=AutoApprove(),
        trace_context=context,
        on_text=None,
    )

    async def create(**_kwargs):
        return _response(None)

    monkeypatch.setattr(
        main_module,
        "client",
        types.SimpleNamespace(responses=types.SimpleNamespace(create=create)),
    )
    state = main_module.LoopState(messages=[{"role": "user", "content": "task"}])
    asyncio.run(main_module.run_one_turn(state, session))

    assert state.last_input_tokens > 0, "compaction trigger still needs an estimate"
    assert sink.by_type("llm.usage")[0]["input_tokens"] is None


def test_compaction_side_call_is_traced_separately(load_module, tmp_path) -> None:
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink)

    class FakeClient:
        class responses:
            @staticmethod
            async def create(**_kwargs):
                return types.SimpleNamespace(
                    output_text="summary",
                    usage={"input_tokens": 5000, "output_tokens": 100, "cost": 0.02},
                )

    state = types.SimpleNamespace(
        messages=[
            {"role": "user", "content": "old request " + "x" * 4000},
            {"role": "assistant", "content": "old answer"},
        ],
        last_input_tokens=5000,
    )
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")

    result = asyncio.run(cc.compact_history_async(
        state,
        todo=todo,
        source="auto",
        client=FakeClient(),
        model="m",
        trace_context=context,
    ))

    assert result is not None
    events = sink.by_type("llm.usage")
    assert len(events) == 1
    assert events[0]["kind"] == "compaction"
    assert events[0]["input_tokens"] == 5000
    assert events[0]["cost"] == 0.02


def test_compaction_without_trace_context_still_works(load_module, tmp_path) -> None:
    # Existing callers pass no trace context; emit_trace(None, ...) is a no-op.
    cc = load_module("context_compact", "context_compact.py")
    cc.TRANSCRIPT_DIR = tmp_path / "transcripts"

    class FakeClient:
        class responses:
            @staticmethod
            async def create(**_kwargs):
                return types.SimpleNamespace(output_text="summary", usage=None)

    state = types.SimpleNamespace(
        messages=[
            {"role": "user", "content": "old request " + "x" * 4000},
            {"role": "assistant", "content": "old answer"},
        ],
        last_input_tokens=5000,
    )
    todo = types.SimpleNamespace(has_active_plan=lambda: False, render=lambda: "")

    assert asyncio.run(cc.compact_history_async(
        state, todo=todo, source="auto", client=FakeClient(), model="m",
    )) is not None
