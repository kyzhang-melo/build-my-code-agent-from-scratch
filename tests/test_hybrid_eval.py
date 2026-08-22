from __future__ import annotations

import asyncio
import importlib
import sys
import types


def _run(awaitable):
    return asyncio.run(awaitable)


def _response(*items):
    return types.SimpleNamespace(output=list(items), output_text="", usage=None)


def _function_call(*, arguments: str = '{"path":"target.txt"}'):
    return types.SimpleNamespace(
        type="function_call",
        id="fc_real",
        call_id="call_real",
        name="edit_file",
        arguments=arguments,
    )


def _client_for(*responses):
    pending = list(responses)

    async def create(**_kwargs):
        item = pending.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return types.SimpleNamespace(responses=types.SimpleNamespace(create=create))


def _load_hybrid_module():
    name = "evals.run_hybrid_malformed_eval"
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_wrapper_injects_first_edit_call_after_preparatory_response() -> None:
    hybrid = _load_hybrid_module()
    first = _response(types.SimpleNamespace(type="message"))
    second = _response(_function_call())
    wrapper = hybrid.MalformFirstToolCallClient(_client_for(first, second))

    assert _run(wrapper.create(input=[])) is first
    mutated = _run(wrapper.create(input=[]))
    call = next(item for item in mutated.output if item.type == "function_call")
    assert call.arguments == '{"path":"target.txt"'
    assert wrapper.injected is True
    assert wrapper.injected_call_id == "call_real"
    assert wrapper.replay_response_received is False


def test_wrapper_marks_replay_accepted_only_after_response() -> None:
    hybrid = _load_hybrid_module()
    first = _response(_function_call())
    second = _response(types.SimpleNamespace(type="message"))
    wrapper = hybrid.MalformFirstToolCallClient(_client_for(first, second))

    mutated = _run(wrapper.create(input=[{"role": "user", "content": "edit"}]))
    call = next(item for item in mutated.output if item.type == "function_call")
    assert call.arguments == '{"path":"target.txt"'
    assert wrapper.injected is True
    assert wrapper.replay_response_received is False

    returned = _run(wrapper.create(input=[{"type": "function_call", "arguments": "{}"}]))
    assert returned is second
    assert wrapper.replay_response_received is True
    assert wrapper.replay_input == [{"type": "function_call", "arguments": "{}"}]


def test_wrapper_does_not_mark_failed_replay_as_accepted() -> None:
    hybrid = _load_hybrid_module()
    wrapper = hybrid.MalformFirstToolCallClient(
        _client_for(_response(_function_call()), RuntimeError("Provider rejected replay")),
    )

    _run(wrapper.create(input=[]))
    try:
        _run(wrapper.create(input=[]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected replay request to fail")

    assert len(wrapper.inputs) == 2
    assert wrapper.replay_response_received is False


def test_hybrid_result_requires_every_protocol_check() -> None:
    hybrid = _load_hybrid_module()
    result = hybrid.HybridResult(
        injection_applied=True,
        tool_layer_recorded_invalid_arguments=True,
        provider_accepted_replay=True,
        replay_input_sanitized=True,
        error_feedback_present=True,
        replay_pairing_order_valid=True,
        corrected_tool_call_completed=True,
        fixture_result_correct=True,
    )
    assert result.passed is True

    required_fields = (
        "injection_applied",
        "tool_layer_recorded_invalid_arguments",
        "provider_accepted_replay",
        "replay_input_sanitized",
        "error_feedback_present",
        "replay_pairing_order_valid",
        "corrected_tool_call_completed",
        "fixture_result_correct",
    )
    for field in required_fields:
        values = {name: getattr(result, name) for name in required_fields}
        values[field] = False
        assert hybrid.HybridResult(**values).passed is False
