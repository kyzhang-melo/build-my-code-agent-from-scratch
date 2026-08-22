from __future__ import annotations

from types import SimpleNamespace

from hypothesis import given, strategies as st


def test_normalize_messages_preserves_boundaries_and_strips(load_module) -> None:
    message_utils = load_module("message_utils", "message_utils.py")

    raw = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"type": "function_call", "id": "fc_x", "call_id": "x", "name": "bash", "arguments": "{}", "noise": 1},
        {"type": "function_call_output", "call_id": "x", "output": "ok", "noise": 1},
        {"role": "assistant", "content": "done"},
        {"role": "assistant", "content": "again"},
    ]

    out = message_utils.normalize_messages(raw)
    assert out[0]["role"] == "user"
    assert out[0]["content"] == "a"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "b"
    assert out[2]["type"] == "function_call"
    assert out[2]["id"] == "fc_x"
    assert "noise" not in out[2]
    assert out[3]["type"] == "function_call_output"
    assert "noise" not in out[3]
    assert out[4]["role"] == "assistant"
    assert out[4]["content"] == "done"
    assert out[5]["role"] == "assistant"
    assert out[5]["content"] == "again"


def test_normalize_messages_preserves_reasoning_items(load_module) -> None:
    message_utils = load_module("message_utils", "message_utils.py")

    raw = [{
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "thought"}],
    }]

    out = message_utils.normalize_messages(raw)

    assert out == raw


def test_response_item_to_dict_serializes_sdk_like_objects(load_module) -> None:
    message_utils = load_module("message_utils", "message_utils.py")

    class Part:
        def __init__(self) -> None:
            self.type = "summary_text"
            self.text = "thought"

    class Item:
        def __init__(self) -> None:
            self.type = "reasoning"
            self.id = "rs_1"
            self.status = "completed"
            self.summary = [Part()]

    out = message_utils.response_item_to_dict(Item())

    assert out == {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "thought"}],
    }


def test_normalize_messages_does_not_merge_summary_with_user(load_module) -> None:
    message_utils = load_module("message_utils", "message_utils.py")

    out = message_utils.normalize_messages([
        {"role": "user", "content": "[CONTEXT SUMMARY]\nold work is done"},
        {"role": "user", "content": "what happened?"},
    ])

    assert len(out) == 2
    assert out[0]["content"].startswith("[CONTEXT SUMMARY]")
    assert out[1]["content"] == "what happened?"


def test_output_text_fallback_has_one_deterministic_boundary(load_module) -> None:
    message_utils = load_module("message_utils", "message_utils.py")
    reasoning = SimpleNamespace(type="reasoning", id="rs_1", summary=[])
    call = SimpleNamespace(
        type="function_call", id="fc_1", call_id="c1", name="bash", arguments="{}",
    )

    assistant, _ = message_utils.build_assistant_message(
        [reasoning, call], "fallback", model_id="m", provider="p",
    )

    assert [block["type"] for block in assistant["content"]] == [
        "reasoning", "text", "tool_call",
    ]
    assert assistant["content"][1]["source"] == "output_text_fallback"


@given(st.lists(st.sampled_from(["reasoning", "message", "call"]), max_size=8))
def test_assistant_serialization_preserves_provider_item_order(kinds) -> None:
    """Logical round-trip never reorders provider-originated response blocks."""
    import message_utils

    items = []
    expected = []
    call_ids = []
    for index, kind in enumerate(kinds):
        if kind == "reasoning":
            items.append(SimpleNamespace(type="reasoning", id=f"rs_{index}", summary=[]))
            expected.append(("reasoning", f"rs_{index}"))
        elif kind == "message":
            text = f"text-{index}"
            items.append(SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            ))
            expected.append(("text", text))
        else:
            call_id = f"call_{index}"
            call_ids.append(call_id)
            items.append(SimpleNamespace(
                type="function_call", id=f"fc_{index}", call_id=call_id,
                name="bash", arguments="{}",
            ))
            expected.append(("call", call_id))

    assistant, _ = message_utils.build_assistant_message(
        items, "fallback", model_id="m", provider="p",
    )
    history = [assistant, *message_utils.build_tool_result_messages(
        [{"call_id": call_id, "output": "ok"} for call_id in call_ids],
        call_order=call_ids,
    )]
    serialized = message_utils.normalize_messages(history)
    actual = []
    for item in serialized:
        if item.get("type") == "reasoning":
            actual.append(("reasoning", item.get("id")))
        elif item.get("type") == "function_call":
            actual.append(("call", item.get("call_id")))
        elif item.get("role") == "assistant" and item.get("content") != "fallback":
            actual.append(("text", item.get("content")))

    assert actual == expected
