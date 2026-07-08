from __future__ import annotations


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
