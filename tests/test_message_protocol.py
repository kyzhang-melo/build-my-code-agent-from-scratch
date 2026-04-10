from __future__ import annotations


def test_normalize_messages_merges_and_strips(load_module) -> None:
    message_utils = load_module("message_utils", "message_utils.py")

    raw = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"type": "function_call", "call_id": "x", "name": "bash", "arguments": "{}", "noise": 1},
        {"type": "function_call_output", "call_id": "x", "output": "ok", "noise": 1},
        {"role": "assistant", "content": "done"},
        {"role": "assistant", "content": "again"},
    ]

    out = message_utils.normalize_messages(raw)
    assert out[0]["role"] == "user"
    assert out[0]["content"] == "a\nb"
    assert out[1]["type"] == "function_call"
    assert "noise" not in out[1]
    assert out[2]["type"] == "function_call_output"
    assert "noise" not in out[2]
    assert out[3]["role"] == "assistant"
    assert out[3]["content"] == "done\nagain"


def test_extract_text_from_blocks(load_module) -> None:
    message_utils = load_module("message_utils", "message_utils.py")
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert message_utils.extract_text(content) == "first\nsecond"
