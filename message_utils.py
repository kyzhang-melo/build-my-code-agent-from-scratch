"""Message protocol adapter helpers."""


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize history before API call.

    - Keep only supported keys for message-like records.
    - Preserve message boundaries; callers rely on adjacent user messages
      staying separate (for example compact summaries vs. new user input).
    """
    cleaned: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type")
        role = msg.get("role")

        if msg_type == "function_call":
            cleaned.append({
                "type": "function_call",
                "call_id": msg.get("call_id", ""),
                "name": msg.get("name", ""),
                "arguments": msg.get("arguments", "{}"),
            })
            continue

        if msg_type == "function_call_output":
            cleaned.append({
                "type": "function_call_output",
                "call_id": msg.get("call_id", ""),
                "output": str(msg.get("output", "")),
            })
            continue

        if role in ("user", "assistant", "system"):
            cleaned.append({
                "role": role,
                "content": msg.get("content", ""),
            })

    return cleaned
