"""Message protocol adapter helpers."""


def _json_safe(value):
    """Convert SDK response objects into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json", exclude_none=True))
        except TypeError:
            return _json_safe(value.model_dump())

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if item is not None
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }

    return str(value)


def response_item_to_dict(item) -> dict:
    """Return a JSON-safe dict for a Responses API output item."""
    data = _json_safe(item)
    return data if isinstance(data, dict) else {}


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

        if msg_type == "reasoning":
            reasoning = response_item_to_dict(msg.get("item", msg))
            if reasoning.get("type") == "reasoning":
                cleaned.append(reasoning)
            continue

        if msg_type == "function_call":
            function_call = {
                "type": "function_call",
                "call_id": msg.get("call_id", ""),
                "name": msg.get("name", ""),
                "arguments": msg.get("arguments", "{}"),
            }
            if msg.get("id"):
                function_call["id"] = msg["id"]
            cleaned.append(function_call)
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
