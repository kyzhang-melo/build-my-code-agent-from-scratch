"""Message protocol adapter helpers."""


def extract_text(content) -> str:
    """
    Extract the final text after the agent loop.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize history before API call.

    - Keep only supported keys for message-like records.
    - Merge consecutive same-role role/content messages.
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

    if not cleaned:
        return cleaned

    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        prev = merged[-1]
        if (
            prev.get("role") in ("user", "assistant", "system")
            and msg.get("role") == prev.get("role")
            and "type" not in prev
            and "type" not in msg
        ):
            prev["content"] = f"{prev.get('content', '')}\n{msg.get('content', '')}".strip()
        else:
            merged.append(msg)
    return merged
