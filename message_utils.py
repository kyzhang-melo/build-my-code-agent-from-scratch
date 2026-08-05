"""Message protocol adapter helpers."""

from __future__ import annotations

from typing import NamedTuple


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


def _usage_attr(source, name):
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def extract_usage(response) -> dict:
    """Return provider-reported usage for one response as a flat dict.

    Every field is preserved verbatim or left ``None``. A missing field means
    the provider did not report it, which is not the same as a reported zero;
    collapsing the two would silently fabricate cache hits and free calls.
    Nothing here is estimated -- callers that need a fallback token count must
    compute it separately.
    """
    usage = _usage_attr(response, "usage")
    # Responses API nests cache counters under ``input_tokens_details``; the
    # Chat Completions shape uses ``prompt_tokens_details``. Accept either so
    # the same reader works if the request style changes.
    input_details = _usage_attr(usage, "input_tokens_details")
    if input_details is None:
        input_details = _usage_attr(usage, "prompt_tokens_details")
    output_details = _usage_attr(usage, "output_tokens_details")
    if output_details is None:
        output_details = _usage_attr(usage, "completion_tokens_details")

    def first(*names, source=usage):
        for name in names:
            value = _usage_attr(source, name)
            if value is not None:
                return value
        return None

    return {
        "cost": first("cost"),
        "input_tokens": first("input_tokens", "prompt_tokens"),
        "output_tokens": first("output_tokens", "completion_tokens"),
        "total_tokens": first("total_tokens"),
        "cached_tokens": first("cached_tokens", source=input_details),
        "cache_write_tokens": first("cache_write_tokens", source=input_details),
        "reasoning_tokens": first("reasoning_tokens", source=output_details),
        # Which upstream host actually served the call. Cache counters are
        # uninterpretable without it: a routing change invalidates the cache.
        "provider": _usage_attr(response, "provider"),
    }


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


# ---------------------------------------------------------------------------
# Resume-time sanitization
# ---------------------------------------------------------------------------

def sanitize_resumed_message(msg: dict, *, same_model: bool) -> dict:
    """Sanitize a single history dict loaded from a session file.

    When the model/provider has changed (``same_model=False``):
    - Drop ``reasoning`` items: they are provider-specific and replaying them
      to a different provider causes 400 errors.
    - Strip the provider-assigned ``id`` from ``function_call`` items: the id
      may be paired with the original record on the provider side. ``call_id``
      is retained because it is the pairing key with ``function_call_output``.

    When the model is the same, the message is returned as-is (a shallow copy).
    """
    if not isinstance(msg, dict):
        return {}

    msg_type = msg.get("type")

    if not same_model and msg_type == "reasoning":
        return {}

    if not same_model and msg_type == "function_call":
        cleaned = dict(msg)
        cleaned.pop("id", None)
        return cleaned

    return dict(msg)


def drop_orphan_tool_calls(messages: list[dict]) -> list[dict]:
    """Drop unpaired tool calls and outputs from resumed history.

    Pairing is order-sensitive: an output is valid only after its call, and a
    call is retained only if a later output completes it. This also handles a
    partially completed parallel batch such as ``call A, call B, output A``.
    """
    cleaned, _, _ = _drop_orphan_tool_calls(messages)
    return cleaned


def _drop_orphan_tool_calls(
    messages: list[dict],
) -> tuple[list[dict], int, int]:
    if not messages:
        return [], 0, 0

    filtered: list[dict] = []
    open_calls: set[str] = set()
    completed_calls: set[str] = set()
    dropped_outputs = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type")
        call_id = str(msg.get("call_id", ""))
        if msg_type == "function_call":
            if call_id:
                open_calls.add(call_id)
            filtered.append(msg)
            continue
        if msg_type == "function_call_output":
            if not call_id or call_id not in open_calls:
                dropped_outputs += 1
                continue
            open_calls.remove(call_id)
            completed_calls.add(call_id)
        filtered.append(msg)

    cleaned: list[dict] = []
    dropped_calls = 0
    for msg in filtered:
        if msg.get("type") == "function_call":
            call_id = str(msg.get("call_id", ""))
            if not call_id or call_id not in completed_calls:
                dropped_calls += 1
                continue
        cleaned.append(msg)
    return cleaned, dropped_calls, dropped_outputs


class ResumeSanitizeDiagnostics(NamedTuple):
    dropped_reasoning: int = 0
    stripped_function_call_ids: int = 0
    dropped_orphan_calls: int = 0
    dropped_orphan_outputs: int = 0
    ignored_invalid_lines: int = 0

    @property
    def changed(self) -> bool:
        return any((
            self.dropped_reasoning,
            self.stripped_function_call_ids,
            self.dropped_orphan_calls,
            self.dropped_orphan_outputs,
            self.ignored_invalid_lines,
        ))


def sanitize_resumed_history(
    messages: list[dict],
    *,
    same_runtime: bool,
    invalid_lines: int = 0,
) -> tuple[list[dict], ResumeSanitizeDiagnostics]:
    """Sanitize provider history and return safe, content-free diagnostics."""
    projected: list[dict] = []
    dropped_reasoning = 0
    stripped_ids = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if not same_runtime and msg.get("type") == "reasoning":
            dropped_reasoning += 1
        if (
            not same_runtime
            and msg.get("type") == "function_call"
            and msg.get("id")
        ):
            stripped_ids += 1
        sanitized = sanitize_resumed_message(msg, same_model=same_runtime)
        if sanitized:
            projected.append(sanitized)

    cleaned, dropped_calls, dropped_outputs = _drop_orphan_tool_calls(projected)
    return cleaned, ResumeSanitizeDiagnostics(
        dropped_reasoning=dropped_reasoning,
        stripped_function_call_ids=stripped_ids,
        dropped_orphan_calls=dropped_calls,
        dropped_orphan_outputs=dropped_outputs,
        ignored_invalid_lines=invalid_lines,
    )
