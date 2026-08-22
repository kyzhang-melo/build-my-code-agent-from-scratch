"""Message protocol adapter helpers."""

from __future__ import annotations

import json
from typing import NamedTuple


ASSISTANT_ROLE = "assistant"
TOOL_ROLE = "tool"


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


def _item_attr(item, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _message_text_parts(item) -> list[str]:
    parts: list[str] = []
    for part in _item_attr(item, "content", None) or []:
        if _item_attr(part, "type") in ("output_text", "text"):
            text = _item_attr(part, "text", "")
            if text:
                parts.append(str(text))
    return parts


def make_text_assistant_message(
    text: str,
    *,
    model_id: str = "",
    provider: str = "",
    source: str = "harness",
) -> dict:
    """Build one provider-neutral assistant message containing visible text."""
    return {
        "role": ASSISTANT_ROLE,
        "content": [{"type": "text", "text": str(text), "source": source}],
        "runtime": {
            "model_id": model_id,
            "provider": provider,
            "protocol": "responses",
        },
    }


def build_assistant_message(
    response_output,
    output_text: str,
    *,
    model_id: str,
    provider: str,
) -> tuple[dict, list]:
    """Capture one response as one ordered assistant message.

    ``output_text`` is an aggregate compatibility fallback. When no text block
    exists in ``response_output``, it is inserted exactly once immediately
    before the first tool call, or at the end when the response has no call.
    Provider-originated blocks are never reordered.
    """
    blocks: list[dict] = []
    tool_calls: list = []
    has_text = False
    for item in response_output or []:
        item_type = _item_attr(item, "type")
        if item_type == "reasoning":
            raw = response_item_to_dict(item)
            if raw.get("type") == "reasoning":
                blocks.append({"type": "reasoning", "provider_item": raw})
            continue
        if item_type == "message":
            for text in _message_text_parts(item):
                blocks.append({"type": "text", "text": text, "source": "response.output"})
                has_text = True
            continue
        if item_type != "function_call":
            continue

        raw_arguments = _item_attr(item, "arguments", "{}")
        try:
            parsed = json.loads(raw_arguments)
        except (ValueError, TypeError):
            replay_arguments = "{}"
            malformed = True
        else:
            replay_arguments = json.dumps(parsed)
            malformed = False
        pairing = {"call_id": str(_item_attr(item, "call_id", ""))}
        item_id = _item_attr(item, "id")
        if item_id and not malformed:
            pairing["item_id"] = str(item_id)
        blocks.append({
            "type": "tool_call",
            "name": str(_item_attr(item, "name", "")),
            "arguments": replay_arguments,
            "pairing": pairing,
            "malformed_arguments": malformed,
        })
        tool_calls.append(item)

    fallback = str(output_text or "")
    if fallback and not has_text:
        insert_at = next(
            (i for i, block in enumerate(blocks) if block.get("type") == "tool_call"),
            len(blocks),
        )
        blocks.insert(insert_at, {
            "type": "text",
            "text": fallback,
            "source": "output_text_fallback",
        })

    call_ids = [
        block.get("pairing", {}).get("call_id", "")
        for block in blocks if block.get("type") == "tool_call"
    ]
    if any(not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
        raise ValueError("response tool-call ids must be non-empty and unique")

    return ({
        "role": ASSISTANT_ROLE,
        "content": blocks,
        "runtime": {
            "model_id": model_id,
            "provider": provider,
            "protocol": "responses",
        },
    }, tool_calls)


def assistant_text(message: dict) -> str:
    """Return visible text from a logical assistant message."""
    content = message.get("content", []) if isinstance(message, dict) else []
    if isinstance(content, str):
        return content
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def assistant_tool_call_ids(message: dict) -> list[str]:
    return [
        str(block.get("pairing", {}).get("call_id", ""))
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_call"
    ]


def build_tool_result_messages(
    results: list[dict], *, call_order: list[str] | None = None,
) -> list[dict]:
    """Convert tool-layer Responses items to provider-neutral result messages."""
    converted = [{
        "role": TOOL_ROLE,
        "call_id": str(result.get("call_id", "")),
        "content": str(result.get("output", "")),
        "is_error": bool(result.get("is_error", False)),
    } for result in results if isinstance(result, dict)]
    if call_order is None:
        return converted
    position = {call_id: index for index, call_id in enumerate(call_order)}
    return sorted(converted, key=lambda result: position.get(result["call_id"], len(position)))


def _logical_assistant(msg: dict) -> bool:
    return msg.get("role") == ASSISTANT_ROLE and isinstance(msg.get("content"), list)


def validate_logical_history(messages: list[dict]) -> None:
    """Reject broken logical call/result exchanges before provider replay."""
    expected: list[str] = []
    seen_results: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if _logical_assistant(msg):
            if expected and seen_results != set(expected):
                raise ValueError("incomplete tool-result batch before assistant message")
            expected = []
            seen_results = set()
            for block in msg["content"]:
                if not isinstance(block, dict) or block.get("type") != "tool_call":
                    continue
                call_id = str(block.get("pairing", {}).get("call_id", ""))
                if not call_id or call_id in expected:
                    raise ValueError("tool-call ids must be non-empty and unique")
                expected.append(call_id)
            continue
        if msg.get("role") == TOOL_ROLE:
            call_id = str(msg.get("call_id", ""))
            if not expected or call_id not in expected or call_id in seen_results:
                raise ValueError("unknown, misplaced, or duplicate tool result")
            seen_results.add(call_id)
            continue
        if expected and seen_results != set(expected):
            raise ValueError("message inserted inside a tool-result batch")
        expected = []
        seen_results = set()
    if expected and seen_results != set(expected):
        raise ValueError("incomplete tool-result batch at end of history")


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize history before API call.

    - Keep only supported keys for message-like records.
    - Preserve message boundaries; callers rely on adjacent user messages
      staying separate (for example compact summaries vs. new user input).
    """
    if any(isinstance(msg, dict) and (_logical_assistant(msg) or msg.get("role") == TOOL_ROLE)
           for msg in messages):
        validate_logical_history(messages)

    cleaned: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type")
        role = msg.get("role")

        if _logical_assistant(msg):
            for block in msg["content"]:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "reasoning":
                    raw = response_item_to_dict(block.get("provider_item", {}))
                    if raw.get("type") == "reasoning":
                        cleaned.append(raw)
                elif block_type == "text":
                    cleaned.append({"role": "assistant", "content": block.get("text", "")})
                elif block_type == "tool_call":
                    pairing = block.get("pairing", {})
                    call = {
                        "type": "function_call",
                        "call_id": pairing.get("call_id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("arguments", "{}"),
                    }
                    if pairing.get("item_id"):
                        call["id"] = pairing["item_id"]
                    cleaned.append(call)
            continue

        if role == TOOL_ROLE:
            cleaned.append({
                "type": "function_call_output",
                "call_id": msg.get("call_id", ""),
                "output": str(msg.get("content", "")),
            })
            continue

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
    dropped_incomplete_exchanges: int = 0
    textualized_cross_runtime_exchanges: int = 0

    @property
    def changed(self) -> bool:
        return any((
            self.dropped_reasoning,
            self.stripped_function_call_ids,
            self.dropped_orphan_calls,
            self.dropped_orphan_outputs,
            self.ignored_invalid_lines,
            self.dropped_incomplete_exchanges,
            self.textualized_cross_runtime_exchanges,
        ))


def migrate_legacy_history(messages: list[dict]) -> list[dict]:
    """Best-effort v1 flat-item projection into v2 logical messages.

    Only calls with a following output in the same contiguous exchange survive.
    The migration never moves a legacy provider item across a user boundary.
    """
    migrated: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict):
            i += 1
            continue
        if _logical_assistant(msg) or msg.get("role") == TOOL_ROLE:
            migrated.append(dict(msg))
            i += 1
            continue
        if msg.get("role") in ("user", "system"):
            migrated.append(dict(msg))
            i += 1
            continue
        if not (msg.get("role") == "assistant" or msg.get("type") in ("reasoning", "function_call")):
            i += 1
            continue

        blocks: list[dict] = []
        while i < len(messages):
            item = messages[i]
            if not isinstance(item, dict):
                i += 1
                continue
            if item.get("role") == "assistant" and isinstance(item.get("content"), str):
                blocks.append({"type": "text", "text": item.get("content", ""), "source": "v1"})
                i += 1
                continue
            if item.get("type") == "reasoning":
                blocks.append({"type": "reasoning", "provider_item": dict(item)})
                i += 1
                continue
            if item.get("type") == "function_call":
                pairing = {"call_id": str(item.get("call_id", ""))}
                if item.get("id"):
                    pairing["item_id"] = item["id"]
                blocks.append({
                    "type": "tool_call",
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                    "pairing": pairing,
                })
                i += 1
                continue
            break

        outputs: list[dict] = []
        while i < len(messages) and isinstance(messages[i], dict) \
                and messages[i].get("type") == "function_call_output":
            item = messages[i]
            outputs.append({
                "role": TOOL_ROLE,
                "call_id": str(item.get("call_id", "")),
                "content": str(item.get("output", "")),
                "is_error": False,
            })
            i += 1

        completed = {item["call_id"] for item in outputs}
        blocks = [
            block for block in blocks
            if block.get("type") != "tool_call"
            or block.get("pairing", {}).get("call_id") in completed
        ]
        if blocks:
            migrated.append({
                "role": ASSISTANT_ROLE,
                "content": blocks,
                "runtime": {"model_id": "", "provider": "", "protocol": "responses"},
            })
        valid_calls = {
            block.get("pairing", {}).get("call_id")
            for block in blocks if block.get("type") == "tool_call"
        }
        migrated.extend(item for item in outputs if item["call_id"] in valid_calls)
    return migrated


def sanitize_resumed_history(
    messages: list[dict],
    *,
    same_runtime: bool,
    invalid_lines: int = 0,
) -> tuple[list[dict], ResumeSanitizeDiagnostics]:
    """Sanitize logical history and return safe, content-free diagnostics."""
    messages = migrate_legacy_history(messages)
    if not same_runtime:
        converted: list[dict] = []
        textualized = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            if not isinstance(msg, dict) or not _logical_assistant(msg):
                if isinstance(msg, dict) and msg.get("role") != TOOL_ROLE:
                    converted.append(dict(msg))
                i += 1
                continue
            text_parts = [
                str(block.get("text", "")) for block in msg["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            calls = [
                block for block in msg["content"]
                if isinstance(block, dict) and block.get("type") == "tool_call"
            ]
            results: list[dict] = []
            j = i + 1
            while j < len(messages) and isinstance(messages[j], dict) \
                    and messages[j].get("role") == TOOL_ROLE:
                results.append(messages[j])
                j += 1
            if calls:
                textualized += 1
                by_id = {str(result.get("call_id", "")): result for result in results}
                for call in calls:
                    call_id = str(call.get("pairing", {}).get("call_id", ""))
                    text_parts.append(
                        f"\n[Historical tool call: {call.get('name', '')}]\n"
                        f"Arguments: {call.get('arguments', '{}')}"
                    )
                    if call_id in by_id:
                        text_parts.append(f"\nTool result: {by_id[call_id].get('content', '')}")
            if text_parts:
                converted.append(make_text_assistant_message(
                    "".join(text_parts), source="cross_runtime_history"
                ))
            i = j
        return converted, ResumeSanitizeDiagnostics(
            dropped_reasoning=sum(
                1 for msg in messages if _logical_assistant(msg)
                for block in msg.get("content", []) if block.get("type") == "reasoning"
            ),
            textualized_cross_runtime_exchanges=textualized,
            ignored_invalid_lines=invalid_lines,
        )

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

    try:
        validate_logical_history(projected)
    except ValueError:
        # A resumed incomplete exchange is safer to drop than to guess/reorder.
        cleaned: list[dict] = []
        for msg in projected:
            if _logical_assistant(msg):
                safe_blocks = [
                    block for block in msg["content"]
                    if block.get("type") in ("text", "reasoning")
                ]
                if safe_blocks:
                    copy = dict(msg)
                    copy["content"] = safe_blocks
                    cleaned.append(copy)
            elif msg.get("role") != TOOL_ROLE:
                cleaned.append(msg)
        return cleaned, ResumeSanitizeDiagnostics(
            ignored_invalid_lines=invalid_lines,
            dropped_incomplete_exchanges=1,
        )
    return projected, ResumeSanitizeDiagnostics(
        dropped_reasoning=dropped_reasoning,
        stripped_function_call_ids=stripped_ids,
        ignored_invalid_lines=invalid_lines,
    )
