"""context_compact.py

Tier 2 of context compaction: ``compact_history_async``. When the conversation
approaches the model's context window (auto) or on user command (``/compact``),
an LLM side-call summarizes the older history, which is then rebuilt as a
checkpoint summary plus a recent verbatim tail. Old user requests that were
summarized are not replayed as active instructions.

Tier 1 (per-output middle-truncation) lives in ``tools.truncate_middle``.

Manual and automatic compaction both call ``compact_history_async``; they differ
only in ``source`` and whether a ``focus`` is supplied.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from message_utils import normalize_messages
from tools import TodoManager

TEMPLATE_PATH = Path(__file__).parent / "templates" / "compact.md"
TRANSCRIPT_DIR = Path(__file__).parent / ".transcripts"

# Marks a message as a compaction summary. It is carried on a user-role message
# for provider compatibility, but normalize_messages preserves its boundary so
# it cannot be merged into the next real user request.
SUMMARY_PREFIX = "[CONTEXT SUMMARY]"

# Approximate recent-context tail retained verbatim after a checkpoint.
DEFAULT_KEEP_RECENT_TOKENS = 20000
SUMMARY_MAX_OUTPUT_TOKENS = 4000
COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following "
    "summary. Treat it as historical context, not as a new user request. Work "
    "listed as Done is complete; do not re-run completed tool work.\n\n"
    "<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

_COMPACTION_DIRECTIVE = (
    "You are a context-compaction assistant. Produce only the handoff summary "
    "asked for in the final message. Do not continue the conversation, do not "
    "answer any question in the history, do not re-execute completed work, and "
    "do not call tools."
)

_template_cache: str | None = None


@dataclass(frozen=True)
class CompactionResult:
    source: str
    tokens_before: int
    tokens_after: int
    transcript_path: str | None


def estimate_tokens(messages: list) -> int:
    """Cheap char-based token estimate (chars ~= 4x tokens)."""
    return len(json.dumps(messages, default=str)) // 4


def _resolve_keep_recent_tokens() -> int:
    raw = os.getenv("COMPACT_KEEP_RECENT_TOKENS", str(DEFAULT_KEEP_RECENT_TOKENS))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_KEEP_RECENT_TOKENS
    return max(0, value)


KEEP_RECENT_TOKENS = _resolve_keep_recent_tokens()


def load_template() -> str:
    global _template_cache
    if _template_cache is None:
        _template_cache = TEMPLATE_PATH.read_text()
    return _template_cache


def render_prompt(focus: str | None, previous_summary: str | None = None) -> str:
    focus_block = (
        f"Focus for this compaction (preserve these above all else): {focus}\n"
        if focus else ""
    )
    previous_block = (
        "<previous-summary>\n"
        f"{previous_summary}\n"
        "</previous-summary>\n\n"
        if previous_summary else ""
    )
    return (
        load_template()
        .replace("{{ focus }}", focus_block)
        .replace("{{ previous_summary }}", previous_block)
    )


def is_summary_message(msg: dict) -> bool:
    return (
        isinstance(msg, dict)
        and msg.get("role") == "user"
        and str(msg.get("content", "")).startswith(SUMMARY_PREFIX)
    )


def _unwrap_summary_content(content: str) -> str:
    body = content[len(SUMMARY_PREFIX):].lstrip() if content.startswith(SUMMARY_PREFIX) else content
    start = body.find("<summary>")
    end = body.rfind("</summary>")
    if start != -1 and end != -1 and end > start:
        return body[start + len("<summary>"):end].strip()
    return body.strip()


def extract_previous_summary(messages: list) -> tuple[str | None, int]:
    """Return a leading summary body and the index where live history starts."""
    if not messages or not isinstance(messages[0], dict) or not is_summary_message(messages[0]):
        return None, 0
    return _unwrap_summary_content(str(messages[0].get("content", ""))), 1


def estimate_message_tokens(message: dict) -> int:
    return estimate_tokens([message])


def find_cut_index(
    messages: list,
    keep_recent_tokens: int = KEEP_RECENT_TOKENS,
    *,
    start_index: int = 0,
) -> int:
    """Choose a safe turn boundary for `[to_summarize] + [tail]`.

    The returned index is the first message of the retained tail. It is either a
    user turn start or `len(messages)` for an empty tail, so the retained tail
    cannot begin with an orphaned function_call_output.
    """
    if start_index >= len(messages):
        return len(messages)

    window = [msg for msg in messages[start_index:] if isinstance(msg, dict)]
    if estimate_tokens(window) <= keep_recent_tokens:
        return len(messages)

    accumulated = 0
    threshold_index = start_index
    for i in range(len(messages) - 1, start_index - 1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        accumulated += estimate_message_tokens(msg)
        if accumulated >= keep_recent_tokens:
            threshold_index = i
            break

    # Keep from the user turn that owns threshold_index. This can keep a little
    # more than the target, but it preserves protocol-safe turn structure.
    for i in range(threshold_index, start_index - 1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user" and not is_summary_message(msg):
            return i

    # No safe user boundary: summarize the whole live history.
    return len(messages)


def build_summary_message(summary: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\n"
            f"{COMPACTION_SUMMARY_PREFIX}"
            f"{summary.strip()}"
            f"{COMPACTION_SUMMARY_SUFFIX}"
        ),
    }


def build_compacted_history(summary: str, tail_messages: list[dict]) -> list[dict]:
    """Checkpoint rebuild: one summary message followed by a verbatim tail."""
    return [build_summary_message(summary), *tail_messages]


async def summarize_async(
    messages: list,
    focus: str | None,
    *,
    client,
    model: str,
    extra_body: dict | None = None,
    previous_summary: str | None = None,
) -> str:
    """Async side-call for a handoff summary. No tools. Raises on empty output."""
    api_input = normalize_messages(
        messages + [{"role": "user", "content": render_prompt(focus, previous_summary)}]
    )
    response = await client.responses.create(
        model=model,
        instructions=_COMPACTION_DIRECTIVE,
        input=api_input,
        max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        extra_body=extra_body,
    )
    summary = (getattr(response, "output_text", "") or "").strip()
    if not summary:
        raise ValueError("compaction summary was empty")
    return summary


def reinject_todo(summary: str, todo: TodoManager) -> str:
    """Append the live TODO list deterministically (not LLM-reconstructed)."""
    if not todo.has_active_plan():
        return summary
    return f"{summary}\n\n## Current TODO state\n{todo.render()}"


def write_transcript(messages: list) -> Path:
    """One-shot JSONL snapshot of history before a destructive compaction."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = TRANSCRIPT_DIR / f"session-{stamp}-{int(time.time() * 1000) % 1000:03d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for msg in messages:
            handle.write(json.dumps(msg, default=str) + "\n")
    return path


def _has_droppable(messages: list) -> bool:
    """True if there's agent work (assistant/tool items) worth compacting away."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("type") in ("reasoning", "function_call", "function_call_output"):
            return True
        if msg.get("role") == "assistant":
            return True
    return False


def prepare_compaction(messages: list) -> tuple[str | None, list[dict], list[dict]]:
    """Split history into previous summary, older messages to summarize, and tail."""
    previous_summary, start_index = extract_previous_summary(messages)
    cut_index = find_cut_index(messages, KEEP_RECENT_TOKENS, start_index=start_index)
    if cut_index <= start_index:
        cut_index = len(messages)

    to_summarize = [
        msg for msg in messages[start_index:cut_index]
        if isinstance(msg, dict)
    ]
    tail = [
        msg for msg in messages[cut_index:]
        if isinstance(msg, dict)
    ]
    return previous_summary, to_summarize, tail


async def compact_history_async(
    state,
    *,
    todo: TodoManager,
    source: str,
    focus: str | None = None,
    client,
    model: str,
    extra_body: dict | None = None,
) -> CompactionResult | None:
    """Summarize and start-fresh-rebuild ``state.messages``.

    Returns ``None`` with the history left untouched when there's nothing worth
    compacting or the summary call fails. Order is failure-safe: snapshot first,
    summarize next, and only replace the live history once a summary is in hand.
    """
    messages = state.messages
    if not _has_droppable(messages):
        return None

    tokens_before = state.last_input_tokens or estimate_tokens(messages)
    transcript_path = write_transcript(messages)

    previous_summary, to_summarize, tail = prepare_compaction(messages)
    if not to_summarize:
        return None

    try:
        summary = await summarize_async(
            to_summarize,
            focus,
            client=client,
            model=model,
            extra_body=extra_body,
            previous_summary=previous_summary,
        )
    except Exception as exc:  # any failure -> keep the original history
        print(f"[compact] summarization failed ({exc}); history left unchanged")
        return None

    summary = reinject_todo(summary, todo)
    new_messages = build_compacted_history(summary, tail)

    state.messages[:] = new_messages
    tokens_after = estimate_tokens(new_messages)
    state.last_input_tokens = tokens_after
    return CompactionResult(
        source=source,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        transcript_path=str(transcript_path),
    )
