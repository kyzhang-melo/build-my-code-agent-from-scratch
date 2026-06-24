"""context_compact.py

Tier 2 of context compaction: ``compact_history``. When the conversation
approaches the model's context window (auto) or on user command (``/compact``),
an LLM side-call summarizes the history, which is then rebuilt "start-fresh" --
the system prompt (carried automatically on the API ``instructions`` field, so
it never lives in the message list) + every user message + the summary --
discarding the agent's own assistant/tool turns, which are regenerable.

Tier 1 (per-output middle-truncation) lives in ``tools.truncate_middle``.

Manual and automatic compaction both call ``compact_history``; they differ only
in ``source`` and whether a ``focus`` is supplied.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from message_utils import normalize_messages
from tools import TODO

TEMPLATE_PATH = Path(__file__).parent / "templates" / "compact.md"
TRANSCRIPT_DIR = Path(__file__).parent / ".transcripts"

# Marks a message as a compaction summary. Carried on a user-role message so it
# reads as a briefing to a fresh instance; also lets re-compaction skip prior
# summaries so they don't stack.
SUMMARY_PREFIX = "[CONTEXT SUMMARY]"

# User messages are kept verbatim but bounded (newest-first). ~20k tokens.
USER_MESSAGE_MAX_CHARS = 80000
SUMMARY_MAX_OUTPUT_TOKENS = 4000

_COMPACTION_DIRECTIVE = (
    "You are a context-compaction assistant. Produce only the handoff summary "
    "asked for in the final message. Do not call tools."
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


def load_template() -> str:
    global _template_cache
    if _template_cache is None:
        _template_cache = TEMPLATE_PATH.read_text()
    return _template_cache


def render_prompt(focus: str | None) -> str:
    focus_block = (
        f"Focus for this compaction (preserve these above all else): {focus}\n"
        if focus else ""
    )
    return load_template().replace("{{ focus }}", focus_block)


def collect_user_messages(messages: list) -> list[str]:
    """User-role message text, in order, skipping prior compaction summaries."""
    out: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            content = str(content)
        if content.startswith(SUMMARY_PREFIX):
            continue
        out.append(content)
    return out


def _cap_user_messages(user_messages: list[str]) -> list[str]:
    """Keep newest-first within the char budget; truncate the one that overflows."""
    kept: list[str] = []
    remaining = USER_MESSAGE_MAX_CHARS
    for msg in reversed(user_messages):
        if remaining <= 0:
            break
        if len(msg) <= remaining:
            kept.append(msg)
            remaining -= len(msg)
        else:
            kept.append(msg[:remaining] + " [...truncated]")
            remaining = 0
    kept.reverse()
    return kept


def build_compacted_history(user_messages: list[str], summary: str) -> list[dict]:
    """start-fresh rebuild: capped user messages + the tagged summary.

    The system prompt is NOT included here -- it rides on the API ``instructions``
    field, so it is preserved automatically.
    """
    history: list[dict] = [
        {"role": "user", "content": msg} for msg in _cap_user_messages(user_messages)
    ]
    history.append({"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary}"})
    return history


def summarize(messages: list, focus: str | None, *, client, model: str) -> str:
    """Side-call the model for a handoff summary. No tools. Raises on empty output."""
    api_input = normalize_messages(
        messages + [{"role": "user", "content": render_prompt(focus)}]
    )
    response = client.responses.create(
        model=model,
        instructions=_COMPACTION_DIRECTIVE,
        input=api_input,
        max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
    )
    summary = (getattr(response, "output_text", "") or "").strip()
    if not summary:
        raise ValueError("compaction summary was empty")
    return summary


def reinject_todo(summary: str) -> str:
    """Append the live TODO list deterministically (not LLM-reconstructed)."""
    if not TODO.has_active_plan():
        return summary
    return f"{summary}\n\n## Current TODO state\n{TODO.render()}"


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
        if msg.get("type") in ("function_call", "function_call_output"):
            return True
        if msg.get("role") == "assistant":
            return True
    return False


def compact_history(
    state,
    *,
    source: str,
    focus: str | None = None,
    client,
    model: str,
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

    try:
        summary = summarize(messages, focus, client=client, model=model)
    except Exception as exc:  # any failure -> keep the original history
        print(f"[compact] summarization failed ({exc}); history left unchanged")
        return None

    summary = reinject_todo(summary)
    new_messages = build_compacted_history(collect_user_messages(messages), summary)

    # Mutate in place so any alias of this list (e.g. the REPL's `history`) sees it.
    state.messages[:] = new_messages
    tokens_after = estimate_tokens(new_messages)
    state.last_input_tokens = tokens_after
    return CompactionResult(
        source=source,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        transcript_path=str(transcript_path),
    )
