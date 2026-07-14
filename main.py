#!/usr/bin/env python3
"""main.py

Split version of the code-agent loop.
"""

import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from dotenv import load_dotenv
from openai import AsyncOpenAI
from message_utils import normalize_messages, response_item_to_dict
from prompts import EXPLORE_SUBAGENT_SYSTEM, GLOB_DISCOVERY_RULES, PARENT_SYSTEM
from tools import (
    EXPLORE_TOOL_REGISTRY,
    EXPLORE_TOOLS,
    TODO,
    TOOLS,
    configure_task_runner,
    execute_tool_calls_async,
)
from context_compact import (
    SUMMARY_PREFIX,
    compact_history_async,
    estimate_tokens,
)


load_dotenv(override=True)
print("[init] .env loaded")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL_ID = os.getenv("MODEL_ID")
OPENROUTER_PROVIDER = os.getenv("OPENROUTER_PROVIDER")  # e.g. "moonshotai" to pin one host

print(f"[init] MODEL_ID={MODEL_ID!r}")
print(f"[init] OPENROUTER_BASE_URL={OPENROUTER_BASE_URL!r}")
print(f"[init] OPENROUTER_API_KEY present={bool(OPENROUTER_API_KEY)}")
print(f"[init] OPENROUTER_PROVIDER={OPENROUTER_PROVIDER!r}")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Please set it in .env")
if not OPENROUTER_BASE_URL:
    raise RuntimeError("OPENROUTER_BASE_URL is not set. Please set it in .env")
if not MODEL_ID:
    raise RuntimeError("MODEL_ID is not set. Please set it in .env")

client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)
print("[init] AsyncOpenAI client initialized")

# OpenRouter provider pinning. When OPENROUTER_PROVIDER is set, every
# responses.create restricts routing to that single upstream host with
# fallbacks off (fail loud rather than silently rerouting), so token
# accounting and the effective context window stay consistent across turns
# and the summarizer side-call. Unset -> None -> OpenRouter's default routing.
PROVIDER_EXTRA_BODY = (
    {"provider": {"only": [OPENROUTER_PROVIDER], "allow_fallbacks": False}}
    if OPENROUTER_PROVIDER
    else None
)
TODO_CONTRACT_MAX_NUDGES = 2
EMPTY_RESPONSE_MAX_NUDGES = 1
EMPTY_RESPONSE_NUDGE = (
    "Your previous response contained no assistant-visible text and no tool "
    "calls. Continue from any reasoning above and provide the required "
    "assistant-visible answer now. If more information is needed, call a tool "
    "instead of returning an empty response."
)
SUMMARY_MIN_LENGTH = 200
SUMMARY_CONTINUATION_ATTEMPTS = 1
SUMMARY_CONTINUATION_PROMPT = (
    "Your previous response was too brief. Please provide a more comprehensive "
    "summary that includes specific technical details, findings, and all "
    "important information that the parent agent should know."
)
MAX_API_CALLS_PER_USER_TURN = 30
MAX_SUBAGENT_API_CALLS = 30
INPUT_PROMPT = "\001\033[36m\002s01 >> \001\033[0m\002"

# --- Context compaction config ---
# Per-model context windows, resolved by PREFIX match against a normalized
# model id (most-specific pattern first, first match wins). Normalizing first
# (drop the "vendor/" prefix and any ":route" suffix like ":exacto"/":nitro")
# means routing/quant tags can't defeat the lookup the way an exact-key dict
# does — e.g. "moonshotai/kimi-k2.5:exacto" still resolves to kimi's 262144
# instead of silently falling through to DEFAULT_CONTEXT_WINDOW. Unknown models
# fall back to a conservative default so they compact early rather than overflow.
CONTEXT_WINDOW_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^hy3$"), 262144),
    (re.compile(r"^kimi-"), 262144),
    (re.compile(r"^deepseek-v4"), 1_000_000),
    (re.compile(r"^minimax-m3"), 524_288),
    (re.compile(r"^glm-5"), 202_800),
]
DEFAULT_CONTEXT_WINDOW = 32000
# Deliberate override, e.g. shrink the window in tests so auto-compaction is
# easy to trigger. 0/unset -> resolve from MODEL_ID via the patterns above.
CONTEXT_WINDOW_OVERRIDE = int(os.getenv("CONTEXT_WINDOW_OVERRIDE", "0"))
RESERVED_OUTPUT_TOKENS = 8000      # mirrors max_output_tokens in run_one_turn
RESERVED_OVERHEAD_TOKENS = 4000    # system prompt + tool schemas
COMPACT_TRIGGER_RATIO = 0.85
# On by default; AUTO_COMPACT=0 disables automatic compaction (manual /compact
# still works). The destructive rewrite is backed by a .transcripts/ snapshot.
AUTO_COMPACT_ENABLED = os.getenv("AUTO_COMPACT", "1") != "0"


def normalize_model_id(model_id: str) -> str:
    """vendor/model:route -> model  (moonshotai/kimi-k2.5:exacto -> kimi-k2.5)."""
    s = model_id.lower().strip()
    s = s.rsplit("/", 1)[-1]   # drop "vendor/" prefix
    s = s.split(":", 1)[0]     # drop ":route"/":quant" suffix
    return s


def context_window() -> int:
    if CONTEXT_WINDOW_OVERRIDE > 0:
        return CONTEXT_WINDOW_OVERRIDE
    norm = normalize_model_id(MODEL_ID)
    for pattern, window in CONTEXT_WINDOW_PATTERNS:
        if pattern.match(norm):
            return window
    return DEFAULT_CONTEXT_WINDOW


def input_budget() -> int:
    # Tokens available for input once the response reservation and fixed overhead
    # are carved out. The trigger ratio applies to THIS, not the raw window, so
    # the output reservation can't be eaten away on small-context models.
    return context_window() - RESERVED_OUTPUT_TOKENS - RESERVED_OVERHEAD_TOKENS


def should_auto_compact(state: "LoopState") -> bool:
    return state.last_input_tokens >= COMPACT_TRIGGER_RATIO * input_budget()


@dataclass
class LoopState:
    # The minimal loop state: history, API call count, and nudge budget.
    messages: list
    api_call_count: int = 0
    nudges: int = 0
    empty_response_nudges: int = 0
    # Input tokens the last API call consumed (from response.usage, or estimated).
    # Drives the auto-compaction trigger.
    last_input_tokens: int = 0


StopReason = Literal["completed", "max_api_calls"]


@dataclass(frozen=True)
class StepOutcome:
    # The turn's answer, evaluated directly from the stopping response --
    # never scanned from history, so a prior turn's message can't leak in.
    stop_reason: StopReason
    final_text: str


@dataclass(frozen=True)
class TurnOutcome:
    stop_reason: StopReason
    final_text: str
    api_calls: int


class StopGate(Protocol):
    # Decides whether a turn may end when the model stops calling tools.
    max_nudges: int

    def check(self, final_text: str) -> str | None:
        """Return None to accept the stop, or a nudge message to push back."""
        ...

    def give_up_note(self) -> str | None:
        """Optional message to append when the nudge budget is exhausted."""
        ...


@dataclass(frozen=True)
class AgentConfig:
    name: str
    system: str
    tools: list[dict]
    max_api_calls: int
    stop_gate: StopGate
    registry: dict | None = None
    # Sink for assistant text as it is produced; None means do not surface it.
    # Display is intentionally decoupled from loop termination: text that rides
    # along with tool calls is shown here, not only the final no-tool message.
    on_text: Callable[[str], None] | None = None


def emit_assistant_text(text: str) -> None:
    print(text)


def _response_item_attr(item, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def debug_empty_output_text_response(response) -> None:
    """Print response.output shape when output_text is empty."""
    if (getattr(response, "output_text", "") or "").strip():
        return

    output = getattr(response, "output", None) or []
    print(f"[debug] output_text empty; response.output has {len(output)} item(s)")
    for i, item in enumerate(output):
        item_type = _response_item_attr(item, "type")
        status = _response_item_attr(item, "status")
        line = f"[debug]  output[{i}] type={item_type!r} status={status!r}"

        if item_type == "message":
            content = _response_item_attr(item, "content", []) or []
            content_types = [
                _response_item_attr(part, "type")
                for part in content
            ]
            line += f" content_types={content_types!r}"

        print(line)


class TodoStopGate:
    def __init__(self, todo, max_nudges: int):
        self.todo = todo
        self.max_nudges = max_nudges

    def check(self, final_text: str) -> str | None:
        if not self.todo.has_active_plan() or self.todo.all_items_completed():
            return None
        return (
            "<contract>Before ending, either complete all todo items, "
            "or call todo to explicitly rewrite/remove items that are no longer needed.</contract>"
        )

    def give_up_note(self) -> str | None:
        return (
            "Warning: Ending with unresolved todo items after repeated contract reminders.\n"
            f"{self.todo.render()}"
        )


class ReportStopGate:
    def __init__(self, min_length: int, max_nudges: int):
        self.min_length = min_length
        self.max_nudges = max_nudges

    def check(self, final_text: str) -> str | None:
        if len(final_text) >= self.min_length:
            return None
        return SUMMARY_CONTINUATION_PROMPT

    def give_up_note(self) -> str | None:
        return None


PARENT_CONFIG = AgentConfig(
    name="parent",
    system=PARENT_SYSTEM,
    tools=TOOLS,
    max_api_calls=MAX_API_CALLS_PER_USER_TURN,
    stop_gate=TodoStopGate(TODO, TODO_CONTRACT_MAX_NUDGES),
    on_text=emit_assistant_text,
)

EXPLORE_SUBAGENT_CONFIG = AgentConfig(
    name="subagent:explore",
    system=EXPLORE_SUBAGENT_SYSTEM,
    tools=EXPLORE_TOOLS,
    registry=EXPLORE_TOOL_REGISTRY,
    max_api_calls=MAX_SUBAGENT_API_CALLS,
    stop_gate=ReportStopGate(SUMMARY_MIN_LENGTH, SUMMARY_CONTINUATION_ATTEMPTS),
)


def build_subagent_prompt(prompt: str) -> str:
    return (
        "Mode: explore. Inspect and analyze only. Do not modify files.\n\n"
        f"{GLOB_DISCOVERY_RULES}\n\n"
        f"Task:\n{prompt}"
    )


async def execute_configured_tool_calls(tool_calls, config: AgentConfig) -> tuple[list[dict], bool]:
    if config.registry is None:
        return await execute_tool_calls_async(tool_calls)
    return await execute_tool_calls_async(tool_calls, config.registry)


async def run_one_turn(state: LoopState, config: AgentConfig = PARENT_CONFIG) -> StepOutcome | None:
    # Returns None to keep looping, or a StepOutcome when the turn ends.
    if state.api_call_count >= config.max_api_calls:
        warning = f"Warning: stopped after max_api_calls={config.max_api_calls}."
        state.messages.append({
            "role": "assistant",
            "content": warning,
        })
        return StepOutcome(stop_reason="max_api_calls", final_text=warning)

    state.api_call_count += 1
    input_messages = normalize_messages(state.messages)
    print(f"[debug] sending {len(input_messages)} messages to LLM")
    for i, msg in enumerate(input_messages[-3:], start=max(0, len(input_messages) - 3)):
        role = msg.get("role") or msg.get("type", "unknown")
        content = msg.get("content") or msg.get("output") or msg.get("arguments") or ""
        if isinstance(content, str):
            preview = content.replace("\n", " ")[:120]
            if len(content) > 120:
                preview += "..."
        else:
            preview = str(content).replace("\n", " ")[:120]
        print(f"[debug]  [{i}] {role}: {preview}")
    response = await client.responses.create(
        model=MODEL_ID,
        instructions=config.system,
        input=input_messages,
        tools=config.tools,
        max_output_tokens=8000,
        extra_body=PROVIDER_EXTRA_BODY,
    )
    debug_empty_output_text_response(response)

    # Track input-token load for the auto-compaction trigger. Prefer the API's
    # reported usage; fall back to a char-based estimate if it's absent.
    usage = getattr(response, "usage", None)
    reported = getattr(usage, "input_tokens", None) if usage is not None else None
    state.last_input_tokens = (
        reported if reported is not None
        else len(json.dumps(input_messages, default=str)) // 4
    )

    output_text = getattr(response, "output_text", "") or ""
    response_output = getattr(response, "output", None) or []

    if output_text:
        state.messages.append({
            "role": "assistant",
            "content": output_text,
        })
    
    tool_calls = []
    for item in response_output:
        item_type = _response_item_attr(item, "type")
        if item_type == "reasoning":
            reasoning = response_item_to_dict(item)
            if reasoning.get("type") == "reasoning":
                state.messages.append(reasoning)
            continue

        if item_type == "function_call":
            function_call = {
                "type": "function_call",
                "call_id": _response_item_attr(item, "call_id", ""),
                "name": _response_item_attr(item, "name", ""),
                "arguments": _response_item_attr(item, "arguments", "{}"),
            }
            item_id = _response_item_attr(item, "id")
            if item_id:
                function_call["id"] = item_id
            state.messages.append(function_call)

            tool_calls.append(item)

    if not tool_calls:
        response_text = output_text.strip()
        if not response_text:
            if state.empty_response_nudges < EMPTY_RESPONSE_MAX_NUDGES:
                state.empty_response_nudges += 1
                state.messages.append({"role": "user", "content": EMPTY_RESPONSE_NUDGE})
                return None

            warning = (
                "Warning: model returned an empty response with no tool calls "
                "after retry."
            )
            state.empty_response_nudges = 0
            state.messages.append({"role": "assistant", "content": warning})
            return StepOutcome(stop_reason="completed", final_text=warning)

        state.empty_response_nudges = 0
        nudge = config.stop_gate.check(response_text)
        if nudge is not None and state.nudges < config.stop_gate.max_nudges:
            # The loop continues, so this text won't become final_text and would
            # otherwise be lost. Surface the rejected answer before nudging.
            if response_text and config.on_text is not None:
                config.on_text(response_text)
            state.nudges += 1
            state.messages.append({"role": "user", "content": nudge})
            return None

        if nudge is not None:
            # Nudge budget exhausted: accept the stop anyway.
            note = config.stop_gate.give_up_note()
            if note is not None:
                state.messages.append({"role": "assistant", "content": note})
        state.nudges = 0
        # Turn ends on a model message with no tool calls: this is the answer.
        return StepOutcome(stop_reason="completed", final_text=response_text)

    state.empty_response_nudges = 0
    # Surface assistant text that accompanies tool calls; otherwise it would be
    # retained in history but never shown, swallowing real deliverables.
    if output_text and config.on_text is not None:
        config.on_text(output_text)

    results, _ = await execute_configured_tool_calls(tool_calls, config)
    if not results:
        return StepOutcome(stop_reason="completed", final_text="")

    state.messages.extend(results)
    state.nudges = 0
    return None


def _drop_oldest_user_message(state: LoopState) -> bool:
    """Drop the oldest retained non-summary user turn.

    Returns True if one turn was removed, False if none remain to drop. The
    compaction summary is protected, and tool outputs stay attached to the turn
    that produced them instead of being left orphaned.
    """
    start = None
    for i, msg in enumerate(state.messages):
        if (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and not str(msg.get("content", "")).startswith(SUMMARY_PREFIX)
        ):
            start = i
            break
    if start is None:
        return False

    end = start + 1
    while end < len(state.messages):
        msg = state.messages[end]
        if (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and not str(msg.get("content", "")).startswith(SUMMARY_PREFIX)
        ):
            break
        end += 1

    del state.messages[start:end]
    return True


async def agent_loop(state: LoopState, config: AgentConfig = PARENT_CONFIG) -> TurnOutcome:
    while (outcome := await run_one_turn(state, config)) is None:
        # Each `None` outcome is a clean turn boundary: the turn's tool outputs
        # are already appended, so the history is well-formed (no orphaned tool
        # call) -- a safe point to compact.
        if AUTO_COMPACT_ENABLED and should_auto_compact(state):
            result = await compact_history_async(
                state, source="auto", focus=None, client=client, model=MODEL_ID,
                extra_body=PROVIDER_EXTRA_BODY,
            )
            if result is not None:
                print(
                    f"[compact] auto-compacted: ~{result.tokens_before} -> "
                    f"~{result.tokens_after} tokens (backup: {result.transcript_path})"
                )
                # If even a fresh summary still doesn't fit, the surviving user
                # messages are the bulk: drop the oldest until we fit, rather
                # than re-summarizing on every subsequent turn.
                while should_auto_compact(state) and _drop_oldest_user_message(state):
                    state.last_input_tokens = estimate_tokens(state.messages)
    return TurnOutcome(
        stop_reason=outcome.stop_reason,
        final_text=outcome.final_text,
        api_calls=state.api_call_count,
    )


async def run_subagent(prompt: str, description: str = "exploration") -> str:
    print(f"\033[35m> task (explore/{description}): {prompt[:120]}\033[0m")
    state = LoopState(messages=[{
        "role": "user",
        "content": build_subagent_prompt(prompt),
    }])
    outcome = await agent_loop(state, EXPLORE_SUBAGENT_CONFIG)

    if outcome.final_text:
        return outcome.final_text
    return f"[subagent produced no output after {outcome.api_calls} turns]"


configure_task_runner(run_subagent)


async def cmd_compact(arg: str, history: list) -> None:
    state = LoopState(history)
    result = await compact_history_async(
        state, source="manual", focus=arg or None, client=client, model=MODEL_ID,
        extra_body=PROVIDER_EXTRA_BODY,
    )
    if result is None:
        print("[compact] nothing to compact yet.")
        return
    history[:] = state.messages
    print(
        f"[compact] context compacted: ~{result.tokens_before} -> "
        f"~{result.tokens_after} tokens (backup: {result.transcript_path})"
    )


async def cmd_help(arg: str, history: list) -> None:
    print("commands: /compact [focus]  |  /help   (q or exit to quit)")


COMMANDS = {
    "compact": cmd_compact,
    "help": cmd_help,
}


async def handle_command(query: str, history: list) -> bool:
    """Dispatch a `/slash` command. Returns True when the input was a command
    (handled here, never forwarded to the model); False for ordinary input."""
    stripped = query.strip()
    if not stripped.startswith("/"):
        return False
    name, _, arg = stripped[1:].partition(" ")
    handler = COMMANDS.get(name.lower())
    if handler is None:
        print(f"unknown command '/{name}' (try /help)")
        return True
    await handler(arg.strip(), history)
    return True


async def repl() -> None:
    history = []
    while True:
        try:
            query = input(INPUT_PROMPT)
            print(f"[debug] query: {query!r}")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if await handle_command(query, history):
            continue
        user_prompt_received_at = datetime.now()
        user_prompt_started = time.perf_counter()
        print(
            "[debug] user_prompt_received_at="
            f"{user_prompt_received_at.isoformat(timespec='seconds')}"
        )
        history.append({
            "role": "user",
            "content": query,
        })

        state = LoopState(history)
        outcome = await agent_loop(state, PARENT_CONFIG)
        final_result_at = datetime.now()
        elapsed = time.perf_counter() - user_prompt_started
        print(
            "[debug] final_result_at="
            f"{final_result_at.isoformat(timespec='seconds')} "
            f"elapsed={elapsed:.3f}s"
        )

        if outcome.final_text:
            print(outcome.final_text)
        print()


if __name__ == "__main__":
    asyncio.run(repl())
