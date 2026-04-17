#!/usr/bin/env python3
"""main.py

Split version of the code-agent loop.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from message_utils import extract_text, normalize_messages
from prompts import SYSTEM
from tools import TODO, TOOLS, execute_tool_calls


load_dotenv(override=True)
print("[init] .env loaded")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL_ID = os.getenv("MODEL_ID")

print(f"[init] MODEL_ID={MODEL_ID!r}")
print(f"[init] OPENROUTER_BASE_URL={OPENROUTER_BASE_URL!r}")
print(f"[init] OPENROUTER_API_KEY present={bool(OPENROUTER_API_KEY)}")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Please set it in .env")
if not OPENROUTER_BASE_URL:
    raise RuntimeError("OPENROUTER_BASE_URL is not set. Please set it in .env")
if not MODEL_ID:
    raise RuntimeError("MODEL_ID is not set. Please set it in .env")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)
print("[init] OpenAI client initialized")
TODO_CONTRACT_MAX_NUDGES = 2


@dataclass
class LoopState:
    # The minimal loop state: history, loop count, and why we continue.
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None
    contract_nudges: int = 0
    todo_rewrite_ack_pending: bool = False


def handle_no_tool_calls(state: LoopState) -> bool:
    if not TODO.has_active_plan() or TODO.all_items_completed() or state.todo_rewrite_ack_pending:
        state.contract_nudges = 0
        state.todo_rewrite_ack_pending = False
        state.transition_reason = None
        return False

    if state.contract_nudges >= TODO_CONTRACT_MAX_NUDGES:
        state.messages.append({
            "role": "assistant",
            "content": (
                "Warning: Ending with unresolved todo items after repeated contract reminders.\n"
                f"{TODO.render()}"
            ),
        })
        state.todo_rewrite_ack_pending = False
        state.transition_reason = None
        return False

    state.contract_nudges += 1
    state.messages.append({
        "role": "user",
        "content": (
            "<contract>Before ending, either complete all todo items, "
            "or call todo to explicitly rewrite/remove items that are no longer needed.</contract>"
        ),
    })
    state.transition_reason = "todo_contract_nudge"
    return True


def handle_tool_calls(state: LoopState, response_output) -> bool:
    todo_signature_before = TODO.snapshot_signature()
    results, used_todo = execute_tool_calls(response_output)
    if not results:
        state.transition_reason = None
        return False

    todo_signature_after = TODO.snapshot_signature()
    state.messages.extend(results)
    if used_todo:
        TODO.state.rounds_since_update = 0
        if todo_signature_before != todo_signature_after:
            state.todo_rewrite_ack_pending = True
    else:
        TODO.note_round_without_update()
        reminder = TODO.reminder()
        if reminder:
            state.messages.append({
                "role": "user",
                "content": reminder,
            })

    if not TODO.has_active_plan() or TODO.all_items_completed():
        state.todo_rewrite_ack_pending = False
    state.contract_nudges = 0

    state.turn_count += 1
    state.transition_reason = "function_call_output"
    return True


def run_one_turn(state: LoopState) -> bool:
    response = client.responses.create(
        model=MODEL_ID,
        instructions=SYSTEM,
        input=normalize_messages(state.messages),
        tools=TOOLS,
        max_output_tokens=8000,
    )

    if response.output_text:
        state.messages.append({
            "role": "assistant",
            "content": response.output_text,
        })
    
    tool_calls = []
    for item in response.output:
        if item.type == "function_call":
            state.messages.append({
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            })

            tool_calls.append(item)

    if not tool_calls:
        return handle_no_tool_calls(state)

    return handle_tool_calls(state, tool_calls)


def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({
            "role": "user",
            "content": query,
        })

        state = LoopState(history)
        agent_loop(state)

        final_text = extract_text(state.messages)
        if final_text:
            print(final_text)
        print()
