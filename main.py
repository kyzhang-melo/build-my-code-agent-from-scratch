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
MAX_API_CALLS_PER_USER_TURN = 30


@dataclass
class LoopState:
    # The minimal loop state: history, API call count, and why we continue.
    messages: list
    api_call_count: int = 0
    transition_reason: str | None = None
    contract_nudges: int = 0
    todo_rewrite_ack_pending: bool = False


class TodoPlanningPolicy:
    def __init__(self, todo, max_contract_nudges: int):
        self.todo = todo
        self.max_contract_nudges = max_contract_nudges

    def handle_no_tool_calls(self, state: LoopState) -> bool:
        if not self.todo.has_active_plan() or self.todo.all_items_completed():
            state.contract_nudges = 0
            state.todo_rewrite_ack_pending = False
            state.transition_reason = None
            return False

        if state.contract_nudges >= self.max_contract_nudges:
            state.messages.append({
                "role": "assistant",
                "content": (
                    "Warning: Ending with unresolved todo items after repeated contract reminders.\n"
                    f"{self.todo.render()}"
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

    def before_tool_calls(self) -> tuple[tuple[str, str, str], ...]:
        return self.todo.snapshot_signature()

    def after_tool_calls(
        self,
        state: LoopState,
        *,
        used_todo: bool,
        signature_before: tuple[tuple[str, str, str], ...],
        signature_after: tuple[tuple[str, str, str], ...],
    ) -> None:
        was_contract_nudge_response = state.transition_reason == "todo_contract_nudge"

        if used_todo:
            self.todo.state.rounds_since_update = 0
            if was_contract_nudge_response and signature_before != signature_after:
                state.todo_rewrite_ack_pending = True
        else:
            self.todo.note_round_without_update()
            reminder = self.todo.reminder()
            if reminder:
                state.messages.append({
                    "role": "user",
                    "content": reminder,
                })

        if not self.todo.has_active_plan() or self.todo.all_items_completed():
            state.todo_rewrite_ack_pending = False
        state.contract_nudges = 0


TODO_POLICY = TodoPlanningPolicy(TODO, TODO_CONTRACT_MAX_NUDGES)


def run_one_turn(state: LoopState) -> bool:
    if state.api_call_count >= MAX_API_CALLS_PER_USER_TURN:
        state.messages.append({
            "role": "assistant",
            "content": (
                "Warning: stopped after "
                f"MAX_API_CALLS_PER_USER_TURN={MAX_API_CALLS_PER_USER_TURN}."
            ),
        })
        state.transition_reason = None
        return False

    state.api_call_count += 1
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
        return TODO_POLICY.handle_no_tool_calls(state)

    todo_signature_before = TODO_POLICY.before_tool_calls()
    results, used_todo = execute_tool_calls(tool_calls)
    if not results:
        state.transition_reason = None
        return False

    todo_signature_after = TODO.snapshot_signature()
    state.messages.extend(results)
    TODO_POLICY.after_tool_calls(
        state,
        used_todo=used_todo,
        signature_before=todo_signature_before,
        signature_after=todo_signature_after,
    )

    state.transition_reason = "function_call_output"
    return True


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
