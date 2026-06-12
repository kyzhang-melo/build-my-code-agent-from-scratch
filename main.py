#!/usr/bin/env python3
"""main.py

Split version of the code-agent loop.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from message_utils import normalize_messages
from prompts import EXPLORE_SUBAGENT_SYSTEM, PARENT_SYSTEM
from tools import (
    EXPLORE_TOOL_REGISTRY,
    EXPLORE_TOOLS,
    TODO,
    TOOLS,
    configure_task_runner,
    execute_tool_calls,
)


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


@dataclass
class LoopState:
    # The minimal loop state: history, API call count, and nudge budget.
    messages: list
    api_call_count: int = 0
    contract_nudges: int = 0
    completion_contract_nudges: int = 0
    # The current turn's answer; None unless the turn ended on a model message
    # with no tool calls. Carried through the loop so we never reconstruct the
    # answer by scanning shared history (which can return a stale prior turn).
    final_text: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    name: str
    system: str
    tools: list[dict]
    max_api_calls: int
    registry: dict | None = None
    todo_policy: object | None = None


class TodoPlanningPolicy:
    def __init__(self, todo, max_contract_nudges: int):
        self.todo = todo
        self.max_contract_nudges = max_contract_nudges

    def handle_no_tool_calls(self, state: LoopState) -> bool:
        if not self.todo.has_active_plan() or self.todo.all_items_completed():
            state.contract_nudges = 0
            return False

        if state.contract_nudges >= self.max_contract_nudges:
            state.messages.append({
                "role": "assistant",
                "content": (
                    "Warning: Ending with unresolved todo items after repeated contract reminders.\n"
                    f"{self.todo.render()}"
                ),
            })
            return False

        state.contract_nudges += 1
        state.messages.append({
            "role": "user",
            "content": (
                "<contract>Before ending, either complete all todo items, "
                "or call todo to explicitly rewrite/remove items that are no longer needed.</contract>"
            ),
        })
        return True


TODO_POLICY = TodoPlanningPolicy(TODO, TODO_CONTRACT_MAX_NUDGES)


PARENT_CONFIG = AgentConfig(
    name="parent",
    system=PARENT_SYSTEM,
    tools=TOOLS,
    max_api_calls=MAX_API_CALLS_PER_USER_TURN,
    todo_policy=TODO_POLICY,
)

EXPLORE_SUBAGENT_CONFIG = AgentConfig(
    name="subagent:explore",
    system=EXPLORE_SUBAGENT_SYSTEM,
    tools=EXPLORE_TOOLS,
    registry=EXPLORE_TOOL_REGISTRY,
    max_api_calls=MAX_SUBAGENT_API_CALLS,
)


def build_subagent_prompt(prompt: str) -> str:
    return (
        "Mode: explore. Inspect and analyze only. Do not modify files.\n\n"
        f"Task:\n{prompt}"
    )


def handle_subagent_no_tool_calls(
    state: LoopState, config: AgentConfig, final_text: str
) -> bool:
    if len(final_text) >= SUMMARY_MIN_LENGTH:
        state.completion_contract_nudges = 0
        return False

    if state.completion_contract_nudges >= SUMMARY_CONTINUATION_ATTEMPTS:
        state.completion_contract_nudges = 0
        return False

    state.completion_contract_nudges += 1
    state.messages.append({
        "role": "user",
        "content": SUMMARY_CONTINUATION_PROMPT,
    })
    return True


def execute_configured_tool_calls(tool_calls, config: AgentConfig) -> tuple[list[dict], bool]:
    if config.registry is None:
        return execute_tool_calls(tool_calls)
    return execute_tool_calls(tool_calls, config.registry)


def run_one_turn(state: LoopState, config: AgentConfig = PARENT_CONFIG) -> bool:
    if state.api_call_count >= config.max_api_calls:
        warning = f"Warning: stopped after max_api_calls={config.max_api_calls}."
        state.messages.append({
            "role": "assistant",
            "content": warning,
        })
        state.final_text = warning
        return False

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
    response = client.responses.create(
        model=MODEL_ID,
        instructions=config.system,
        input=input_messages,
        tools=config.tools,
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
        # The current turn's text, evaluated directly -- never scanned from
        # history, so a prior turn's message can't leak in.
        response_text = (response.output_text or "").strip()
        if config.todo_policy is None:
            should_continue = handle_subagent_no_tool_calls(state, config, response_text)
        else:
            should_continue = config.todo_policy.handle_no_tool_calls(state)
        if not should_continue:
            # Turn ends on a model message with no tool calls: this is the answer.
            state.final_text = response_text
        return should_continue

    results, _ = execute_configured_tool_calls(tool_calls, config)
    if not results:
        return False

    state.messages.extend(results)
    state.contract_nudges = 0
    state.completion_contract_nudges = 0
    return True


def agent_loop(state: LoopState, config: AgentConfig = PARENT_CONFIG) -> None:
    while run_one_turn(state, config):
        pass


def run_subagent(prompt: str, description: str = "exploration") -> str:
    print(f"\033[35m> task (explore/{description}): {prompt[:120]}\033[0m")
    state = LoopState(messages=[{
        "role": "user",
        "content": build_subagent_prompt(prompt),
    }])
    agent_loop(state, EXPLORE_SUBAGENT_CONFIG)

    if state.final_text:
        return state.final_text
    return f"[subagent produced no output after {state.api_call_count} turns]"


configure_task_runner(run_subagent)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input(INPUT_PROMPT)
            print(f"[debug] query: {query!r}")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({
            "role": "user",
            "content": query,
        })

        state = LoopState(history)
        agent_loop(state, PARENT_CONFIG)

        if state.final_text:
            print(state.final_text)
        print()
