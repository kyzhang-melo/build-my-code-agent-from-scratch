#!/usr/bin/env python3
"""main.py

Split version of the code-agent loop.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from prompts import SYSTEM
from tools import TOOLS, execute_tool_calls


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


@dataclass
class LoopState:
    # The minimal loop state: history, loop count, and why we continue.
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None


def extract_text(content) -> str:
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


def run_one_turn(state: LoopState) -> bool:
    response = client.responses.create(
        model=MODEL_ID,
        instructions=SYSTEM,
        input=state.messages,
        tools=TOOLS,
        max_output_tokens=8000,
    )
    for item in response.output:
        if item.type == "function_call":
            state.messages.append({
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            })

    if response.output_text:
        state.messages.append({
            "role": "assistant",
            "content": response.output_text,
        })

    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        state.transition_reason = None
        return False

    results = execute_tool_calls(response.output)
    if not results:
        state.transition_reason = None
        return False
    state.messages.extend(results)

    state.turn_count += 1
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

        final_text = extract_text(state.messages[-1]["content"])
        if final_text:
            print(final_text)
        print()
