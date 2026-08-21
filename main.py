#!/usr/bin/env python3
"""main.py

Split version of the code-agent loop.
"""

import asyncio
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from message_utils import extract_usage, normalize_messages, response_item_to_dict
from permissions import PermissionManager, PermissionMode, PermissionService, TerminalApprovalHandler
from prompts import GLOB_DISCOVERY_RULES, build_explore_system, build_parent_system
from sandbox import LocalSandbox, Sandbox
from session import (
    AgentSession,
    ReportStopGate,
    TodoStopGate,
    TurnSteeringPolicy,
    generate_session_id,
)
from session_store import (
    NullSessionStore,
    SESSION_DIR_ENV,
    SessionStore,
    SessionStoreProtocol,
    find_most_recent_session,
    get_default_session_dir,
    list_session_headers,
)
from tools import (
    EXPLORE_TOOLS,
    READ_ONLY_TOOL_NAMES,
    TOOLS,
    TodoManager,
    TodoParams,
    build_tool_registry,
    execute_tool_calls_async,
    select_tool_schemas,
)
from terminal_input import TerminalInput
from trace import TraceContext, emit_trace
from workspace import Workspace
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
MAX_API_CALLS_PER_USER_TURN = 0  # 0 means unlimited
MAX_SUBAGENT_API_CALLS = 0  # 0 means unlimited
INPUT_PROMPT = FormattedText([("class:input-prompt", "s01 >> ")])
INPUT_STYLE = Style.from_dict({"input-prompt": "ansicyan"})

# --- Context compaction config ---
# Per-model limits, resolved by PREFIX match against a normalized
# model id (most-specific pattern first, first match wins). Normalizing first
# (drop the "vendor/" prefix and any ":route" suffix like ":exacto"/":nitro")
# means routing/quant tags can't defeat the lookup the way an exact-key dict
# does — e.g. "moonshotai/kimi-k2.5:exacto" still resolves to kimi's 262144
# instead of silently falling through to DEFAULT_CONTEXT_WINDOW. Unknown models
# fall back to a conservative default so they compact early rather than overflow.
@dataclass(frozen=True)
class ModelLimits:
    context_window_tokens: int
    max_input_tokens: int
    max_output_tokens: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "context_window_tokens": self.context_window_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
        }


def _limits(context_window_tokens: int, max_output_tokens: int | None = None) -> ModelLimits:
    return ModelLimits(context_window_tokens, context_window_tokens, max_output_tokens)


MODEL_LIMIT_PATTERNS: list[tuple[re.Pattern[str], ModelLimits]] = [
    # TokenHub documents hy3's maximum input separately from its total window.
    (re.compile(r"^hy3$"), ModelLimits(262_144, 192_000, 128_000)),
    (re.compile(r"^kimi-"), _limits(262_144)),
    (re.compile(r"^deepseek-v4"), _limits(1_000_000)),
    (re.compile(r"^minimax-m3"), _limits(524_288)),
    (re.compile(r"^glm-5\.2"), _limits(1_000_000)),
    (re.compile(r"^glm-5"), _limits(202_800)),
    (re.compile(r"^nemotron-3-ultra"), _limits(1_048_576)),
    (re.compile(r"^laguna-m"), _limits(262_144)),
    (re.compile(r"^longcat-"), _limits(1_048_576)),
]
DEFAULT_MODEL_LIMITS = _limits(32_000)
DEFAULT_CONTEXT_WINDOW = DEFAULT_MODEL_LIMITS.context_window_tokens
# Full JSON override for a single run. It replaces the model catalog result;
# partial overrides are rejected so related limits cannot become inconsistent.
MODEL_LIMITS_OVERRIDE = os.getenv("MODEL_LIMITS_OVERRIDE", "")
DEFAULT_MAX_OUTPUT_TOKENS: int | None = None
AUTO_MAX_OUTPUT_TOKEN_RESERVATION = 32768
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
RESERVED_OVERHEAD_TOKENS = 4000    # system prompt + tool schemas
COMPACT_TRIGGER_RATIO = 0.85
# On by default; AUTO_COMPACT=0 disables automatic compaction (manual /compact
# still works). The destructive rewrite is backed by a .transcripts/ snapshot.
AUTO_COMPACT_ENABLED = os.getenv("AUTO_COMPACT", "1") != "0"


def create_terminal_input() -> TerminalInput:
    """Create one terminal input owner for a CLI run."""
    return TerminalInput(style=INPUT_STYLE)


def normalize_model_id(model_id: str) -> str:
    """vendor/model:route -> model  (moonshotai/kimi-k2.5:exacto -> kimi-k2.5)."""
    s = model_id.lower().strip()
    s = s.rsplit("/", 1)[-1]   # drop "vendor/" prefix
    s = s.split(":", 1)[0]     # drop ":route"/":quant" suffix
    return s


def _parse_model_limits_override(raw: str) -> ModelLimits | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("MODEL_LIMITS_OVERRIDE must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("MODEL_LIMITS_OVERRIDE must be a JSON object")
    required = {"context_window_tokens", "max_input_tokens", "max_output_tokens"}
    if set(value) != required:
        raise ValueError(
            "MODEL_LIMITS_OVERRIDE must contain exactly: "
            "context_window_tokens, max_input_tokens, max_output_tokens"
        )
    context_window_tokens = value["context_window_tokens"]
    max_input_tokens = value["max_input_tokens"]
    max_output_tokens = value["max_output_tokens"]
    if (
        isinstance(context_window_tokens, bool)
        or not isinstance(context_window_tokens, int)
        or context_window_tokens <= 0
        or isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or max_input_tokens <= 0
        or max_input_tokens > context_window_tokens
        or (
            max_output_tokens is not None
            and (isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int)
                 or max_output_tokens <= 0)
        )
    ):
        raise ValueError(
            "MODEL_LIMITS_OVERRIDE requires positive integer context_window_tokens "
            "and max_input_tokens (max_input_tokens <= context_window_tokens), "
            "and a positive integer or null max_output_tokens"
        )
    return ModelLimits(context_window_tokens, max_input_tokens, max_output_tokens)


def model_limits(model_id: str | None = None) -> ModelLimits:
    override = _parse_model_limits_override(MODEL_LIMITS_OVERRIDE)
    if override is not None:
        return override
    norm = normalize_model_id(model_id if model_id is not None else MODEL_ID)
    for pattern, limits in MODEL_LIMIT_PATTERNS:
        if pattern.match(norm):
            return limits
    return DEFAULT_MODEL_LIMITS


def context_window() -> int:
    """Compatibility helper for callers that only need the total window."""
    return model_limits().context_window_tokens


def output_token_reservation(max_output_tokens: int | None) -> int:
    limits = model_limits()
    return (
        min(
            AUTO_MAX_OUTPUT_TOKEN_RESERVATION,
            limits.context_window_tokens // 2,
            limits.max_output_tokens or AUTO_MAX_OUTPUT_TOKEN_RESERVATION,
        )
        if max_output_tokens is None
        else max_output_tokens
    )


def input_budget(
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
) -> int:
    # Tokens available for input once the response reservation and fixed overhead
    # are carved out. The trigger ratio applies to THIS, not the raw window, so
    # the output reservation can't be eaten away on small-context models.
    limits = model_limits()
    return min(
        limits.max_input_tokens,
        limits.context_window_tokens - output_token_reservation(max_output_tokens),
    ) - RESERVED_OVERHEAD_TOKENS


def should_auto_compact(
    state: "LoopState",
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
) -> bool:
    # API usage describes the previous request. New tool results and steering
    # directives are not represented there, so also estimate the payload that
    # would be sent on the next request.
    current_estimate = estimate_tokens(normalize_messages(state.messages))
    return max(state.last_input_tokens, current_estimate) >= (
        COMPACT_TRIGGER_RATIO * input_budget(max_output_tokens)
    )


def validate_generation_config(
    reasoning_effort: str | None,
    max_output_tokens: int | None,
) -> None:
    if (
        reasoning_effort is not None
        and reasoning_effort not in REASONING_EFFORTS
    ):
        raise ValueError(f"unsupported reasoning_effort: {reasoning_effort}")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    known_max_output = model_limits().max_output_tokens
    if (
        max_output_tokens is not None
        and known_max_output is not None
        and max_output_tokens > known_max_output
    ):
        raise ValueError(
            f"max_output_tokens={max_output_tokens} exceeds this model's "
            f"maximum of {known_max_output}"
        )
    if input_budget(max_output_tokens) <= 0:
        raise ValueError(
            "max_output_tokens plus reserved overhead must be smaller than "
            f"the {context_window()}-token context window"
        )


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


def llm_request_failure_details(exc: BaseException) -> dict[str, str | int]:
    """Return safe, actionable metadata for a failed Responses API request."""
    details: dict[str, str | int] = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    if isinstance(exc, json.JSONDecodeError):
        # Do not write a malformed provider body to logs or trace files. Its
        # size and digest still let us correlate repeated bad responses with a
        # provider while preserving the agent's potentially sensitive context.
        document = exc.doc
        details.update({
            "json_error_line": exc.lineno,
            "json_error_column": exc.colno,
            "json_error_position": exc.pos,
            "json_document_chars": len(document),
            "json_document_sha256": hashlib.sha256(
                document.encode("utf-8", "replace")
            ).hexdigest(),
        })

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            details["http_status_code"] = status_code
        headers = getattr(response, "headers", None)
        if headers is not None:
            for header_name, field_name in (
                ("x-request-id", "request_id"),
                ("x-openrouter-request-id", "openrouter_request_id"),
                ("cf-ray", "cloudflare_ray"),
                ("content-type", "response_content_type"),
                ("content-length", "response_content_length"),
            ):
                value = headers.get(header_name)
                if value:
                    details[field_name] = str(value)
    return details


def create_explore_session(
    workspace: Workspace,
    permission_service: PermissionService,
    trace_context: TraceContext,
    session_id: str,
    *,
    sandbox: Sandbox | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
) -> AgentSession:
    """Create an isolated read-only exploration session."""
    validate_generation_config(reasoning_effort, max_output_tokens)
    todo = TodoManager()
    runtime = sandbox or LocalSandbox(workspace)
    child_trace = replace(trace_context, agent_id="subagent:explore")
    return AgentSession(
        name="subagent:explore",
        session_id=session_id,
        workspace=workspace,
        sandbox=runtime,
        todo=todo,
        system=build_explore_system(workspace.root),
        tools=EXPLORE_TOOLS,
        registry=build_tool_registry(
            workspace, todo, READ_ONLY_TOOL_NAMES, file_backend=runtime.file_backend,
            command_runner=runtime.command_runner,
        ),
        permission_service=permission_service,
        permission_source="subagent:explore",
        trace_context=child_trace,
        max_api_calls=MAX_SUBAGENT_API_CALLS,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        stop_gate=ReportStopGate(
            SUMMARY_MIN_LENGTH,
            SUMMARY_CONTINUATION_ATTEMPTS,
            SUMMARY_CONTINUATION_PROMPT,
        ),
    )


def create_parent_session(
    workdir: str | Path,
    *,
    approval_handler,
    trace_context: TraceContext | None = None,
    on_text: Callable[[str], None] | None = emit_assistant_text,
    session_id: str | None = None,
    store: SessionStoreProtocol | None = None,
    session_dir: Path | None = None,
    max_api_calls: int = MAX_API_CALLS_PER_USER_TURN,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
    steering_policy: TurnSteeringPolicy | None = None,
    system_addendum: str | None = None,
    tool_names: set[str] | frozenset[str] | None = None,
    sandbox: Sandbox | None = None,
) -> AgentSession:
    """Build one fully isolated parent-agent session."""
    validate_generation_config(reasoning_effort, max_output_tokens)
    workspace = Workspace(Path(workdir))
    runtime = sandbox or LocalSandbox(workspace)
    todo = TodoManager()
    permission_service = PermissionService(
        manager=PermissionManager(workspace.root),
        handler=approval_handler,
    )
    sid = session_id or generate_session_id()
    trace = trace_context or TraceContext()
    # Only fill run_id when the caller did not set one (e.g. evals pass their
    # own run_id for attribution). This keeps subagent traces under the same
    # run_id as the parent via replace(trace_context, agent_id=...).
    if not trace.run_id:
        trace = replace(trace, run_id=sid)
    session_store = store if store is not None else NullSessionStore()

    async def task_runner(prompt: str, description: str) -> str:
        explore = create_explore_session(
            workspace,
            permission_service,
            trace,
            sid,
            sandbox=runtime,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        return await run_subagent(prompt, description, explore)

    system = build_parent_system(workspace.root)
    if system_addendum:
        system = f"{system}\n\n{system_addendum.strip()}"
    selected_names = (
        frozenset(tool["name"] for tool in TOOLS)
        if tool_names is None
        else frozenset(tool_names)
    )

    return AgentSession(
        name="parent",
        session_id=sid,
        workspace=workspace,
        sandbox=runtime,
        todo=todo,
        system=system,
        tools=select_tool_schemas(selected_names),
        registry=build_tool_registry(
            workspace,
            todo,
            selected_names,
            task_runner=task_runner,
            file_backend=runtime.file_backend,
            command_runner=runtime.command_runner,
        ),
        permission_service=permission_service,
        permission_source="parent",
        trace_context=trace,
        max_api_calls=max_api_calls,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        stop_gate=TodoStopGate(todo, TODO_CONTRACT_MAX_NUDGES),
        steering_policy=steering_policy,
        store=session_store,
        on_text=on_text,
        session_dir=session_dir,
    )


def build_subagent_prompt(prompt: str) -> str:
    return (
        "Mode: explore. Inspect and analyze only. Do not modify files.\n\n"
        f"{GLOB_DISCOVERY_RULES}\n\n"
        f"Task:\n{prompt}"
    )


async def execute_configured_tool_calls(
    tool_calls,
    session: AgentSession,
    api_call: int | None = None,
) -> tuple[list[dict], bool]:
    return await execute_tool_calls_async(
        tool_calls,
        session.registry,
        session.todo,
        permission_service=session.permission_service,
        permission_source=session.permission_source,
        trace_context=session.trace_context,
        api_call=api_call,
    )


async def run_one_turn(state: LoopState, session: AgentSession) -> StepOutcome | None:
    # Returns None to keep looping, or a StepOutcome when the turn ends.
    if session.max_api_calls and state.api_call_count >= session.max_api_calls:
        warning = f"Warning: stopped after max_api_calls={session.max_api_calls}."
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
    request = dict(
        model=MODEL_ID,
        instructions=session.system,
        input=input_messages,
        tools=session.tools,
        extra_body=PROVIDER_EXTRA_BODY,
    )
    if session.max_output_tokens is not None:
        request["max_output_tokens"] = session.max_output_tokens
    if session.reasoning_effort is not None:
        request["reasoning"] = {"effort": session.reasoning_effort}
    try:
        response = await client.responses.create(**request)
    except Exception as exc:
        details = llm_request_failure_details(exc)
        details.update({
            "api_call": state.api_call_count,
            "input_message_count": len(input_messages),
            "input_chars": len(json.dumps(input_messages, default=str)),
            "model": MODEL_ID or "",
            "provider": OPENROUTER_PROVIDER or "default",
        })
        print(f"[debug] LLM request failed: {json.dumps(details, sort_keys=True)}")
        emit_trace(session.trace_context, "llm.request_failed", **details)
        raise
    debug_empty_output_text_response(response)

    # Track input-token load for the auto-compaction trigger. Prefer the API's
    # reported usage; fall back to a char-based estimate if it's absent.
    usage = getattr(response, "usage", None)
    reported = getattr(usage, "input_tokens", None) if usage is not None else None
    state.last_input_tokens = (
        reported if reported is not None
        else len(json.dumps(input_messages, default=str)) // 4
    )

    # Record what the call actually cost. This is observational only: the
    # estimate above still drives compaction, and no estimated value is ever
    # reported as usage.
    emit_trace(
        session.trace_context,
        "llm.usage",
        kind="turn",
        api_call=state.api_call_count,
        model=MODEL_ID or "",
        configured_provider=OPENROUTER_PROVIDER or "default",
        **extract_usage(response),
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
            raw_arguments = _response_item_attr(item, "arguments", "{}")
            # Provider-facing history must only contain valid JSON arguments;
            # a malformed function_call replayed to the Provider causes a 400
            # that kills the agent loop before the model ever sees the tool
            # error output. Normalize here: the tool layer still receives the
            # original SDK item (via tool_calls below) so it can report the
            # exact parse error; only the replayed history is sanitized.
            try:
                parsed = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                replay_arguments = "{}"
                malformed = True
            else:
                replay_arguments = json.dumps(parsed)
                malformed = False
            function_call = {
                "type": "function_call",
                "call_id": _response_item_attr(item, "call_id", ""),
                "name": _response_item_attr(item, "name", ""),
                "arguments": replay_arguments,
            }
            # Drop the Provider-assigned item id when arguments were malformed.
            # The id may be paired with the original malformed record on the
            # Provider side; replaying it with sanitized arguments can violate
            # pairing constraints. call_id is retained so the function_call
            # still pairs with its function_call_output.
            if not malformed:
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
        nudge = session.stop_gate.check(response_text)
        if nudge is None:
            gate_decision = "allow"
            gate_reason = "requirements_satisfied"
            gate_nudge_count = state.nudges
        elif state.nudges < session.stop_gate.max_nudges:
            gate_decision = "block"
            gate_reason = (
                "unresolved_todos"
                if session.stop_gate.name == "todo"
                else "response_too_brief"
            )
            gate_nudge_count = state.nudges + 1
        else:
            gate_decision = "give_up"
            gate_reason = (
                "unresolved_todos"
                if session.stop_gate.name == "todo"
                else "nudge_budget_exhausted"
            )
            gate_nudge_count = state.nudges
        emit_trace(
            session.trace_context,
            "stop_gate.checked",
            source=session.permission_source,
            gate=session.stop_gate.name,
            decision=gate_decision,
            nudge_count=gate_nudge_count,
            reason=gate_reason,
        )
        if nudge is not None and state.nudges < session.stop_gate.max_nudges:
            # The loop continues, so this text won't become final_text and would
            # otherwise be lost. Surface the rejected answer before nudging.
            if response_text and session.on_text is not None:
                session.on_text(response_text)
            state.nudges += 1
            state.messages.append({"role": "user", "content": nudge})
            return None

        if nudge is not None:
            # Nudge budget exhausted: accept the stop anyway.
            note = session.stop_gate.give_up_note()
            if note is not None:
                state.messages.append({"role": "assistant", "content": note})
        state.nudges = 0
        # Turn ends on a model message with no tool calls: this is the answer.
        return StepOutcome(stop_reason="completed", final_text=response_text)

    state.empty_response_nudges = 0
    # Surface assistant text that accompanies tool calls; otherwise it would be
    # retained in history but never shown, swallowing real deliverables.
    if output_text and session.on_text is not None:
        session.on_text(output_text)

    results, _ = await execute_configured_tool_calls(tool_calls, session, api_call=state.api_call_count)
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


async def agent_loop(state: LoopState, session: AgentSession) -> TurnOutcome:
    while (outcome := await run_one_turn(state, session)) is None:
        if session.steering_policy is not None:
            try:
                directive = session.steering_policy.after_turn(
                    state.api_call_count
                )
            except Exception as exc:  # noqa: BLE001 - optional policy is best effort
                emit_trace(
                    session.trace_context,
                    "steering.failed",
                    source=session.permission_source,
                    policy=session.steering_policy.name,
                    api_call=state.api_call_count,
                    error_type=type(exc).__name__,
                )
            else:
                if directive is not None:
                    state.messages.append({
                        "role": "user",
                        "content": directive.content,
                    })
                    emit_trace(
                        session.trace_context,
                        "steering.injected",
                        source=session.permission_source,
                        policy=session.steering_policy.name,
                        api_call=state.api_call_count,
                        reason=directive.reason,
                    )
        # This is a clean turn boundary: tool outputs and any steering directive
        # have already been appended, so history is well-formed and represents
        # the next request exactly enough for a conservative preflight check.
        if AUTO_COMPACT_ENABLED and should_auto_compact(
            state, session.max_output_tokens
        ):
            result = await compact_history_async(
                state, source="auto", focus=None, client=client, model=MODEL_ID,
                extra_body=PROVIDER_EXTRA_BODY, todo=session.todo,
                trace_context=session.trace_context,
            )
            if result is not None:
                print(
                    f"[compact] auto-compacted: ~{result.tokens_before} -> "
                    f"~{result.tokens_after} tokens (backup: {result.transcript_path})"
                )
                # If even a fresh summary still doesn't fit, the surviving user
                # messages are the bulk: drop the oldest until we fit, rather
                # than re-summarizing on every subsequent turn.
                while should_auto_compact(
                    state, session.max_output_tokens
                ) and _drop_oldest_user_message(state):
                    state.last_input_tokens = estimate_tokens(state.messages)
    return TurnOutcome(
        stop_reason=outcome.stop_reason,
        final_text=outcome.final_text,
        api_calls=state.api_call_count,
    )


async def run_subagent(
    prompt: str,
    description: str,
    session: AgentSession,
) -> str:
    print(f"\033[35m> task (explore/{description}): {prompt[:120]}\033[0m")
    state = LoopState(messages=[{
        "role": "user",
        "content": build_subagent_prompt(prompt),
    }])
    outcome = await agent_loop(state, session)

    if outcome.final_text:
        return outcome.final_text
    return f"[subagent produced no output after {outcome.api_calls} turns]"


async def cmd_compact(arg: str, history: list, session: AgentSession) -> None:
    state = LoopState(history)
    result = await compact_history_async(
        state, source="manual", focus=arg or None, client=client, model=MODEL_ID,
        extra_body=PROVIDER_EXTRA_BODY, todo=session.todo,
        trace_context=session.trace_context,
    )
    if result is None:
        print("[compact] nothing to compact yet.")
        return
    history[:] = state.messages
    print(
        f"[compact] context compacted: ~{result.tokens_before} -> "
        f"~{result.tokens_after} tokens (backup: {result.transcript_path})"
    )


async def cmd_help(arg: str, history: list, session: AgentSession) -> None:
    del arg, history, session
    print(
        "commands: /compact [focus]  |  /mode <default|plan>  |  "
        "/permissions  |  /sessions  |  /help   (q or exit to quit)"
    )


async def cmd_mode(arg: str, history: list, session: AgentSession) -> None:
    del history
    value = arg.strip().lower()
    if value not in {mode.value for mode in PermissionMode}:
        print("usage: /mode <default|plan>")
        return
    session.permission_service.manager.set_mode(value)
    print(f"[permissions] mode={session.permission_service.manager.mode.value}")


async def cmd_permissions(arg: str, history: list, session: AgentSession) -> None:
    del arg, history
    manager = session.permission_service.manager
    print(f"[permissions] mode={manager.mode.value}")
    paths = sorted(
        action.removeprefix("file_mutation:")
        for action in manager.auto_approve_actions
        if action.startswith("file_mutation:")
    )
    if not paths:
        print("[permissions] session-approved paths: none")
    else:
        print("[permissions] session-approved paths:")
        for path in paths:
            print(f"  {path}")
    commands = sorted(
        action.removeprefix("bash:")
        for action in manager.auto_approve_actions
        if action.startswith("bash:")
    )
    if not commands:
        print("[permissions] session-approved bash commands: none")
        return
    print("[permissions] session-approved bash commands:")
    for command in commands:
        print(f"  {command}")


async def cmd_sessions(arg: str, history: list, session: AgentSession) -> None:
    del arg, history
    if session.session_dir is None:
        print("[sessions] no session directory is configured")
        return
    sessions_dir = session.session_dir
    headers = list_session_headers(sessions_dir, cwd=session.workspace.root)
    if not headers:
        print("[sessions] no saved sessions in this workspace")
        return
    print(f"[sessions] {len(headers)} session(s) in {sessions_dir}:")
    for h in headers:
        sid = h.get("session_id", "?")
        display = h.get("session_name") or sid
        updated = (h.get("updated_at") or h.get("created_at", "?"))[:19]
        model = h.get("model_id", "?")
        print(f"  {display}  id={sid}  updated={updated}  model={model}")


COMMANDS = {
    "compact": cmd_compact,
    "help": cmd_help,
    "mode": cmd_mode,
    "permissions": cmd_permissions,
    "sessions": cmd_sessions,
}


async def handle_command(query: str, history: list, session: AgentSession) -> bool:
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
    await handler(arg.strip(), history, session)
    return True


def _resolve_session_path(
    session_arg: str,
    sessions_dir: Path,
    cwd: Path,
) -> Path | None:
    """Resolve a --resume argument to a session file path.

    Accepts a bare session id (looked up in the configured session directory),
    a relative path, or an absolute path.
    """
    # Bare session id: look for <id>.jsonl in the configured directory.
    candidate = sessions_dir / f"{session_arg}.jsonl"
    if candidate.is_file():
        headers = list_session_headers(sessions_dir, cwd=cwd)
        if any(Path(str(header["_path"])) == candidate for header in headers):
            return candidate
    # Human-readable session name (case-insensitive within this workspace).
    folded = session_arg.casefold()
    matches = [
        Path(str(header["_path"]))
        for header in list_session_headers(sessions_dir, cwd=cwd)
        if str(header.get("session_name", "")).casefold() == folded
    ]
    if len(matches) == 1:
        return matches[0]
    # Relative or absolute path
    path = Path(session_arg).resolve()
    if path.is_file():
        return path
    return None


def _print_session_list(sessions_dir: Path, cwd: Path) -> None:
    headers = list_session_headers(sessions_dir, cwd=cwd)
    if not headers:
        print("No saved sessions in this workspace.")
        return
    print(f"Sessions in {sessions_dir}:")
    for h in headers:
        sid = h.get("session_id", "?")
        display = h.get("session_name") or sid
        updated = (h.get("updated_at") or h.get("created_at", "?"))[:19]
        model = h.get("model_id", "?")
        print(f"  {display}  id={sid}  updated={updated}  model={model}")


def _print_resume_diagnostics(store: SessionStoreProtocol) -> None:
    diagnostics = store.resume_diagnostics
    details = [
        ("reasoning", diagnostics.dropped_reasoning),
        ("function_call ids", diagnostics.stripped_function_call_ids),
        ("orphan calls", diagnostics.dropped_orphan_calls),
        ("orphan outputs", diagnostics.dropped_orphan_outputs),
        ("invalid JSONL lines", diagnostics.ignored_invalid_lines),
    ]
    changed = [f"{label}={count}" for label, count in details if count]
    if changed:
        print(f"[session] resume sanitized: {', '.join(changed)}")


def _resolve_sessions_dir(
    workspace: Workspace,
    override: str | Path | None,
) -> Path:
    configured = override or os.getenv(SESSION_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return get_default_session_dir(workspace.root)


async def repl(
    *,
    resume: str | None = None,
    continue_recent: bool = False,
    list_only: bool = False,
    no_session: bool = False,
    session_name: str | None = None,
    session_dir: str | Path | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
) -> None:
    cwd = Path.cwd()
    workspace = Workspace(cwd)
    sessions_dir = _resolve_sessions_dir(workspace, session_dir)

    if list_only:
        if session_name:
            print("Error: --name cannot be combined with --list-sessions")
            return
        _print_session_list(sessions_dir, workspace.root)
        return

    # Determine session store: resume, continue, new, or null.
    store: SessionStoreProtocol = NullSessionStore()
    resumed_history: list[dict] = []
    session_id = generate_session_id()

    if no_session:
        if session_name:
            print("Error: --name requires session persistence")
            return
        print("[session] persistence disabled (--no-session)")
    elif resume is not None:
        if session_name:
            print("Error: --name cannot be combined with --resume")
            return
        path = _resolve_session_path(resume, sessions_dir, workspace.root)
        if path is None:
            print(f"Error: session not found: {resume}")
            return
        try:
            store = SessionStore.open(
                path,
                workspace,
                MODEL_ID,
                OPENROUTER_PROVIDER or "",
                acquire_lock=True,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        resumed_history = store.messages()
        session_id = store.session_id
        print(f"[session] resumed {session_id} ({len(resumed_history)} messages)")
        _print_resume_diagnostics(store)
        store.sync(resumed_history)  # re-sync to capture sanitized state
    elif continue_recent:
        if session_name:
            print("Error: --name cannot be combined with --continue")
            return
        path = find_most_recent_session(sessions_dir, workspace.root)
        if path is None:
            print("[session] no recent session found; starting new")
            store = SessionStore.create(
                workspace,
                session_id,
                MODEL_ID,
                OPENROUTER_PROVIDER or "",
                session_dir=sessions_dir,
                acquire_lock=True,
            )
            session_id = store.session_id
        else:
            try:
                store = SessionStore.open(
                    path,
                    workspace,
                    MODEL_ID,
                    OPENROUTER_PROVIDER or "",
                    acquire_lock=True,
                )
            except ValueError as exc:
                print(f"Error resuming {path}: {exc}")
                return
            resumed_history = store.messages()
            session_id = store.session_id
            print(f"[session] continued {session_id} ({len(resumed_history)} messages)")
            _print_resume_diagnostics(store)
            store.sync(resumed_history)
    else:
        try:
            store = SessionStore.create(
                workspace,
                session_id,
                MODEL_ID,
                OPENROUTER_PROVIDER or "",
                session_name,
                session_dir=sessions_dir,
                acquire_lock=True,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        session_id = store.session_id

    terminal_input = create_terminal_input()

    history = list(resumed_history)
    session = create_parent_session(
        cwd,
        approval_handler=TerminalApprovalHandler(
            interactive=sys.stdin.isatty(),
            prompt_fn=terminal_input.prompt,
        ),
        session_id=session_id,
        store=store,
        session_dir=sessions_dir,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )

    # Restore todo state if the session had an active plan.
    if store.is_persistent:
        saved_items = store.last_todo_items()
        if saved_items is not None:
            try:
                session.todo.update(TodoParams.model_validate({"items": saved_items}))
                if saved_items:
                    print(f"[session] restored todo plan ({len(saved_items)} items)")
            except Exception as exc:
                print(f"[session] warning: could not restore todo state: {exc}")

    def _todo_snapshot() -> list[dict]:
        return [
            item.model_dump(by_alias=True)
            for item in session.todo.state.items
        ]

    committed_history = copy.deepcopy(history)
    committed_todo = copy.deepcopy(_todo_snapshot())
    end_reason = "error"
    completed_turns = 0
    total_api_calls = 0
    emit_trace(
        session.trace_context,
        "session.started",
        source="parent",
        session_id=session.session_id,
        workspace_root=str(session.workspace.root),
        session_storage_dir=str(sessions_dir),
        model_id=MODEL_ID,
        reasoning_effort=session.reasoning_effort,
        max_output_tokens=session.max_output_tokens,
        resumed=bool(resumed_history),
    )

    try:
        while True:
            try:
                query = await terminal_input.prompt(INPUT_PROMPT)
                print(f"[debug] query: {query!r}")
            except EOFError:
                end_reason = "eof"
                break
            except KeyboardInterrupt:
                end_reason = "interrupt"
                break
            if query.strip().lower() in ("q", "exit", ""):
                end_reason = "user_exit"
                break
            if await handle_command(query, history, session):
                store.sync(history)
                store.sync_todo(_todo_snapshot())
                committed_history = copy.deepcopy(history)
                committed_todo = copy.deepcopy(_todo_snapshot())
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
            try:
                with terminal_input.processing():
                    outcome = await agent_loop(state, session)
            except KeyboardInterrupt:
                end_reason = "interrupt"
                history[:] = copy.deepcopy(committed_history)
                break
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

            # Persist at the turn boundary: history is well-formed (no orphaned
            # tool calls) and the next iteration starts a fresh user turn.
            store.sync(history)
            store.sync_todo(_todo_snapshot())
            committed_history = copy.deepcopy(history)
            committed_todo = copy.deepcopy(_todo_snapshot())
            completed_turns += 1
            total_api_calls += outcome.api_calls
    except KeyboardInterrupt:
        end_reason = "interrupt"
        history[:] = copy.deepcopy(committed_history)
    finally:
        await terminal_input.close()
        store.sync(committed_history)
        store.sync_todo(committed_todo)
        emit_trace(
            session.trace_context,
            "session.ended",
            source="parent",
            session_id=session.session_id,
            reason=end_reason,
            turns=completed_turns,
            agent_api_calls=total_api_calls,
        )
        store.close()
        if store.is_persistent:
            print(f"[session] saved {session_id}")


def _positive_int(value: str) -> int:
    import argparse

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> dict:
    import argparse
    parser = argparse.ArgumentParser(description="Code agent CLI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        metavar="TARGET",
        help="Resume a session by name, id, or file path",
    )
    group.add_argument("--continue", action="store_true", dest="continue_recent", help="Continue the most recent session")
    group.add_argument("--list-sessions", action="store_true", dest="list_only", help="List saved sessions and exit")
    group.add_argument("--no-session", action="store_true", dest="no_session", help="Disable session persistence")
    parser.add_argument("--name", dest="session_name", help="Optional name for a new session")
    parser.add_argument(
        "--session-dir",
        dest="session_dir",
        help=f"session storage directory (overrides {SESSION_DIR_ENV})",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="reasoning effort sent to the Responses API",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="maximum output tokens per agent call (default: provider limit)",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    asyncio.run(repl(**parse_args()))
