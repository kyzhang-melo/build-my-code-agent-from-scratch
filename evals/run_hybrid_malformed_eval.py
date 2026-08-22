#!/usr/bin/env python3
"""Hybrid Provider-protocol regression eval for malformed tool-call recovery.

Unlike the deterministic pytest suite and the behavioral `run_evals.py`
scenarios, this hybrid eval deliberately injects a fault into a *real*
Provider response to verify that the agent loop can recover from a
malformed `function_call.arguments` without crashing the session.

Flow:

    1st request  -> real Provider (e.g. Tencent via OpenRouter)
                   |
             local wrapper corrupts the first edit_file arguments
                   |
    Runtime normalizes the malformed function_call in history
    Tool layer rejects the malformed call, emits an error output
                   |
    2nd request  -> real Provider receives sanitized history
                   |
    Provider must accept:  reasoning + function_call(args="{}", no id)
                           + function_call_output(error)
                   |
    Model emits a corrected edit_file call -> tool succeeds
                   |
    Agent returns a final answer

This is the only test that exercises the real Provider's validation of
the synthetic history shape produced by the malformed-arguments fix
(commit 7194671). The deterministic pytest tests verify our state
construction but cannot confirm Provider-side acceptance.

Usage:
    ./.venv/bin/python evals/run_hybrid_malformed_eval.py
    ./.venv/bin/python evals/run_hybrid_malformed_eval.py --model tencent/...
    ./.venv/bin/python evals/run_hybrid_malformed_eval.py --keep-workspace
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402
from trace import MemoryTraceSink, TraceContext  # noqa: E402
from permissions import ApprovalRequest, ApprovalResponse  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "hybrid_fixtures" / "malformed-tool-recovery"
RUNS_DIR = Path(__file__).resolve().parent / ".runs_hybrid"

PROMPT = (
    'Use edit_file to change "Hello" to "Goodbye" in target.txt. '
    "If a tool call fails because its arguments are invalid, correct the "
    "arguments and retry the tool call. Then read the file to verify the result."
)
DEFAULT_TIMEOUT = 240


class AutoApproveHandler:
    async def request(self, request: ApprovalRequest) -> ApprovalResponse:
        if request.allow_for_session:
            return ApprovalResponse("approve_for_session")
        return ApprovalResponse("approve")


class MalformFirstToolCallClient:
    """Wrapper around the real OpenAI client that corrupts the first edit_file call.

    Only the `arguments` field of the first `edit_file` function_call is
    replaced with malformed JSON. The real `id`, `call_id`, `name`, and all
    reasoning items are preserved so the second request to the Provider
    matches the protocol shape produced by a genuine model response.

    Injection happens exactly once, on the first real `edit_file` call. Earlier
    responses may contain preparatory read-only calls; those pass through
    unchanged. If no `edit_file` call appears, `injected` stays False and the
    eval must report `injection_not_applied` so the run is not silently
    considered passing.
    """

    def __init__(self, real_client):
        self._real_client = real_client
        self.responses = self
        self.injected = False
        self.injected_call_id: str | None = None
        self.replay_response_received = False
        self.replay_input: list[dict] | None = None
        self.inputs: list[list[dict]] = []

    async def create(self, **kwargs):
        api_input = kwargs.get("input", [])
        self.inputs.append(api_input)
        is_replay_request = self.injected
        if is_replay_request and self.replay_input is None:
            self.replay_input = api_input
        response = await self._real_client.responses.create(**kwargs)

        if is_replay_request:
            # Recording the input only proves the request started. Reaching
            # this point proves the Provider accepted it and returned a
            # response.
            self.replay_response_received = True
            return response

        output = list(getattr(response, "output", None) or [])
        mutated: list = []
        found_edit_file = False

        for item in output:
            if (
                not found_edit_file
                and getattr(item, "type", None) == "function_call"
                and getattr(item, "name", "") == "edit_file"
            ):
                found_edit_file = True
                self.injected = True
                self.injected_call_id = getattr(item, "call_id", "")
                mutated.append(SimpleNamespace(
                    type="function_call",
                    id=getattr(item, "id", None),
                    call_id=getattr(item, "call_id", ""),
                    name="edit_file",
                    arguments='{"path":"target.txt"',
                ))
            else:
                mutated.append(item)

        if not found_edit_file:
            return response

        return SimpleNamespace(
            output=mutated,
            output_text=getattr(response, "output_text", "") or "",
            usage=getattr(response, "usage", None),
        )


@dataclass
class HybridResult:
    injection_applied: bool = False
    tool_layer_recorded_invalid_arguments: bool = False
    provider_accepted_replay: bool = False
    replay_input_sanitized: bool = False
    error_feedback_present: bool = False
    replay_pairing_order_valid: bool = False
    corrected_tool_call_completed: bool = False
    fixture_result_correct: bool = False
    api_calls: int = 0
    final_text: str = ""
    error: str = ""
    workspace: str = ""
    checks: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.injection_applied
            and self.tool_layer_recorded_invalid_arguments
            and self.provider_accepted_replay
            and self.replay_input_sanitized
            and self.error_feedback_present
            and self.replay_pairing_order_valid
            and self.corrected_tool_call_completed
            and self.fixture_result_correct
        )


def _record(result: HybridResult, label: str, passed: bool, detail: str = "") -> None:
    result.checks.append({"label": label, "passed": passed, "detail": detail})


def evaluate(
    result: HybridResult,
    hybrid_client: MalformFirstToolCallClient,
    trace_sink: MemoryTraceSink,
    workspace: Path,
) -> None:
    # 1. Injection actually happened.
    _record(result, "injection_applied", result.injection_applied,
            "" if result.injection_applied else "first response had no edit_file call")

    # 2. Tool layer recorded invalid_arguments with json_parse error_type.
    completed = trace_sink.by_type("tool.completed")
    invalid = [
        e for e in completed
        if e.get("tool_name") == "edit_file"
        and e.get("status") == "invalid_arguments"
        and e.get("error_type") == "json_parse"
    ]
    result.tool_layer_recorded_invalid_arguments = bool(invalid)
    _record(result, "tool_layer_recorded_invalid_arguments",
            result.tool_layer_recorded_invalid_arguments,
            "" if invalid else "no tool.completed with status=invalid_arguments, error_type=json_parse")

    # 3. Provider accepted the first request after fault injection.
    _record(result, "provider_accepted_replay", result.provider_accepted_replay,
            "" if result.provider_accepted_replay else "second Provider request raised or no second request")

    # 4. The first API input after fault injection was sanitized.
    if hybrid_client.replay_input is not None:
        replay_input = hybrid_client.replay_input
        fc_items = [m for m in replay_input if isinstance(m, dict) and m.get("type") == "function_call"]
        injected_items = [
            m for m in fc_items
            if m.get("call_id") == hybrid_client.injected_call_id
        ]
        sanitized = (
            len(injected_items) == 1
            and injected_items[0].get("arguments") == "{}"
            and "id" not in injected_items[0]
        )
        result.replay_input_sanitized = sanitized
        _record(result, "replay_input_sanitized", result.replay_input_sanitized,
                "" if sanitized else f"injected call records={injected_items!r}")
        fco_items = [m for m in replay_input if isinstance(m, dict) and m.get("type") == "function_call_output"]
        has_error = any(
            m.get("call_id") == hybrid_client.injected_call_id
            and "invalid arguments" in str(m.get("output", ""))
            for m in fco_items
        )
        result.error_feedback_present = has_error
        _record(result, "error_feedback_in_replay_input", has_error,
                "" if has_error else "no function_call_output with 'invalid arguments' in replay input")
        call_index = next((
            index for index, item in enumerate(replay_input)
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("call_id") == hybrid_client.injected_call_id
        ), -1)
        output_index = next((
            index for index, item in enumerate(replay_input)
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == hybrid_client.injected_call_id
        ), -1)
        pairing_order_valid = (
            0 <= call_index < output_index
            and not any(
                isinstance(item, dict) and item.get("role") in ("assistant", "user", "system")
                for item in replay_input[call_index + 1:output_index]
            )
        )
        result.replay_pairing_order_valid = pairing_order_valid
        _record(result, "replay_pairing_order_valid", pairing_order_valid,
                "" if pairing_order_valid else "message boundary appeared inside call/result exchange")
    else:
        _record(result, "replay_input_sanitized", False, "no replay API request captured")
        _record(result, "error_feedback_in_replay_input", False, "no replay API request captured")
        _record(result, "replay_pairing_order_valid", False, "no replay API request captured")

    # 5. Corrected edit_file call completed successfully.
    successful_edits = [
        e for e in completed
        if e.get("tool_name") == "edit_file"
        and e.get("status") == "success"
    ]
    result.corrected_tool_call_completed = bool(successful_edits)
    _record(result, "corrected_edit_file_completed", bool(successful_edits),
            "" if successful_edits else "no successful edit_file call after recovery")

    # 6. Fixture result: target.txt contains "Goodbye, friend."
    target = workspace / "target.txt"
    if target.exists():
        content = target.read_text()
        result.fixture_result_correct = "Goodbye, friend." in content
        _record(result, "fixture_result_correct", result.fixture_result_correct,
                "" if result.fixture_result_correct else f"target.txt content: {content!r}")
    else:
        _record(result, "fixture_result_correct", False, "target.txt not found")


async def run_hybrid_eval(args: argparse.Namespace) -> int:
    if args.model:
        main.MODEL_ID = args.model

    run_root = RUNS_DIR / datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    workspace = run_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURE_DIR / "template", workspace, dirs_exist_ok=True)

    print(f"[hybrid] model={main.MODEL_ID!r} workspace={workspace}")

    real_client = main.client
    hybrid_client = MalformFirstToolCallClient(real_client)
    main.client = hybrid_client

    trace_sink = MemoryTraceSink()
    trace_context = TraceContext(
        sink=trace_sink,
        run_id=f"hybrid-malformed-tool:{run_root.name}",
        agent_id="parent",
    )
    session = main.create_parent_session(
        workspace.resolve(),
        approval_handler=AutoApproveHandler(),
        trace_context=trace_context,
        on_text=None,
    )
    state = main.LoopState(messages=[{"role": "user", "content": PROMPT}])

    result = HybridResult(workspace=str(workspace))
    try:
        outcome = await asyncio.wait_for(
            main.agent_loop(state, session),
            timeout=args.timeout,
        )
        result.final_text = outcome.final_text
        result.api_calls = outcome.api_calls
    except asyncio.TimeoutError:
        result.error = f"timed out after {args.timeout}s"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        main.client = real_client

    result.injection_applied = hybrid_client.injected
    result.provider_accepted_replay = hybrid_client.replay_response_received
    evaluate(result, hybrid_client, trace_sink, workspace)

    # Report
    status = "PASS" if result.passed else "FAIL"
    color = "\033[32m" if result.passed else "\033[31m"
    print(f"\n{color}[{status}]\033[0m hybrid-malformed-tool-recovery  "
          f"(api_calls={result.api_calls})")
    if result.error:
        print(f"    error: {result.error}")
    for check in result.checks:
        mark = "\033[32m✓\033[0m" if check["passed"] else "\033[31m✗\033[0m"
        line = f"    {mark} {check['label']}"
        if not check["passed"] and check["detail"]:
            line += f"  ({check['detail']})"
        print(line)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": main.MODEL_ID,
        "injection_applied": result.injection_applied,
        "tool_layer_recorded_invalid_arguments": result.tool_layer_recorded_invalid_arguments,
        "provider_accepted_replay": result.provider_accepted_replay,
        "replay_input_sanitized": result.replay_input_sanitized,
        "error_feedback_present": result.error_feedback_present,
        "replay_pairing_order_valid": result.replay_pairing_order_valid,
        "corrected_tool_call_completed": result.corrected_tool_call_completed,
        "fixture_result_correct": result.fixture_result_correct,
        "api_calls": result.api_calls,
        "final_text": result.final_text,
        "error": result.error,
        "passed": result.passed,
        "checks": result.checks,
    }
    (run_root / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[hybrid] report: {run_root}/report.json")

    if not args.keep_workspace and result.passed:
        shutil.rmtree(workspace, ignore_errors=True)

    return 0 if result.passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Provider-protocol regression eval for malformed tool-call recovery.",
    )
    parser.add_argument("--model", help="override MODEL_ID for this run")
    parser.add_argument("--keep-workspace", action="store_true",
                        help="retain the workspace after the run")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"timeout in seconds (default {DEFAULT_TIMEOUT})")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run_hybrid_eval(parse_args())))
