"""Decision-point micro-eval runner.

Reproduces the exact harness failure patterns surfaced by the Phase 1
Harness Diagnostic Report, but deterministically — no live model needed.

Each micro-eval calls the real tool dispatcher (``run_tool_call_async``)
with synthetic arguments that mimic what models actually submit, then
checks both the **tool output** (what the model sees) and the **trace
events** (what the offline analyzer sees).

These deterministic checks are the baseline-characterization layer.  The
live-model baseline/candidate experiment in :mod:`evals.micro.paired` tests
whether changing the decision point improves agent behavior.

Micro-evals are NOT scenario evals (``evals/scenarios/``). Scenario
evals test end-to-end agent behavior with a live model. Micro-evals
test individual harness decision points with synthetic inputs.
"""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass, field
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools  # noqa: E402
import trace as runtime_trace  # noqa: E402
from workspace import Workspace  # noqa: E402


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MicroCheck:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class MicroResult:
    name: str
    defect_id: str  # links back to Phase 1 report section
    tool_name: str
    raw_arguments: str
    passed: bool
    checks: list[MicroCheck] = field(default_factory=list)
    output: str = ""
    trace_events: list[dict] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _make_call(name: str, call_id: str, arguments: str):
    """Build a duck-typed tool call item matching what the dispatcher expects."""
    return types.SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=arguments,
    )


def run_micro_eval(
    name: str,
    defect_id: str,
    tool_name: str,
    raw_arguments: str,
    checks: list,
    *,
    workspace: Path,
    call_id: str = "micro-1",
) -> MicroResult:
    """Run one tool call against the real dispatcher and evaluate checks.

    ``checks`` is a list of callables ``(output, trace_events) -> MicroCheck``.
    Each check receives the model-facing output string and the list of trace
    events, and returns a ``MicroCheck`` with pass/fail + detail.
    """
    sink = runtime_trace.MemoryTraceSink()
    context = runtime_trace.TraceContext(sink=sink, run_id=f"micro:{name}")
    ws = Workspace(workspace)
    todo = tools.TodoManager()
    registry = tools.build_tool_registry(ws, todo)

    result = MicroResult(
        name=name,
        defect_id=defect_id,
        tool_name=tool_name,
        raw_arguments=raw_arguments,
        passed=False,
    )

    try:
        response, _used_todo = asyncio.run(tools.run_tool_call_async(
            _make_call(tool_name, call_id, raw_arguments),
            registry,
            todo,
            trace_context=context,
        ))
        result.output = response.get("output", "")
        result.trace_events = list(sink.events)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        return result

    for check_fn in checks:
        check = check_fn(result.output, result.trace_events)
        result.checks.append(check)

    result.passed = all(c.passed for c in result.checks) and bool(result.checks)
    if not result.checks:
        result.error = "no checks defined"
    return result
