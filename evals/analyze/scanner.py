"""Offline scanner for SWE-bench eval artifacts.

Loads runs, attempts, traces, and result metadata from
``evals/.runs/swebench/**`` into structured data objects so the report
renderer can compute deterministic metrics.

Design constraints:
- Every metric must be provable by structured trace evidence.
- Legacy traces (pre-Phase-0) lack ``validation_issues``,
  ``raw_arguments_*``, ``api_call``, ``step_index``, and the split
  truncation fields. The scanner records which new fields are present
  per attempt so the report can mark unavailable data as ``legacy``
  instead of guessing.
- No ``agent.log`` text parsing. If a field is missing from the trace,
  it is reported as unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / ".runs" / "swebench"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AttemptRef:
    """One attempt directory and its loaded artifacts."""

    run_id: str
    instance_id: str
    attempt: int
    run_dir: Path
    manifest: dict[str, Any]
    result: dict[str, Any]
    trace_events: list[dict[str, Any]]
    model: str
    harness_commit: str
    harness_dirty: bool
    # Whether the trace has Phase-0 fields (validation_issues, raw_arguments_*,
    # api_call, step_index, split truncation).
    has_phase0_fields: bool = False

    @property
    def agent_status(self) -> str:
        return self.result.get("agent_status", "")

    @property
    def patch_status(self) -> str:
        return self.result.get("patch_status", "")

    @property
    def api_calls(self) -> int:
        return self.result.get("api_calls", 0)


@dataclass
class ToolCallPair:
    """A matched tool.requested + tool.completed pair for one call."""

    tool_name: str
    call_id: str
    source: str
    requested: dict[str, Any]
    completed: dict[str, Any] | None
    # Convenience fields extracted at load time.
    status: str = ""
    error_type: str | None = None
    success: bool = True
    api_call: int | None = None
    step_index: int | None = None
    validation_issues: list[dict] = field(default_factory=list)
    raw_arguments_sha256: str = ""
    raw_arguments_chars: int = 0
    runtime_output_truncated: bool | None = None
    tool_internal_truncated: bool | None = None
    truncated_chars: int | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[AttemptRef]:
    """Load all attempts from all runs under ``runs_dir``.

    Returns a list sorted by (run_id, instance_id, attempt) for stable
    output.
    """
    attempts: list[AttemptRef] = []
    if not runs_dir.is_dir():
        return attempts

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        instances_dir = run_dir / "instances"
        if not instances_dir.is_dir():
            continue
        for inst_dir in sorted(instances_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            for attempt_dir in sorted(inst_dir.iterdir()):
                if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt-"):
                    continue
                ref = _load_attempt(run_dir, inst_dir.name, attempt_dir, manifest)
                if ref is not None:
                    attempts.append(ref)
    return attempts


def _load_attempt(
    run_dir: Path,
    instance_id: str,
    attempt_dir: Path,
    manifest: dict[str, Any],
) -> AttemptRef | None:
    result_path = attempt_dir / "result.json"
    trace_path = attempt_dir / "trace.jsonl"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    trace_events: list[dict[str, Any]] = []
    if trace_path.is_file():
        for line in trace_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                trace_events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    has_phase0 = _detect_phase0_fields(trace_events)
    return AttemptRef(
        run_id=run_dir.name,
        instance_id=instance_id,
        attempt=int(attempt_dir.name.removeprefix("attempt-")),
        run_dir=run_dir,
        manifest=manifest,
        result=result,
        trace_events=trace_events,
        model=manifest.get("model", ""),
        harness_commit=str(manifest.get("harness_commit", ""))[:8],
        harness_dirty=bool(manifest.get("harness_worktree_dirty", False)),
        has_phase0_fields=has_phase0,
    )


def _detect_phase0_fields(trace_events: list[dict[str, Any]]) -> bool:
    """Return True if any trace event has Phase-0 instrumentation fields."""
    for event in trace_events:
        if event.get("event") == "tool.requested":
            if "validation_issues" in event or "raw_arguments_sha256" in event:
                return True
        if event.get("event") == "tool.completed":
            if "runtime_output_truncated" in event or "tool_internal_truncated" in event:
                return True
    return False


def extract_tool_calls(attempt: AttemptRef) -> list[ToolCallPair]:
    """Match tool.requested and tool.completed events by call_id."""
    requested: dict[str, dict] = {}
    completed: dict[str, dict] = {}
    for event in attempt.trace_events:
        if event.get("event") == "tool.requested":
            cid = event.get("call_id", "")
            if cid:
                requested[cid] = event
        elif event.get("event") == "tool.completed":
            cid = event.get("call_id", "")
            if cid:
                completed[cid] = event

    pairs: list[ToolCallPair] = []
    for cid, req in requested.items():
        comp = completed.get(cid)
        pair = ToolCallPair(
            tool_name=req.get("tool_name", ""),
            call_id=cid,
            source=req.get("source", ""),
            requested=req,
            completed=comp,
        )
        if comp is not None:
            pair.status = comp.get("status", "")
            pair.error_type = comp.get("error_type")
            pair.success = comp.get("success", True)
            pair.api_call = comp.get("api_call")
            pair.step_index = comp.get("step_index")
            pair.runtime_output_truncated = comp.get("runtime_output_truncated")
            pair.tool_internal_truncated = comp.get("tool_internal_truncated")
            pair.truncated_chars = comp.get("truncated_chars")
        # Phase-0 fields on requested
        pair.validation_issues = req.get("validation_issues", [])
        pair.raw_arguments_sha256 = req.get("raw_arguments_sha256", "")
        pair.raw_arguments_chars = req.get("raw_arguments_chars", 0)
        pair.api_call = pair.api_call if pair.api_call is not None else req.get("api_call")
        pair.step_index = pair.step_index if pair.step_index is not None else req.get("step_index")
        pairs.append(pair)
    # Sort by sequence (trace order) if available, else by call_id
    pairs.sort(key=lambda p: p.requested.get("sequence", 0))
    return pairs
