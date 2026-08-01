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
    created_at: str = ""
    trace_parse_errors: int = 0
    trace_missing: bool = False
    result_unavailable: bool = False
    manifest_unavailable: bool = False
    # Phase-0 was delivered as four independent commits. Track each
    # capability separately so a partially instrumented trace is never treated
    # as if all fields were available.
    has_validation_issues: bool = False
    has_raw_fingerprint: bool = False
    has_api_step: bool = False
    has_split_truncation: bool = False

    @property
    def agent_status(self) -> str:
        return self.result.get("agent_status", "")

    @property
    def patch_status(self) -> str:
        return self.result.get("patch_status", "")

    @property
    def api_calls(self) -> int:
        return self.result.get("api_calls", 0)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.run_id, self.instance_id, self.attempt)

    @property
    def has_phase0_fields(self) -> bool:
        """Backward-compatible shorthand for a complete Phase-0 trace."""
        return all((
            self.has_validation_issues,
            self.has_raw_fingerprint,
            self.has_api_step,
            self.has_split_truncation,
        ))


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
    success: bool | None = None
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
        manifest: dict[str, Any] = {}
        manifest_unavailable = not manifest_path.is_file()
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                manifest_unavailable = True
        instances_dir = run_dir / "instances"
        if not instances_dir.is_dir():
            continue
        for inst_dir in sorted(instances_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            for attempt_dir in sorted(inst_dir.iterdir()):
                if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt-"):
                    continue
                suffix = attempt_dir.name.removeprefix("attempt-")
                if not suffix.isdigit():
                    continue
                ref = _load_attempt(
                    run_dir,
                    inst_dir.name,
                    attempt_dir,
                    manifest,
                    manifest_unavailable=manifest_unavailable,
                )
                if ref is not None:
                    attempts.append(ref)
    attempts.sort(key=lambda a: (
        a.created_at or "9999",
        a.run_id,
        a.instance_id,
        a.attempt,
    ))
    return attempts


def _load_attempt(
    run_dir: Path,
    instance_id: str,
    attempt_dir: Path,
    manifest: dict[str, Any],
    *,
    manifest_unavailable: bool,
) -> AttemptRef | None:
    result_path = attempt_dir / "result.json"
    trace_path = attempt_dir / "trace.jsonl"
    result: dict[str, Any] = {}
    result_unavailable = not result_path.is_file()
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            result_unavailable = True
    trace_events: list[dict[str, Any]] = []
    trace_parse_errors = 0
    if trace_path.is_file():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                trace_events.append(json.loads(line))
            except json.JSONDecodeError:
                trace_parse_errors += 1

    capabilities = _detect_phase0_fields(trace_events)
    created_at = str(manifest.get("created_at", ""))
    if not created_at and trace_events:
        created_at = str(trace_events[0].get("timestamp", ""))
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
        created_at=created_at,
        trace_parse_errors=trace_parse_errors,
        trace_missing=not trace_path.is_file(),
        result_unavailable=result_unavailable,
        manifest_unavailable=manifest_unavailable,
        **capabilities,
    )


def _detect_phase0_fields(trace_events: list[dict[str, Any]]) -> dict[str, bool]:
    """Return per-capability availability for one attempt."""
    requested = [e for e in trace_events if e.get("event") == "tool.requested"]
    completed = [e for e in trace_events if e.get("event") == "tool.completed"]
    tool_events = [*requested, *completed]
    return {
        "has_validation_issues": bool(requested) and all(
            "validation_issues" in e for e in requested
        ),
        "has_raw_fingerprint": bool(requested) and all(
            "raw_arguments_sha256" in e and "raw_arguments_chars" in e
            for e in requested
        ),
        "has_api_step": bool(tool_events) and all(
            "api_call" in e and "step_index" in e for e in tool_events
        ),
        "has_split_truncation": bool(completed) and all(
            "runtime_output_truncated" in e
            and "tool_internal_truncated" in e
            and "truncated_chars" in e
            for e in completed
        ),
    }


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
