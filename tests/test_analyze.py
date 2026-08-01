from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.analyze.report import analyze, render_report
from evals.analyze.scanner import (
    AttemptRef,
    ToolCallPair,
    extract_tool_calls,
    load_runs,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic run directories
# ---------------------------------------------------------------------------


def _write_manifest(run_dir: Path, *, model: str, commit: str = "abcdef12", dirty: bool = True) -> None:
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": run_dir.name,
        "model": model,
        "harness_commit": commit,
        "harness_worktree_dirty": dirty,
        "instance_ids": [],
    }))


def _write_result(attempt_dir: Path, *, agent_status: str, patch_status: str, api_calls: int = 10) -> None:
    (attempt_dir / "result.json").write_text(json.dumps({
        "instance_id": attempt_dir.parent.name,
        "attempt": int(attempt_dir.name.removeprefix("attempt-")),
        "agent_status": agent_status,
        "patch_status": patch_status,
        "stop_reason": agent_status,
        "api_calls": api_calls,
        "duration_seconds": 100.0,
        "patch_bytes": 500,
        "error": "",
    }))


def _write_trace(attempt_dir: Path, events: list[dict]) -> None:
    lines = [json.dumps(e, ensure_ascii=True) for e in events]
    (attempt_dir / "trace.jsonl").write_text("\n".join(lines) + "\n")


def _tc_event(
    event: str,
    call_id: str,
    tool_name: str,
    *,
    source: str = "parent",
    sequence: int = 1,
    status: str = "success",
    success: bool = True,
    error_type: str | None = None,
    validation_issues: list[dict] | None = None,
    raw_arguments_sha256: str = "",
    raw_arguments_chars: int = 0,
    api_call: int | None = None,
    step_index: int | None = None,
    runtime_output_truncated: bool | None = None,
    tool_internal_truncated: bool | None = None,
    truncated_chars: int | None = None,
    arguments: dict | None = None,
) -> dict:
    e = {
        "event": event,
        "timestamp": "2026-08-01T00:00:00+00:00",
        "run_id": "test-run",
        "agent_id": "parent",
        "sequence": sequence,
        "source": source,
        "tool_name": tool_name,
        "call_id": call_id,
    }
    if arguments is not None:
        e["arguments"] = arguments
    if event == "tool.requested":
        e["argument_error"] = None
        e["validation_issues"] = validation_issues or []
        e["raw_arguments_sha256"] = raw_arguments_sha256
        e["raw_arguments_chars"] = raw_arguments_chars
        e["api_call"] = api_call
        e["step_index"] = step_index
    if event == "tool.completed":
        e["status"] = status
        e["success"] = success
        e["error_type"] = error_type
        e["duration_ms"] = 100
        e["output_chars"] = 50
        e["output_truncated"] = False
        e["api_call"] = api_call
        e["step_index"] = step_index
        if runtime_output_truncated is not None:
            e["runtime_output_truncated"] = runtime_output_truncated
            e["tool_internal_truncated"] = tool_internal_truncated
            e["truncated_chars"] = truncated_chars
    return e


def _make_run(
    runs_dir: Path,
    run_id: str,
    model: str,
    instances: dict[str, list[dict]],
    *,
    commit: str = "abcdef12",
    dirty: bool = True,
    agent_status: str = "completed",
    patch_status: str = "produced",
    api_calls: int = 10,
) -> Path:
    """Create a synthetic run with the given trace events per instance.

    ``instances`` maps instance_id -> list of trace events.
    """
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, model=model, commit=commit, dirty=dirty)
    for instance_id, events in instances.items():
        inst_dir = run_dir / "instances" / instance_id
        attempt_dir = inst_dir / "attempt-1"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _write_result(attempt_dir, agent_status=agent_status, patch_status=patch_status, api_calls=api_calls)
        _write_trace(attempt_dir, events)
    return run_dir


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------


def test_scanner_loads_attempts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-1", "test-model", {
        "inst-1": [_tc_event("tool.requested", "c1", "read_file", sequence=1)],
    })
    attempts = load_runs(runs_dir)
    assert len(attempts) == 1
    a = attempts[0]
    assert a.run_id == "run-1"
    assert a.instance_id == "inst-1"
    assert a.attempt == 1
    assert a.model == "test-model"
    assert a.harness_commit == "abcdef12"
    assert a.harness_dirty is True


def test_scanner_detects_phase0_fields(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    # Phase-0 trace (has validation_issues)
    _make_run(runs_dir, "run-new", "m", {
        "i": [_tc_event("tool.requested", "c1", "grep", sequence=1, validation_issues=[{"path": "-n", "type": "extra_forbidden"}])],
    })
    # Legacy trace (no Phase-0 fields)
    legacy_events = [{
        "event": "tool.requested", "timestamp": "2026-08-01T00:00:00+00:00",
        "run_id": "r", "agent_id": "parent", "sequence": 1, "source": "parent",
        "tool_name": "grep", "call_id": "c1", "arguments": {}, "argument_error": None,
    }]
    _make_run(runs_dir, "run-old", "m", {"i": legacy_events})
    attempts = load_runs(runs_dir)
    by_run = {a.run_id: a for a in attempts}
    assert by_run["run-new"].has_phase0_fields is True
    assert by_run["run-old"].has_phase0_fields is False


def test_extract_tool_calls_matches_pairs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    events = [
        _tc_event("tool.requested", "c1", "read_file", sequence=1, raw_arguments_sha256="abc123", raw_arguments_chars=20),
        _tc_event("tool.completed", "c1", "read_file", sequence=2, status="success"),
        _tc_event("tool.requested", "c2", "grep", sequence=3, validation_issues=[{"path": "-n", "type": "extra_forbidden"}]),
        _tc_event("tool.completed", "c2", "grep", sequence=4, status="invalid_arguments", success=False, error_type="validation"),
    ]
    _make_run(runs_dir, "r", "m", {"i": events})
    attempts = load_runs(runs_dir)
    calls = extract_tool_calls(attempts[0])
    assert len(calls) == 2
    assert calls[0].tool_name == "read_file"
    assert calls[0].success is True
    assert calls[0].raw_arguments_sha256 == "abc123"
    assert calls[1].tool_name == "grep"
    assert calls[1].success is False
    assert calls[1].error_type == "validation"
    assert calls[1].validation_issues == [{"path": "-n", "type": "extra_forbidden"}]


# ---------------------------------------------------------------------------
# Report analysis tests
# ---------------------------------------------------------------------------


def test_report_coverage_counts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1", "model-a", {"i1": [_tc_event("tool.requested", "c1", "read_file")]})
    _make_run(runs_dir, "r2", "model-b", {"i2": [_tc_event("tool.requested", "c1", "read_file")]})
    data = analyze(runs_dir)
    assert data.total_runs == 2
    assert data.total_attempts == 2
    assert data.total_tool_calls == 2
    assert sorted(data.models) == ["model-a", "model-b"]


def test_report_tool_failure_matrix(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    events = [
        _tc_event("tool.requested", "c1", "grep", sequence=1),
        _tc_event("tool.completed", "c1", "grep", sequence=2, status="invalid_arguments", success=False, error_type="validation"),
        _tc_event("tool.requested", "c2", "read_file", sequence=3),
        _tc_event("tool.completed", "c2", "read_file", sequence=4, status="success"),
    ]
    _make_run(runs_dir, "r1", "model-a", {"i1": events})
    data = analyze(runs_dir)
    # Should have one failure row: model-a / grep / validation
    grep_rows = [r for r in data.failure_rows if r.tool_name == "grep" and r.error_type == "validation"]
    assert len(grep_rows) == 1
    assert grep_rows[0].failed_calls == 1
    assert grep_rows[0].total_calls == 1
    assert grep_rows[0].model == "model-a"


def test_report_param_friction_with_phase0(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    events = [
        _tc_event("tool.requested", "c1", "grep", sequence=1, validation_issues=[{"path": "-n", "type": "extra_forbidden"}]),
        _tc_event("tool.completed", "c1", "grep", sequence=2, status="invalid_arguments", success=False, error_type="validation"),
        _tc_event("tool.requested", "c2", "grep", sequence=3, validation_issues=[{"path": "-n", "type": "extra_forbidden"}]),
        _tc_event("tool.completed", "c2", "grep", sequence=4, status="invalid_arguments", success=False, error_type="validation"),
    ]
    _make_run(runs_dir, "r1", "model-a", {"i1": events})
    data = analyze(runs_dir)
    assert len(data.friction_rows) == 1
    row = data.friction_rows[0]
    assert row.tool_name == "grep"
    assert row.field_path == "-n"
    assert row.issue_type == "extra_forbidden"
    assert row.occurrences == 2
    assert row.affected_attempts == 1
    assert row.affected_models == 1


def test_report_param_friction_legacy_excluded(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    # Legacy trace without validation_issues
    legacy_events = [{
        "event": "tool.requested", "timestamp": "2026-08-01T00:00:00+00:00",
        "run_id": "r", "agent_id": "parent", "sequence": 1, "source": "parent",
        "tool_name": "grep", "call_id": "c1", "arguments": {}, "argument_error": None,
    }]
    _make_run(runs_dir, "r1", "model-a", {"i1": legacy_events})
    data = analyze(runs_dir)
    assert len(data.friction_rows) == 0
    assert data.friction_legacy_note is True


def test_report_consecutive_same_error(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    events = [
        _tc_event("tool.requested", "c1", "grep", sequence=1),
        _tc_event("tool.completed", "c1", "grep", sequence=2, status="invalid_arguments", success=False, error_type="validation"),
        _tc_event("tool.requested", "c2", "grep", sequence=3),
        _tc_event("tool.completed", "c2", "grep", sequence=4, status="invalid_arguments", success=False, error_type="validation"),
    ]
    _make_run(runs_dir, "r1", "model-a", {"i1": events})
    data = analyze(runs_dir)
    consec = [r for r in data.anomaly_rows if r.kind == "consecutive_same_error"]
    assert len(consec) == 1
    assert consec[0].tool_name == "grep"
    assert consec[0].error_type == "validation"
    assert consec[0].count == 1  # one pair of back-to-back


def test_report_exact_repeat_with_hash(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    events = [
        _tc_event("tool.requested", "c1", "read_file", sequence=1, raw_arguments_sha256="samehash", raw_arguments_chars=20),
        _tc_event("tool.completed", "c1", "read_file", sequence=2),
        _tc_event("tool.requested", "c2", "read_file", sequence=3, raw_arguments_sha256="samehash", raw_arguments_chars=20),
        _tc_event("tool.completed", "c2", "read_file", sequence=4),
    ]
    _make_run(runs_dir, "r1", "model-a", {"i1": events})
    data = analyze(runs_dir)
    repeats = [r for r in data.anomaly_rows if r.kind == "exact_repeat"]
    assert len(repeats) == 1
    assert repeats[0].tool_name == "read_file"
    assert repeats[0].count == 1  # one repeat beyond the first


def test_report_exact_repeat_not_detected_without_hash(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    # Legacy trace: no raw_arguments_sha256
    events = [
        _tc_event("tool.requested", "c1", "read_file", sequence=1, raw_arguments_sha256="", raw_arguments_chars=0),
        _tc_event("tool.completed", "c1", "read_file", sequence=2),
        _tc_event("tool.requested", "c2", "read_file", sequence=3, raw_arguments_sha256="", raw_arguments_chars=0),
        _tc_event("tool.completed", "c2", "read_file", sequence=4),
    ]
    _make_run(runs_dir, "r1", "model-a", {"i1": events})
    data = analyze(runs_dir)
    repeats = [r for r in data.anomaly_rows if r.kind == "exact_repeat"]
    assert len(repeats) == 0


def test_report_budget_bound_candidate(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    events = [
        _tc_event("tool.requested", "c1", "read_file", sequence=1),
        _tc_event("tool.completed", "c1", "read_file", sequence=2),
        _tc_event("tool.requested", "c2", "edit_file", sequence=3),
        _tc_event("tool.completed", "c2", "edit_file", sequence=4),
        _tc_event("tool.requested", "c3", "edit_file", sequence=5),
        _tc_event("tool.completed", "c3", "edit_file", sequence=6),
    ]
    _make_run(runs_dir, "r1", "model-a", {"i1": events}, agent_status="max_api_calls", api_calls=30)
    data = analyze(runs_dir)
    budget = [r for r in data.defect_candidates if r.kind == "budget_bound"]
    assert len(budget) == 1
    assert budget[0].agent_status == "max_api_calls"
    assert "editing near budget exhaustion" in budget[0].detail


def test_report_no_patch_candidate(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "r1", "model-a", {"i1": []}, agent_status="error", patch_status="empty")
    data = analyze(runs_dir)
    no_patch = [r for r in data.defect_candidates if r.kind == "no_patch"]
    assert len(no_patch) == 1
    assert no_patch[0].patch_status == "empty"


def test_report_observation_gaps_legacy(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    legacy_events = [{
        "event": "tool.requested", "timestamp": "2026-08-01T00:00:00+00:00",
        "run_id": "r", "agent_id": "parent", "sequence": 1, "source": "parent",
        "tool_name": "grep", "call_id": "c1", "arguments": {}, "argument_error": None,
    }]
    _make_run(runs_dir, "r1", "model-a", {"i1": legacy_events})
    data = analyze(runs_dir)
    assert any("legacy" in g.lower() for g in data.observation_gaps)


def test_report_renders_markdown(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    events = [
        _tc_event("tool.requested", "c1", "grep", sequence=1, validation_issues=[{"path": "-n", "type": "extra_forbidden"}]),
        _tc_event("tool.completed", "c1", "grep", sequence=2, status="invalid_arguments", success=False, error_type="validation"),
    ]
    _make_run(runs_dir, "r1", "model-a", {"i1": events})
    data = analyze(runs_dir)
    md = render_report(data)
    assert "# Harness Diagnostic Report" in md
    assert "## 1. Data Coverage" in md
    assert "## 2. Tool Failure Matrix" in md
    assert "## 3. Parameter Friction" in md
    assert "## 4. Anomaly Sequences" in md
    assert "## Defect Candidates" in md
    assert "## Observation Gaps" in md
    assert "model-a" in md
    assert "grep" in md
    assert "-n" in md


# ---------------------------------------------------------------------------
# Acceptance test: grep friction persists post-fix, visible in failure matrix
# ---------------------------------------------------------------------------


def test_acceptance_grep_friction_visible_in_failure_matrix(tmp_path: Path) -> None:
    """The report must surface grep validation failures grouped by model,
    so that a human can see grep -n issues persist in longcat/hy3 even
    after the 3d67a66 prompt-text fix.

    This is the acceptance criterion from the Phase 1 plan: the report
    must automatically point out that grep parameter friction persists
    in specific models after a harness change.
    """
    runs_dir = tmp_path / "runs"
    # Simulate two models: one with grep validation failures, one without
    grep_fail_events = [
        _tc_event("tool.requested", "c1", "grep", sequence=1),
        _tc_event("tool.completed", "c1", "grep", sequence=2, status="invalid_arguments", success=False, error_type="validation"),
    ]
    ok_events = [
        _tc_event("tool.requested", "c1", "grep", sequence=1),
        _tc_event("tool.completed", "c1", "grep", sequence=2, status="success"),
    ]
    _make_run(runs_dir, "run-longcat", "meituan/longcat-2.0", {"inst-1": grep_fail_events})
    _make_run(runs_dir, "run-glm", "z-ai/glm-5.2", {"inst-2": ok_events})
    data = analyze(runs_dir)

    # The failure matrix must show grep/validation failures for longcat but not glm
    longcat_grep = [
        r for r in data.failure_rows
        if r.model == "meituan/longcat-2.0" and r.tool_name == "grep" and r.error_type == "validation"
    ]
    glm_grep = [
        r for r in data.failure_rows
        if r.model == "z-ai/glm-5.2" and r.tool_name == "grep" and r.error_type == "validation"
    ]
    assert len(longcat_grep) == 1
    assert longcat_grep[0].failed_calls == 1
    assert len(glm_grep) == 0

    # The rendered report must contain the model name and the failure
    md = render_report(data)
    assert "meituan/longcat-2.0" in md
    assert "grep" in md
    assert "validation" in md
