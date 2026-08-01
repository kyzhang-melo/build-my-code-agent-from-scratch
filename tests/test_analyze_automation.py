from __future__ import annotations

import json
from pathlib import Path

from evals.analyze.automation import (
    DIAGNOSTIC_JSON,
    DIAGNOSTIC_MARKDOWN,
    DIFF_JSON,
    DIFF_MARKDOWN,
    diff_snapshots,
    find_previous_comparable_run,
    generate_run_diagnostics,
)


def _manifest(
    run_dir: Path,
    *,
    created_at: str,
    model: str = "vendor/model",
    provider: str = "provider/fp8",
    commit: str = "abcdef123456",
) -> dict:
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": created_at,
        "dataset": "SWE-bench/SWE-bench_Verified",
        "split": "test",
        "model": model,
        "provider": provider,
        "instance_ids": ["owner__repo-1"],
        "max_api_calls": 30,
        "reasoning_effort": "low",
        "max_output_tokens": 8000,
        "harness_commit": commit,
        "harness_worktree_dirty": False,
    }
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _tool_event(event: str, call_id: str, sequence: int) -> dict:
    row = {
        "event": event,
        "timestamp": "2026-08-01T00:00:00+00:00",
        "run_id": "test",
        "agent_id": "parent",
        "source": "parent",
        "sequence": sequence,
        "call_id": call_id,
        "tool_name": "grep",
        "api_call": sequence,
        "step_index": 0,
    }
    if event == "tool.requested":
        row.update({
            "arguments": {"pattern": "x"},
            "argument_error": None,
            "validation_issues": [{"path": "-n", "type": "extra_forbidden"}],
            "raw_arguments_chars": 25,
            "raw_arguments_sha256": f"hash-{call_id}",
        })
    else:
        row.update({
            "status": "invalid_arguments",
            "success": False,
            "error_type": "validation",
            "duration_ms": 1,
            "output_chars": 10,
            "output_truncated": False,
            "runtime_output_truncated": False,
            "tool_internal_truncated": False,
            "truncated_chars": 0,
        })
    return row


def _attempt(run_dir: Path, failures: int) -> None:
    attempt_dir = run_dir / "instances" / "owner__repo-1" / "attempt-1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "result.json").write_text(json.dumps({
        "instance_id": "owner__repo-1",
        "attempt": 1,
        "agent_status": "completed",
        "patch_status": "produced",
        "api_calls": failures + 1,
    }), encoding="utf-8")
    events = []
    for index in range(failures):
        call_id = f"call-{index}"
        events.extend([
            _tool_event("tool.requested", call_id, index * 2 + 1),
            _tool_event("tool.completed", call_id, index * 2 + 2),
        ])
    (attempt_dir / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_diff_snapshots_reports_new_disappeared_and_frequency_changes() -> None:
    previous = {
        "run_id": "old",
        "signals": [
            {"signal_id": "same", "category": "tool_failure", "label": "same",
             "count": 1, "rate": 0.1},
            {"signal_id": "gone", "category": "anomaly_sequence", "label": "gone",
             "count": 2, "rate": None},
        ],
        "observation_gaps": ["closed gap"],
    }
    current = {
        "run_id": "new",
        "signals": [
            {"signal_id": "same", "category": "tool_failure", "label": "same",
             "count": 3, "rate": 0.3},
            {"signal_id": "added", "category": "parameter_friction", "label": "added",
             "count": 1, "rate": None},
        ],
        "observation_gaps": ["new gap"],
    }

    diff = diff_snapshots(current, previous)

    assert [row["signal_id"] for row in diff["new_signals"]] == ["added"]
    assert [row["signal_id"] for row in diff["disappeared_signals"]] == ["gone"]
    assert diff["frequency_changes"][0]["count_delta"] == 2
    assert round(diff["frequency_changes"][0]["rate_delta"], 2) == 0.2
    assert diff["new_observation_gaps"] == ["new gap"]
    assert diff["closed_observation_gaps"] == ["closed gap"]


def test_previous_run_requires_same_controlled_configuration(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    old = runs / "old"
    wrong_provider = runs / "wrong-provider"
    current = runs / "current"
    _manifest(old, created_at="2026-08-01T00:00:00+00:00")
    _manifest(
        wrong_provider,
        created_at="2026-08-02T00:00:00+00:00",
        provider="other/fp8",
    )
    current_manifest = _manifest(
        current,
        created_at="2026-08-03T00:00:00+00:00",
    )
    for run in (old, wrong_provider, current):
        (run / "instances").mkdir()

    assert find_previous_comparable_run(current, current_manifest) == old


def test_generate_run_diagnostics_compares_with_previous_raw_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    previous = runs / "previous"
    current = runs / "current"
    _manifest(previous, created_at="2026-08-01T00:00:00+00:00", commit="oldcommit")
    _attempt(previous, failures=1)
    _manifest(current, created_at="2026-08-02T00:00:00+00:00", commit="newcommit")
    _attempt(current, failures=2)

    artifacts = generate_run_diagnostics(current)

    assert artifacts["previous_run"] == previous
    for name in (DIAGNOSTIC_JSON, DIAGNOSTIC_MARKDOWN, DIFF_JSON, DIFF_MARKDOWN):
        assert (current / name).is_file()
    diff = json.loads((current / DIFF_JSON).read_text(encoding="utf-8"))
    failure_change = next(
        row for row in diff["frequency_changes"]
        if row["category"] == "tool_failure"
    )
    assert failure_change["before_count"] == 1
    assert failure_change["after_count"] == 2
    rendered = (current / DIFF_MARKDOWN).read_text(encoding="utf-8")
    assert "previous" in rendered
    assert "Frequency changes" in rendered


def test_first_comparable_run_explicitly_reports_missing_baseline(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "first"
    _manifest(run, created_at="2026-08-01T00:00:00+00:00")
    _attempt(run, failures=1)

    generate_run_diagnostics(run)

    diff = json.loads((run / DIFF_JSON).read_text(encoding="utf-8"))
    assert diff["comparable"] is False
    assert diff["previous_run"] is None
    assert "A baseline was not guessed" in (
        run / DIFF_MARKDOWN
    ).read_text(encoding="utf-8")
