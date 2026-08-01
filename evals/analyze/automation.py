"""Phase-3 automatic per-run diagnostics and structured run-to-run diffs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.analyze.report import ReportData, analyze_attempts, render_report
from evals.analyze.scanner import load_run


DIAGNOSTIC_JSON = "harness-diagnostic.json"
DIAGNOSTIC_MARKDOWN = "harness-diagnostic.md"
DIFF_JSON = "harness-diagnostic-diff.json"
DIFF_MARKDOWN = "harness-diagnostic-diff.md"

COMPARISON_FIELDS = (
    "dataset",
    "split",
    "model",
    "provider",
    "instance_ids",
    "max_api_calls",
    "reasoning_effort",
    "max_output_tokens",
    "instance_timeout_seconds",
    "auto_compact",
)


@dataclass(frozen=True)
class Signal:
    signal_id: str
    category: str
    label: str
    count: int
    affected_attempts: int | None = None
    rate: float | None = None


@dataclass(frozen=True)
class SignalChange:
    signal_id: str
    category: str
    label: str
    before_count: int
    after_count: int
    count_delta: int
    before_rate: float | None
    after_rate: float | None
    rate_delta: float | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def _signal_id(category: str, *parts: str) -> str:
    return json.dumps([category, *parts], ensure_ascii=False, separators=(",", ":"))


def signals_from_report(data: ReportData) -> list[Signal]:
    """Project directly observed report rows into stable comparable signals."""
    signals: list[Signal] = []
    for row in data.failure_rows:
        signals.append(Signal(
            signal_id=_signal_id(
                "tool_failure", row.model, row.tool_name, row.error_type,
            ),
            category="tool_failure",
            label=f"{row.model} / {row.tool_name} / {row.error_type}",
            count=row.failed_calls,
            affected_attempts=row.affected_attempts,
            rate=row.failure_rate,
        ))
    for row in data.friction_rows:
        signals.append(Signal(
            signal_id=_signal_id(
                "parameter_friction",
                row.model,
                row.tool_name,
                row.field_path,
                row.issue_type,
            ),
            category="parameter_friction",
            label=(
                f"{row.model} / {row.tool_name} / {row.field_path} / "
                f"{row.issue_type}"
            ),
            count=row.occurrences,
            affected_attempts=row.affected_attempts,
        ))
    for row in data.anomaly_rows:
        signals.append(Signal(
            signal_id=_signal_id(
                "anomaly_sequence",
                row.model,
                row.kind,
                row.tool_name,
                row.error_type,
                row.detail,
            ),
            category="anomaly_sequence",
            label=(
                f"{row.model} / {row.kind} / {row.tool_name} / "
                f"{row.error_type} / {row.detail}"
            ),
            count=row.count,
            affected_attempts=row.affected_attempts,
        ))
    for row in data.defect_candidates:
        signals.append(Signal(
            signal_id=_signal_id(
                "defect_candidate", row.model, row.kind, row.instance_id,
            ),
            category="defect_candidate",
            label=f"{row.model} / {row.kind} / {row.instance_id}",
            count=1,
            affected_attempts=1,
        ))
    return sorted(signals, key=lambda signal: signal.signal_id)


def comparison_key(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return configuration fields that must match for a meaningful diff."""
    key = {field: manifest.get(field) for field in COMPARISON_FIELDS}
    key["provider"] = manifest.get("provider") or ""
    return key


def build_snapshot(
    run_dir: Path,
    manifest: dict[str, Any],
    data: ReportData,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id", run_dir.name),
        "created_at": manifest.get("created_at", ""),
        "harness_commit": manifest.get("harness_commit", ""),
        "harness_worktree_dirty": bool(
            manifest.get("harness_worktree_dirty", False)
        ),
        "comparison_key": comparison_key(manifest),
        "coverage": {
            "attempts": data.total_attempts,
            "tool_calls": data.total_tool_calls,
            "legacy_attempts": data.legacy_attempts,
            "malformed_trace_lines": data.malformed_trace_lines,
            "no_trace_attempts": data.no_trace_attempts,
            "incomplete_tool_calls": data.incomplete_tool_calls,
        },
        "signals": [asdict(signal) for signal in signals_from_report(data)],
        "observation_gaps": list(data.observation_gaps),
    }


def _manifest_time(manifest: dict[str, Any], run_dir: Path) -> tuple[str, str]:
    return str(manifest.get("created_at", "")), run_dir.name


def find_previous_comparable_run(
    current_run: Path,
    current_manifest: dict[str, Any],
) -> Path | None:
    """Find the newest older run with the same controlled configuration."""
    candidates: list[tuple[tuple[str, str], Path]] = []
    current_time = _manifest_time(current_manifest, current_run)
    current_key = comparison_key(current_manifest)
    for candidate in current_run.parent.iterdir():
        if candidate == current_run or not candidate.is_dir():
            continue
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
        except (json.JSONDecodeError, OSError):
            continue
        candidate_time = _manifest_time(manifest, candidate)
        if candidate_time >= current_time:
            continue
        if comparison_key(manifest) != current_key:
            continue
        if not (candidate / "instances").is_dir():
            continue
        candidates.append((candidate_time, candidate))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def diff_snapshots(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute additions, removals, and frequency changes by stable signal ID."""
    base = {
        "schema_version": 1,
        "current_run": current["run_id"],
        "previous_run": previous["run_id"] if previous else None,
        "current_harness_commit": current.get("harness_commit", ""),
        "current_harness_worktree_dirty": bool(
            current.get("harness_worktree_dirty", False)
        ),
        "previous_harness_commit": (
            previous.get("harness_commit", "") if previous else None
        ),
        "previous_harness_worktree_dirty": (
            bool(previous.get("harness_worktree_dirty", False))
            if previous else None
        ),
        "comparable": previous is not None,
        "new_signals": [],
        "disappeared_signals": [],
        "frequency_changes": [],
        "new_observation_gaps": [],
        "closed_observation_gaps": [],
    }
    if previous is None:
        return base

    before = {row["signal_id"]: row for row in previous.get("signals", [])}
    after = {row["signal_id"]: row for row in current.get("signals", [])}
    base["new_signals"] = [after[key] for key in sorted(after.keys() - before.keys())]
    base["disappeared_signals"] = [
        before[key] for key in sorted(before.keys() - after.keys())
    ]
    changes: list[dict[str, Any]] = []
    for key in sorted(before.keys() & after.keys()):
        old = before[key]
        new = after[key]
        old_rate = old.get("rate")
        new_rate = new.get("rate")
        count_delta = int(new["count"]) - int(old["count"])
        rate_delta = (
            float(new_rate) - float(old_rate)
            if old_rate is not None and new_rate is not None
            else None
        )
        if count_delta == 0 and (rate_delta is None or rate_delta == 0):
            continue
        changes.append(asdict(SignalChange(
            signal_id=key,
            category=new["category"],
            label=new["label"],
            before_count=int(old["count"]),
            after_count=int(new["count"]),
            count_delta=count_delta,
            before_rate=old_rate,
            after_rate=new_rate,
            rate_delta=rate_delta,
        )))
    base["frequency_changes"] = changes

    old_gaps = set(previous.get("observation_gaps", []))
    new_gaps = set(current.get("observation_gaps", []))
    base["new_observation_gaps"] = sorted(new_gaps - old_gaps)
    base["closed_observation_gaps"] = sorted(old_gaps - new_gaps)
    return base


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_diff(diff: dict[str, Any], current: dict[str, Any]) -> str:
    lines = [
        "# Harness Diagnostic Diff",
        "",
        f"- Current run: `{diff['current_run']}`",
        f"- Previous comparable run: `{diff['previous_run'] or 'unavailable'}`",
        f"- Current harness: `{current.get('harness_commit') or 'unavailable'}` "
        f"({'dirty' if current.get('harness_worktree_dirty') else 'clean'})",
        "",
    ]
    if not diff["comparable"]:
        lines += [
            "> No older run has the same dataset, instance set, model, provider, "
            "API budget, reasoning effort, and output-token configuration. "
            "A baseline was not guessed.",
            "",
        ]
        return "\n".join(lines)

    lines.insert(4, (
        f"- Previous harness: `{diff.get('previous_harness_commit') or 'unavailable'}` "
        f"({'dirty' if diff.get('previous_harness_worktree_dirty') else 'clean'})"
    ))
    if (
        diff.get("current_harness_worktree_dirty")
        or diff.get("previous_harness_worktree_dirty")
    ):
        lines += [
            "> **Warning**: at least one compared run used a dirty harness. "
            "The diff is descriptive and cannot independently prove a regression.",
            "",
        ]

    lines += [
        "## Summary",
        "",
        f"- New diagnostic signals: **{len(diff['new_signals'])}**",
        f"- Disappeared diagnostic signals: **{len(diff['disappeared_signals'])}**",
        f"- Frequency changes: **{len(diff['frequency_changes'])}**",
        f"- New observation gaps: **{len(diff['new_observation_gaps'])}**",
        f"- Closed observation gaps: **{len(diff['closed_observation_gaps'])}**",
        "",
    ]
    for title, key in (
        ("New diagnostic signals", "new_signals"),
        ("Disappeared diagnostic signals", "disappeared_signals"),
    ):
        lines += [f"## {title}", ""]
        rows = diff[key]
        if not rows:
            lines += ["_None._", ""]
            continue
        lines += ["| Category | Signal | Count | Rate |", "|---|---|---:|---:|"]
        for row in rows:
            lines.append(
                f"| {row['category']} | {row['label']} | {row['count']} | "
                f"{_rate(row.get('rate'))} |"
            )
        lines.append("")

    lines += ["## Frequency changes", ""]
    if not diff["frequency_changes"]:
        lines += ["_None._", ""]
    else:
        lines += [
            "| Category | Signal | Count | Delta | Rate | Rate delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in diff["frequency_changes"]:
            rate_delta = row.get("rate_delta")
            lines.append(
                f"| {row['category']} | {row['label']} | "
                f"{row['before_count']} → {row['after_count']} | "
                f"{row['count_delta']:+d} | "
                f"{_rate(row.get('before_rate'))} → {_rate(row.get('after_rate'))} | "
                f"{'n/a' if rate_delta is None else f'{rate_delta:+.1%}'} |"
            )
        lines.append("")

    for title, key in (
        ("New observation gaps", "new_observation_gaps"),
        ("Closed observation gaps", "closed_observation_gaps"),
    ):
        lines += [f"## {title}", ""]
        rows = diff[key]
        lines += ([f"- {row}" for row in rows] if rows else ["_None._"])
        lines.append("")
    return "\n".join(lines)


def generate_run_diagnostics(run_dir: Path) -> dict[str, Path | None]:
    """Generate per-run diagnostics and compare with the previous like-for-like run."""
    manifest = _read_json(run_dir / "manifest.json")
    data = analyze_attempts(load_run(run_dir))
    snapshot = build_snapshot(run_dir, manifest, data)
    _atomic_write(run_dir / DIAGNOSTIC_MARKDOWN, render_report(data))
    _atomic_write(
        run_dir / DIAGNOSTIC_JSON,
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
    )

    previous_dir = find_previous_comparable_run(run_dir, manifest)
    previous_snapshot = None
    if previous_dir is not None:
        # Re-analyze the raw previous run with the current analyzer. Cached
        # snapshots may predate a retry or use an older signal schema.
        previous_manifest = _read_json(previous_dir / "manifest.json")
        previous_data = analyze_attempts(load_run(previous_dir))
        previous_snapshot = build_snapshot(
            previous_dir, previous_manifest, previous_data,
        )
    diff = diff_snapshots(snapshot, previous_snapshot)
    _atomic_write(
        run_dir / DIFF_JSON,
        json.dumps(diff, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write(run_dir / DIFF_MARKDOWN, render_diff(diff, snapshot))
    return {
        "diagnostic": run_dir / DIAGNOSTIC_MARKDOWN,
        "diff": run_dir / DIFF_MARKDOWN,
        "previous_run": previous_dir,
    }
