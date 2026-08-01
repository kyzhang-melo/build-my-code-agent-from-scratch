"""Harness Diagnostic Report generator.

Produces a Markdown report from scanned SWE-bench artifacts. The report
has three layers:

- **Observed anomalies**: facts directly provable from trace events.
- **Defect candidates**: patterns that *may* indicate a harness issue,
  requiring human or micro-eval confirmation.
- **Observation gaps**: questions the current trace cannot answer.

Every metric is computed from structured trace fields only. Legacy
traces (pre-Phase-0) lack ``validation_issues``, ``raw_arguments_*``,
``api_call``, ``step_index``, and split truncation fields; the report
marks these as ``legacy`` rather than guessing.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.analyze.scanner import (
    AttemptRef,
    DEFAULT_RUNS_DIR,
    ToolCallPair,
    extract_tool_calls,
    load_runs,
)


# ---------------------------------------------------------------------------
# Analysis data structures
# ---------------------------------------------------------------------------


@dataclass
class SeenRef:
    run_id: str
    model: str
    harness_commit: str
    harness_dirty: bool
    created_at: str


@dataclass
class ToolFailureRow:
    model: str
    tool_name: str
    error_type: str
    total_calls: int
    failed_calls: int
    failure_rate: float
    affected_attempts: int
    incomplete_calls: int
    first_seen: SeenRef
    last_seen: SeenRef


@dataclass
class ParamFrictionRow:
    model: str
    tool_name: str
    field_path: str
    issue_type: str
    occurrences: int
    affected_attempts: int
    first_seen: SeenRef
    last_seen: SeenRef


@dataclass
class AnomalyRow:
    model: str
    kind: str  # "consecutive_same_error" | "exact_repeat" | "permission_deny_loop"
    tool_name: str
    error_type: str
    count: int
    affected_attempts: int
    first_seen: SeenRef
    last_seen: SeenRef
    detail: str


@dataclass
class DefectCandidateRow:
    kind: str  # "budget_bound" | "no_patch"
    run_id: str
    instance_id: str
    model: str
    agent_status: str
    patch_status: str
    api_calls: int
    detail: str


@dataclass
class ReportData:
    # Section 1: coverage
    total_runs: int = 0
    total_attempts: int = 0
    total_tool_calls: int = 0
    legacy_attempts: int = 0
    dirty_attempts: int = 0
    validation_unavailable_attempts: int = 0
    fingerprint_unavailable_attempts: int = 0
    api_step_unavailable_attempts: int = 0
    truncation_unavailable_attempts: int = 0
    incomplete_tool_calls: int = 0
    malformed_trace_lines: int = 0
    no_trace_attempts: int = 0
    unavailable_result_attempts: int = 0
    unavailable_manifest_attempts: int = 0
    models: list[str] = field(default_factory=list)
    runs_per_model: dict[str, int] = field(default_factory=dict)
    attempts_per_model: dict[str, int] = field(default_factory=dict)

    # Section 2: tool failure matrix
    failure_rows: list[ToolFailureRow] = field(default_factory=list)

    # Section 3: parameter friction
    friction_rows: list[ParamFrictionRow] = field(default_factory=list)
    friction_legacy_note: bool = False

    # Section 4: anomaly sequences
    anomaly_rows: list[AnomalyRow] = field(default_factory=list)
    anomaly_legacy_note: bool = False

    # Defect candidates
    defect_candidates: list[DefectCandidateRow] = field(default_factory=list)

    # Observation gaps
    observation_gaps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def _seen_ref(attempt: AttemptRef) -> SeenRef:
    return SeenRef(
        run_id=attempt.run_id,
        model=attempt.model,
        harness_commit=attempt.harness_commit,
        harness_dirty=attempt.harness_dirty,
        created_at=attempt.created_at,
    )


def _first_last(attempts: list[AttemptRef]) -> tuple[SeenRef, SeenRef]:
    ordered = sorted(attempts, key=lambda a: (
        a.created_at or "9999",
        a.run_id,
        a.instance_id,
        a.attempt,
    ))
    return _seen_ref(ordered[0]), _seen_ref(ordered[-1])


def _has_tool_events(attempt: AttemptRef) -> bool:
    return any(
        event.get("event") in {"tool.requested", "tool.completed"}
        for event in attempt.trace_events
    )


def analyze(runs_dir: Path = DEFAULT_RUNS_DIR) -> ReportData:
    """Load all artifacts and compute the full report data."""
    return analyze_attempts(load_runs(runs_dir))


def analyze_attempts(attempts: list[AttemptRef]) -> ReportData:
    """Compute a report from an explicit, already-scanned attempt set."""
    report = ReportData()
    if not attempts:
        return report

    _analyze_coverage(attempts, report)
    _analyze_tool_failures(attempts, report)
    _analyze_param_friction(attempts, report)
    _analyze_anomaly_sequences(attempts, report)
    _analyze_defect_candidates(attempts, report)
    _collect_observation_gaps(attempts, report)
    return report


def _analyze_coverage(attempts: list[AttemptRef], report: ReportData) -> None:
    report.total_runs = len({a.run_id for a in attempts})
    report.total_attempts = len(attempts)
    relevant = [a for a in attempts if _has_tool_events(a)]
    report.legacy_attempts = sum(1 for a in relevant if not a.has_phase0_fields)
    report.dirty_attempts = sum(1 for a in attempts if a.harness_dirty)
    report.validation_unavailable_attempts = sum(
        1 for a in relevant if not a.has_validation_issues
    )
    report.fingerprint_unavailable_attempts = sum(
        1 for a in relevant if not a.has_raw_fingerprint
    )
    report.api_step_unavailable_attempts = sum(
        1 for a in relevant if not a.has_api_step
    )
    report.truncation_unavailable_attempts = sum(
        1 for a in relevant if not a.has_split_truncation
    )
    report.malformed_trace_lines = sum(a.trace_parse_errors for a in attempts)
    report.no_trace_attempts = sum(1 for a in attempts if a.trace_missing)
    report.unavailable_result_attempts = sum(
        1 for a in attempts if a.result_unavailable
    )
    report.unavailable_manifest_attempts = sum(
        1 for a in attempts if a.manifest_unavailable
    )

    models = sorted({a.model for a in attempts if a.model})
    report.models = models
    report.runs_per_model = dict(collections.Counter(
        a.model for a in attempts if a.model
    ))
    # Count unique runs per model
    runs_per_model: dict[str, set[str]] = collections.defaultdict(set)
    for a in attempts:
        if a.model:
            runs_per_model[a.model].add(a.run_id)
    report.attempts_per_model = dict(collections.Counter(
        a.model for a in attempts if a.model
    ))
    report.runs_per_model = {m: len(runs) for m, runs in runs_per_model.items()}

    total_calls = 0
    for a in attempts:
        total_calls += sum(
            1 for e in a.trace_events if e.get("event") == "tool.requested"
        )
        report.incomplete_tool_calls += sum(
            1 for call in extract_tool_calls(a) if call.completed is None
        )
    report.total_tool_calls = total_calls


def _analyze_tool_failures(attempts: list[AttemptRef], report: ReportData) -> None:
    """Build model x tool x error_type failure matrix."""
    # Denominators are all requests for one model+tool. Error-specific rows use
    # that shared denominator, otherwise every error bucket reports 100%.
    totals: collections.Counter[tuple[str, str]] = collections.Counter()
    incomplete: collections.Counter[tuple[str, str]] = collections.Counter()
    failures: dict[tuple[str, str, str], dict[str, Any]] = {}
    for a in attempts:
        calls = extract_tool_calls(a)
        for c in calls:
            tool_key = (a.model, c.tool_name)
            totals[tool_key] += 1
            if c.completed is None or c.success is None:
                incomplete[tool_key] += 1
                continue
            if c.success:
                continue
            et = c.error_type or "unknown"
            key = (a.model, c.tool_name, et)
            if key not in failures:
                failures[key] = {
                    "count": 0,
                    "attempt_ids": set(),
                    "attempts": [],
                }
            failures[key]["count"] += 1
            failures[key]["attempt_ids"].add(a.identity)
            failures[key]["attempts"].append(a)

    rows: list[ToolFailureRow] = []
    for (model, tool, et), data in failures.items():
        total = totals[(model, tool)]
        failed = data["count"]
        first, last = _first_last(data["attempts"])
        rows.append(ToolFailureRow(
            model=model,
            tool_name=tool,
            error_type=et,
            total_calls=total,
            failed_calls=failed,
            failure_rate=failed / total if total else 0.0,
            affected_attempts=len(data["attempt_ids"]),
            incomplete_calls=incomplete[(model, tool)],
            first_seen=first,
            last_seen=last,
        ))
    # Sort by failed_calls desc, then model, then tool
    rows.sort(key=lambda r: (-r.failed_calls, r.model, r.tool_name, r.error_type))
    report.failure_rows = rows


def _analyze_param_friction(attempts: list[AttemptRef], report: ReportData) -> None:
    """Top-N parameter validation issues from validation_issues field."""
    # Model is part of the key: model-specific rows are the primary report.
    agg: dict[tuple[str, str, str, str], dict] = {}
    legacy_count = 0
    for a in attempts:
        if _has_tool_events(a) and not a.has_validation_issues:
            legacy_count += 1
            continue
        calls = extract_tool_calls(a)
        for c in calls:
            for issue in c.validation_issues:
                path = issue.get("path", "")
                itype = issue.get("type", "")
                key = (a.model, c.tool_name, path, itype)
                if key not in agg:
                    agg[key] = {
                        "occurrences": 0,
                        "attempt_ids": set(),
                        "attempts": [],
                    }
                agg[key]["occurrences"] += 1
                agg[key]["attempt_ids"].add(a.identity)
                agg[key]["attempts"].append(a)

    report.friction_legacy_note = legacy_count > 0

    rows: list[ParamFrictionRow] = []
    for (model, tool, path, itype), data in agg.items():
        first, last = _first_last(data["attempts"])
        rows.append(ParamFrictionRow(
            model=model,
            tool_name=tool,
            field_path=path,
            issue_type=itype,
            occurrences=data["occurrences"],
            affected_attempts=len(data["attempt_ids"]),
            first_seen=first,
            last_seen=last,
        ))
    rows.sort(key=lambda r: (
        r.model,
        -r.affected_attempts,
        -r.occurrences,
        r.tool_name,
        r.field_path,
    ))
    report.friction_rows = rows


def _analyze_anomaly_sequences(attempts: list[AttemptRef], report: ReportData) -> None:
    """Detect consecutive same-type errors, exact repeats, and permission deny loops."""
    consec_agg: dict[tuple[str, str, str], dict] = {}
    repeat_agg: dict[tuple[str, str, str, int, str], dict] = {}
    deny_agg: dict[tuple[str, str], dict] = {}

    legacy_count = 0
    for a in attempts:
        if _has_tool_events(a) and not a.has_raw_fingerprint:
            legacy_count += 1
        calls = extract_tool_calls(a)

        # Consecutive same-type errors
        prev_key: tuple[str, str] | None = None
        for c in calls:
            if c.success is False and c.error_type:
                key = (c.tool_name, c.error_type)
                if key == prev_key:
                    agg_key = (a.model, *key)
                    if agg_key not in consec_agg:
                        consec_agg[agg_key] = {
                            "count": 0, "attempt_ids": set(), "attempts": [],
                        }
                    consec_agg[agg_key]["count"] += 1
                    consec_agg[agg_key]["attempt_ids"].add(a.identity)
                    consec_agg[agg_key]["attempts"].append(a)
                prev_key = key
            else:
                prev_key = None

        # Exact repeat: same source+tool+length+hash within one attempt.
        if a.has_raw_fingerprint:
            seen_hashes: collections.Counter[tuple[str, str, int, str]] = (
                collections.Counter()
            )
            for c in calls:
                h = c.raw_arguments_sha256
                if not h:
                    continue
                seen_hashes[(
                    c.source,
                    c.tool_name,
                    c.raw_arguments_chars,
                    h,
                )] += 1
            for (source, tool, chars, h), count in seen_hashes.items():
                if count > 1:
                    key = (a.model, source, tool, chars, h)
                    if key not in repeat_agg:
                        repeat_agg[key] = {
                            "count": 0, "attempt_ids": set(), "attempts": [],
                        }
                    repeat_agg[key]["count"] += count - 1
                    repeat_agg[key]["attempt_ids"].add(a.identity)
                    repeat_agg[key]["attempts"].append(a)

        # Permission deny follow-up: only count the same tool in a strictly
        # later API call. Calls submitted in the same model response are not a
        # reaction to the denial and therefore are not counted.
        if not a.has_api_step:
            continue
        requests_by_call = {
            c.call_id: c for c in calls
        }
        for e in a.trace_events:
            if (
                e.get("event") == "permission.decided"
                and e.get("decision") == "deny"
            ):
                denied = requests_by_call.get(e.get("call_id", ""))
                if denied is None or not isinstance(denied.api_call, int):
                    continue
                followed_up = any(
                    later.tool_name == denied.tool_name
                    and later.source == denied.source
                    and isinstance(later.api_call, int)
                    and later.api_call > denied.api_call
                    for later in calls
                )
                if not followed_up:
                    continue
                key = (a.model, denied.tool_name)
                if key not in deny_agg:
                    deny_agg[key] = {
                        "count": 0, "attempt_ids": set(), "attempts": [],
                    }
                deny_agg[key]["count"] += 1
                deny_agg[key]["attempt_ids"].add(a.identity)
                deny_agg[key]["attempts"].append(a)

    report.anomaly_legacy_note = legacy_count > 0

    rows: list[AnomalyRow] = []
    for (model, tool, et), data in consec_agg.items():
        first, last = _first_last(data["attempts"])
        rows.append(AnomalyRow(
            model=model,
            kind="consecutive_same_error",
            tool_name=tool,
            error_type=et,
            count=data["count"],
            affected_attempts=len(data["attempt_ids"]),
            first_seen=first,
            last_seen=last,
            detail=f"Same tool+error back-to-back: {tool} / {et}",
        ))
    for (model, source, tool, chars, h), data in repeat_agg.items():
        first, last = _first_last(data["attempts"])
        rows.append(AnomalyRow(
            model=model,
            kind="exact_repeat",
            tool_name=tool,
            error_type="",
            count=data["count"],
            affected_attempts=len(data["attempt_ids"]),
            first_seen=first,
            last_seen=last,
            detail=(
                f"Repeated identical raw arguments {data['count']} time(s) "
                f"beyond the first (source={source}, chars={chars}, "
                f"sha256={h[:8]}...)"
            ),
        ))
    for (model, tool), data in deny_agg.items():
        first, last = _first_last(data["attempts"])
        rows.append(AnomalyRow(
            model=model,
            kind="permission_deny_loop",
            tool_name=tool,
            error_type="permission_denied",
            count=data["count"],
            affected_attempts=len(data["attempt_ids"]),
            first_seen=first,
            last_seen=last,
            detail=f"Permission denied, then {tool} requested in a later API call",
        ))
    rows.sort(key=lambda r: (r.model, -r.count, -r.affected_attempts, r.kind))
    report.anomaly_rows = rows


def _analyze_defect_candidates(attempts: list[AttemptRef], report: ReportData) -> None:
    """Budget-bound and no-patch candidates."""
    rows: list[DefectCandidateRow] = []
    for a in attempts:
        if a.result_unavailable:
            continue
        # Budget-bound: max_api_calls AND trace shows tool activity near the end
        if a.agent_status == "max_api_calls":
            calls = extract_tool_calls(a)
            # Check if last 3 calls include an edit_file or write_file
            last_tools = [c.tool_name for c in calls[-3:]] if calls else []
            editing_near_end = any(t in ("edit_file", "write_file") for t in last_tools)
            detail = "last tools: " + ", ".join(last_tools) if last_tools else "no tool calls"
            if editing_near_end:
                detail += " (editing near budget exhaustion)"
            rows.append(DefectCandidateRow(
                kind="budget_bound",
                run_id=a.run_id,
                instance_id=a.instance_id,
                model=a.model,
                agent_status=a.agent_status,
                patch_status=a.patch_status,
                api_calls=a.api_calls,
                detail=detail,
            ))
        # No patch
        if a.patch_status != "produced":
            rows.append(DefectCandidateRow(
                kind="no_patch",
                run_id=a.run_id,
                instance_id=a.instance_id,
                model=a.model,
                agent_status=a.agent_status,
                patch_status=a.patch_status,
                api_calls=a.api_calls,
                detail=f"patch_status={a.patch_status}",
            ))
    report.defect_candidates = rows


def _collect_observation_gaps(attempts: list[AttemptRef], report: ReportData) -> None:
    """List questions the current trace cannot answer."""
    gaps: list[str] = []
    unavailable = (
        ("validation_issues", report.validation_unavailable_attempts),
        ("raw_arguments fingerprint", report.fingerprint_unavailable_attempts),
        ("api_call/step_index", report.api_step_unavailable_attempts),
        ("split truncation", report.truncation_unavailable_attempts),
    )
    for field_name, count in unavailable:
        if count:
            gaps.append(
                f"{count} attempts with tool events lack complete {field_name} "
                "instrumentation; that metric is unavailable for those attempts."
            )
    if report.no_trace_attempts > 0:
        gaps.append(
            f"{report.no_trace_attempts} attempts have no trace file. "
            "Tool-level analysis is unavailable for these."
        )
    if report.malformed_trace_lines > 0:
        gaps.append(
            f"{report.malformed_trace_lines} malformed trace lines were excluded. "
            "Metrics for their attempts may be incomplete."
        )
    if report.incomplete_tool_calls > 0:
        gaps.append(
            f"{report.incomplete_tool_calls} tool requests have no matching "
            "tool.completed event. Their outcomes are classified as incomplete, "
            "not successful."
        )
    if report.unavailable_result_attempts > 0:
        gaps.append(
            f"{report.unavailable_result_attempts} attempt results are missing or "
            "malformed; result-level candidate analysis is unavailable for them."
        )
    if report.unavailable_manifest_attempts > 0:
        gaps.append(
            f"{report.unavailable_manifest_attempts} attempts have a missing or "
            "malformed run manifest; model and harness provenance may be unavailable."
        )
    # Context compaction events
    has_compact = any(
        any(e.get("event", "").startswith("compact") for e in a.trace_events)
        for a in attempts
    )
    if not has_compact:
        gaps.append(
            "No context compaction events in any trace. "
            "Cannot assess whether compaction dropped task-critical information."
        )
    # Token usage per call
    has_token_usage = any(
        any("input_tokens" in e or "output_tokens" in e for e in a.trace_events)
        for a in attempts
    )
    if not has_token_usage:
        gaps.append(
            "No per-call token usage in trace. "
            "Cannot compute cost efficiency or token-budget curves."
        )
    report.observation_gaps = gaps


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_report(data: ReportData) -> str:
    """Render the report data as a Markdown string."""
    lines: list[str] = []
    lines.append("# Harness Diagnostic Report")
    lines.append("")
    lines.append("> Deterministic scan of SWE-bench eval artifacts.")
    lines.append("> Every metric is provable by structured trace evidence.")
    lines.append("> Defect candidates require human or micro-eval confirmation.")
    lines.append("")

    _render_coverage(data, lines)
    _render_tool_failures(data, lines)
    _render_param_friction(data, lines)
    _render_anomaly_sequences(data, lines)
    _render_defect_candidates(data, lines)
    _render_observation_gaps(data, lines)

    return "\n".join(lines) + "\n"


def _render_seen(ref: SeenRef) -> str:
    dirty = "dirty" if ref.harness_dirty else "clean"
    commit = ref.harness_commit or "unknown"
    return f"{ref.run_id} ({commit}, {dirty})"


def _render_coverage(data: ReportData, lines: list[str]) -> None:
    lines.append("## 1. Data Coverage")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Runs | {data.total_runs} |")
    lines.append(f"| Attempts | {data.total_attempts} |")
    lines.append(f"| Tool calls | {data.total_tool_calls} |")
    lines.append(f"| Legacy attempts (pre-Phase-0) | {data.legacy_attempts} |")
    lines.append(f"| Dirty harness attempts | {data.dirty_attempts} |")
    lines.append(f"| Incomplete tool calls | {data.incomplete_tool_calls} |")
    lines.append(f"| Malformed trace lines | {data.malformed_trace_lines} |")
    lines.append(f"| Attempts without trace file | {data.no_trace_attempts} |")
    lines.append("")
    lines.append("| Phase-0 capability | Attempts unavailable |")
    lines.append("|---|---|")
    lines.append(
        f"| validation_issues | {data.validation_unavailable_attempts} |"
    )
    lines.append(
        f"| raw arguments fingerprint | {data.fingerprint_unavailable_attempts} |"
    )
    lines.append(f"| api_call / step_index | {data.api_step_unavailable_attempts} |")
    lines.append(
        f"| split truncation | {data.truncation_unavailable_attempts} |"
    )
    lines.append("")
    if data.models:
        lines.append("| Model | Runs | Attempts |")
        lines.append("|---|---|---|")
        for m in data.models:
            lines.append(
                f"| {m} | {data.runs_per_model.get(m, 0)} | "
                f"{data.attempts_per_model.get(m, 0)} |"
            )
        lines.append("")
    if data.dirty_attempts > 0:
        lines.append(
            f"> **Warning**: {data.dirty_attempts} attempts ran with a dirty harness "
            "worktree. Their commit labels do not uniquely identify the executed "
            "harness and cannot independently prove a regression."
        )
        lines.append("")


def _render_tool_failures(data: ReportData, lines: list[str]) -> None:
    lines.append("## 2. Tool Failure Matrix (model x tool x error_type)")
    lines.append("")
    if not data.failure_rows:
        lines.append("_No tool failures detected._")
        lines.append("")
        return
    lines.append(
        "| Model | Tool | Error type | Requests | Failed | Rate | Incomplete | "
        "Attempts | First seen | Last seen |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in data.failure_rows:
        lines.append(
            f"| {r.model} | {r.tool_name} | {r.error_type} | "
            f"{r.total_calls} | {r.failed_calls} | "
            f"{r.failure_rate:.1%} | {r.incomplete_calls} | "
            f"{r.affected_attempts} | {_render_seen(r.first_seen)} | "
            f"{_render_seen(r.last_seen)} |"
        )
    lines.append("")


def _render_param_friction(data: ReportData, lines: list[str]) -> None:
    lines.append("## 3. Parameter Friction Top-N (validation_issues)")
    lines.append("")
    if data.friction_legacy_note:
        lines.append(
            "> **Availability note**: attempts without structured "
            "`validation_issues` are excluded from this section; no values are "
            "inferred from `agent.log`."
        )
        lines.append("")
    if not data.friction_rows:
        lines.append("_No structured validation_issues detected._")
        lines.append("")
        return
    lines.append(
        "| Model | Tool | Field | Type | Occurrences | Attempts | "
        "First seen | Last seen |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in data.friction_rows:
        lines.append(
            f"| {r.model} | {r.tool_name} | `{r.field_path}` | {r.issue_type} | "
            f"{r.occurrences} | {r.affected_attempts} | "
            f"{_render_seen(r.first_seen)} | {_render_seen(r.last_seen)} |"
        )
    lines.append("")


def _render_anomaly_sequences(data: ReportData, lines: list[str]) -> None:
    lines.append("## 4. Anomaly Sequences")
    lines.append("")
    if data.anomaly_legacy_note:
        lines.append(
            "> **Availability note**: exact-repeat detection requires "
            "`raw_arguments_sha256`. Attempts without it are excluded from "
            "exact-repeat counts. Consecutive-error detection still uses all "
            "completed tool traces."
        )
        lines.append("")
    if not data.anomaly_rows:
        lines.append("_No anomaly sequences detected._")
        lines.append("")
        return
    lines.append(
        "| Model | Kind | Tool | Error type | Count | Attempts | "
        "First seen | Last seen | Detail |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in data.anomaly_rows:
        lines.append(
            f"| {r.model} | {r.kind} | {r.tool_name} | {r.error_type} | "
            f"{r.count} | {r.affected_attempts} | "
            f"{_render_seen(r.first_seen)} | {_render_seen(r.last_seen)} | "
            f"{r.detail} |"
        )
    lines.append("")


def _render_defect_candidates(data: ReportData, lines: list[str]) -> None:
    lines.append("## Defect Candidates")
    lines.append("")
    lines.append(
        "> These patterns **may** indicate harness issues. "
        "They require human review or micro-eval confirmation before "
        "being treated as confirmed defects."
    )
    lines.append("")
    if not data.defect_candidates:
        lines.append("_No defect candidates detected._")
        lines.append("")
        return
    budget = [r for r in data.defect_candidates if r.kind == "budget_bound"]
    no_patch = [r for r in data.defect_candidates if r.kind == "no_patch"]
    if budget:
        lines.append("### Budget-bound candidates")
        lines.append("")
        lines.append("| Run | Instance | Model | API calls | Patch | Detail |")
        lines.append("|---|---|---|---|---|---|")
        for r in budget:
            lines.append(
                f"| {r.run_id} | {r.instance_id} | {r.model} | "
                f"{r.api_calls} | {r.patch_status} | {r.detail} |"
            )
        lines.append("")
    if no_patch:
        lines.append("### No-patch candidates")
        lines.append("")
        lines.append("| Run | Instance | Model | Agent status | Patch | Detail |")
        lines.append("|---|---|---|---|---|---|")
        for r in no_patch:
            lines.append(
                f"| {r.run_id} | {r.instance_id} | {r.model} | "
                f"{r.agent_status} | {r.patch_status} | {r.detail} |"
            )
        lines.append("")


def _render_observation_gaps(data: ReportData, lines: list[str]) -> None:
    lines.append("## Observation Gaps")
    lines.append("")
    lines.append(
        "> Questions the current trace cannot answer. "
        "These identify instrumentation to add in future phases."
    )
    lines.append("")
    if not data.observation_gaps:
        lines.append("_No observation gaps identified._")
        lines.append("")
        return
    for i, gap in enumerate(data.observation_gaps, 1):
        lines.append(f"{i}. {gap}")
    lines.append("")
