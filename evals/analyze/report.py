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
class ToolFailureRow:
    model: str
    tool_name: str
    error_type: str
    total_calls: int
    failed_calls: int
    failure_rate: float
    affected_attempts: int


@dataclass
class ParamFrictionRow:
    tool_name: str
    field_path: str
    issue_type: str
    occurrences: int
    affected_attempts: int
    affected_models: int
    models: list[str]
    first_seen: str
    last_seen: str


@dataclass
class AnomalyRow:
    kind: str  # "consecutive_same_error" | "exact_repeat" | "permission_deny_loop"
    tool_name: str
    error_type: str
    count: int
    affected_attempts: int
    affected_models: int
    models: list[str]
    first_seen: str
    last_seen: str
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


def analyze(runs_dir: Path = DEFAULT_RUNS_DIR) -> ReportData:
    """Load all artifacts and compute the full report data."""
    attempts = load_runs(runs_dir)
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
    report.legacy_attempts = sum(1 for a in attempts if not a.has_phase0_fields)
    report.dirty_attempts = sum(1 for a in attempts if a.harness_dirty)

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
    report.total_tool_calls = total_calls


def _analyze_tool_failures(attempts: list[AttemptRef], report: ReportData) -> None:
    """Build model x tool x error_type failure matrix."""
    # (model, tool, error_type) -> [total, failed, attempt_ids]
    matrix: dict[tuple[str, str, str], list] = {}
    for a in attempts:
        calls = extract_tool_calls(a)
        for c in calls:
            et = c.error_type or "none"
            key = (a.model, c.tool_name, et)
            if key not in matrix:
                matrix[key] = [0, 0, set()]
            matrix[key][0] += 1
            if not c.success:
                matrix[key][1] += 1
            matrix[key][2].add(a.instance_id)

    rows: list[ToolFailureRow] = []
    for (model, tool, et), (total, failed, attempt_ids) in matrix.items():
        if et == "none" and failed == 0:
            continue  # skip pure-success rows with no error type
        rows.append(ToolFailureRow(
            model=model,
            tool_name=tool,
            error_type=et,
            total_calls=total,
            failed_calls=failed,
            failure_rate=failed / total if total else 0.0,
            affected_attempts=len(attempt_ids),
        ))
    # Sort by failed_calls desc, then model, then tool
    rows.sort(key=lambda r: (-r.failed_calls, r.model, r.tool_name, r.error_type))
    report.failure_rows = rows


def _analyze_param_friction(attempts: list[AttemptRef], report: ReportData) -> None:
    """Top-N parameter validation issues from validation_issues field."""
    # (tool, field_path, issue_type) -> {occurrences, attempt_ids, models, runs}
    agg: dict[tuple[str, str, str], dict] = {}
    legacy_count = 0
    for a in attempts:
        if not a.has_phase0_fields:
            legacy_count += 1
            continue
        calls = extract_tool_calls(a)
        for c in calls:
            for issue in c.validation_issues:
                path = issue.get("path", "")
                itype = issue.get("type", "")
                key = (c.tool_name, path, itype)
                if key not in agg:
                    agg[key] = {
                        "occurrences": 0,
                        "attempt_ids": set(),
                        "models": set(),
                        "runs": [],
                    }
                agg[key]["occurrences"] += 1
                agg[key]["attempt_ids"].add(a.instance_id)
                agg[key]["models"].add(a.model)
                agg[key]["runs"].append(a.run_id)

    report.friction_legacy_note = legacy_count > 0

    rows: list[ParamFrictionRow] = []
    for (tool, path, itype), data in agg.items():
        runs = data["runs"]
        rows.append(ParamFrictionRow(
            tool_name=tool,
            field_path=path,
            issue_type=itype,
            occurrences=data["occurrences"],
            affected_attempts=len(data["attempt_ids"]),
            affected_models=len(data["models"]),
            models=sorted(data["models"]),
            first_seen=runs[0] if runs else "",
            last_seen=runs[-1] if runs else "",
        ))
    # Sort by affected_attempts desc, then affected_models, then occurrences
    rows.sort(key=lambda r: (-r.affected_attempts, -r.affected_models, -r.occurrences))
    report.friction_rows = rows


def _analyze_anomaly_sequences(attempts: list[AttemptRef], report: ReportData) -> None:
    """Detect consecutive same-type errors, exact repeats, and permission deny loops."""
    # Consecutive same-type errors: same tool + same error_type back-to-back
    consec_agg: dict[tuple[str, str], dict] = {}
    # Exact repeat: same raw_arguments_sha256 submitted more than once in one attempt
    repeat_agg: dict[tuple[str, str], dict] = {}
    # Permission deny loop: permission denied then same tool called again
    deny_agg: dict[tuple[str], dict] = {}

    legacy_count = 0
    for a in attempts:
        if not a.has_phase0_fields:
            legacy_count += 1
        calls = extract_tool_calls(a)

        # Consecutive same-type errors
        prev_key: tuple[str, str] | None = None
        for c in calls:
            if not c.success and c.error_type:
                key = (c.tool_name, c.error_type)
                if key == prev_key:
                    if key not in consec_agg:
                        consec_agg[key] = {
                            "count": 0, "attempt_ids": set(), "models": set(), "runs": [],
                        }
                    consec_agg[key]["count"] += 1
                    consec_agg[key]["attempt_ids"].add(a.instance_id)
                    consec_agg[key]["models"].add(a.model)
                    consec_agg[key]["runs"].append(a.run_id)
                prev_key = key
            else:
                prev_key = None

        # Exact repeat (only if raw_arguments_sha256 is available)
        if a.has_phase0_fields:
            seen_hashes: dict[str, int] = {}
            for c in calls:
                h = c.raw_arguments_sha256
                if not h:
                    continue
                seen_hashes[h] = seen_hashes.get(h, 0) + 1
            for h, count in seen_hashes.items():
                if count > 1:
                    # Find the tool name for this hash
                    tool = next((c.tool_name for c in calls if c.raw_arguments_sha256 == h), "")
                    key = (tool, h)
                    if key not in repeat_agg:
                        repeat_agg[key] = {
                            "count": count - 1,  # repeats beyond the first
                            "attempt_ids": set(), "models": set(), "runs": [],
                        }
                    repeat_agg[key]["attempt_ids"].add(a.instance_id)
                    repeat_agg[key]["models"].add(a.model)
                    repeat_agg[key]["runs"].append(a.run_id)

        # Permission deny loop
        events = a.trace_events
        for i, e in enumerate(events):
            if (
                e.get("event") == "permission.decided"
                and e.get("decision") == "deny"
            ):
                tool = e.get("tool_name", "")
                # Check if the same tool is requested again within the next 5 events
                for j in range(i + 1, min(i + 6, len(events))):
                    ne = events[j]
                    if (
                        ne.get("event") == "tool.requested"
                        and ne.get("tool_name") == tool
                    ):
                        key = (tool,)
                        if key not in deny_agg:
                            deny_agg[key] = {
                                "count": 0, "attempt_ids": set(), "models": set(), "runs": [],
                            }
                        deny_agg[key]["count"] += 1
                        deny_agg[key]["attempt_ids"].add(a.instance_id)
                        deny_agg[key]["models"].add(a.model)
                        deny_agg[key]["runs"].append(a.run_id)
                        break

    report.anomaly_legacy_note = legacy_count > 0

    rows: list[AnomalyRow] = []
    for (tool, et), data in consec_agg.items():
        rows.append(AnomalyRow(
            kind="consecutive_same_error",
            tool_name=tool,
            error_type=et,
            count=data["count"],
            affected_attempts=len(data["attempt_ids"]),
            affected_models=len(data["models"]),
            models=sorted(data["models"]),
            first_seen=data["runs"][0] if data["runs"] else "",
            last_seen=data["runs"][-1] if data["runs"] else "",
            detail=f"Same tool+error back-to-back: {tool} / {et}",
        ))
    for (tool, h), data in repeat_agg.items():
        rows.append(AnomalyRow(
            kind="exact_repeat",
            tool_name=tool,
            error_type="",
            count=data["count"],
            affected_attempts=len(data["attempt_ids"]),
            affected_models=len(data["models"]),
            models=sorted(data["models"]),
            first_seen=data["runs"][0] if data["runs"] else "",
            last_seen=data["runs"][-1] if data["runs"] else "",
            detail=f"Identical raw arguments submitted {data['count'] + 1}x (sha256={h[:8]}...)",
        ))
    for (tool,), data in deny_agg.items():
        rows.append(AnomalyRow(
            kind="permission_deny_loop",
            tool_name=tool,
            error_type="permission_denied",
            count=data["count"],
            affected_attempts=len(data["attempt_ids"]),
            affected_models=len(data["models"]),
            models=sorted(data["models"]),
            first_seen=data["runs"][0] if data["runs"] else "",
            last_seen=data["runs"][-1] if data["runs"] else "",
            detail=f"Permission denied then same tool re-requested: {tool}",
        ))
    rows.sort(key=lambda r: (-r.count, -r.affected_attempts))
    report.anomaly_rows = rows


def _analyze_defect_candidates(attempts: list[AttemptRef], report: ReportData) -> None:
    """Budget-bound and no-patch candidates."""
    rows: list[DefectCandidateRow] = []
    for a in attempts:
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
    legacy = sum(1 for a in attempts if not a.has_phase0_fields)
    if legacy > 0:
        gaps.append(
            f"{legacy} attempts have legacy traces (pre-Phase-0). "
            "validation_issues, raw_arguments fingerprint, api_call/step_index, "
            "and split truncation fields are unavailable for these."
        )
    # Check if any attempt has no trace at all
    no_trace = sum(1 for a in attempts if not a.trace_events)
    if no_trace > 0:
        gaps.append(
            f"{no_trace} attempts have no trace events. "
            "Tool-level analysis is unavailable for these."
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
            "worktree. Run-level grouping by commit is unreliable; use observable "
            "harness features (e.g. error message text) for regression detection."
        )
        lines.append("")


def _render_tool_failures(data: ReportData, lines: list[str]) -> None:
    lines.append("## 2. Tool Failure Matrix (model x tool x error_type)")
    lines.append("")
    if not data.failure_rows:
        lines.append("_No tool failures detected._")
        lines.append("")
        return
    lines.append("| Model | Tool | Error type | Total | Failed | Rate | Attempts |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in data.failure_rows:
        lines.append(
            f"| {r.model} | {r.tool_name} | {r.error_type} | "
            f"{r.total_calls} | {r.failed_calls} | "
            f"{r.failure_rate:.1%} | {r.affected_attempts} |"
        )
    lines.append("")


def _render_param_friction(data: ReportData, lines: list[str]) -> None:
    lines.append("## 3. Parameter Friction Top-N (validation_issues)")
    lines.append("")
    if data.friction_legacy_note:
        lines.append(
            "> **Legacy note**: some attempts predate Phase-0 instrumentation "
            "and are excluded from this section. Historical `-n`/`-C` grep issues "
            "are only visible in `agent.log` text, not in structured trace."
        )
        lines.append("")
    if not data.friction_rows:
        lines.append("_No structured validation_issues detected._")
        lines.append("")
        return
    lines.append(
        "| Tool | Field | Type | Occurrences | Attempts | Models | "
        "First seen | Last seen |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in data.friction_rows:
        lines.append(
            f"| {r.tool_name} | `{r.field_path}` | {r.issue_type} | "
            f"{r.occurrences} | {r.affected_attempts} | {r.affected_models} | "
            f"{r.first_seen} | {r.last_seen} |"
        )
    lines.append("")


def _render_anomaly_sequences(data: ReportData, lines: list[str]) -> None:
    lines.append("## 4. Anomaly Sequences")
    lines.append("")
    if data.anomaly_legacy_note:
        lines.append(
            "> **Legacy note**: exact-repeat detection requires "
            "`raw_arguments_sha256` (Phase-0). Legacy attempts are excluded "
            "from exact-repeat counts. Consecutive-error detection works on "
            "all traces."
        )
        lines.append("")
    if not data.anomaly_rows:
        lines.append("_No anomaly sequences detected._")
        lines.append("")
        return
    lines.append(
        "| Kind | Tool | Error type | Count | Attempts | Models | "
        "First seen | Last seen | Detail |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in data.anomaly_rows:
        lines.append(
            f"| {r.kind} | {r.tool_name} | {r.error_type} | "
            f"{r.count} | {r.affected_attempts} | {r.affected_models} | "
            f"{r.first_seen} | {r.last_seen} | {r.detail} |"
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
