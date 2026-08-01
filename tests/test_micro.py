"""Pytest coverage for the decision-point micro-eval framework.

These tests verify the micro-eval runner, check helpers, and case
definitions work correctly. They do NOT call a live model — they use
the real tool dispatcher with synthetic inputs.
"""

from __future__ import annotations

import asyncio
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import tools

from evals.micro.cases import (
    ALL_CASES,
    EDIT_FILE_CASES,
    GREP_CASES,
    TRACE_CASES,
    cases_by_defect_id,
    prepare_workspace,
)
from evals.micro.checks import (
    output_contains,
    output_not_contains,
    output_starts_with,
    trace_error_type_is,
    trace_has_raw_fingerprint,
    trace_has_validation_issues,
    trace_status_is,
    trace_success_is,
    trace_validation_issue_contains,
)
from evals.micro.runner import MicroCheck, MicroResult, run_micro_eval
from workspace import Workspace
from evals.micro.paired import (
    CriterionResult,
    PairedTrialResult,
    VariantMetrics,
    _candidate_grep_schema,
    aggregate,
    apply_harness_variant,
    evaluate_acceptance,
    normalize_grep_n_alias,
)


# ---------------------------------------------------------------------------
# Case definition tests
# ---------------------------------------------------------------------------


def test_all_cases_have_unique_names() -> None:
    names = [c.name for c in ALL_CASES]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


def test_all_cases_have_defect_id() -> None:
    for case in ALL_CASES:
        assert case.defect_id, f"case {case.name} has no defect_id"


def test_all_cases_have_checks() -> None:
    for case in ALL_CASES:
        assert case.checks, f"case {case.name} has no checks"


def test_cases_by_defect_id_filters_correctly() -> None:
    grep_cases = cases_by_defect_id("failure_matrix:grep/validation")
    assert len(grep_cases) == 5
    assert all(c.tool_name == "grep" for c in grep_cases)


def test_grep_cases_cover_all_cli_flags() -> None:
    flag_names = {c.name for c in GREP_CASES}
    assert "grep-rejects-n-flag" in flag_names
    assert "grep-rejects-C-flag" in flag_names
    assert "grep-rejects-A-flag" in flag_names
    assert "grep-rejects-B-flag" in flag_names
    assert "grep-accepts-valid-params" in flag_names


def test_edit_file_cases_cover_json_parse_and_validation() -> None:
    defect_ids = {c.defect_id for c in EDIT_FILE_CASES}
    assert "failure_matrix:edit_file/json_parse" in defect_ids
    assert "failure_matrix:edit_file/validation" in defect_ids


# ---------------------------------------------------------------------------
# Check helper tests
# ---------------------------------------------------------------------------


def test_output_contains_passes() -> None:
    check = output_contains("Error")
    result = check("Error: something bad", [])
    assert result.passed
    assert "Error" in result.label


def test_output_contains_fails() -> None:
    check = output_contains("Error")
    result = check("all good", [])
    assert not result.passed
    assert "not in output" in result.detail


def test_output_not_contains_passes() -> None:
    check = output_not_contains("SECRET")
    result = check("all good", [])
    assert result.passed


def test_output_not_contains_fails() -> None:
    check = output_not_contains("SECRET")
    result = check("SECRET data", [])
    assert not result.passed


def test_output_starts_with_passes() -> None:
    check = output_starts_with("Error:")
    result = check("Error: bad thing", [])
    assert result.passed


def test_trace_status_is_passes() -> None:
    events = [{"event": "tool.completed", "status": "success"}]
    check = trace_status_is("success")
    result = check("", events)
    assert result.passed


def test_trace_status_is_fails_on_wrong_status() -> None:
    events = [{"event": "tool.completed", "status": "invalid_arguments"}]
    check = trace_status_is("success")
    result = check("", events)
    assert not result.passed
    assert "invalid_arguments" in result.detail


def test_trace_status_is_fails_on_missing_event() -> None:
    check = trace_status_is("success")
    result = check("", [])
    assert not result.passed
    assert "no tool.completed" in result.detail


def test_trace_error_type_is_passes() -> None:
    events = [{"event": "tool.completed", "error_type": "validation"}]
    check = trace_error_type_is("validation")
    result = check("", events)
    assert result.passed


def test_trace_has_validation_issues_passes() -> None:
    events = [{"event": "tool.requested", "validation_issues": [{"path": "-n"}]}]
    check = trace_has_validation_issues()
    result = check("", events)
    assert result.passed


def test_trace_has_validation_issues_fails_on_empty() -> None:
    events = [{"event": "tool.requested", "validation_issues": []}]
    check = trace_has_validation_issues()
    result = check("", events)
    assert not result.passed


def test_trace_has_raw_fingerprint_passes() -> None:
    events = [{"event": "tool.requested", "raw_arguments_sha256": "abc123", "raw_arguments_chars": 10}]
    check = trace_has_raw_fingerprint()
    result = check("", events)
    assert result.passed


def test_trace_has_raw_fingerprint_fails_on_missing() -> None:
    events = [{"event": "tool.requested", "raw_arguments_sha256": "", "raw_arguments_chars": 0}]
    check = trace_has_raw_fingerprint()
    result = check("", events)
    assert not result.passed


def test_trace_validation_issue_contains_passes() -> None:
    events = [{"event": "tool.requested", "validation_issues": [{"path": "-n", "type": "extra_forbidden"}]}]
    check = trace_validation_issue_contains("-n")
    result = check("", events)
    assert result.passed


def test_trace_validation_issue_contains_fails_on_missing() -> None:
    events = [{"event": "tool.requested", "validation_issues": [{"path": "-C", "type": "extra_forbidden"}]}]
    check = trace_validation_issue_contains("-n")
    result = check("", events)
    assert not result.passed


# ---------------------------------------------------------------------------
# Runner integration tests (real tool dispatcher)
# ---------------------------------------------------------------------------


def test_run_micro_eval_grep_rejects_n_flag(tmp_path: Path) -> None:
    """The runner must correctly invoke the real dispatcher and capture
    both the output and trace events for a grep -n rejection."""
    result = run_micro_eval(
        name="test-grep-n",
        defect_id="test",
        tool_name="grep",
        raw_arguments='{"pattern":"TODO","-n":true}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("validation"),
            trace_success_is(False),
            trace_has_validation_issues(),
            trace_validation_issue_contains("-n"),
            output_contains("Error:"),
            output_contains("line_number"),
            output_contains("read_file"),
        ],
        workspace=tmp_path,
    )
    assert result.passed
    assert result.error == ""
    assert "Error:" in result.output
    assert "line_number" in result.output


def test_run_micro_eval_grep_valid_params(tmp_path: Path) -> None:
    """Valid grep arguments must succeed — the positive control."""
    (tmp_path / "app.py").write_text("# TODO: fix this\n")
    result = run_micro_eval(
        name="test-grep-valid",
        defect_id="test",
        tool_name="grep",
        raw_arguments='{"pattern":"TODO","output_mode":"content"}',
        checks=[
            trace_status_is("success"),
            trace_success_is(True),
            trace_has_raw_fingerprint(),
        ],
        workspace=tmp_path,
    )
    assert result.passed
    assert "TODO" in result.output


def test_run_micro_eval_edit_file_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON must produce a json_parse error, not a crash."""
    (tmp_path / "greeting.txt").write_text("Hello, friend.\n")
    result = run_micro_eval(
        name="test-edit-bad-json",
        defect_id="test",
        tool_name="edit_file",
        raw_arguments='{"path":"greeting.txt","edits":[{"old_text":"Hello","new_text":"Goodbye",}]}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("json_parse"),
            trace_success_is(False),
            output_contains("Error:"),
        ],
        workspace=tmp_path,
    )
    assert result.passed


def test_run_micro_eval_edit_file_valid(tmp_path: Path) -> None:
    """Valid edit_file call with escaped quotes must succeed."""
    (tmp_path / "greeting.txt").write_text("Hello, friend.\n")
    result = run_micro_eval(
        name="test-edit-valid",
        defect_id="test",
        tool_name="edit_file",
        raw_arguments='{"path":"greeting.txt","edits":[{"old_text":"Hello","new_text":"Goodbye \\\"world\\\""}]}',
        checks=[
            trace_status_is("success"),
            trace_success_is(True),
        ],
        workspace=tmp_path,
    )
    assert result.passed


def test_run_micro_eval_captures_trace_events(tmp_path: Path) -> None:
    """The runner must capture trace events from the sink."""
    result = run_micro_eval(
        name="test-trace",
        defect_id="test",
        tool_name="grep",
        raw_arguments='{"pattern":"x"}',
        checks=[trace_status_is("success")],
        workspace=tmp_path,
    )
    assert len(result.trace_events) >= 2  # requested + completed
    events = [e["event"] for e in result.trace_events]
    assert "tool.requested" in events
    assert "tool.completed" in events


def test_run_micro_eval_no_checks_returns_error(tmp_path: Path) -> None:
    result = run_micro_eval(
        name="test-no-checks",
        defect_id="test",
        tool_name="grep",
        raw_arguments='{"pattern":"x"}',
        checks=[],
        workspace=tmp_path,
    )
    assert not result.passed
    assert "no checks" in result.error


# ---------------------------------------------------------------------------
# Full suite acceptance test
# ---------------------------------------------------------------------------


def test_all_micro_cases_pass(tmp_path: Path) -> None:
    """Every defined micro-eval case must pass against the real dispatcher.

    This is the deterministic baseline-characterization layer for Phase 2.
    Behavioral improvement is decided separately by the paired live runner.
    """
    failures: list[str] = []
    for case in ALL_CASES:
        with tempfile.TemporaryDirectory(prefix="micro-test-") as tmp:
            ws = Path(tmp)
            prepare_workspace(case, ws)
            result = run_micro_eval(
                name=case.name,
                defect_id=case.defect_id,
                tool_name=case.tool_name,
                raw_arguments=case.raw_arguments,
                checks=case.checks,
                workspace=ws,
            )
            if not result.passed:
                failed_checks = [c.label for c in result.checks if not c.passed]
                failures.append(
                    f"{case.name}: {failed_checks}"
                )
    assert not failures, "Micro-eval failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Acceptance test: micro-evals reproduce grep friction from Phase 1 report
# ---------------------------------------------------------------------------


def test_micro_evals_reproduce_grep_friction_from_report(tmp_path: Path) -> None:
    """The grep micro-evals must test the exact failure patterns that the
    Phase 1 report surfaced: models passing -n, -C, -A, -B as JSON keys.

    This preserves the exact baseline behavior behind the report finding. The
    paired live experiment separately tests whether accepting the alias is
    better for model behavior.
    """
    grep_validation_cases = [
        c for c in GREP_CASES
        if c.defect_id == "failure_matrix:grep/validation"
        and "rejects" in c.name
    ]
    # Must cover all four CLI flags the report identified
    flag_names = {c.name for c in grep_validation_cases}
    assert "grep-rejects-n-flag" in flag_names
    assert "grep-rejects-C-flag" in flag_names
    assert "grep-rejects-A-flag" in flag_names
    assert "grep-rejects-B-flag" in flag_names

    # Each must pass — the harness currently handles these correctly
    for case in grep_validation_cases:
        with tempfile.TemporaryDirectory(prefix="micro-accept-") as tmp:
            ws = Path(tmp)
            prepare_workspace(case, ws)
            result = run_micro_eval(
                name=case.name,
                defect_id=case.defect_id,
                tool_name=case.tool_name,
                raw_arguments=case.raw_arguments,
                checks=case.checks,
                workspace=ws,
            )
            assert result.passed, f"{case.name} failed: {[c.label for c in result.checks if not c.passed]}"


# ---------------------------------------------------------------------------
# Live paired experiment (no provider calls in pytest)
# ---------------------------------------------------------------------------


def test_candidate_grep_schema_exposes_n_without_mutating_baseline() -> None:
    baseline = [{
        "name": "grep",
        "description": "Do not pass command-line flags such as -n, -A, -B, or -C. ",
        "parameters": {"properties": {"pattern": {"type": "string"}}},
    }]
    candidate = _candidate_grep_schema(baseline)

    assert "-n" not in baseline[0]["parameters"]["properties"]
    assert candidate[0]["parameters"]["properties"]["-n"]["type"] == "boolean"
    assert "-n is accepted" in candidate[0]["description"]


def test_candidate_normalizes_n_to_line_number() -> None:
    normalized = normalize_grep_n_alias({
        "pattern": "TODO",
        "output_mode": "content",
        "-n": True,
    })

    assert normalized == {
        "pattern": "TODO",
        "output_mode": "content",
        "line_number": True,
    }


def test_candidate_preserves_conflicting_alias_for_strict_rejection() -> None:
    normalized = normalize_grep_n_alias({
        "pattern": "TODO",
        "-n": True,
        "line_number": False,
    })

    assert normalized["-n"] is True
    assert normalized["line_number"] is False


def test_candidate_variant_accepts_n_at_real_dispatcher(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class DummySession:
        tools: list[dict]
        registry: dict

    (tmp_path / "app.py").write_text("# TODO: candidate path\n")
    todo = tools.TodoManager()
    baseline = DummySession(
        tools=tools.select_tool_schemas({"grep"}),
        registry=tools.build_tool_registry(Workspace(tmp_path), todo, {"grep"}),
    )
    candidate = apply_harness_variant(baseline, "candidate")
    call = types.SimpleNamespace(
        type="function_call",
        name="grep",
        call_id="candidate-grep",
        arguments='{"pattern":"TODO","output_mode":"content","-n":true}',
    )

    response, _ = asyncio.run(tools.run_tool_call_async(
        call,
        candidate.registry,
        todo,
    ))

    assert "TODO" in response["output"]
    assert candidate.registry["grep"] is not baseline.registry["grep"]


def _trial(
    model: str,
    variant: str,
    *,
    passed: bool,
    api_calls: int,
    alias_failures: int,
    errors: list[str] | None = None,
) -> PairedTrialResult:
    return PairedTrialResult(
        model=model,
        provider="fixed-provider",
        scenario="grep-n-find-definition",
        trial=1,
        variant=variant,
        passed=passed,
        api_calls=api_calls,
        alias_validation_failures=alias_failures,
        tool_errors=errors or [],
    )


def test_aggregate_keeps_model_and_variant_separate() -> None:
    results = [
        _trial("model-a", "baseline", passed=False, api_calls=4, alias_failures=1,
               errors=["grep/validation"]),
        _trial("model-a", "candidate", passed=True, api_calls=2, alias_failures=0),
        _trial("model-a", "candidate", passed=True, api_calls=3, alias_failures=0),
    ]

    metrics = aggregate(results)
    baseline = next(item for item in metrics if item.variant == "baseline")
    candidate = next(item for item in metrics if item.variant == "candidate")

    assert baseline.success_rate == 0.0
    assert baseline.alias_validation_failures == 1
    assert candidate.success_rate == 1.0
    assert candidate.median_api_calls == 2.5


def _passing_metrics() -> list[VariantMetrics]:
    rows: list[VariantMetrics] = []
    for model in ("model-a", "model-b"):
        rows.extend([
            VariantMetrics(
                model=model,
                variant="baseline",
                attempts=15,
                passed=12,
                success_rate=0.8,
                median_api_calls=3,
                alias_validation_failures=5,
                tool_errors=["grep/validation"],
            ),
            VariantMetrics(
                model=model,
                variant="candidate",
                attempts=15,
                passed=14,
                success_rate=14 / 15,
                median_api_calls=2,
                alias_validation_failures=0,
                tool_errors=[],
            ),
        ])
    return rows


def test_acceptance_requires_all_preregistered_criteria() -> None:
    criteria = evaluate_acceptance(_passing_metrics())

    assert len(criteria) == 5
    assert all(item.passed for item in criteria)


def test_acceptance_rejects_candidate_only_error() -> None:
    metrics = _passing_metrics()
    candidate = next(
        item for item in metrics
        if item.model == "model-b" and item.variant == "candidate"
    )
    candidate.tool_errors = ["grep/execution_error"]

    criteria = evaluate_acceptance(metrics)

    new_error = next(
        item for item in criteria
        if item.label == "no per-model candidate-only tool error signature"
    )
    assert not new_error.passed


def test_acceptance_rejects_single_model() -> None:
    metrics = [item for item in _passing_metrics() if item.model == "model-a"]

    criteria = evaluate_acceptance(metrics)

    assert criteria == [CriterionResult(
        label="at least two models",
        passed=False,
        detail="paired models=1",
    )]
