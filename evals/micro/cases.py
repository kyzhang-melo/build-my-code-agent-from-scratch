"""Micro-eval case definitions.

Each case is derived from a specific defect pattern surfaced by the
Phase 1 Harness Diagnostic Report. The ``defect_id`` field links each
case back to the report section that motivated it.

Cases are grouped by the tool they test. Each case specifies:
- ``tool_name``: the tool to call
- ``raw_arguments``: the exact JSON string a model would submit
- ``checks``: list of check functions evaluating output + trace
- ``defect_id``: which Phase 1 report finding this reproduces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class MicroCase:
    name: str
    defect_id: str
    tool_name: str
    raw_arguments: str
    checks: list
    needs_files: dict[str, str] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# grep parameter validation (Phase 1: grep/validation, 18 failures in longcat+hy3)
# ---------------------------------------------------------------------------

GREP_CASES: list[MicroCase] = [
    MicroCase(
        name="grep-rejects-n-flag",
        defect_id="failure_matrix:grep/validation",
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
        description=(
            "Models (especially longcat/hy3) pass -n as a JSON key. "
            "The harness must reject it with a validation error that "
            "guides the model to use line_number=true instead."
        ),
    ),
    MicroCase(
        name="grep-rejects-C-flag",
        defect_id="failure_matrix:grep/validation",
        tool_name="grep",
        raw_arguments='{"pattern":"TODO","-C":3}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("validation"),
            trace_success_is(False),
            trace_has_validation_issues(),
            output_contains("Error:"),
            output_contains("CLI flags"),
        ],
        description=(
            "Models pass -C (context lines) as a JSON key. "
            "The harness must reject it and mention CLI flags are not supported."
        ),
    ),
    MicroCase(
        name="grep-rejects-A-flag",
        defect_id="failure_matrix:grep/validation",
        tool_name="grep",
        raw_arguments='{"pattern":"TODO","-A":2}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("validation"),
            trace_success_is(False),
            trace_has_validation_issues(),
            output_contains("Error:"),
        ],
        description="Models pass -A (after-context) as a JSON key.",
    ),
    MicroCase(
        name="grep-rejects-B-flag",
        defect_id="failure_matrix:grep/validation",
        tool_name="grep",
        raw_arguments='{"pattern":"TODO","-B":2}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("validation"),
            trace_success_is(False),
            trace_has_validation_issues(),
            output_contains("Error:"),
        ],
        description="Models pass -B (before-context) as a JSON key.",
    ),
    MicroCase(
        name="grep-accepts-valid-params",
        defect_id="failure_matrix:grep/validation",
        tool_name="grep",
        raw_arguments='{"pattern":"TODO","output_mode":"content"}',
        checks=[
            trace_status_is("success"),
            trace_success_is(True),
            trace_has_raw_fingerprint(),
        ],
        needs_files={"app.py": "def foo():\n    # TODO: implement\n    pass\n"},
        description=(
            "Valid grep arguments must succeed. This is the positive "
            "control — if this fails, the grep tool itself is broken, "
            "not just the validation."
        ),
    ),
]


# ---------------------------------------------------------------------------
# edit_file JSON parsing (Phase 1: edit_file/json_parse, 7 failures)
# ---------------------------------------------------------------------------

EDIT_FILE_CASES: list[MicroCase] = [
    MicroCase(
        name="edit_file-rejects-malformed-json",
        defect_id="failure_matrix:edit_file/json_parse",
        tool_name="edit_file",
        raw_arguments='{"path":"greeting.txt","edits":[{"old_text":"Hello","new_text":"Goodbye",}]}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("json_parse"),
            trace_success_is(False),
            output_contains("Error:"),
            output_contains("invalid"),
        ],
        needs_files={"greeting.txt": "Hello, friend.\n"},
        description=(
            "Models submit JSON with trailing commas or other syntax "
            "errors. The harness must reject with json_parse error, "
            "not crash."
        ),
    ),
    MicroCase(
        name="edit_file-rejects-unescaped-newlines",
        defect_id="failure_matrix:edit_file/json_parse",
        tool_name="edit_file",
        raw_arguments='{"path":"greeting.txt","edits":[{"old_text":"Hello","new_text":"Goodbye\nWorld"}]}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("json_parse"),
            trace_success_is(False),
            output_contains("Error:"),
        ],
        needs_files={"greeting.txt": "Hello, friend.\n"},
        description=(
            "Models submit raw newlines inside JSON string values "
            "instead of \\n. This is invalid JSON and must be rejected."
        ),
    ),
    MicroCase(
        name="edit_file-accepts-valid-json-with-special-chars",
        defect_id="failure_matrix:edit_file/json_parse",
        tool_name="edit_file",
        raw_arguments='{"path":"greeting.txt","edits":[{"old_text":"Hello","new_text":"Goodbye \\\"world\\\""}]}',
        checks=[
            trace_status_is("success"),
            trace_success_is(True),
        ],
        needs_files={"greeting.txt": "Hello, friend.\n"},
        description=(
            "Valid JSON with escaped quotes must succeed. This is the "
            "positive control for edit_file."
        ),
    ),
    MicroCase(
        name="edit_file-rejects-missing-edits-field",
        defect_id="failure_matrix:edit_file/validation",
        tool_name="edit_file",
        raw_arguments='{"path":"greeting.txt"}',
        checks=[
            trace_status_is("invalid_arguments"),
            trace_error_type_is("validation"),
            trace_success_is(False),
            trace_has_validation_issues(),
            output_contains("Error:"),
        ],
        needs_files={"greeting.txt": "Hello, friend.\n"},
        description="Missing required 'edits' field must be rejected.",
    ),
]


# ---------------------------------------------------------------------------
# Trace instrumentation (Phase 0: validation_issues + raw_arguments fingerprint)
# ---------------------------------------------------------------------------

TRACE_CASES: list[MicroCase] = [
    MicroCase(
        name="trace-records-validation-issues-on-grep-failure",
        defect_id="param_friction:grep/validation_issues",
        tool_name="grep",
        raw_arguments='{"pattern":"x","-n":true}',
        checks=[
            trace_has_validation_issues(),
            trace_validation_issue_contains("-n"),
            trace_has_raw_fingerprint(),
        ],
        description=(
            "The trace must record structured validation_issues with the "
            "field path, so the offline analyzer can detect parameter "
            "friction without parsing agent.log."
        ),
    ),
    MicroCase(
        name="trace-records-raw-arguments-fingerprint",
        defect_id="anomaly:exact_repeat",
        tool_name="grep",
        raw_arguments='{"pattern":"TODO"}',
        checks=[
            trace_has_raw_fingerprint(),
            trace_success_is(True),
        ],
        needs_files={"app.py": "# TODO: fix\n"},
        description=(
            "The trace must record raw_arguments_sha256 and "
            "raw_arguments_chars so the offline analyzer can detect "
            "exact-repeat calls."
        ),
    ),
]


ALL_CASES: list[MicroCase] = GREP_CASES + EDIT_FILE_CASES + TRACE_CASES


def cases_by_defect_id(defect_id: str) -> list[MicroCase]:
    """Return all cases linked to a given defect_id."""
    return [c for c in ALL_CASES if c.defect_id == defect_id]


def prepare_workspace(case: MicroCase, workspace: Path) -> None:
    """Create any files the case needs in its workspace."""
    for rel, content in case.needs_files.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
