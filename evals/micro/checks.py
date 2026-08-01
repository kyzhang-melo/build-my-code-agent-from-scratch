"""Reusable check functions for micro-evals.

Each check is a callable ``(output: str, trace_events: list[dict]) -> MicroCheck``
that returns a pass/fail result with a detail string.
"""

from __future__ import annotations

from evals.micro.runner import MicroCheck


# ---------------------------------------------------------------------------
# Output checks (what the model sees)
# ---------------------------------------------------------------------------


def output_contains(needle: str, *, label: str | None = None) -> callable:
    """Check that the tool output contains ``needle``."""
    def check(output: str, _events: list[dict]) -> MicroCheck:
        found = needle in output
        return MicroCheck(
            label=label or f"output contains {needle!r}",
            passed=found,
            detail="" if found else f"{needle!r} not in output",
        )
    return check


def output_not_contains(needle: str, *, label: str | None = None) -> callable:
    """Check that the tool output does NOT contain ``needle``."""
    def check(output: str, _events: list[dict]) -> MicroCheck:
        found = needle in output
        return MicroCheck(
            label=label or f"output does not contain {needle!r}",
            passed=not found,
            detail="" if not found else f"{needle!r} unexpectedly in output",
        )
    return check


def output_starts_with(prefix: str, *, label: str | None = None) -> callable:
    def check(output: str, _events: list[dict]) -> MicroCheck:
        ok = output.startswith(prefix)
        return MicroCheck(
            label=label or f"output starts with {prefix!r}",
            passed=ok,
            detail="" if ok else f"output starts with {output[:40]!r}",
        )
    return check


# ---------------------------------------------------------------------------
# Trace checks (what the offline analyzer sees)
# ---------------------------------------------------------------------------


def _completed_event(events: list[dict]) -> dict | None:
    for e in events:
        if e.get("event") == "tool.completed":
            return e
    return None


def _requested_event(events: list[dict]) -> dict | None:
    for e in events:
        if e.get("event") == "tool.requested":
            return e
    return None


def trace_status_is(expected: str, *, label: str | None = None) -> callable:
    """Check that tool.completed has the expected status."""
    def check(_output: str, events: list[dict]) -> MicroCheck:
        comp = _completed_event(events)
        if comp is None:
            return MicroCheck(
                label=label or f"trace status is {expected!r}",
                passed=False,
                detail="no tool.completed event",
            )
        actual = comp.get("status", "")
        return MicroCheck(
            label=label or f"trace status is {expected!r}",
            passed=actual == expected,
            detail="" if actual == expected else f"status={actual!r}",
        )
    return check


def trace_error_type_is(expected: str, *, label: str | None = None) -> callable:
    """Check that tool.completed has the expected error_type."""
    def check(_output: str, events: list[dict]) -> MicroCheck:
        comp = _completed_event(events)
        if comp is None:
            return MicroCheck(
                label=label or f"trace error_type is {expected!r}",
                passed=False,
                detail="no tool.completed event",
            )
        actual = comp.get("error_type")
        return MicroCheck(
            label=label or f"trace error_type is {expected!r}",
            passed=actual == expected,
            detail="" if actual == expected else f"error_type={actual!r}",
        )
    return check


def trace_success_is(expected: bool, *, label: str | None = None) -> callable:
    """Check that tool.completed has the expected success flag."""
    def check(_output: str, events: list[dict]) -> MicroCheck:
        comp = _completed_event(events)
        if comp is None:
            return MicroCheck(
                label=label or f"trace success is {expected}",
                passed=False,
                detail="no tool.completed event",
            )
        actual = comp.get("success")
        return MicroCheck(
            label=label or f"trace success is {expected}",
            passed=actual == expected,
            detail="" if actual == expected else f"success={actual!r}",
        )
    return check


def trace_has_validation_issues(*, label: str | None = None) -> callable:
    """Check that tool.requested has a non-empty validation_issues list."""
    def check(_output: str, events: list[dict]) -> MicroCheck:
        req = _requested_event(events)
        if req is None:
            return MicroCheck(
                label=label or "trace has validation_issues",
                passed=False,
                detail="no tool.requested event",
            )
        issues = req.get("validation_issues", [])
        return MicroCheck(
            label=label or "trace has validation_issues",
            passed=bool(issues),
            detail="" if issues else "validation_issues is empty",
        )
    return check


def trace_has_raw_fingerprint(*, label: str | None = None) -> callable:
    """Check that tool.requested has raw_arguments_sha256 and raw_arguments_chars."""
    def check(_output: str, events: list[dict]) -> MicroCheck:
        req = _requested_event(events)
        if req is None:
            return MicroCheck(
                label=label or "trace has raw_arguments fingerprint",
                passed=False,
                detail="no tool.requested event",
            )
        has_hash = bool(req.get("raw_arguments_sha256"))
        has_chars = "raw_arguments_chars" in req
        ok = has_hash and has_chars
        return MicroCheck(
            label=label or "trace has raw_arguments fingerprint",
            passed=ok,
            detail="" if ok else f"sha256={req.get('raw_arguments_sha256')!r}, chars_present={has_chars}",
        )
    return check


def trace_validation_issue_contains(
    field_path: str,
    *,
    label: str | None = None,
) -> callable:
    """Check that validation_issues contains an entry with the given path."""
    def check(_output: str, events: list[dict]) -> MicroCheck:
        req = _requested_event(events)
        if req is None:
            return MicroCheck(
                label=label or f"validation issue path={field_path!r}",
                passed=False,
                detail="no tool.requested event",
            )
        issues = req.get("validation_issues", [])
        found = any(
            issue.get("path") == field_path or field_path in issue.get("path", "")
            for issue in issues
        )
        return MicroCheck(
            label=label or f"validation issue path={field_path!r}",
            passed=found,
            detail="" if found else f"no issue with path={field_path!r} in {issues}",
        )
    return check
