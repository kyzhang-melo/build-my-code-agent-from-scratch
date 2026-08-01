"""CLI for the decision-point micro-eval runner.

Runs deterministic micro-evals against the real tool dispatcher and
produces a structured report. No live model needed.

Usage:
    python evals/micro/run_micro.py [--case NAME] [--defect-id ID]
                                    [--output FILE] [--list]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from evals.micro.cases import ALL_CASES, prepare_workspace
from evals.micro.runner import MicroResult, run_micro_eval


def run_all_cases(only_case: str | None = None, only_defect: str | None = None) -> list[MicroResult]:
    """Run all (or filtered) micro-eval cases and return results."""
    results: list[MicroResult] = []
    for case in ALL_CASES:
        if only_case and case.name != only_case:
            continue
        if only_defect and case.defect_id != only_defect:
            continue
        with tempfile.TemporaryDirectory(prefix="micro-eval-") as tmp:
            workspace = Path(tmp)
            prepare_workspace(case, workspace)
            result = run_micro_eval(
                name=case.name,
                defect_id=case.defect_id,
                tool_name=case.tool_name,
                raw_arguments=case.raw_arguments,
                checks=case.checks,
                workspace=workspace,
            )
            results.append(result)
    return results


def render_markdown(results: list[MicroResult]) -> str:
    """Render micro-eval results as a Markdown report."""
    lines: list[str] = []
    lines.append("# Decision-Point Micro-Eval Report")
    lines.append("")
    lines.append(
        "> Deterministic tests of harness decision points identified by the "
        "Phase 1 Harness Diagnostic Report."
    )
    lines.append("> No live model required — these reproduce exact failure "
                 "patterns with synthetic inputs.")
    lines.append("")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    lines.append(f"**Passed: {passed}/{total}**")
    lines.append("")

    # Group by defect_id
    by_defect: dict[str, list[MicroResult]] = {}
    for r in results:
        by_defect.setdefault(r.defect_id, []).append(r)

    for defect_id in sorted(by_defect.keys()):
        group = by_defect[defect_id]
        group_passed = sum(1 for r in group if r.passed)
        lines.append(f"## {defect_id} ({group_passed}/{len(group)} passed)")
        lines.append("")
        lines.append("| Case | Tool | Result | Checks | Arguments |")
        lines.append("|---|---|---|---|---|")
        for r in group:
            status = "PASS" if r.passed else "FAIL"
            check_summary = f"{sum(1 for c in r.checks if c.passed)}/{len(r.checks)}"
            args_preview = r.raw_arguments[:60].replace("|", "\\|")
            if len(r.raw_arguments) > 60:
                args_preview += "..."
            lines.append(
                f"| {r.name} | {r.tool_name} | {status} | "
                f"{check_summary} | `{args_preview}` |"
            )
        lines.append("")

        # Show failed check details
        failed = [r for r in group if not r.passed]
        if failed:
            lines.append("### Failed checks")
            lines.append("")
            for r in failed:
                lines.append(f"**{r.name}**:")
                lines.append("")
                for c in r.checks:
                    if not c.passed:
                        lines.append(f"- [ ] {c.label}: {c.detail}")
                if r.error:
                    lines.append(f"- error: {r.error}")
                lines.append("")
            lines.append("")

    return "\n".join(lines) + "\n"


def render_json(results: list[MicroResult]) -> str:
    """Render micro-eval results as JSON."""
    return json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "cases": [
            {
                "name": r.name,
                "defect_id": r.defect_id,
                "tool_name": r.tool_name,
                "raw_arguments": r.raw_arguments,
                "passed": r.passed,
                "error": r.error,
                "checks": [
                    {"label": c.label, "passed": c.passed, "detail": c.detail}
                    for c in r.checks
                ],
            }
            for r in results
        ],
    }, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_micro",
        description="Run decision-point micro-evals against the real tool dispatcher.",
    )
    parser.add_argument("--case", help="run only the named case")
    parser.add_argument("--defect-id", help="run only cases linked to this defect ID")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="write report to this file (format inferred from extension)")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--json", action="store_true",
                        help="output JSON instead of Markdown (for stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for case in ALL_CASES:
            print(f"  {case.name:45s}  [{case.defect_id}]  {case.tool_name}")
        return 0

    results = run_all_cases(only_case=args.case, only_defect=args.defect_id)
    if not results:
        target = args.case or args.defect_id or "<any>"
        print(f"No micro-eval cases found for '{target}'.", file=sys.stderr)
        return 1

    if args.output is not None:
        if args.output.suffix == ".json":
            report = render_json(results)
        else:
            report = render_markdown(results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    elif args.json:
        print(render_json(results))
    else:
        print(render_markdown(results))

    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1
