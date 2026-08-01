"""CLI for the Harness Diagnostic Report analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals.analyze.report import analyze, render_report
from evals.analyze.scanner import DEFAULT_RUNS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_analyze",
        description="Generate a Harness Diagnostic Report from SWE-bench eval artifacts.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory containing SWE-bench runs (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write report to this file instead of stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir
    if not runs_dir.is_dir():
        print(f"Error: runs directory not found: {runs_dir}", file=sys.stderr)
        return 1

    data = analyze(runs_dir)
    report = render_report(data)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)
    return 0
