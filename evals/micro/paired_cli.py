"""CLI for live-model baseline/candidate paired micro-evals."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from evals.micro.paired import (
    DEFAULT_RUNS_DIR,
    GREP_ALIAS_HYPOTHESIS,
    aggregate,
    discover_scenarios,
    evaluate_acceptance,
    run_experiment,
    write_report,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_paired",
        description="Run a preregistered old/new harness experiment with live models.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model ID; pass exactly twice for the Phase 2 acceptance run",
    )
    parser.add_argument("--k", type=positive_int, default=5, help="repeats per pair (default: 5)")
    parser.add_argument("--max-api-calls", type=positive_int, default=8)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--max-output-tokens", type=positive_int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list", action="store_true", help="show the hypothesis and scenarios")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(f"Hypothesis: {GREP_ALIAS_HYPOTHESIS.hypothesis_id}")
        print(GREP_ALIAS_HYPOTHESIS.statement)
        print("Scenarios:")
        for scenario in discover_scenarios():
            print(f"  {scenario.name}")
        print("Preregistered criteria:")
        for criterion in GREP_ALIAS_HYPOTHESIS.preregistered_criteria:
            print(f"  - {criterion}")
        return 0

    models = list(dict.fromkeys(args.model))
    if len(models) != 2:
        parser.error("Phase 2 requires exactly two distinct --model values")
    if args.k < 5:
        parser.error("Phase 2 requires --k >= 5")
    if not os.getenv("OPENROUTER_PROVIDER"):
        parser.error(
            "Phase 2 requires OPENROUTER_PROVIDER so both variants use one fixed provider"
        )

    run_root = args.output_dir or (
        DEFAULT_RUNS_DIR / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    results = asyncio.run(run_experiment(
        models=models,
        k=args.k,
        run_root=run_root,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        max_api_calls=args.max_api_calls,
    ))
    metrics = aggregate(results)
    criteria = evaluate_acceptance(metrics)
    write_report(
        run_root,
        results,
        metrics,
        criteria,
        k=args.k,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        max_api_calls=args.max_api_calls,
    )
    accepted = bool(criteria) and all(item.passed for item in criteria)
    print(f"Report: {run_root / 'summary.md'}")
    return 0 if accepted else 1
