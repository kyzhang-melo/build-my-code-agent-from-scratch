from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .core import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RUNS_DIR,
    atomic_write_json,
    create_manifest,
    ensure_manifest,
    ensure_mirror,
    generate_report,
    latest_result,
    load_subset,
    load_tasks_via_bridge,
    next_attempt,
    remove_prediction,
    run_agent_attempt,
    run_official_evaluator,
    should_run,
    upsert_prediction,
)
from .models import AgentResult


DEFAULT_SWEBENCH_REPO = Path(__file__).resolve().parents[2].parent / "SWE-bench"


def resolve_swebench_repo(value: str | None) -> Path:
    path = Path(value or os.getenv("SWEBENCH_REPO", "") or DEFAULT_SWEBENCH_REPO).expanduser().resolve()
    if not (path / "swebench" / "harness" / "run_evaluation.py").is_file():
        raise SystemExit(f"SWE-bench repository not found at {path}")
    return path


def resolve_swebench_python(value: str | None, repo: Path) -> Path:
    raw = value or os.getenv("SWEBENCH_PYTHON")
    # Do not call Path.resolve() here: venv Python executables are commonly
    # symlinks to the base interpreter. Resolving the symlink would launch the
    # base interpreter without the venv's site-packages.
    candidate = Path(raw).expanduser() if raw else repo / ".venv" / "bin" / "python"
    path = Path(os.path.abspath(candidate))
    if not path.exists():
        raise SystemExit(f"SWE-bench Python not found at {path}; pass --swebench-python")
    return path


def run_dir(args: argparse.Namespace) -> Path:
    return Path(args.runs_dir).expanduser().resolve() / args.run_id


async def cmd_generate(args: argparse.Namespace) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    swebench_repo = resolve_swebench_repo(args.swebench_repo)
    swebench_python = resolve_swebench_python(args.swebench_python, swebench_repo)
    dataset, split, instance_ids, selection = load_subset(Path(args.subset))
    target = run_dir(args)
    target.mkdir(parents=True, exist_ok=True)

    model = args.model or os.getenv("MODEL_ID")
    if not model:
        raise SystemExit("MODEL_ID is not set; pass --model or configure .env")
    manifest = create_manifest(
        run_id=args.run_id,
        dataset=dataset,
        split=split,
        selection=selection,
        instance_ids=instance_ids,
        model=model,
        max_api_calls=args.max_api_calls,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        timeout=args.instance_timeout,
        swebench_repo=swebench_repo,
        provider=os.getenv("OPENROUTER_PROVIDER"),
    )
    ensure_manifest(target / "manifest.json", manifest)
    tasks = load_tasks_via_bridge(
        swebench_python,
        swebench_repo,
        dataset,
        split,
        instance_ids,
        target / "tasks.json",
    )

    cache_dir = Path(args.repo_cache).expanduser().resolve()
    for task in tasks:
        instance_dir = target / "instances" / task.instance_id
        if not should_run(instance_dir, args.rerun_failed):
            print(f"[skip] {task.instance_id}")
            continue
        attempt = next_attempt(instance_dir)
        attempt_dir = instance_dir / f"attempt-{attempt}"
        print(f"[run] {task.instance_id} attempt={attempt}")
        try:
            mirror = ensure_mirror(task, cache_dir)
            result = await run_agent_attempt(
                task,
                mirror,
                attempt_dir,
                run_id=args.run_id,
                model=model,
                max_api_calls=args.max_api_calls,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                timeout=args.instance_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - isolate batch failures
            attempt_dir.mkdir(parents=True, exist_ok=True)
            result = AgentResult(
                task.instance_id,
                attempt,
                "error",
                "not_exported",
                stop_reason="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            atomic_write_json(attempt_dir / "result.json", result.to_dict())
        if result.patch_status == "produced":
            patch = (attempt_dir / "patch.diff").read_text(encoding="utf-8")
            upsert_prediction(
                target / "predictions.jsonl",
                {
                    "instance_id": task.instance_id,
                    "model_name_or_path": model,
                    "model_patch": patch,
                },
            )
        else:
            remove_prediction(target / "predictions.jsonl", task.instance_id)
        print(
            f"[{result.agent_status}] {task.instance_id} "
            f"patch={result.patch_status} api_calls={result.api_calls}"
        )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    swebench_repo = resolve_swebench_repo(args.swebench_repo)
    swebench_python = resolve_swebench_python(args.swebench_python, swebench_repo)
    summary = run_official_evaluator(
        swebench_python=swebench_python,
        swebench_repo=swebench_repo,
        run_dir=run_dir(args),
        namespace=args.namespace,
        max_workers=args.max_workers,
        cache_level=args.cache_level,
    )
    print(f"Official report: {summary}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = generate_report(run_dir(args))
    diagnostic_path = None
    try:
        from evals.analyze.automation import generate_run_diagnostics

        artifacts = generate_run_diagnostics(run_dir(args))
        diagnostic_path = artifacts["diff"]
    except Exception as exc:  # noqa: BLE001 - diagnostics must not erase eval success
        print(
            f"[diagnostics warning] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    print(
        f"Resolved {report['resolved']}/{report['total_instances']}. "
        f"Evaluated {report['evaluated']}/{report['total_instances']}. "
        f"Report: {run_dir(args) / 'summary.md'}"
    )
    if diagnostic_path is not None:
        print(f"Harness diagnostic diff: {diagnostic_path}")
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    """Run prediction generation, official evaluation, and reporting in order."""
    print("[pipeline] generate")
    generate_status = await cmd_generate(args)
    if generate_status:
        return generate_status

    predictions = run_dir(args) / "predictions.jsonl"
    if not predictions.exists() or not any(
        line.strip() for line in predictions.read_text(encoding="utf-8").splitlines()
    ):
        raise SystemExit(
            "generation produced no predictions; official evaluation was not started"
        )

    print("[pipeline] evaluate")
    evaluate_status = cmd_evaluate(args)
    if evaluate_status:
        return evaluate_status

    print("[pipeline] report")
    return cmd_report(args)


def _shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))


def _swebench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--swebench-repo")
    parser.add_argument("--swebench-python")


def _generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subset", required=True)
    parser.add_argument("--model")
    parser.add_argument("--repo-cache", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--max-api-calls", type=int, default=30)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--max-output-tokens", type=_positive_int)
    parser.add_argument(
        "--instance-timeout",
        type=_positive_int,
        help="optional agent instance timeout in seconds (default: no limit)",
    )
    parser.add_argument("--rerun-failed", action="store_true")


def _evaluate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--cache-level",
        choices=("none", "base", "env", "instance"),
        default="instance",
        help="Docker image cache level (default: retain instance images)",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run myCodeAgent against SWE-bench.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="run the agent and produce predictions")
    _shared_run_args(generate)
    _swebench_args(generate)
    _generate_args(generate)
    generate.set_defaults(func=cmd_generate)

    evaluate = sub.add_parser("evaluate", help="run the official Docker evaluator")
    _shared_run_args(evaluate)
    _swebench_args(evaluate)
    _evaluate_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    report = sub.add_parser("report", help="merge agent and official results")
    _shared_run_args(report)
    report.set_defaults(func=cmd_report)

    pipeline = sub.add_parser(
        "run",
        help="generate predictions, evaluate them, and produce a report",
    )
    _shared_run_args(pipeline)
    _swebench_args(pipeline)
    _generate_args(pipeline)
    _evaluate_args(pipeline)
    pipeline.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return asyncio.run(result) if asyncio.iscoroutine(result) else int(result)
