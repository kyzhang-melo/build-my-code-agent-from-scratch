from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from evals.swebench.cli import resolve_swebench_python, resolve_swebench_repo
from evals.swebench.core import (
    DEFAULT_CACHE_DIR,
    atomic_write_json,
    create_manifest,
    ensure_mirror,
    load_tasks_via_bridge,
    remove_prediction,
    run_official_evaluator,
    upsert_prediction,
)

from .core import (
    DEFAULT_RUNS_DIR,
    SESSION_ID,
    generate_sequence_report,
    load_sequence_spec,
    run_sequence_episode,
    validate_task_sequence,
)


def run_dir(args: argparse.Namespace) -> Path:
    return Path(args.runs_dir).expanduser().resolve() / args.run_id


async def cmd_generate(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=True)
    target = run_dir(args)
    if target.exists():
        raise SystemExit(
            f"sequence run already exists: {target}; use a new --run-id "
            "(mid-sequence resume is intentionally unsupported)"
        )
    swebench_repo = resolve_swebench_repo(args.swebench_repo)
    swebench_python = resolve_swebench_python(args.swebench_python, swebench_repo)
    spec = load_sequence_spec(Path(args.subset))
    model = args.model or os.getenv("MODEL_ID")
    if not model:
        raise SystemExit("MODEL_ID is not set; pass --model or configure .env")
    provider = os.getenv("OPENROUTER_PROVIDER", "")
    target.mkdir(parents=True)
    tasks = load_tasks_via_bridge(
        swebench_python,
        swebench_repo,
        spec.dataset,
        spec.split,
        spec.instance_ids,
        target / "tasks.json",
    )
    cache_dir = Path(args.repo_cache).expanduser().resolve()
    mirror = ensure_mirror(tasks[0], cache_dir)
    sequence_commits = validate_task_sequence(spec, tasks, mirror)

    manifest = create_manifest(
        run_id=args.run_id,
        dataset=spec.dataset,
        split=spec.split,
        selection=spec.selection,
        instance_ids=spec.instance_ids,
        model=model,
        max_api_calls=args.max_api_calls,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        timeout=args.instance_timeout,
        swebench_repo=swebench_repo,
        provider=provider,
    )
    manifest.update(
        {
            "eval_mode": "warm_context_sequence",
            "sequence_id": spec.sequence_id,
            "repo": spec.repo,
            "subsystem": spec.subsystem,
            "session_id": SESSION_ID,
            "session_mode": "persist_and_resume",
            "transition_mode": "clean_base",
            "feedback_mode": "none",
            "sequence_commits": sequence_commits,
        }
    )
    atomic_write_json(target / "manifest.json", manifest)

    for position, task in enumerate(tasks, 1):
        print(f"[run] episode={position}/{len(tasks)} {task.instance_id}")
        result = await run_sequence_episode(
            task,
            mirror,
            target,
            position=position,
            run_id=args.run_id,
            session_id=SESSION_ID,
            model=model,
            provider=provider,
            max_api_calls=args.max_api_calls,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            timeout=args.instance_timeout,
        )
        if result.patch_status == "produced":
            patch_path = (
                target / "instances" / task.instance_id / "attempt-1" / "patch.diff"
            )
            upsert_prediction(
                target / "predictions.jsonl",
                {
                    "instance_id": task.instance_id,
                    "model_name_or_path": model,
                    "model_patch": patch_path.read_text(encoding="utf-8"),
                },
            )
        else:
            remove_prediction(target / "predictions.jsonl", task.instance_id)
        print(
            f"[{result.agent_status}] {task.instance_id} patch={result.patch_status} "
            f"calls={result.api_calls} history_committed={result.history_committed}"
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
    report = generate_sequence_report(run_dir(args))
    print(
        f"Resolved {report['resolved']}/{report['total_instances']}. "
        f"Report: {run_dir(args) / 'summary.md'}"
    )
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    print("[pipeline] generate")
    status = await cmd_generate(args)
    if status:
        return status
    predictions = run_dir(args) / "predictions.jsonl"
    if not predictions.exists() or not predictions.read_text(encoding="utf-8").strip():
        raise SystemExit(
            "generation produced no predictions; official evaluation was not started"
        )
    print("[pipeline] evaluate")
    status = cmd_evaluate(args)
    if status:
        return status
    print("[pipeline] report")
    return cmd_report(args)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))


def _swebench(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--swebench-repo")
    parser.add_argument("--swebench-python")


def _generate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subset", required=True)
    parser.add_argument("--model")
    parser.add_argument("--repo-cache", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--max-api-calls", type=_positive_int, default=30)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--max-output-tokens", type=_positive_int)
    parser.add_argument("--instance-timeout", type=_positive_int)


def _evaluate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--max-workers", type=_positive_int, default=4)
    parser.add_argument(
        "--cache-level", choices=("none", "base", "env", "instance"), default="instance"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chronological SWE-bench tasks with persisted agent context."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    _shared(generate)
    _swebench(generate)
    _generate(generate)
    generate.set_defaults(func=cmd_generate)
    evaluate = sub.add_parser("evaluate")
    _shared(evaluate)
    _swebench(evaluate)
    _evaluate(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)
    report = sub.add_parser("report")
    _shared(report)
    report.set_defaults(func=cmd_report)
    pipeline = sub.add_parser("run")
    _shared(pipeline)
    _swebench(pipeline)
    _generate(pipeline)
    _evaluate(pipeline)
    pipeline.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return asyncio.run(result) if asyncio.iscoroutine(result) else int(result)
