from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .core import CompareConfig, run_comparison


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / ".runs" / "judge"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blindly compare code-agent eval trajectories.")
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--judge-run-id", required=True)
    compare.add_argument("--factor", choices=("reasoning", "context"), required=True)
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--run-b", required=True)
    compare.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    compare.add_argument("--judge-model")
    compare.add_argument("--judge-provider")
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--max-input-chars", type=_positive_int, default=400_000)
    compare.add_argument("--concurrency", type=_positive_int, default=2)
    compare.add_argument("--rerun-failed", action="store_true")
    return parser


async def cmd_compare(args: argparse.Namespace) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    model = args.judge_model or os.getenv("JUDGE_MODEL_ID")
    if not model:
        raise SystemExit("JUDGE_MODEL_ID is not set; pass --judge-model or configure .env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("OPENROUTER_API_KEY and OPENROUTER_BASE_URL are required")
    provider = args.judge_provider or os.getenv("JUDGE_PROVIDER", "")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def create_response(prompt: str) -> str:
        extra_body = (
            {"provider": {"only": [provider], "allow_fallbacks": False}}
            if provider else None
        )
        response = await client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            extra_body=extra_body,
        )
        return getattr(response, "output_text", "") or ""

    output_dir = Path(args.runs_dir).expanduser().resolve() / args.judge_run_id
    config = CompareConfig(
        judge_run_id=args.judge_run_id,
        factor=args.factor,
        run_a=Path(args.run_a).expanduser().resolve(),
        run_b=Path(args.run_b).expanduser().resolve(),
        output_dir=output_dir,
        judge_model=model,
        judge_provider=provider,
        seed=args.seed,
        max_input_chars=args.max_input_chars,
        concurrency=args.concurrency,
        rerun_failed=args.rerun_failed,
    )
    summary = await run_comparison(config, create_response)
    print(f"Judge report: {output_dir / 'summary.md'}")
    print(f"Statuses: {summary['status_counts']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        return asyncio.run(cmd_compare(args))
    raise AssertionError(args.command)

