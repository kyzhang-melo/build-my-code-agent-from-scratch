from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import ValidationError

from .models import AggregateJudgment, PairJudgment
from .prompts import (
    AGGREGATE_PROMPT_VERSION,
    RUBRIC_VERSION,
    build_aggregate_prompt,
    build_pair_prompt,
)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ResponseCreate = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class CompareConfig:
    judge_run_id: str
    factor: str
    run_a: Path
    run_b: Path
    output_dir: Path
    judge_model: str
    judge_provider: str
    seed: int = 0
    max_input_chars: int = 400_000
    concurrency: int = 2
    rerun_failed: bool = False


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mode(manifest: dict) -> str:
    return "warm" if manifest.get("eval_mode") == "warm_context_sequence" else "cold"


def _factor_value(manifest: dict, factor: str) -> str:
    if factor == "reasoning":
        return str(manifest.get("reasoning_effort") or "default")
    return _mode(manifest)


def validate_compatible_runs(a: dict, b: dict, factor: str) -> tuple[str, str]:
    if factor not in {"reasoning", "context"}:
        raise ValueError("factor must be 'reasoning' or 'context'")
    common = (
        "dataset", "split", "instance_ids", "model", "provider", "max_api_calls",
        "max_output_tokens", "instance_timeout_seconds", "harness_commit",
        "swebench_commit", "auto_compact",
    )
    conflicts = [key for key in common if a.get(key) != b.get(key)]
    if conflicts:
        raise ValueError(f"incompatible run manifests: {', '.join(conflicts)}")

    value_a = _factor_value(a, factor)
    value_b = _factor_value(b, factor)
    if value_a == value_b:
        raise ValueError(f"factor {factor!r} has the same value in both runs: {value_a}")
    if factor == "reasoning" and _mode(a) != _mode(b):
        raise ValueError("reasoning comparison requires the same context mode")
    if factor == "context" and a.get("reasoning_effort") != b.get("reasoning_effort"):
        raise ValueError("context comparison requires the same reasoning effort")
    if factor == "context" and {value_a, value_b} != {"cold", "warm"}:
        raise ValueError("context comparison requires one cold and one warm run")
    return value_a, value_b


def _latest_attempt(instance_dir: Path) -> Path:
    attempts = []
    for path in instance_dir.glob("attempt-*"):
        try:
            attempts.append((int(path.name.split("-", 1)[1]), path))
        except ValueError:
            continue
    if not attempts:
        raise FileNotFoundError(f"no attempts found under {instance_dir}")
    return max(attempts)[1]


def _task_map(run_dir: Path) -> dict[str, dict]:
    tasks = _load_json(run_dir / "tasks.json")
    if not isinstance(tasks, list):
        raise ValueError(f"tasks.json must contain a list: {run_dir}")
    return {str(task["instance_id"]): task for task in tasks}


def _summary_map(run_dir: Path) -> dict[str, dict]:
    summary = _load_json(run_dir / "summary.json")
    if not isinstance(summary, dict):
        raise ValueError(f"summary.json must contain an object: {run_dir}")
    return {str(row["instance_id"]): row for row in summary.get("instances", [])}


def _clean_string(value: str, replacements: list[tuple[str, str]]) -> str:
    cleaned = ANSI_RE.sub("", value)
    for source, target in replacements:
        if source:
            cleaned = cleaned.replace(source, target)
    return cleaned


def _line_number(text: str) -> str:
    return "\n".join(f"L{index}: {line}" for index, line in enumerate(text.splitlines(), 1))


def _normalized_trace(path: Path, replacements: list[tuple[str, str]]) -> str:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        event = json.loads(raw)
        if event.get("event") == "llm.usage":
            continue
        event.pop("run_id", None)
        event.pop("timestamp", None)
        for hidden in (
            "model", "model_id", "provider", "configured_provider",
            "reasoning_effort", "cost", "input_tokens", "output_tokens",
            "total_tokens", "cached_tokens", "cache_write_tokens",
            "reasoning_tokens",
        ):
            event.pop(hidden, None)
        cleaned = _clean_string(
            json.dumps(event, ensure_ascii=False, sort_keys=True), replacements
        )
        sequence = event.get("sequence", "?")
        lines.append(f"T{sequence}: {cleaned}")
    return "\n".join(lines)


def _trajectory(run_dir: Path, instance_id: str, side: str) -> tuple[str, dict]:
    attempt = _latest_attempt(run_dir / "instances" / instance_id)
    result = _load_json(attempt / "result.json")
    if not isinstance(result, dict):
        raise ValueError(f"result.json must contain an object: {attempt}")
    replacements = [
        (str(attempt / "workspace"), "<WORKSPACE>"),
        (str(run_dir), f"<TRAJECTORY_{side}>"),
        (str(run_dir.resolve()), f"<TRAJECTORY_{side}>"),
        (run_dir.name, f"<TRAJECTORY_{side}>")
    ]
    log_path = attempt / "agent.log"
    trace_path = attempt / "trace.jsonl"
    if not log_path.is_file() or not trace_path.is_file():
        missing = [str(p) for p in (log_path, trace_path) if not p.is_file()]
        raise FileNotFoundError(f"missing trajectory artifacts: {', '.join(missing)}")
    log = _line_number(_clean_string(log_path.read_text(encoding="utf-8"), replacements))
    trace = _normalized_trace(trace_path, replacements)
    patch_path = attempt / "patch.diff"
    final_path = attempt / "final_text.txt"
    patch = _clean_string(
        patch_path.read_text(encoding="utf-8") if patch_path.is_file() else "<no patch>",
        replacements,
    )
    final = _clean_string(
        final_path.read_text(encoding="utf-8") if final_path.is_file() else "",
        replacements,
    )
    safe_result = {
        key: value for key, value in result.items()
        if key not in {"official_status", "error"}
    }
    content = (
        f"<run-result>\n{json.dumps(safe_result, ensure_ascii=False, sort_keys=True)}\n</run-result>\n"
        f"<final-text>\n{final}\n</final-text>\n"
        f"<patch>\n{patch}\n</patch>\n"
        f"<agent-log>\n{log}\n</agent-log>\n"
        f"<structured-trace>\n{trace}\n</structured-trace>"
    )
    return content, {"attempt": attempt.name, "result": result}


def _a_is_first(config: CompareConfig, instance_id: str) -> bool:
    material = (
        f"{config.seed}:{config.factor}:{instance_id}:"
        f"{config.run_a.name}:{config.run_b.name}"
    )
    return int(hashlib.sha256(material.encode()).hexdigest(), 16) % 2 == 0


def _parse_pair(text: str) -> PairJudgment:
    return PairJudgment.model_validate_json(text.strip())


def _parse_aggregate(text: str) -> AggregateJudgment:
    return AggregateJudgment.model_validate_json(text.strip())


async def _judge_call(create_response: ResponseCreate, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return await create_response(prompt)
        except Exception as exc:  # noqa: BLE001 - provider boundary
            last_error = exc
            if attempt < 3:
                await asyncio.sleep(1.2 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _existing_record(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _official_row(summary: dict[str, dict], instance_id: str) -> dict:
    row = summary.get(instance_id, {})
    return {
        "official_status": row.get("official_status", "unknown"),
        "agent_status": row.get("agent_status", "unknown"),
        "api_calls": row.get("api_calls"),
        "duration_seconds": row.get("duration_seconds"),
        "patch_status": row.get("patch_status", "unknown"),
    }


async def run_comparison(
    config: CompareConfig, create_response: ResponseCreate
) -> dict:
    manifest_a = _load_json(config.run_a / "manifest.json")
    manifest_b = _load_json(config.run_b / "manifest.json")
    if not isinstance(manifest_a, dict) or not isinstance(manifest_b, dict):
        raise ValueError("run manifests must contain JSON objects")
    value_a, value_b = validate_compatible_runs(manifest_a, manifest_b, config.factor)
    tasks_a = _task_map(config.run_a)
    tasks_b = _task_map(config.run_b)
    summary_a = _summary_map(config.run_a)
    summary_b = _summary_map(config.run_b)
    instance_ids = list(manifest_a["instance_ids"])
    required_instances = set(instance_ids)
    if not required_instances.issubset(tasks_a) or not required_instances.issubset(tasks_b):
        raise ValueError("tasks.json does not cover every manifest instance")
    task_conflicts = [
        instance_id for instance_id in instance_ids
        if any(
            tasks_a[instance_id].get(key) != tasks_b[instance_id].get(key)
            for key in ("repo", "base_commit", "problem_statement")
        )
    ]
    if task_conflicts:
        raise ValueError(
            "task artifacts differ between runs for: " + ", ".join(task_conflicts)
        )

    judge_manifest = {
        "schema_version": 1,
        "judge_run_id": config.judge_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "factor": config.factor,
        "run_a": str(config.run_a),
        "run_b": str(config.run_b),
        "first_value": value_a,
        "second_value": value_b,
        "instance_ids": instance_ids,
        "judge_model": config.judge_model,
        "judge_provider": config.judge_provider,
        "seed": config.seed,
        "max_input_chars": config.max_input_chars,
        "concurrency": config.concurrency,
        "rubric_version": RUBRIC_VERSION,
        "aggregate_prompt_version": AGGREGATE_PROMPT_VERSION,
        "calibration_status": "uncalibrated",
    }
    manifest_path = config.output_dir / "manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        if not isinstance(existing, dict):
            raise ValueError("judge manifest must contain a JSON object")
        comparison_keys = (
            "factor", "run_a", "run_b", "judge_model", "judge_provider", "seed",
            "max_input_chars", "rubric_version", "aggregate_prompt_version",
        )
        conflicts = [
            key for key in comparison_keys
            if existing.get(key) != judge_manifest.get(key)
        ]
        if conflicts:
            raise ValueError(f"judge manifest conflicts in: {', '.join(conflicts)}")
    else:
        atomic_write_json(manifest_path, judge_manifest)

    semaphore = asyncio.Semaphore(config.concurrency)

    async def judge_instance(instance_id: str) -> dict:
        result_path = config.output_dir / "instances" / instance_id / "judgment.json"
        existing_record = _existing_record(result_path)
        if existing_record is not None and (
            existing_record.get("status") == "completed" or not config.rerun_failed
        ):
            return existing_record
        target_dir = result_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        a_first = _a_is_first(config, instance_id)
        source_a = config.run_a if a_first else config.run_b
        source_b = config.run_b if a_first else config.run_a
        mapping = {
            "A": {"run": source_a.name, "factor_value": value_a if a_first else value_b},
            "B": {"run": source_b.name, "factor_value": value_b if a_first else value_a},
        }
        try:
            trajectory_a, meta_a = _trajectory(source_a, instance_id, "A")
            trajectory_b, meta_b = _trajectory(source_b, instance_id, "B")
            problem = str(tasks_a[instance_id].get("problem_statement", ""))
            prompt = build_pair_prompt(
                problem=problem, trajectory_a=trajectory_a, trajectory_b=trajectory_b
            )
            (target_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            if len(prompt) > config.max_input_chars:
                record = {
                    "schema_version": 1, "instance_id": instance_id,
                    "status": "input_too_large", "input_chars": len(prompt),
                    "mapping": mapping, "attempts": {"A": meta_a["attempt"], "B": meta_b["attempt"]},
                }
                atomic_write_json(result_path, record)
                return record
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            record = {
                "schema_version": 1, "instance_id": instance_id,
                "status": "missing_or_invalid_artifact", "mapping": mapping,
                "error_type": type(exc).__name__, "error": str(exc),
            }
            atomic_write_json(result_path, record)
            return record
        try:
            async with semaphore:
                raw = await _judge_call(create_response, prompt)
            (target_dir / "raw-response.txt").write_text(raw, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - provider boundary
            record = {
                "schema_version": 1, "instance_id": instance_id,
                "status": "provider_error", "mapping": mapping,
                "error_type": type(exc).__name__, "error": str(exc),
            }
            atomic_write_json(result_path, record)
            return record
        try:
            judgment = _parse_pair(raw)
            record = {
                "schema_version": 1, "instance_id": instance_id, "status": "completed",
                "input_chars": len(prompt), "mapping": mapping,
                "attempts": {"A": meta_a["attempt"], "B": meta_b["attempt"]},
                "judgment": judgment.model_dump(),
            }
        except ValidationError as exc:
            record = {
                "schema_version": 1, "instance_id": instance_id,
                "status": "invalid_response", "mapping": mapping,
                "error_type": type(exc).__name__, "error": str(exc),
            }
        atomic_write_json(result_path, record)
        return record

    records = await asyncio.gather(*(judge_instance(i) for i in instance_ids))
    revealed_rows = []
    preference_counts: Counter[str] = Counter()
    for record in records:
        instance_id = record["instance_id"]
        row = {"instance_id": instance_id, "status": record["status"]}
        if record["status"] == "completed":
            preference = record["judgment"]["overall_preference"]
            preferred_value = (
                record["mapping"][preference]["factor_value"]
                if preference in {"A", "B"} else preference
            )
            preference_counts[preferred_value] += 1
            row.update({
                "mapping": record["mapping"],
                "blind_judgment": record["judgment"],
                "preferred_factor_value": preferred_value,
                "run_a_outcome": _official_row(summary_a, instance_id),
                "run_b_outcome": _official_row(summary_b, instance_id),
            })
        revealed_rows.append(row)

    aggregate_dir = config.output_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_prompt = build_aggregate_prompt(
        factor=config.factor, first_value=value_a, second_value=value_b, rows=revealed_rows
    )
    (aggregate_dir / "prompt.txt").write_text(aggregate_prompt, encoding="utf-8")
    aggregate_record: dict
    if not any(record["status"] == "completed" for record in records):
        aggregate_record = {"status": "no_completed_instances"}
    else:
        try:
            async with semaphore:
                raw = await _judge_call(create_response, aggregate_prompt)
            (aggregate_dir / "raw-response.txt").write_text(raw, encoding="utf-8")
            aggregate = _parse_aggregate(raw)
            aggregate_record = {"status": "completed", "judgment": aggregate.model_dump()}
        except Exception as exc:  # noqa: BLE001
            aggregate_record = {
                "status": "judge_error", "error_type": type(exc).__name__, "error": str(exc)
            }
    atomic_write_json(aggregate_dir / "judgment.json", aggregate_record)

    summary = {
        "schema_version": 1,
        "judge_run_id": config.judge_run_id,
        "factor": config.factor,
        "first_value": value_a,
        "second_value": value_b,
        "calibration_status": "uncalibrated",
        "status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "preference_counts": dict(sorted(preference_counts.items())),
        "instances": revealed_rows,
        "aggregate": aggregate_record,
    }
    atomic_write_json(config.output_dir / "summary.json", summary)
    lines = [
        f"# Process Judge: `{config.judge_run_id}`", "",
        f"- Factor: `{config.factor}` (`{value_a}` vs `{value_b}`)",
        f"- Judge: `{config.judge_model}`", "- Calibration: **uncalibrated**", 
        f"- Statuses: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        f"- Process preferences: `{json.dumps(summary['preference_counts'], sort_keys=True)}`", "",
        "| Instance | Judge status | Preferred factor value | Run A outcome | Run B outcome |",
        "|---|---|---|---|---|",
    ]
    for row in revealed_rows:
        lines.append(
            f"| {row['instance_id']} | {row['status']} | "
            f"{row.get('preferred_factor_value', '-')} | "
            f"{row.get('run_a_outcome', {}).get('official_status', '-')} | "
            f"{row.get('run_b_outcome', {}).get('official_status', '-')} |"
        )
    if aggregate_record.get("status") == "completed":
        lines.extend(["", "## Aggregate review", "", aggregate_record["judgment"]["summary"]])
    (config.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
