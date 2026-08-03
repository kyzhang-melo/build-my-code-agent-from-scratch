from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from context_compact import estimate_tokens
from evals.swebench.core import (
    AutoApproveHandler,
    SWEBENCH_SYSTEM_ADDENDUM,
    SWEBENCH_TOOL_NAMES,
    atomic_write_json,
    build_prompt,
    create_isolated_workspace,
    git_output,
    run_command,
    validate_patch,
)
from evals.swebench.models import Task
from session_store import SessionStore
from tools import TodoParams
from trace import JsonlTraceSink, TraceContext
from workspace import Workspace

from .models import SequenceAgentResult, SequenceSpec


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / ".runs" / "swebench-sequence"
SESSION_ID = "warm-context-sequence"


def load_sequence_spec(path: Path) -> SequenceSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = raw.get("instance_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(x, str) and x for x in ids)
    ):
        raise ValueError(
            "sequence subset must contain a non-empty string list 'instance_ids'"
        )
    if len(ids) != len(set(ids)):
        raise ValueError("sequence subset contains duplicate instance IDs")
    required = ("sequence_id", "repo", "subsystem")
    missing = [
        key
        for key in required
        if not isinstance(raw.get(key), str) or not raw[key].strip()
    ]
    if missing:
        raise ValueError(f"sequence subset is missing metadata: {', '.join(missing)}")
    return SequenceSpec(
        dataset=raw.get("dataset", "SWE-bench/SWE-bench_Verified"),
        split=raw.get("split", "test"),
        selection=raw.get("selection", ""),
        sequence_id=raw["sequence_id"],
        repo=raw["repo"],
        subsystem=raw["subsystem"],
        instance_ids=ids,
    )


def validate_task_sequence(
    spec: SequenceSpec,
    tasks: list[Task],
    mirror: Path,
) -> list[dict]:
    if [task.instance_id for task in tasks] != spec.instance_ids:
        raise ValueError("bridge tasks do not preserve the declared sequence order")
    mismatched = [task.instance_id for task in tasks if task.repo != spec.repo]
    if mismatched:
        raise ValueError(f"sequence contains tasks outside {spec.repo}: {mismatched}")

    metadata: list[dict] = []
    previous: Task | None = None
    previous_date: datetime | None = None
    for task in tasks:
        date = git_output(
            ["--git-dir", str(mirror), "show", "-s", "--format=%cI", task.base_commit]
        )
        parsed_date = datetime.fromisoformat(date)
        if previous_date is not None and parsed_date < previous_date:
            raise ValueError(
                f"base commit dates are out of order: "
                f"{previous.instance_id} -> {task.instance_id}"
            )
        distance = 0
        if previous is not None:
            try:
                git_output([
                    "--git-dir", str(mirror), "merge-base", "--is-ancestor",
                    previous.base_commit, task.base_commit,
                ])
            except RuntimeError as exc:
                raise ValueError(
                    f"base commits are not chronological ancestors: "
                    f"{previous.instance_id} -> {task.instance_id}"
                ) from exc
            distance = int(git_output([
                "--git-dir", str(mirror), "rev-list", "--count",
                f"{previous.base_commit}..{task.base_commit}",
            ]))
        metadata.append({
            "instance_id": task.instance_id,
            "base_commit": task.base_commit,
            "base_date": date,
            "commits_since_previous": distance,
        })
        previous = task
        previous_date = parsed_date
    return metadata


def build_sequence_prompt(task: Task, position: int) -> str:
    prefix = f"SWE-bench warm-context sequence episode {position}: {task.instance_id}.\n\n"
    if position > 1:
        prefix += (
            "The previous issue's patch has been submitted. The repository has now "
            "been advanced to the clean base commit for this issue. Treat the prior "
            "conversation as historical context: re-inspect current files when their "
            "state matters, and do not assume earlier workspace edits are present.\n\n"
        )
    return prefix + build_prompt(task)


def _todo_snapshot(session) -> list[dict]:
    return [item.model_dump(by_alias=True) for item in session.todo.state.items]


def export_sequence_patch(workspace: Path, base_commit: str) -> str:
    """Export source changes without runtime-owned compaction snapshots."""
    fd, index_name = tempfile.mkstemp(prefix="swebench-sequence-index-")
    os.close(fd)
    index_path = Path(index_name)
    index_path.unlink()
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = str(index_path)
    try:
        run_command(["git", "read-tree", base_commit], cwd=workspace, env=env)
        run_command(
            [
                "git",
                "add",
                "-A",
                "--",
                ".",
                ":(exclude).transcripts",
                ":(exclude).transcripts/**",
            ],
            cwd=workspace,
            env=env,
        )
        return run_command(
            ["git", "diff", "--cached", "--binary", "--full-index", base_commit],
            cwd=workspace,
            env=env,
        ).stdout
    finally:
        index_path.unlink(missing_ok=True)


async def run_sequence_episode(
    task: Task,
    mirror: Path,
    run_dir: Path,
    *,
    position: int,
    run_id: str,
    session_id: str,
    model: str,
    provider: str,
    max_api_calls: int,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    timeout: int | None,
) -> SequenceAgentResult:
    import main

    attempt_dir = run_dir / "instances" / task.instance_id / "attempt-1"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stable_workspace = run_dir / "workspace"
    archived_workspace = attempt_dir / "workspace"
    create_isolated_workspace(mirror, stable_workspace, task.base_commit)

    result = SequenceAgentResult(
        instance_id=task.instance_id,
        attempt=1,
        agent_status="error",
        patch_status="not_exported",
        sequence_position=position,
        session_id=session_id,
    )
    started = time.monotonic()
    final_text = ""
    store = None
    state = None
    main.MODEL_ID = model
    session_path = run_dir / "session" / f"{session_id}.jsonl"
    try:
        workspace = Workspace(stable_workspace)
        if session_path.exists():
            store = SessionStore.open(
                session_path, workspace, model, provider, acquire_lock=True
            )
        else:
            store = SessionStore.create(
                workspace, session_id, model, provider,
                session_dir=run_dir / "session", acquire_lock=True,
            )
            # Materialize the external session file even before any messages exist.
            store.sync_todo([])

        history = store.messages()
        result.resumed_message_count = len(history)
        result.history_tokens_before = estimate_tokens(history)
        result.history_tokens_after = result.history_tokens_before
        result.resume_diagnostics = store.resume_diagnostics._asdict()
        store.sync(history)

        trace = TraceContext(
            sink=JsonlTraceSink(attempt_dir / "trace.jsonl"),
            run_id=f"{run_id}:episode-{position}:{task.instance_id}",
            agent_id="parent",
        )
        session = main.create_parent_session(
            stable_workspace,
            approval_handler=AutoApproveHandler(),
            trace_context=trace,
            on_text=None,
            session_id=session_id,
            store=store,
            session_dir=run_dir / "session",
            max_api_calls=max_api_calls,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            system_addendum=SWEBENCH_SYSTEM_ADDENDUM,
            tool_names=SWEBENCH_TOOL_NAMES,
        )
        saved_items = store.last_todo_items()
        if saved_items is not None:
            session.todo.update(TodoParams.model_validate({"items": saved_items}))

        history.append({"role": "user", "content": build_sequence_prompt(task, position)})
        state = main.LoopState(messages=history)
        log_path = attempt_dir / "agent.log"
        try:
            with log_path.open("w", encoding="utf-8") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if timeout is None:
                    outcome = await main.agent_loop(state, session)
                else:
                    outcome = await asyncio.wait_for(
                        main.agent_loop(state, session), timeout
                    )
            final_text = outcome.final_text
            result.agent_status = outcome.stop_reason
            result.stop_reason = outcome.stop_reason
            result.api_calls = outcome.api_calls
        except asyncio.TimeoutError:
            result.agent_status = "timeout"
            result.stop_reason = "timeout"
            result.api_calls = state.api_call_count
            result.error = f"instance timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001 - rollback this turn and continue sequence
            result.agent_status = "error"
            result.stop_reason = "error"
            result.api_calls = state.api_call_count
            result.error = f"{type(exc).__name__}: {exc}"

        # Persistence is part of the eval protocol, not an agent failure. Let a
        # store error abort generation instead of silently continuing with an
        # uncertain context boundary.
        if result.agent_status in {"completed", "max_api_calls"}:
            store.sync(state.messages)
            store.sync_todo(_todo_snapshot(session))
            result.history_committed = True
            result.history_tokens_after = estimate_tokens(state.messages)

        (attempt_dir / "final_text.txt").write_text(final_text, encoding="utf-8")
        try:
            patch = export_sequence_patch(stable_workspace, task.base_commit)
            if not patch.strip():
                result.patch_status = "empty"
            else:
                validate_patch(mirror, task.base_commit, patch)
                (attempt_dir / "patch.diff").write_text(patch, encoding="utf-8")
                result.patch_status = "produced"
                result.patch_bytes = len(patch.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - preserve agent result and invalid patch details
            result.patch_status = "invalid"
            if not result.error:
                result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if store is not None:
            store.close()
        if stable_workspace.exists():
            shutil.move(str(stable_workspace), str(archived_workspace))
        result.duration_seconds = round(time.monotonic() - started, 3)
        atomic_write_json(attempt_dir / "result.json", result.to_dict())
    return result


def generate_sequence_report(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["model"].replace("/", "__")
    official_path = run_dir / "official" / f"{model}.{manifest['run_id']}.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    status_sets = {
        "resolved": set(official.get("resolved_ids", [])),
        "unresolved": set(official.get("unresolved_ids", [])),
        "evaluator_error": set(official.get("error_ids", [])),
        "empty_patch": set(official.get("empty_patch_ids", [])),
    }
    rows = []
    for position, instance_id in enumerate(manifest["instance_ids"], 1):
        result_path = run_dir / "instances" / instance_id / "attempt-1" / "result.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {}
        )
        official_status = next(
            (name for name, ids in status_sets.items() if instance_id in ids),
            "not_submitted",
        )
        rows.append({
            **result,
            "instance_id": instance_id,
            "sequence_position": position,
            "official_status": official_status,
        })
    agent_counts = Counter(row.get("agent_status", "not_run") for row in rows)
    official_counts = Counter(row["official_status"] for row in rows)
    resolved = len(status_sets["resolved"])
    evaluated = len(set().union(*status_sets.values()))
    report = {
        "schema_version": 1,
        "eval_mode": "warm_context_sequence",
        "run_id": manifest["run_id"],
        "sequence_id": manifest["sequence_id"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "resolved": resolved,
        "evaluated": evaluated,
        "total_instances": len(rows),
        "agent_status_counts": dict(sorted(agent_counts.items())),
        "official_status_counts": dict(sorted(official_counts.items())),
        "instances": rows,
    }
    atomic_write_json(run_dir / "summary.json", report)
    lines = [
        f"# Warm-context SWE-bench Report: `{manifest['run_id']}`",
        "",
        f"- Sequence: `{manifest['sequence_id']}`",
        f"- Model: `{manifest['model']}`",
        f"- Resolved: **{resolved}/{len(rows)}**",
        "",
        "| # | Instance | Agent | Calls | Resumed messages | Before tokens | History committed | Patch | Official |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence_position']} | {row['instance_id']} | "
            f"{row.get('agent_status', 'not_run')} | {row.get('api_calls', 0)} | "
            f"{row.get('resumed_message_count', 0)} | "
            f"{row.get('history_tokens_before', 0)} | "
            f"{row.get('history_committed', False)} | {row.get('patch_status', 'none')} | "
            f"{row['official_status']} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
