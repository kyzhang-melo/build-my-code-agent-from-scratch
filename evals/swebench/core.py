from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from session import SteeringDirective

from .models import AgentResult, Task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "evals" / ".runs" / "swebench"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "evals" / ".cache" / "swebench" / "repos"
BRIDGE_PATH = Path(__file__).with_name("bridge.py")
TERMINAL_AGENT_STATUSES = {"completed", "max_api_calls", "timeout", "error"}
FAILED_AGENT_STATUSES = {"max_api_calls", "timeout", "error"}
SWEBENCH_SYSTEM_ADDENDUM = (
    "SWE-bench evaluation mode: solve the issue by producing the smallest correct "
    "source patch. The solve workspace is provided only for source inspection and "
    "editing. Do not install dependencies, create virtual environments, download "
    "packages, run the project's tests, or execute or import project code to "
    "validate the change. Do not attempt to repair the host environment. Use source "
    "inspection, static reasoning, and review of the final diff instead. Do not "
    "commit changes. The patch will be tested later by the official SWE-bench Docker "
    "harness."
)
SWEBENCH_TOOL_NAMES = frozenset({
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "bash",
    "todo",
    "task",
})
DOCKER_GENERATION_REQUIRED = (
    "Legacy local SWE-bench runs cannot be resumed by the Docker-only runner; "
    "choose a new run ID."
)
STAGED_STEERING_THRESHOLDS = (45, 60)


class SwebenchSteeringPolicy:
    """Prompt an eval agent to converge at fixed call-count milestones."""

    name = "staged"

    def __init__(self, patch_provider: Callable[[], str]):
        self.patch_provider = patch_provider
        self._emitted: set[int] = set()

    def after_turn(self, api_call_count: int) -> SteeringDirective | None:
        if api_call_count in self._emitted:
            return None
        if api_call_count == STAGED_STEERING_THRESHOLDS[0]:
            directive = SteeringDirective(
                content=(
                    "You have used 45 model calls. Stop broad exploration and "
                    "converge on the most likely root cause. Inspect only the "
                    "files needed to validate that hypothesis, then implement "
                    "the smallest correct source patch."
                ),
                reason="converge_on_root_cause",
            )
        elif api_call_count == STAGED_STEERING_THRESHOLDS[1]:
            has_patch = bool(self.patch_provider().strip())
            if has_patch:
                directive = SteeringDirective(
                    content=(
                        "You have used 60 model calls and the workspace already "
                        "contains a patch. Prioritize delivery now: review the "
                        "final diff, correct only concrete issues, and finish "
                        "without starting optional investigation."
                    ),
                    reason="review_patch_and_finish",
                )
            else:
                directive = SteeringDirective(
                    content=(
                        "You have used 60 model calls and the workspace still "
                        "has no patch. Stop open-ended searching, choose the "
                        "most defensible diagnosis, implement the smallest "
                        "correct source fix, review its diff, and finish."
                    ),
                    reason="implement_patch_and_finish",
                )
        else:
            return None
        self._emitted.add(api_call_count)
        return directive


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=check,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"command failed with exit code {exc.returncode}: {' '.join(args)}"
        if detail:
            message += f"\n{detail}"
        raise RuntimeError(message) from exc


def git_output(args: list[str], *, cwd: Path | None = None) -> str:
    return run_command(["git", *args], cwd=cwd).stdout.strip()


def load_subset(path: Path) -> tuple[str, str, list[str], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = raw.get("instance_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("subset must contain a non-empty string list 'instance_ids'")
    if len(ids) != len(set(ids)):
        raise ValueError("subset contains duplicate instance IDs")
    return (
        raw.get("dataset", "SWE-bench/SWE-bench_Verified"),
        raw.get("split", "test"),
        ids,
        raw.get("selection", ""),
    )


def load_tasks_via_bridge(
    swebench_python: Path,
    swebench_repo: Path,
    dataset: str,
    split: str,
    instance_ids: list[str],
    output_path: Path,
    namespace: str = "swebench",
) -> list[Task]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(swebench_repo), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    run_command(
        [
            str(swebench_python),
            str(BRIDGE_PATH),
            "--dataset",
            dataset,
            "--split",
            split,
            "--instance-ids",
            *instance_ids,
            "--output",
            str(output_path),
            "--namespace",
            namespace,
        ],
        cwd=swebench_repo,
        env=env,
    )
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    tasks = [Task.from_dict(row) for row in rows]
    returned = {task.instance_id for task in tasks}
    missing = set(instance_ids) - returned
    if missing:
        raise ValueError(f"SWE-bench bridge did not return: {sorted(missing)}")
    by_id = {task.instance_id: task for task in tasks}
    return [by_id[instance_id] for instance_id in instance_ids]


def _repo_key(repo: str) -> str:
    if "/" not in repo:
        raise ValueError(f"invalid GitHub repo name: {repo!r}")
    return repo.replace("/", "__") + ".git"


def ensure_mirror(task: Task, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mirror = cache_dir / _repo_key(task.repo)
    if not mirror.exists():
        run_command(["git", "clone", "--mirror", f"https://github.com/{task.repo}.git", str(mirror)])
    try:
        git_output(["--git-dir", str(mirror), "cat-file", "-e", f"{task.base_commit}^{{commit}}"])
    except RuntimeError:
        run_command(["git", "--git-dir", str(mirror), "fetch", "--prune", "origin"])
        git_output(["--git-dir", str(mirror), "cat-file", "-e", f"{task.base_commit}^{{commit}}"])
    return mirror


def create_isolated_workspace(
    mirror: Path,
    workspace: Path,
    base_commit: str,
) -> Path:
    """Create a repository whose visible history ends at ``base_commit``.

    The full mirror is only a download cache. Fetching the base commit into a
    new repository preserves its ancestors without exposing the mirror's
    branches, pull-request refs, or descendant commit objects to the agent.
    """
    if workspace.exists():
        raise FileExistsError(f"attempt workspace already exists: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "init", "--quiet", str(workspace)])
    run_command(
        [
            "git",
            "fetch",
            "--quiet",
            "--no-tags",
            mirror.resolve().as_uri(),
            base_commit,
        ],
        cwd=workspace,
    )
    run_command(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=workspace)
    (workspace / ".git" / "FETCH_HEAD").unlink(missing_ok=True)

    head = git_output(["rev-parse", "HEAD"], cwd=workspace)
    if head != base_commit:
        raise RuntimeError(f"workspace HEAD {head} does not match base_commit {base_commit}")
    if git_output(["status", "--porcelain"], cwd=workspace):
        raise RuntimeError("new task workspace is not clean")
    if git_output(["remote"], cwd=workspace):
        raise RuntimeError("isolated task repository unexpectedly has a remote")
    if (workspace / ".git" / "objects" / "info" / "alternates").exists():
        raise RuntimeError("isolated task repository unexpectedly shares Git objects")
    return workspace


def next_attempt(instance_dir: Path) -> int:
    attempts = []
    if instance_dir.exists():
        for child in instance_dir.iterdir():
            if child.is_dir() and child.name.startswith("attempt-"):
                try:
                    attempts.append(int(child.name.split("-", 1)[1]))
                except ValueError:
                    continue
    return max(attempts, default=0) + 1


def latest_result(instance_dir: Path) -> dict | None:
    candidates = sorted(
        instance_dir.glob("attempt-*/result.json"),
        key=lambda path: int(path.parent.name.split("-", 1)[1]),
    )
    return json.loads(candidates[-1].read_text(encoding="utf-8")) if candidates else None


def should_run(instance_dir: Path, rerun_failed: bool) -> bool:
    result = latest_result(instance_dir)
    if result is None:
        return True
    failed = (
        result.get("agent_status") in FAILED_AGENT_STATUSES
        or result.get("patch_status") != "produced"
    )
    return rerun_failed and failed


def build_prompt(task: Task) -> str:
    return (
        "Resolve the following issue in the current repository.\n\n"
        "Inspect the repository, locate the relevant source code, and implement the "
        "smallest general fix. Do not install dependencies, create virtual "
        "environments, download packages, run the project's tests, or execute or "
        "import project code for validation. The official SWE-bench Docker "
        "environment will test the resulting patch separately. Do not commit "
        "changes. Do not modify tests merely to make them pass. Work only inside "
        "the current workspace.\n\n"
        "Use bash to inspect the workspace and to review the final patch with "
        "`git diff` before finishing. Then summarize what you changed and why.\n\n"
        f"Issue:\n{task.problem_statement}"
    )


def export_patch(workspace: Path, base_commit: str) -> str:
    fd, index_name = tempfile.mkstemp(prefix="swebench-index-")
    os.close(fd)
    index_path = Path(index_name)
    index_path.unlink()
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = str(index_path)
    try:
        run_command(["git", "read-tree", base_commit], cwd=workspace, env=env)
        run_command(["git", "add", "-A"], cwd=workspace, env=env)
        patch = run_command(
            ["git", "diff", "--cached", "--binary", "--full-index", base_commit],
            cwd=workspace,
            env=env,
        ).stdout
    finally:
        index_path.unlink(missing_ok=True)
    return patch


def validate_patch(mirror: Path, base_commit: str, patch: str) -> None:
    if not patch.strip():
        raise ValueError("agent produced an empty patch")
    with tempfile.TemporaryDirectory(prefix="swebench-patch-check-") as temp:
        index_path = Path(temp) / "index"
        patch_path = Path(temp) / "patch.diff"
        patch_path.write_text(patch, encoding="utf-8")
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(index_path)
        run_command(
            ["git", "--git-dir", str(mirror), "read-tree", base_commit],
            env=env,
        )
        run_command(
            [
                "git",
                "--git-dir",
                str(mirror),
                "apply",
                "--check",
                "--cached",
                "--binary",
                str(patch_path),
            ],
            env=env,
        )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def upsert_prediction(path: Path, prediction: dict) -> None:
    records: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["instance_id"]] = record
    records[prediction["instance_id"]] = prediction
    temp = path.with_name(f".{path.name}.tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[key] for key in sorted(records)]
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    os.replace(temp, path)


def remove_prediction(path: Path, instance_id: str) -> None:
    if not path.exists():
        return
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [record for record in records if record["instance_id"] != instance_id]
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    os.replace(temp, path)


def repository_state(path: Path) -> tuple[str, bool]:
    commit = git_output(["rev-parse", "HEAD"], cwd=path)
    dirty = bool(git_output(["status", "--porcelain"], cwd=path))
    return commit, dirty


def create_manifest(
    *,
    run_id: str,
    dataset: str,
    split: str,
    selection: str,
    instance_ids: list[str],
    model: str,
    max_api_calls: int,
    steering_policy: str,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    timeout: int | None,
    swebench_repo: Path,
    model_limits: dict[str, int | None],
    provider: str | None = None,
    image_namespace: str = "swebench",
) -> dict:
    harness_commit, harness_dirty = repository_state(PROJECT_ROOT)
    swebench_commit, swebench_dirty = repository_state(swebench_repo)
    return {
        "schema_version": 4,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": split,
        "selection": selection,
        "instance_ids": instance_ids,
        "model": model,
        "provider": provider or "",
        "max_api_calls": max_api_calls,
        "steering_policy": steering_policy,
        "steering_thresholds": (
            list(STAGED_STEERING_THRESHOLDS)
            if steering_policy == "staged"
            else []
        ),
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "model_limits": model_limits,
        "instance_timeout_seconds": timeout,
        "harness_commit": harness_commit,
        "harness_worktree_dirty": harness_dirty,
        "swebench_commit": swebench_commit,
        "swebench_worktree_dirty": swebench_dirty,
        "platform": platform.platform(),
        "auto_compact": os.getenv("AUTO_COMPACT", "1") != "0",
        "generate_environment": "docker",
        "solve_network_mode": "none",
        "solve_workspace_mode": "image_testbed",
        "image_namespace": image_namespace,
    }


def ensure_manifest(path: Path, proposed: dict) -> dict:
    proposed = dict(proposed)
    if proposed.get("generate_environment", "docker") != "docker":
        raise ValueError(DOCKER_GENERATION_REQUIRED)
    proposed.setdefault("generate_environment", "docker")
    proposed.setdefault("solve_network_mode", "none")
    proposed.setdefault("solve_workspace_mode", "image_testbed")
    proposed.setdefault("image_namespace", "swebench")
    if not path.exists():
        atomic_write_json(path, proposed)
        return proposed
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("generate_environment") != "docker":
        raise ValueError(DOCKER_GENERATION_REQUIRED)
    needs_workspace_migration = "solve_workspace_mode" not in existing
    if needs_workspace_migration:
        existing["solve_workspace_mode"] = "bind_mount"
    needs_model_limits_migration = "model_limits" not in existing
    keys = (
        "dataset",
        "split",
        "instance_ids",
        "model",
        "provider",
        "max_api_calls",
        "steering_policy",
        "steering_thresholds",
        "reasoning_effort",
        "max_output_tokens",
        "model_limits",
        "instance_timeout_seconds",
        "harness_commit",
        "swebench_commit",
        "generate_environment",
        "solve_network_mode",
        "solve_workspace_mode",
        "image_namespace",
    )
    conflicts = [
        key
        for key in keys
        if not (
            needs_model_limits_migration
            and key in {"model_limits", "harness_commit"}
        )
        if (
            (existing.get(key) or "") != (proposed.get(key) or "")
            if key == "provider"
            else existing.get(key) != proposed.get(key)
        )
    ]
    if conflicts:
        detail = f"run manifest conflicts in: {', '.join(conflicts)}"
        if "solve_workspace_mode" in conflicts and existing.get("solve_workspace_mode") == "bind_mount":
            detail += "; legacy Docker runs used bind mounts; choose a new run id for image /testbed"
        raise ValueError(detail)
    if needs_model_limits_migration:
        previous_harness_commit = existing.get("harness_commit")
        existing["schema_version"] = 2
        existing["model_limits"] = proposed["model_limits"]
        existing["harness_commit"] = proposed["harness_commit"]
        existing["harness_worktree_dirty"] = proposed["harness_worktree_dirty"]
        migrations = existing.setdefault("migrations", [])
        if not isinstance(migrations, list):
            raise ValueError("run manifest has invalid migrations metadata")
        migrations.append({
            "kind": "model_limits_and_harness_upgrade",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "previous_harness_commit": previous_harness_commit,
            "new_harness_commit": proposed["harness_commit"],
        })
    if needs_workspace_migration:
        existing["schema_version"] = 4
        migrations = existing.setdefault("migrations", [])
        if not isinstance(migrations, list):
            raise ValueError("run manifest has invalid migrations metadata")
        migrations.append({
            "kind": "solve_workspace_upgrade",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "generate_environment": existing["generate_environment"],
            "solve_workspace_mode": existing["solve_workspace_mode"],
        })
    if needs_model_limits_migration or needs_workspace_migration:
        atomic_write_json(path, existing)
    return existing


class AutoApproveHandler:
    async def request(self, request):
        from permissions import ApprovalResponse

        if request.allow_for_session:
            return ApprovalResponse("approve_for_session")
        return ApprovalResponse("approve")


async def run_agent_attempt(
    task: Task,
    attempt_dir: Path,
    *,
    run_id: str,
    model: str,
    max_api_calls: int,
    steering_policy: str,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    timeout: int | None,
) -> AgentResult:
    import main
    from trace import JsonlTraceSink, TraceContext
    from sandbox import DockerSandbox

    attempt_dir.mkdir(parents=True, exist_ok=True)
    workspace = None
    attempt = int(attempt_dir.name.split("-", 1)[1])
    result = AgentResult(task.instance_id, attempt, "error", "not_exported")
    started = time.monotonic()
    final_text = ""
    main.MODEL_ID = model
    trace = TraceContext(
        sink=JsonlTraceSink(attempt_dir / "trace.jsonl"),
        run_id=f"{run_id}:{task.instance_id}:attempt-{attempt}",
        agent_id="parent",
    )
    state = main.LoopState(messages=[{"role": "user", "content": build_prompt(task)}])
    log_path = attempt_dir / "agent.log"
    sandbox = None
    try:
        if not task.instance_image_key:
            raise RuntimeError("task is missing its SWE-bench instance image key")
        sandbox = DockerSandbox(
            task.instance_image_key,
            f"{run_id}:{task.instance_id}:attempt-{attempt}",
            task.base_commit,
            platform=task.platform,
            run_id=run_id,
        )
        sandbox.start()
        workspace = Path(sandbox.workspace_root)
        session = main.create_parent_session(
            workspace,
            approval_handler=AutoApproveHandler(),
            trace_context=trace,
            on_text=None,
            max_api_calls=max_api_calls,
            steering_policy=(
                SwebenchSteeringPolicy(sandbox.export_patch)
                if steering_policy == "staged"
                else None
            ),
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            system_addendum=SWEBENCH_SYSTEM_ADDENDUM,
            tool_names=SWEBENCH_TOOL_NAMES,
            sandbox=sandbox,
        )
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if timeout is None:
                outcome = await main.agent_loop(state, session)
            else:
                outcome = await asyncio.wait_for(
                    main.agent_loop(state, session),
                    timeout=timeout,
                )
        final_text = outcome.final_text
        result.agent_status = outcome.stop_reason
        result.stop_reason = outcome.stop_reason
        result.api_calls = outcome.api_calls
    except asyncio.TimeoutError:
        result.agent_status = "timeout"
        result.stop_reason = "timeout"
        result.error = f"instance timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - record one task failure and continue batch
        result.agent_status = "error"
        result.stop_reason = "error"
        # agent_loop did not return a TurnOutcome, but its state still records
        # every request that was attempted before the failure.
        result.api_calls = state.api_call_count
        result.error = f"{type(exc).__name__}: {exc}"
    try:
        try:
            (attempt_dir / "final_text.txt").write_text(final_text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - retain cleanup and result recording
            if not result.error:
                result.error = f"{type(exc).__name__}: {exc}"

        # A Docker startup failure never created a solve workspace, so there is
        # no patch to classify. Preserve `not_exported` rather than reporting
        # an invalid agent patch.
        if workspace is not None:
            try:
                patch = sandbox.export_patch()
                if not patch.strip():
                    result.patch_status = "empty"
                else:
                    (attempt_dir / "patch.diff").write_text(patch, encoding="utf-8")
                    result.patch_status = "produced"
                    result.patch_bytes = len(patch.encode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                result.patch_status = "invalid"
                if not result.error:
                    result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            sandbox.close()
    result.duration_seconds = round(time.monotonic() - started, 3)
    atomic_write_json(attempt_dir / "result.json", result.to_dict())
    return result


def run_agent_attempt_worker(
    task: Task,
    attempt_dir: Path,
    *,
    run_id: str,
    model: str,
    max_api_calls: int,
    steering_policy: str,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    timeout: int | None,
) -> AgentResult:
    """Run one attempt in a dedicated worker process.

    Process isolation is intentional: the agent runtime currently configures
    the model and redirects stdout/stderr at process scope.
    """
    return asyncio.run(
        run_agent_attempt(
            task,
            attempt_dir,
            run_id=run_id,
            model=model,
            max_api_calls=max_api_calls,
            steering_policy=steering_policy,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
    )


def run_official_evaluator(
    *,
    swebench_python: Path,
    swebench_repo: Path,
    run_dir: Path,
    namespace: str,
    max_workers: int,
    cache_level: str,
) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    predictions = run_dir / "predictions.jsonl"
    if not predictions.exists():
        raise FileNotFoundError(f"predictions not found: {predictions}")
    ids = [
        json.loads(line)["instance_id"]
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not ids:
        raise ValueError("predictions file is empty")
    official_dir = run_dir / "official"
    official_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(swebench_python),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        manifest["dataset"],
        "--split",
        manifest["split"],
        "--predictions_path",
        str(predictions),
        "--instance_ids",
        *ids,
        "--max_workers",
        str(max_workers),
        "--cache_level",
        cache_level,
        "--namespace",
        namespace,
        "--run_id",
        manifest["run_id"],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(swebench_repo), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    log_path = official_dir / "evaluator.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=official_dir, env=env, check=True, text=True, stdout=log, stderr=subprocess.STDOUT)
    model = manifest["model"].replace("/", "__")
    summary = official_dir / f"{model}.{manifest['run_id']}.json"
    if not summary.exists():
        raise FileNotFoundError(f"official summary not found: {summary}")
    return summary


def generate_report(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["model"].replace("/", "__")
    official_path = run_dir / "official" / f"{model}.{manifest['run_id']}.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    resolved = set(official.get("resolved_ids", []))
    unresolved = set(official.get("unresolved_ids", []))
    errors = set(official.get("error_ids", []))
    empty = set(official.get("empty_patch_ids", []))
    rows = []
    for instance_id in manifest["instance_ids"]:
        result = latest_result(run_dir / "instances" / instance_id) or {}
        if instance_id in resolved:
            status = "resolved"
        elif instance_id in unresolved:
            status = "unresolved"
        elif instance_id in errors:
            status = "evaluator_error"
        elif instance_id in empty:
            status = "empty_patch"
        else:
            status = "not_submitted"
        rows.append({**result, "instance_id": instance_id, "official_status": status})
    agent_counts = Counter(row.get("agent_status", "not_run") for row in rows)
    patch_counts = Counter(row.get("patch_status", "none") for row in rows)
    official_counts = Counter(row["official_status"] for row in rows)
    api_values = [int(row.get("api_calls", 0)) for row in rows if row.get("agent_status")]
    duration_values = [
        float(row.get("duration_seconds", 0)) for row in rows if row.get("agent_status")
    ]
    report = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "resolved": len(resolved),
        "total_instances": len(manifest["instance_ids"]),
        "evaluated": len(resolved | unresolved | errors | empty),
        "agent_status_counts": dict(sorted(agent_counts.items())),
        "patch_status_counts": dict(sorted(patch_counts.items())),
        "official_status_counts": dict(sorted(official_counts.items())),
        "average_api_calls": round(sum(api_values) / len(api_values), 2) if api_values else 0,
        "average_duration_seconds": (
            round(sum(duration_values) / len(duration_values), 2)
            if duration_values
            else 0
        ),
        "instances": rows,
    }
    atomic_write_json(run_dir / "summary.json", report)
    lines = [
        f"# SWE-bench Report: `{manifest['run_id']}`",
        "",
        f"- Dataset: `{manifest['dataset']}`",
        f"- Model: `{manifest['model']}`",
        f"- Resolved: **{report['resolved']}/{report['total_instances']}**",
        f"- Evaluated: **{report['evaluated']}/{report['total_instances']}**",
        f"- Average API calls: **{report['average_api_calls']}**",
        f"- Average duration: **{report['average_duration_seconds']}s**",
        f"- Agent statuses: `{json.dumps(report['agent_status_counts'], sort_keys=True)}`",
        f"- Patch statuses: `{json.dumps(report['patch_status_counts'], sort_keys=True)}`",
        "",
        "| Instance | Agent | API calls | Patch | Official |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance_id']} | {row.get('agent_status', 'not_run')} | "
            f"{row.get('api_calls', 0)} | {row.get('patch_status', 'none')} | "
            f"{row['official_status']} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
