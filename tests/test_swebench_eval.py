from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evals.swebench import core
from evals.swebench.models import Task


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "change.txt").write_text("before\n", encoding="utf-8")
    (source / "delete.txt").write_text("delete me\n", encoding="utf-8")
    (source / "binary.bin").write_bytes(b"\x00old")
    git(source, "add", ".")
    git(source, "commit", "-m", "base")
    commit = git(source, "rev-parse", "HEAD")
    mirror = tmp_path / "mirror.git"
    subprocess.run(["git", "clone", "--mirror", str(source), str(mirror)], check=True)
    return source, mirror, commit


def test_task_from_dict_preserves_optional_defaults() -> None:
    task = Task.from_dict({
        "instance_id": "id",
        "repo": "owner/repo",
        "base_commit": "abc",
        "problem_statement": "Fix it",
    })

    assert task.platform == "linux/x86_64"
    assert task.instance_image_key == ""


def test_build_prompt_contains_issue_but_not_gold_fields() -> None:
    task = Task("id", "owner/repo", "abc", "Fix the actual issue", "1")
    prompt = core.build_prompt(task)
    assert "Fix the actual issue" in prompt
    assert "FAIL_TO_PASS" not in prompt
    assert "test_patch" not in prompt
    assert "gold" not in prompt.lower()
    assert "Do not install dependencies" in prompt
    assert "run the project's tests" in prompt
    assert "execute or import project code" in prompt
    assert "Use bash to inspect the workspace" in prompt
    assert "git diff" in prompt
    assert "tests run" not in prompt


def test_patch_export_includes_modify_add_delete_binary_and_preserves_index(tmp_path) -> None:
    _, mirror, commit = make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    core.create_isolated_workspace(mirror, workspace, commit)
    (workspace / "change.txt").write_text("after\n", encoding="utf-8")
    (workspace / "delete.txt").unlink()
    (workspace / "new.txt").write_text("new\n", encoding="utf-8")
    (workspace / "binary.bin").write_bytes(b"\x00new")

    before_index = git(workspace, "diff", "--cached")
    patch = core.export_patch(workspace, commit)

    assert "change.txt" in patch
    assert "delete.txt" in patch
    assert "new.txt" in patch
    assert "binary.bin" in patch
    assert git(workspace, "diff", "--cached") == before_index == ""
    core.validate_patch(mirror, commit, patch)


def test_git_diff_tool_reviews_all_changes_without_touching_index(
    load_module, tmp_path,
) -> None:
    tools = load_module("tools_swebench_git_diff", "tools.py")
    _, mirror, commit = make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    core.create_isolated_workspace(mirror, workspace, commit)
    (workspace / "change.txt").write_text("after\n", encoding="utf-8")
    (workspace / "delete.txt").unlink()
    (workspace / "new.txt").write_text("new\n", encoding="utf-8")
    (workspace / "binary.bin").write_bytes(b"\x00new")
    before_index = git(workspace, "diff", "--cached")

    output = tools.run_git_diff(tools.Workspace(workspace))

    assert "change.txt" in output
    assert "delete.txt" in output
    assert "new.txt" in output
    assert "binary.bin" in output
    assert "GIT binary patch" in output
    assert git(workspace, "diff", "--cached") == before_index == ""
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        tools.GitDiffParams.model_validate({"path": "../other-run"})


def test_isolated_workspace_preserves_ancestors_but_hides_future(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")

    history = source / "history.txt"
    history.write_text("ancestor\n", encoding="utf-8")
    git(source, "add", "history.txt")
    git(source, "commit", "-m", "ancestor")
    ancestor = git(source, "rev-parse", "HEAD")

    history.write_text("ancestor\nbase\n", encoding="utf-8")
    git(source, "add", "history.txt")
    git(source, "commit", "-m", "base")
    base = git(source, "rev-parse", "HEAD")

    history.write_text("ancestor\nbase\nfuture\n", encoding="utf-8")
    git(source, "add", "history.txt")
    git(source, "commit", "-m", "future")
    future = git(source, "rev-parse", "HEAD")

    mirror = tmp_path / "mirror.git"
    subprocess.run(["git", "clone", "--mirror", str(source), str(mirror)], check=True)
    workspace = tmp_path / "workspace"
    core.create_isolated_workspace(mirror, workspace, base)

    visible_history = git(workspace, "log", "--all", "--format=%H").splitlines()
    assert git(workspace, "rev-parse", "HEAD") == base
    assert visible_history == [base, ancestor]
    assert git(workspace, "show", "HEAD~1:history.txt") == "ancestor"
    assert ancestor in git(workspace, "blame", "--porcelain", "history.txt")
    assert git(workspace, "remote") == ""
    refs = subprocess.run(
        ["git", "show-ref"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    assert refs.returncode == 1
    assert refs.stdout == ""
    assert not (workspace / ".git" / "FETCH_HEAD").exists()
    assert not (workspace / ".git" / "objects" / "info" / "alternates").exists()
    assert str(mirror.resolve()) not in (
        workspace / ".git" / "config"
    ).read_text(encoding="utf-8")

    future_lookup = subprocess.run(
        ["git", "cat-file", "-e", f"{future}^{{commit}}"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    assert future_lookup.returncode != 0
    fsck = subprocess.run(
        ["git", "fsck", "--unreachable", "--no-reflogs"],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    )
    assert future not in fsck.stdout


def test_upsert_prediction_replaces_duplicate_atomically(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    core.upsert_prediction(
        path,
        {"instance_id": "b", "model_name_or_path": "m", "model_patch": "old"},
    )
    core.upsert_prediction(
        path,
        {"instance_id": "a", "model_name_or_path": "m", "model_patch": "a"},
    )
    core.upsert_prediction(
        path,
        {"instance_id": "b", "model_name_or_path": "m", "model_patch": "new"},
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == ["a", "b"]
    assert rows[1]["model_patch"] == "new"
    assert not path.with_name(".predictions.jsonl.tmp").exists()


def test_remove_prediction_drops_stale_retry_output(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    for instance_id in ("a", "b"):
        core.upsert_prediction(
            path,
            {"instance_id": instance_id, "model_name_or_path": "m", "model_patch": instance_id},
        )
    core.remove_prediction(path, "a")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == ["b"]


def test_failed_attempt_requires_explicit_rerun(tmp_path) -> None:
    instance = tmp_path / "instance"
    first = instance / "attempt-1"
    first.mkdir(parents=True)
    core.atomic_write_json(
        first / "result.json",
        {
            "agent_status": "timeout",
            "patch_status": "empty",
        },
    )
    assert core.should_run(instance, rerun_failed=False) is False
    assert core.should_run(instance, rerun_failed=True) is True
    assert core.next_attempt(instance) == 2


def test_agent_attempt_records_api_calls_when_the_agent_errors(
    monkeypatch, tmp_path,
) -> None:
    _, mirror, commit = make_repo(tmp_path)
    task = Task("owner__repo-1", "owner/repo", commit, "Fix it")
    attempt_dir = tmp_path / "attempt-1"

    import main
    import sandbox

    class FakeDockerSandbox:
        workspace_root = "/testbed"

        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            return self

        def close(self):
            pass

        def export_patch(self):
            return ""

    class FakeState:
        def __init__(self, messages):
            del messages
            self.api_call_count = 0

    async def fail_agent_loop(state, _session):
        state.api_call_count = 7
        raise json.JSONDecodeError("Expecting value", "not-json", 0)

    captured_session = {}

    def fake_create_parent_session(*_args, **kwargs):
        captured_session.update(kwargs)
        return object()

    monkeypatch.setattr(main, "LoopState", FakeState)
    monkeypatch.setattr(main, "create_parent_session", fake_create_parent_session)
    monkeypatch.setattr(main, "agent_loop", fail_agent_loop)
    monkeypatch.setattr(sandbox, "DockerSandbox", FakeDockerSandbox)
    task = Task(
        task.instance_id, task.repo, task.base_commit, task.problem_statement,
        task.version, instance_image_key="example/image:latest",
    )

    result = asyncio.run(core.run_agent_attempt(
        task,
        mirror,
        attempt_dir,
        run_id="test",
        model="test-model",
        max_api_calls=30,
        steering_policy="staged",
        reasoning_effort=None,
        max_output_tokens=None,
        timeout=None,
        generate_environment="docker",
    ))

    assert result.agent_status == "error"
    assert result.api_calls == 7
    assert isinstance(
        captured_session["steering_policy"], core.SwebenchSteeringPolicy
    )


def test_docker_start_failure_is_not_misclassified_as_invalid_patch(
    monkeypatch, tmp_path,
) -> None:
    import sandbox

    closed = []

    class FailingDockerSandbox:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("docker failed to start")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(sandbox, "DockerSandbox", FailingDockerSandbox)
    task = Task(
        "owner__repo-1", "owner/repo", "base", "Fix it",
        instance_image_key="example/image:latest",
    )

    result = asyncio.run(core.run_agent_attempt(
        task, tmp_path / "mirror", tmp_path / "attempt-1",
        run_id="run", model="model", max_api_calls=1,
        steering_policy="none", reasoning_effort=None,
        max_output_tokens=None, timeout=None, generate_environment="docker",
    ))

    assert result.agent_status == "error"
    assert result.patch_status == "not_exported"
    assert "docker failed to start" in result.error
    assert closed == [True]


def test_local_agent_attempt_fails_before_workspace_or_agent_start(
    monkeypatch, tmp_path,
) -> None:
    import main

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("local generation reached agent or workspace setup")

    monkeypatch.setattr(core, "create_isolated_workspace", unexpected_call)
    monkeypatch.setattr(main, "create_parent_session", unexpected_call)
    task = Task("owner__repo-1", "owner/repo", "base", "Fix it")

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(core.run_agent_attempt(
            task, tmp_path / "mirror", tmp_path / "attempt-1",
            run_id="run", model="model", max_api_calls=1,
            steering_policy="none", reasoning_effort=None,
            max_output_tokens=None, timeout=None,
            generate_environment="local",
        ))

    assert str(exc_info.value) == core.DOCKER_GENERATION_REQUIRED
    assert not (tmp_path / "attempt-1").exists()


def test_final_text_write_failure_still_closes_docker_sandbox(
    monkeypatch, tmp_path,
) -> None:
    import main
    import sandbox

    closed = []

    class FakeDockerSandbox:
        workspace_root = "/testbed"

        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            return self

        def close(self):
            closed.append(True)

        def export_patch(self):
            return ""

    def fail_session(*_args, **_kwargs):
        raise RuntimeError("agent setup failed")

    original_write_text = Path.write_text
    failed = False

    def fail_final_text_once(path, data, *args, **kwargs):
        nonlocal failed
        if path.name == "final_text.txt" and not failed:
            failed = True
            raise OSError("disk full")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(sandbox, "DockerSandbox", FakeDockerSandbox)
    monkeypatch.setattr(main, "create_parent_session", fail_session)
    monkeypatch.setattr(Path, "write_text", fail_final_text_once)
    task = Task(
        "owner__repo-1", "owner/repo", "base", "Fix it",
        instance_image_key="example/image:latest",
    )

    result = asyncio.run(core.run_agent_attempt(
        task, tmp_path / "mirror", tmp_path / "attempt-1",
        run_id="run", model="model", max_api_calls=1,
        steering_policy="none", reasoning_effort=None,
        max_output_tokens=None, timeout=None, generate_environment="docker",
    ))

    assert result.agent_status == "error"
    assert result.patch_status == "empty"
    assert closed == [True]


def test_successful_attempt_is_never_rerun(tmp_path) -> None:
    result = tmp_path / "instance" / "attempt-1" / "result.json"
    core.atomic_write_json(
        result,
        {
            "agent_status": "completed",
            "patch_status": "produced",
        },
    )
    assert core.should_run(result.parents[1], rerun_failed=True) is False


def test_manifest_rejects_material_config_changes(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    base = {
        "schema_version": 2,
        "dataset": "verified",
        "split": "test",
        "instance_ids": ["a"],
        "model": "m",
        "max_api_calls": 30,
        "steering_policy": "staged",
        "steering_thresholds": [45, 60],
        "reasoning_effort": None,
        "max_output_tokens": None,
        "model_limits": {
            "context_window_tokens": 32000,
            "max_input_tokens": 32000,
            "max_output_tokens": None,
        },
        "instance_timeout_seconds": 10,
        "harness_commit": "h",
        "swebench_commit": "s",
    }
    core.ensure_manifest(path, base)
    changed = {**base, "model": "other"}
    with pytest.raises(ValueError, match="model"):
        core.ensure_manifest(path, changed)
    changed = {**base, "reasoning_effort": "high"}
    with pytest.raises(ValueError, match="reasoning_effort"):
        core.ensure_manifest(path, changed)
    changed = {
        **base,
        "steering_policy": "none",
        "steering_thresholds": [],
    }
    with pytest.raises(ValueError, match="steering_policy"):
        core.ensure_manifest(path, changed)
    changed = {**base, "max_output_tokens": 8000}
    with pytest.raises(ValueError, match="max_output_tokens"):
        core.ensure_manifest(path, changed)
    changed = {**base, "model_limits": {**base["model_limits"], "max_input_tokens": 16000}}
    with pytest.raises(ValueError, match="model_limits"):
        core.ensure_manifest(path, changed)
    # Legacy manifests without provider remain resumable when no pin is used.
    core.ensure_manifest(path, {**base, "provider": ""})
    changed = {**base, "provider": "provider/fp8"}
    with pytest.raises(ValueError, match="provider"):
        core.ensure_manifest(path, changed)


def test_manifest_migrates_legacy_resume_after_config_check(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    core.atomic_write_json(path, {
        "schema_version": 1,
        "dataset": "verified",
        "model": "hy3",
        "harness_commit": "old-harness",
        "generate_environment": "docker",
        "solve_network_mode": "none",
        "solve_workspace_mode": "image_testbed",
        "image_namespace": "swebench",
    })
    proposed = {
        "schema_version": 2,
        "dataset": "verified",
        "model": "hy3",
        "harness_commit": "new-harness",
        "harness_worktree_dirty": True,
        "generate_environment": "docker",
        "solve_network_mode": "none",
        "solve_workspace_mode": "image_testbed",
        "image_namespace": "swebench",
        "model_limits": {
            "context_window_tokens": 262144,
            "max_input_tokens": 192000,
            "max_output_tokens": 128000,
        },
    }
    assert core.ensure_manifest(path, proposed)["schema_version"] == 2
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["model_limits"] == proposed["model_limits"]
    assert migrated["harness_commit"] == "new-harness"
    harness_migration = next(
        item for item in migrated["migrations"]
        if item["kind"] == "model_limits_and_harness_upgrade"
    )
    assert harness_migration["previous_harness_commit"] == "old-harness"

    with pytest.raises(ValueError, match="model"):
        core.ensure_manifest(path, {**proposed, "model": "other"})
    with pytest.raises(ValueError, match="harness_commit"):
        core.ensure_manifest(path, {**proposed, "harness_commit": "another-harness"})


def test_legacy_local_manifest_requires_docker(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    core.atomic_write_json(path, {"schema_version": 2})

    with pytest.raises(ValueError) as exc_info:
        core.ensure_manifest(path, {
            "schema_version": 3,
            "generate_environment": "docker",
            "solve_network_mode": "none",
        })

    assert str(exc_info.value) == core.DOCKER_GENERATION_REQUIRED


def test_manifest_with_explicit_local_generation_is_rejected(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    core.atomic_write_json(path, {
        "schema_version": 4,
        "generate_environment": "local",
    })

    with pytest.raises(ValueError) as exc_info:
        core.ensure_manifest(path, {"schema_version": 4})

    assert str(exc_info.value) == core.DOCKER_GENERATION_REQUIRED


def test_bind_mount_manifest_requires_new_run_for_image_testbed(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    core.atomic_write_json(path, {
        "schema_version": 3,
        "generate_environment": "docker",
        "solve_network_mode": "none",
        "image_namespace": "swebench",
    })

    with pytest.raises(ValueError, match="legacy Docker runs used bind mounts"):
        core.ensure_manifest(path, {
            "schema_version": 4,
            "generate_environment": "docker",
            "solve_network_mode": "none",
            "solve_workspace_mode": "image_testbed",
            "image_namespace": "swebench",
        })


def test_report_merges_agent_and_official_statuses(tmp_path) -> None:
    run_dir = tmp_path / "run"
    core.atomic_write_json(
        run_dir / "manifest.json",
        {
            "run_id": "r1",
            "dataset": "verified",
            "model": "vendor/model",
            "instance_ids": ["a", "b"],
        },
    )
    for instance_id, status in (("a", "completed"), ("b", "max_api_calls")):
        core.atomic_write_json(
            run_dir / "instances" / instance_id / "attempt-1" / "result.json",
            {
                "instance_id": instance_id,
                "attempt": 1,
                "agent_status": status,
                "patch_status": "produced",
                "api_calls": 2,
                "duration_seconds": 3,
            },
        )
    core.atomic_write_json(
        run_dir / "official" / "vendor__model.r1.json",
        {
            "resolved_ids": ["a"],
            "unresolved_ids": ["b"],
            "error_ids": [],
            "empty_patch_ids": [],
        },
    )
    report = core.generate_report(run_dir)
    assert report["resolved"] == 1
    assert report["total_instances"] == 2
    assert report["evaluated"] == 2
    assert report["agent_status_counts"] == {"completed": 1, "max_api_calls": 1}
    assert [row["official_status"] for row in report["instances"]] == [
        "resolved",
        "unresolved",
    ]
    assert (run_dir / "summary.md").is_file()


def test_report_uses_planned_tasks_as_resolved_denominator(tmp_path) -> None:
    """An unsubmitted empty patch must not disappear from the resolve rate."""
    run_dir = tmp_path / "run"
    instance_ids = ["resolved-a", "resolved-b", "resolved-c", "unresolved", "empty"]
    core.atomic_write_json(
        run_dir / "manifest.json",
        {
            "run_id": "historical-empty-patch",
            "dataset": "verified",
            "model": "vendor/model",
            "instance_ids": instance_ids,
        },
    )
    for instance_id in instance_ids:
        patch_status = "empty" if instance_id == "empty" else "produced"
        core.atomic_write_json(
            run_dir / "instances" / instance_id / "attempt-1" / "result.json",
            {
                "instance_id": instance_id,
                "attempt": 1,
                "agent_status": "completed",
                "patch_status": patch_status,
                "api_calls": 2,
                "duration_seconds": 3,
            },
        )
    core.atomic_write_json(
        run_dir / "official" / "vendor__model.historical-empty-patch.json",
        {
            "resolved_ids": ["resolved-a", "resolved-b", "resolved-c"],
            "unresolved_ids": ["unresolved"],
            "error_ids": [],
            "empty_patch_ids": [],
        },
    )

    report = core.generate_report(run_dir)

    assert report["resolved"] == 3
    assert report["total_instances"] == 5
    assert report["evaluated"] == 4
    assert report["official_status_counts"] == {
        "not_submitted": 1,
        "resolved": 3,
        "unresolved": 1,
    }
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "- Resolved: **3/5**" in summary
    assert "- Evaluated: **4/5**" in summary
    assert "- Resolved: **3/4**" not in summary


def test_load_subset_validates_duplicates(tmp_path) -> None:
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"instance_ids": ["a", "a"]}))
    with pytest.raises(ValueError, match="duplicate"):
        core.load_subset(path)


def test_small_subset_has_five_distinct_repositories_and_smoke_control() -> None:
    subsets = core.PROJECT_ROOT / "evals" / "swebench" / "subsets"
    dataset, split, instance_ids, selection = core.load_subset(subsets / "small.json")
    _, _, smoke_ids, _ = core.load_subset(subsets / "smoke.json")

    repositories = {instance_id.split("__", 1)[0] for instance_id in instance_ids}
    assert dataset == "SWE-bench/SWE-bench_Verified"
    assert split == "test"
    assert len(instance_ids) == 5
    assert len(repositories) == 5
    assert set(smoke_ids) <= set(instance_ids)
    assert "not a representative benchmark sample" in selection


def test_small_11_subset_extends_small_with_ten_distinct_repositories() -> None:
    subsets = core.PROJECT_ROOT / "evals" / "swebench" / "subsets"
    dataset, split, small_ids, _ = core.load_subset(subsets / "small.json")
    _, _, instance_ids, selection = core.load_subset(subsets / "small_11.json")

    repositories = {instance_id.split("__", 1)[0] for instance_id in instance_ids}
    assert dataset == "SWE-bench/SWE-bench_Verified"
    assert split == "test"
    assert instance_ids[: len(small_ids)] == small_ids
    assert len(instance_ids) == 11
    assert len(repositories) == 10
    assert "not a representative benchmark sample" in selection


def test_parent_session_accepts_per_session_api_budget(load_module, tmp_path) -> None:
    main = load_module("main_swebench_budget", "main.py")
    session = main.create_parent_session(
        tmp_path,
        approval_handler=None,
        on_text=None,
        max_api_calls=7,
    )
    assert session.max_api_calls == 7


def test_parent_session_system_addendum_is_per_session(load_module, tmp_path) -> None:
    main = load_module("main_swebench_system", "main.py")
    regular = main.create_parent_session(
        tmp_path,
        approval_handler=None,
        on_text=None,
    )
    swebench = main.create_parent_session(
        tmp_path,
        approval_handler=None,
        on_text=None,
        system_addendum=core.SWEBENCH_SYSTEM_ADDENDUM,
        tool_names=core.SWEBENCH_TOOL_NAMES,
    )

    assert "SWE-bench evaluation mode" not in regular.system
    assert "SWE-bench evaluation mode" in swebench.system
    assert "Do not install dependencies" in swebench.system
    assert "run the project's tests" in swebench.system
    assert "execute or import project code" in swebench.system
    assert "review of the final diff" in swebench.system
    regular_names = {tool["name"] for tool in regular.tools}
    swebench_names = {tool["name"] for tool in swebench.tools}
    assert "bash" in regular_names
    assert "git_diff" not in regular_names
    assert swebench_names == set(core.SWEBENCH_TOOL_NAMES)
    assert set(swebench.registry) == set(core.SWEBENCH_TOOL_NAMES)
    assert "bash" in swebench_names
    assert "git_diff" not in swebench_names
    assert {"bash", "task", "todo"} <= swebench_names


def test_run_command_executes_all_pipeline_phases(monkeypatch, tmp_path) -> None:
    from evals.swebench import cli

    calls = []
    args = argparse.Namespace(run_id="pipeline", runs_dir=str(tmp_path))

    async def generate(_args):
        calls.append("generate")
        predictions = tmp_path / "pipeline" / "predictions.jsonl"
        predictions.parent.mkdir(parents=True)
        predictions.write_text('{"instance_id": "task"}\n', encoding="utf-8")
        return 0

    def evaluate(_args):
        calls.append("evaluate")
        return 0

    def report(_args):
        calls.append("report")
        return 0

    monkeypatch.setattr(cli, "cmd_generate", generate)
    monkeypatch.setattr(cli, "cmd_evaluate", evaluate)
    monkeypatch.setattr(cli, "cmd_report", report)

    assert asyncio.run(cli.cmd_run(args)) == 0
    assert calls == ["generate", "evaluate", "report"]


def test_report_command_automatically_generates_harness_diagnostics(
    monkeypatch, tmp_path, capsys,
) -> None:
    from evals.analyze import automation
    from evals.swebench import cli

    target = tmp_path / "run"
    target.mkdir()
    monkeypatch.setattr(cli, "generate_report", lambda _path: {
        "resolved": 1,
        "total_instances": 1,
        "evaluated": 1,
    })
    called = []

    def generate_diagnostics(path):
        called.append(path)
        return {"diff": path / "harness-diagnostic-diff.md"}

    monkeypatch.setattr(automation, "generate_run_diagnostics", generate_diagnostics)
    args = argparse.Namespace(run_id="run", runs_dir=str(tmp_path))

    assert cli.cmd_report(args) == 0
    assert called == [target]
    assert "Harness diagnostic diff" in capsys.readouterr().out


def test_diagnostic_failure_does_not_change_swebench_report_status(
    monkeypatch, tmp_path, capsys,
) -> None:
    from evals.analyze import automation
    from evals.swebench import cli

    monkeypatch.setattr(cli, "generate_report", lambda _path: {
        "resolved": 0,
        "total_instances": 1,
        "evaluated": 1,
    })
    monkeypatch.setattr(
        automation,
        "generate_run_diagnostics",
        lambda _path: (_ for _ in ()).throw(ValueError("bad trace")),
    )
    args = argparse.Namespace(run_id="run", runs_dir=str(tmp_path))

    assert cli.cmd_report(args) == 0
    assert "diagnostics warning" in capsys.readouterr().err


def test_run_command_stops_when_generation_has_no_predictions(
    monkeypatch, tmp_path
) -> None:
    from evals.swebench import cli

    calls = []
    args = argparse.Namespace(run_id="pipeline", runs_dir=str(tmp_path))

    async def generate(_args):
        calls.append("generate")
        return 0

    def evaluate(_args):
        calls.append("evaluate")
        return 0

    monkeypatch.setattr(cli, "cmd_generate", generate)
    monkeypatch.setattr(cli, "cmd_evaluate", evaluate)

    with pytest.raises(SystemExit, match="no predictions"):
        asyncio.run(cli.cmd_run(args))
    assert calls == ["generate"]


def test_run_parser_combines_generation_and_evaluation_options() -> None:
    from evals.swebench import cli

    args = cli.build_parser().parse_args(
        [
            "run",
            "--run-id",
            "pipeline",
            "--subset",
            "smoke.json",
            "--max-api-calls",
            "7",
            "--max-workers",
            "2",
            "--agent-workers",
            "3",
            "--reasoning-effort",
            "high",
            "--max-output-tokens",
            "16000",
            "--cache-level",
            "env",
        ]
    )

    assert args.func is cli.cmd_run
    assert args.max_api_calls == 7
    assert args.steering_policy == "staged"
    assert args.max_workers == 2
    assert args.agent_workers == 3
    assert args.reasoning_effort == "high"
    assert args.max_output_tokens == 16000
    assert args.cache_level == "env"
    assert args.namespace == "swebench"
    assert args.generate_environment == "docker"


def test_swebench_parser_accepts_none_reasoning_effort() -> None:
    from evals.swebench import cli

    args = cli.build_parser().parse_args(
        ["run", "--run-id", "pipeline", "--subset", "small.json", "--reasoning-effort", "none"]
    )

    assert args.reasoning_effort == "none"


def test_swebench_defaults_to_five_workers() -> None:
    from evals.swebench import cli

    evaluate = cli.build_parser().parse_args(["evaluate", "--run-id", "pipeline"])
    pipeline = cli.build_parser().parse_args(
        ["run", "--run-id", "pipeline", "--subset", "small.json"]
    )

    assert evaluate.max_workers == 5
    assert pipeline.max_workers == 5
    assert pipeline.agent_workers == 5
    assert pipeline.max_api_calls == 90
    assert pipeline.steering_policy == "staged"
    assert pipeline.reasoning_effort is None
    assert pipeline.max_output_tokens is None
    assert pipeline.instance_timeout is None
    assert pipeline.generate_environment == "docker"
    assert evaluate.cache_level == "instance"
    assert pipeline.cache_level == "instance"


def test_swebench_parser_accepts_local_for_actionable_runtime_error() -> None:
    from evals.swebench import cli

    args = cli.build_parser().parse_args([
        "generate", "--run-id", "local", "--subset", "small.json",
        "--generate-environment", "local",
    ])

    assert args.generate_environment == "local"


def test_generate_rejects_local_environment_before_setup() -> None:
    from evals.swebench import cli

    args = argparse.Namespace(generate_environment="local")

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(cli.cmd_generate(args))

    assert str(exc_info.value) == core.DOCKER_GENERATION_REQUIRED


def test_swebench_parser_can_disable_staged_steering() -> None:
    from evals.swebench import cli

    args = cli.build_parser().parse_args([
        "generate",
        "--run-id",
        "pipeline",
        "--subset",
        "small.json",
        "--steering-policy",
        "none",
    ])

    assert args.steering_policy == "none"


def test_staged_steering_uses_patch_aware_second_message() -> None:
    policy = core.SwebenchSteeringPolicy(lambda: "")

    first = policy.after_turn(45)
    no_patch = policy.after_turn(60)

    assert first is not None
    assert first.reason == "converge_on_root_cause"
    assert no_patch is not None
    assert no_patch.reason == "implement_patch_and_finish"
    assert policy.after_turn(60) is None

    patch_policy = core.SwebenchSteeringPolicy(lambda: "diff --git a/x b/x")
    has_patch = patch_policy.after_turn(60)
    assert has_patch is not None
    assert has_patch.reason == "review_patch_and_finish"


@pytest.mark.parametrize("option", ["--agent-workers", "--max-workers"])
def test_swebench_cli_rejects_nonpositive_worker_counts(option) -> None:
    from evals.swebench import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["run", "--run-id", "invalid", "--subset", "small.json", option, "0"]
        )


def test_generate_runs_five_workers_and_continues_after_error(
    monkeypatch, tmp_path
) -> None:
    from evals.swebench import cli
    from evals.swebench.models import AgentResult

    tasks = [Task(f"task-{index}", "owner/repo", "abc", "Fix it") for index in range(7)]
    active = 0
    maximum = 0
    calls = []
    lock = threading.Lock()

    def fake_worker(task, _mirror, attempt_dir, **_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(task.instance_id)
        time.sleep(0.03)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        status = "error" if task.instance_id == "task-1" else "completed"
        result = AgentResult(
            task.instance_id,
            1,
            status,
            "empty",
            stop_reason=status,
            error="boom" if status == "error" else "",
        )
        core.atomic_write_json(attempt_dir / "result.json", result.to_dict())
        with lock:
            active -= 1
        return result

    monkeypatch.setattr(cli, "resolve_swebench_repo", lambda _value: tmp_path)
    monkeypatch.setattr(cli, "resolve_swebench_python", lambda _value, _repo: tmp_path / "python")
    monkeypatch.setattr(
        cli,
        "load_subset",
        lambda _path: ("dataset", "test", [task.instance_id for task in tasks], "test"),
    )
    monkeypatch.setattr(cli, "create_manifest", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "ensure_manifest", lambda *_args: {})
    monkeypatch.setattr(cli, "load_tasks_via_bridge", lambda *_args: tasks)
    monkeypatch.setattr(cli, "ensure_mirror", lambda _task, _cache: tmp_path / "mirror")
    monkeypatch.setattr(cli, "run_agent_attempt_worker", fake_worker)
    monkeypatch.setattr(
        cli,
        "ProcessPoolExecutor",
        lambda max_workers, mp_context: ThreadPoolExecutor(max_workers=max_workers),
    )
    args = argparse.Namespace(
        run_id="parallel",
        runs_dir=str(tmp_path / "runs"),
        swebench_repo=None,
        swebench_python=None,
        subset="subset.json",
        model="test-model",
        repo_cache=str(tmp_path / "cache"),
        max_api_calls=3,
        steering_policy="staged",
        reasoning_effort=None,
        max_output_tokens=None,
        instance_timeout=None,
        rerun_failed=False,
        agent_workers=5,
    )

    assert asyncio.run(cli.cmd_generate(args)) == 0
    assert maximum == 5
    assert sorted(calls) == sorted(task.instance_id for task in tasks)


def test_swebench_cli_rejects_nonpositive_output_tokens() -> None:
    from evals.swebench import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "generate",
                "--run-id",
                "invalid",
                "--subset",
                "small.json",
                "--max-output-tokens",
                "0",
            ]
        )


def test_swebench_cli_accepts_optional_instance_timeout() -> None:
    from evals.swebench import cli

    args = cli.build_parser().parse_args(
        [
            "generate",
            "--run-id",
            "bounded",
            "--subset",
            "small.json",
            "--instance-timeout",
            "3600",
        ]
    )

    assert args.instance_timeout == 3600


def test_swebench_cli_rejects_auto_output_tokens() -> None:
    from evals.swebench import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "generate",
                "--run-id",
                "auto-output",
                "--subset",
                "small.json",
                "--max-output-tokens",
                "auto",
            ]
        )


def test_official_evaluator_passes_cache_level_to_harness(
    monkeypatch, tmp_path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "SWE-bench/SWE-bench_Verified",
                "split": "test",
                "run_id": "cache-test",
                "model": "provider/model",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "owner__repo-1",
                "model_name_or_path": "provider/model",
                "model_patch": "patch",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        summary = run_dir / "official" / "provider__model.cache-test.json"
        summary.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    core.run_official_evaluator(
        swebench_python=Path("/swebench/python"),
        swebench_repo=tmp_path / "SWE-bench",
        run_dir=run_dir,
        namespace="swebench",
        max_workers=4,
        cache_level="instance",
    )

    command = captured["command"]
    cache_index = command.index("--cache_level")
    assert command[cache_index + 1] == "instance"


def test_swebench_python_keeps_venv_symlink(tmp_path) -> None:
    from evals.swebench.cli import resolve_swebench_python

    repo = tmp_path / "SWE-bench"
    executable = repo / ".venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    target = tmp_path / "base-python"
    target.write_text("", encoding="utf-8")
    executable.symlink_to(target)

    resolved = resolve_swebench_python(None, repo)

    assert resolved == executable
    assert resolved.is_symlink()
