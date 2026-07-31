from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
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
    assert "shell tool is unavailable" in prompt
    assert "Use git_diff to review the final patch" in prompt
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
        "dataset": "verified",
        "split": "test",
        "instance_ids": ["a"],
        "model": "m",
        "max_api_calls": 30,
        "reasoning_effort": None,
        "max_output_tokens": None,
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
    changed = {**base, "max_output_tokens": 8000}
    with pytest.raises(ValueError, match="max_output_tokens"):
        core.ensure_manifest(path, changed)


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


def test_small_10_subset_extends_small_with_five_distinct_repositories() -> None:
    subsets = core.PROJECT_ROOT / "evals" / "swebench" / "subsets"
    dataset, split, small_ids, _ = core.load_subset(subsets / "small.json")
    _, _, instance_ids, selection = core.load_subset(subsets / "small_10.json")

    repositories = {instance_id.split("__", 1)[0] for instance_id in instance_ids}
    assert dataset == "SWE-bench/SWE-bench_Verified"
    assert split == "test"
    assert instance_ids[: len(small_ids)] == small_ids
    assert len(instance_ids) == 10
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
    assert "bash" not in swebench_names
    assert {"git_diff", "task", "todo"} <= swebench_names


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
    assert args.max_workers == 2
    assert args.reasoning_effort == "high"
    assert args.max_output_tokens == 16000
    assert args.cache_level == "env"
    assert args.namespace == "swebench"


def test_evaluation_defaults_to_four_workers() -> None:
    from evals.swebench import cli

    evaluate = cli.build_parser().parse_args(["evaluate", "--run-id", "pipeline"])
    pipeline = cli.build_parser().parse_args(
        ["run", "--run-id", "pipeline", "--subset", "small.json"]
    )

    assert evaluate.max_workers == 4
    assert pipeline.max_workers == 4
    assert pipeline.reasoning_effort is None
    assert pipeline.max_output_tokens is None
    assert pipeline.instance_timeout is None
    assert evaluate.cache_level == "instance"
    assert pipeline.cache_level == "instance"


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
