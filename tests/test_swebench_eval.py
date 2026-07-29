from __future__ import annotations

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
    assert "testing is not required" in prompt
    assert "Do not install dependencies" in prompt


def test_patch_export_includes_modify_add_delete_binary_and_preserves_index(tmp_path) -> None:
    _, mirror, commit = make_repo(tmp_path)
    workspace = tmp_path / "workspace"
    core.create_worktree(mirror, workspace, commit)
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
        "instance_timeout_seconds": 10,
        "harness_commit": "h",
        "swebench_commit": "s",
    }
    core.ensure_manifest(path, base)
    changed = {**base, "model": "other"}
    with pytest.raises(ValueError, match="model"):
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
    assert report["evaluated"] == 2
    assert report["agent_status_counts"] == {"completed": 1, "max_api_calls": 1}
    assert [row["official_status"] for row in report["instances"]] == [
        "resolved",
        "unresolved",
    ]
    assert (run_dir / "summary.md").is_file()


def test_load_subset_validates_duplicates(tmp_path) -> None:
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"instance_ids": ["a", "a"]}))
    with pytest.raises(ValueError, match="duplicate"):
        core.load_subset(path)


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
    )

    assert "SWE-bench evaluation mode" not in regular.system
    assert "SWE-bench evaluation mode" in swebench.system
    assert "Do not install dependencies" in swebench.system
    assert "Tests are optional" in swebench.system


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
