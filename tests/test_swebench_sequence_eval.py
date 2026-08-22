from __future__ import annotations

import json
import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.swebench.models import Task
from evals.swebench_sequence import core
from evals.swebench_sequence.cli import build_parser
from evals.swebench_sequence.models import SequenceSpec
from tools import TodoManager, TodoParams
from workspace import Workspace


CURATED_IDS = [
    "django__django-15277",
    "django__django-15280",
    "django__django-15161",
    "django__django-15368",
    "django__django-15375",
    "django__django-15382",
]


def assistant_text(text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text, "source": "test"}],
        "runtime": {"model_id": "old", "provider": "", "protocol": "responses"},
    }


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def make_history(tmp_path: Path, count: int = 3) -> tuple[Path, list[str]]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    commits = []
    for number in range(1, count + 1):
        (source / "state.txt").write_text(f"base-{number}\n", encoding="utf-8")
        git(source, "add", "state.txt")
        git(source, "commit", "-m", f"base {number}")
        commits.append(git(source, "rev-parse", "HEAD"))
    mirror = tmp_path / "mirror.git"
    subprocess.run(["git", "clone", "--mirror", str(source), str(mirror)], check=True)
    return mirror, commits


def task(number: int, commit: str) -> Task:
    return Task(f"owner__repo-{number}", "owner/repo", commit, f"Issue {number}")


def fake_main(agent_loop):
    class LoopState:
        def __init__(self, messages):
            self.messages = messages
            self.api_call_count = 0

    def create_parent_session(workdir, **kwargs):
        del kwargs
        return SimpleNamespace(todo=TodoManager(), workspace=Workspace(Path(workdir)))

    return SimpleNamespace(
        MODEL_ID="old",
        LoopState=LoopState,
        create_parent_session=create_parent_session,
        agent_loop=agent_loop,
    )


def episode_kwargs(run_dir: Path, position: int) -> dict:
    return {
        "position": position,
        "run_id": "run",
        "session_id": "test-sequence",
        "model": "test/model",
        "provider": "",
        "max_api_calls": 5,
        "reasoning_effort": None,
        "max_output_tokens": None,
        "timeout": None,
    }


def test_curated_django_sequence_is_ordered_and_scoped() -> None:
    path = Path("evals/swebench/subsets/django_db_models_6.json")
    spec = core.load_sequence_spec(path)
    assert spec.sequence_id == "django-db-models-6"
    assert spec.repo == "django/django"
    assert spec.subsystem == "django/db/models"
    assert spec.instance_ids == CURATED_IDS


def test_load_sequence_spec_rejects_missing_metadata_and_duplicates(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"instance_ids": ["a", "a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        core.load_sequence_spec(path)
    path.write_text(json.dumps({"instance_ids": ["a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing metadata"):
        core.load_sequence_spec(path)


def test_validate_task_sequence_records_ancestry_and_rejects_wrong_repo(tmp_path) -> None:
    mirror, commits = make_history(tmp_path)
    tasks = [task(1, commits[0]), task(2, commits[2])]
    spec = SequenceSpec("dataset", "test", "", "seq", "owner/repo", "area", [t.instance_id for t in tasks])
    rows = core.validate_task_sequence(spec, tasks, mirror)
    assert [row["commits_since_previous"] for row in rows] == [0, 2]

    bad = [tasks[0], Task(tasks[1].instance_id, "other/repo", commits[2], "Issue")]
    with pytest.raises(ValueError, match="outside owner/repo"):
        core.validate_task_sequence(spec, bad, mirror)


def test_episode_reopens_history_but_uses_clean_next_base(monkeypatch, tmp_path) -> None:
    mirror, commits = make_history(tmp_path, 2)
    run_dir = tmp_path / "run"
    seen_states = []

    async def agent_loop(state, session):
        seen_states.append(state.api_call_count)
        position = len(seen_states)
        assert session.workspace.root == (run_dir / "workspace").resolve()
        if position == 1:
            assert (session.workspace.root / "state.txt").read_text() == "base-1\n"
            (session.workspace.root / "first.txt").write_text("first patch\n")
            (session.workspace.root / ".transcripts").mkdir()
            (session.workspace.root / ".transcripts" / "snapshot.md").write_text("runtime only\n")
            state.messages.extend([{
                "role": "assistant",
                "content": [{
                    "type": "tool_call", "name": "read_file", "arguments": "{}",
                    "pairing": {"call_id": "call-1"},
                }],
                "runtime": {"model_id": "old", "provider": "", "protocol": "responses"},
            }, {
                "role": "tool", "call_id": "call-1",
                "content": "inspected models", "is_error": False,
            }, assistant_text("first episode complete")])
        else:
            assert (session.workspace.root / "state.txt").read_text() == "base-2\n"
            assert not (session.workspace.root / "first.txt").exists()
            assert any("first episode complete" in str(msg.get("content")) for msg in state.messages)
            assert any(msg.get("content") == "inspected models" for msg in state.messages)
            (session.workspace.root / "second.txt").write_text("second patch\n")
            state.messages.append(assistant_text("second episode complete"))
        state.api_call_count = 1
        return SimpleNamespace(stop_reason="completed", final_text="done", api_calls=1)

    monkeypatch.setitem(sys.modules, "main", fake_main(agent_loop))
    first = asyncio.run(core.run_sequence_episode(
        task(1, commits[0]), mirror, run_dir, **episode_kwargs(run_dir, 1)
    ))
    second = asyncio.run(core.run_sequence_episode(
        task(2, commits[1]), mirror, run_dir, **episode_kwargs(run_dir, 2)
    ))

    assert seen_states == [0, 0]
    assert first.resumed_message_count == 0
    assert second.resumed_message_count > 0
    assert first.history_committed and second.history_committed
    assert "first.txt" in (run_dir / "instances" / task(1, commits[0]).instance_id / "attempt-1" / "patch.diff").read_text()
    assert ".transcripts" not in (run_dir / "instances" / task(1, commits[0]).instance_id / "attempt-1" / "patch.diff").read_text()
    assert (run_dir / "instances" / task(1, commits[0]).instance_id / "attempt-1" / "workspace" / ".transcripts" / "snapshot.md").exists()
    second_patch = (run_dir / "instances" / task(2, commits[1]).instance_id / "attempt-1" / "patch.diff").read_text()
    assert "second.txt" in second_patch
    assert "first.txt" not in second_patch


def test_error_episode_rolls_back_history_and_todo(monkeypatch, tmp_path) -> None:
    mirror, commits = make_history(tmp_path, 3)
    run_dir = tmp_path / "run"
    calls = 0

    async def agent_loop(state, session):
        nonlocal calls
        calls += 1
        if calls == 1:
            state.messages.append(assistant_text("committed marker"))
            session.todo.update(TodoParams.model_validate({"items": [{
                "content": "kept", "status": "pending", "active_form": "Keeping"
            }]}))
            return SimpleNamespace(stop_reason="max_api_calls", final_text="budget", api_calls=5)
        if calls == 2:
            assert [item.content for item in session.todo.state.items] == ["kept"]
            state.messages.append(assistant_text("failed marker"))
            session.todo.update(TodoParams.model_validate({"items": [{
                "content": "discarded", "status": "pending", "active_form": "Discarding"
            }]}))
            state.api_call_count = 2
            raise RuntimeError("provider failed")
        contents = [str(msg.get("content", "")) for msg in state.messages]
        assert any("committed marker" in content for content in contents)
        assert not any("failed marker" in content for content in contents)
        assert not any("owner__repo-2" in content for content in contents)
        assert [item.content for item in session.todo.state.items] == ["kept"]
        state.messages.append(assistant_text("after rollback"))
        return SimpleNamespace(stop_reason="completed", final_text="done", api_calls=1)

    monkeypatch.setitem(sys.modules, "main", fake_main(agent_loop))
    results = []
    for position, commit in enumerate(commits, 1):
        results.append(asyncio.run(core.run_sequence_episode(
            task(position, commit), mirror, run_dir, **episode_kwargs(run_dir, position)
        )))
    assert [result.history_committed for result in results] == [True, False, True]
    assert results[0].agent_status == "max_api_calls"
    assert results[1].agent_status == "error"


def test_sequence_report_preserves_manifest_order(tmp_path) -> None:
    run_dir = tmp_path / "run"
    ids = ["second", "first"]
    (run_dir / "official").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "r", "sequence_id": "s", "dataset": "d", "model": "owner/model",
        "instance_ids": ids,
    }), encoding="utf-8")
    (run_dir / "official" / "owner__model.r.json").write_text(json.dumps({
        "resolved_ids": ["first"], "unresolved_ids": ["second"]
    }), encoding="utf-8")
    for instance_id in ids:
        result_dir = run_dir / "instances" / instance_id / "attempt-1"
        result_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text(json.dumps({
            "agent_status": "completed", "patch_status": "produced"
        }), encoding="utf-8")
    report = core.generate_sequence_report(run_dir)
    assert [row["instance_id"] for row in report["instances"]] == ids
    assert [row["sequence_position"] for row in report["instances"]] == [1, 2]


def test_cli_has_no_rerun_failed_option() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate", "--run-id", "r", "--subset", "s.json"])
    assert not hasattr(args, "rerun_failed")
    with pytest.raises(SystemExit):
        parser.parse_args([
            "generate", "--run-id", "r", "--subset", "s.json", "--rerun-failed"
        ])
