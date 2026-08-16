from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from grep_engine import GrepRequest, search
from sandbox import CommandResult, DockerFileBackend, DockerSandbox
from workspace import Workspace


def test_python_grep_fallback_supports_all_output_modes(monkeypatch, tmp_path) -> None:
    import grep_engine

    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)
    (tmp_path / "a.py").write_text("Needle needle\nnone\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "binary.py").write_bytes(b"needle\0hidden")

    files = search(tmp_path, GrepRequest("needle", glob="*.py", ignore_case=True))
    counts = search(tmp_path, GrepRequest(
        "needle", glob="*.py", ignore_case=True, output_mode="count_matches",
    ))
    content = search(tmp_path, GrepRequest(
        "needle", glob="*.py", ignore_case=True, output_mode="content",
    ))

    assert files.output == "a.py"
    assert counts.output == "a.py:2"
    assert content.output == "a.py:1:Needle needle"
    assert files.implementation == "python_fallback"


def test_python_grep_fallback_respects_gitignore(monkeypatch, tmp_path) -> None:
    import grep_engine

    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "kept.txt").write_text("needle\n", encoding="utf-8")

    assert search(tmp_path, GrepRequest("needle")).output == "kept.txt"


def test_grep_results_remain_workspace_relative(monkeypatch, tmp_path) -> None:
    import grep_engine

    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "target.py").write_text("needle\n", encoding="utf-8")

    result = search(tmp_path, GrepRequest("needle", path="pkg"))

    assert result.output == "pkg/target.py"


def test_python_grep_rejects_invalid_pattern_and_symlink_escape(monkeypatch, tmp_path) -> None:
    import grep_engine

    monkeypatch.setattr(grep_engine.shutil, "which", lambda _name: None)
    outside = tmp_path.parent / "outside-grep.txt"
    outside.write_text("needle\n", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)

    assert search(tmp_path, GrepRequest("["), restrict_to_root=True).output.startswith(
        "Error: invalid grep pattern"
    )
    assert search(tmp_path, GrepRequest("needle"), restrict_to_root=True).output == "No matches found."


def test_docker_file_backend_uses_json_bridge() -> None:
    class Runner:
        def run(self, argv, **kwargs):
            assert argv == ["/opt/miniconda3/bin/python", "/tmp/mycodeagent-grep-engine.py"]
            assert kwargs["cwd"] == "/testbed"
            payload = json.loads(kwargs["stdin"])
            assert payload["root"] == "/testbed"
            assert payload["request"]["pattern"] == "needle"
            return CommandResult(0, json.dumps({"output": "pkg/a.py"}), "")

    assert DockerFileBackend(Runner()).grep(GrepRequest("needle")) == "pkg/a.py"


def test_docker_sandbox_uses_network_none_and_always_removes(monkeypatch, tmp_path) -> None:
    import sandbox

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(argv, 0, stdout="container-id\n", stderr="")
        if argv[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps({"output": "No matches found."}), stderr="",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    runtime = DockerSandbox(Workspace(tmp_path), "example/image:latest").start()
    runtime.close()

    run = next(call for call in calls if call[:2] == ["docker", "run"])
    assert calls[0][:2] == ["docker", "ps"]
    assert ["--network", "none"] == run[run.index("--network"):run.index("--network") + 2]
    assert ["--user", f"{os.getuid()}:{os.getgid()}"] == run[
        run.index("--user"):run.index("--user") + 2
    ]
    assert f"type=bind,src={tmp_path},dst=/testbed" in run
    assert any(item.startswith("mycodeagent.workspace=") for item in run)
    assert any(call[:2] == ["docker", "cp"] for call in calls)
    assert calls[-1] == ["docker", "rm", "-f", "container-id"]


def test_docker_command_timeout_is_normalized(monkeypatch) -> None:
    import sandbox

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["docker", "exec"], 3)

    monkeypatch.setattr(sandbox.subprocess, "run", timeout)

    result = sandbox.DockerCommandRunner("container-id").run(["true"], timeout=3)

    assert result.exit_code == 124
    assert result.stderr == "container command timed out after 3s"
