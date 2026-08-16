"""Session-scoped local and Docker execution backends."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from grep_engine import GrepRequest, search
from workspace import Workspace


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        stdin: str | None = None,
        timeout: float = 20,
    ) -> CommandResult: ...


class FileBackend(Protocol):
    def grep(self, request: GrepRequest) -> str: ...


class Sandbox(Protocol):
    @property
    def command_runner(self) -> CommandRunner: ...

    @property
    def file_backend(self) -> FileBackend: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LocalCommandRunner:
    workspace: Workspace

    def run(self, argv, *, cwd=None, stdin=None, timeout=20) -> CommandResult:
        completed = subprocess.run(
            argv,
            cwd=cwd or self.workspace.root,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class LocalFileBackend:
    workspace: Workspace

    def grep(self, request: GrepRequest) -> str:
        return search(self.workspace.root, request).output


@dataclass(frozen=True)
class LocalSandbox:
    workspace: Workspace

    @property
    def command_runner(self) -> LocalCommandRunner:
        return LocalCommandRunner(self.workspace)

    @property
    def file_backend(self) -> LocalFileBackend:
        return LocalFileBackend(self.workspace)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class DockerCommandRunner:
    container_id: str

    def run(self, argv, *, cwd=None, stdin=None, timeout=20) -> CommandResult:
        command = ["docker", "exec", "-i"]
        if cwd:
            command.extend(["--workdir", cwd])
        command.extend([self.container_id, *argv])
        try:
            completed = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", f"container command timed out after {timeout}s")
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class DockerFileBackend:
    runner: DockerCommandRunner
    bridge_path: str = "/tmp/mycodeagent-grep-engine.py"
    root: str = "/testbed"

    def grep(self, request: GrepRequest) -> str:
        payload = json.dumps({"root": self.root, "request": request.__dict__})
        result = self.runner.run(
            ["/opt/miniconda3/bin/python", self.bridge_path],
            cwd=self.root,
            stdin=payload,
            timeout=25,
        )
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            return f"Error: container grep failed. {detail}".strip()
        try:
            return str(json.loads(result.stdout)["output"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return f"Error: container grep returned an invalid response: {exc}"


class DockerSandbox:
    """One solve container sharing an attempt workspace at /testbed."""

    def __init__(self, workspace: Workspace, image: str, *, platform: str = "linux/x86_64"):
        self.workspace = workspace
        self.image = image
        self.platform = platform
        self.container_id: str | None = None

    @property
    def command_runner(self) -> DockerCommandRunner:
        if self.container_id is None:
            raise RuntimeError("Docker sandbox is not started")
        return DockerCommandRunner(self.container_id)

    @property
    def file_backend(self) -> DockerFileBackend:
        return DockerFileBackend(self.command_runner)

    def start(self) -> "DockerSandbox":
        name = f"mycodeagent-swe-{uuid.uuid4().hex[:12]}"
        workspace_label = hashlib.sha256(
            str(self.workspace.root).encode("utf-8")
        ).hexdigest()[:16]
        self._remove_orphans(workspace_label)
        completed = subprocess.run(
            [
                "docker", "run", "--detach", "--network", "none",
                "--platform", self.platform, "--name", name,
                "--label", "mycodeagent.role=swebench-solve",
                "--label", f"mycodeagent.workspace={workspace_label}",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--mount", f"type=bind,src={self.workspace.root},dst=/testbed",
                "--workdir", "/testbed", self.image, "tail", "-f", "/dev/null",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "failed to start solve container")
        self.container_id = completed.stdout.strip()
        bridge = Path(__file__).with_name("grep_engine.py")
        copied = subprocess.run(
            ["docker", "cp", str(bridge), f"{self.container_id}:/tmp/mycodeagent-grep-engine.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if copied.returncode != 0:
            self.close()
            raise RuntimeError(copied.stderr.strip() or "failed to install grep bridge")
        check = self.file_backend.grep(
            GrepRequest(pattern="__mycodeagent_bridge_self_check_7f4d2b9a__", path=".")
        )
        if check.startswith("Error:"):
            self.close()
            raise RuntimeError(f"grep bridge self-check failed: {check}")
        return self

    @staticmethod
    def _remove_orphans(workspace_label: str) -> None:
        listed = subprocess.run(
            [
                "docker", "ps", "--all", "--quiet", "--filter",
                f"label=mycodeagent.workspace={workspace_label}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for container_id in listed.stdout.split():
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                check=False,
            )

    def close(self) -> None:
        container_id = self.container_id
        self.container_id = None
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                check=False,
            )

    def __enter__(self) -> "DockerSandbox":
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
