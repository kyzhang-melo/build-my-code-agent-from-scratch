"""Session-scoped local and Docker execution backends."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from grep_engine import GrepRequest, search
from workspace import Workspace


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandRunner(Protocol):
    @property
    def execution_location(self) -> Literal["local", "remote"]: ...

    @property
    def default_cwd(self) -> str: ...

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        stdin: str | None = None,
        timeout: float = 20,
    ) -> CommandResult: ...


class FileBackend(Protocol):
    @property
    def execution_location(self) -> Literal["local", "remote"]: ...

    def grep(self, request: GrepRequest) -> str: ...


class RemoteFileBackend(FileBackend, Protocol):
    def call(self, operation: str, args: dict) -> str: ...


class Sandbox(Protocol):
    @property
    def workspace_root(self) -> str: ...

    @property
    def command_runner(self) -> CommandRunner: ...

    @property
    def file_backend(self) -> FileBackend: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LocalCommandRunner:
    workspace: Workspace

    @property
    def execution_location(self) -> Literal["local"]:
        return "local"

    @property
    def default_cwd(self) -> str:
        return str(self.workspace.root)

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

    @property
    def execution_location(self) -> Literal["local"]:
        return "local"

    def grep(self, request: GrepRequest) -> str:
        return search(self.workspace.root, request).output


@dataclass(frozen=True)
class LocalSandbox:
    workspace: Workspace

    @property
    def workspace_root(self) -> str:
        return str(self.workspace.root)

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

    @property
    def execution_location(self) -> Literal["remote"]:
        return "remote"

    @property
    def default_cwd(self) -> str:
        return "/testbed"

    def run(self, argv, *, cwd=None, stdin=None, timeout=20) -> CommandResult:
        token = uuid.uuid4().hex
        pid_path = f"/tmp/mycodeagent-command-{token}.pid"
        launcher = (
            "import os,sys\n"
            "p=sys.argv[1]; a=sys.argv[2:]\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " os.setsid()\n"
            " open(p,'w').write(str(os.getpid()))\n"
            " os.execvp(a[0],a)\n"
            "_,status=os.waitpid(pid,0)\n"
            "try: os.unlink(p)\n"
            "except FileNotFoundError: pass\n"
            "os._exit(os.waitstatus_to_exitcode(status))\n"
        )
        command = ["docker", "exec", "-i"]
        if cwd:
            command.extend(["--workdir", cwd])
        command.extend([
            self.container_id,
            "/opt/miniconda3/bin/python", "-c", launcher, pid_path, *argv,
        ])
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
            self._terminate_process_group(pid_path)
            return CommandResult(
                124, "", f"[Command killed by timeout ({timeout}s)]", timed_out=True,
            )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def _terminate_process_group(self, pid_path: str) -> None:
        terminator = (
            "import os,signal,sys,time\n"
            "p=sys.argv[1]; pid=int(open(p).read())\n"
            "try: os.killpg(pid,signal.SIGTERM)\n"
            "except ProcessLookupError: pass\n"
            "time.sleep(0.5)\n"
            "try: os.killpg(pid,signal.SIGKILL)\n"
            "except ProcessLookupError: pass\n"
            "try: os.unlink(p)\n"
            "except FileNotFoundError: pass\n"
        )
        try:
            subprocess.run(
                [
                    "docker", "exec", self.container_id,
                    "/opt/miniconda3/bin/python", "-c", terminator, pid_path,
                ],
                capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass


@dataclass(frozen=True)
class DockerFileBackend:
    # SWE-bench instance images conventionally provide both this interpreter
    # and the prepared repository at /testbed; this is not a generic Docker API.
    runner: DockerCommandRunner
    bridge_path: str = "/tmp/mycodeagent-file-bridge.py"
    root: str = "/testbed"

    @property
    def execution_location(self) -> Literal["remote"]:
        return "remote"

    def call(self, operation: str, args: dict) -> str:
        payload = json.dumps({"operation": operation, "args": args})
        result = self.runner.run(
            ["/opt/miniconda3/bin/python", self.bridge_path],
            cwd=self.root,
            stdin=payload,
            timeout=25,
        )
        if result.exit_code != 0:
            try:
                detail = json.loads(result.stdout).get("error", "")
            except (json.JSONDecodeError, TypeError):
                detail = result.stderr.strip() or result.stdout.strip()
            return f"Error: container {operation} failed. {detail}".strip()
        try:
            return str(json.loads(result.stdout)["output"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return f"Error: container {operation} returned an invalid response: {exc}"

    def grep(self, request: GrepRequest) -> str:
        return self.call("grep", request.__dict__)


class DockerSandbox:
    """One solve container using the instance image's prepared /testbed."""

    def __init__(self, image: str, sandbox_id: str, base_commit: str, *, platform: str = "linux/x86_64", run_id: str | None = None):
        self.image = image
        self.sandbox_id = sandbox_id
        self.base_commit = base_commit
        self.platform = platform
        self.run_id = run_id
        self.container_id: str | None = None

    @property
    def workspace_root(self) -> str:
        return "/testbed"

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
        self._remove_orphans(self.sandbox_id)
        completed = subprocess.run(
            [
                "docker", "run", "--detach", "--network", "none",
                "--platform", self.platform, "--name", name,
                "--label", "mycodeagent.role=swebench-solve",
                "--label", f"mycodeagent.sandbox={self.sandbox_id}",
                *(["--label", f"mycodeagent.run={self.run_id}"] if self.run_id else []),
                "--workdir", "/testbed", self.image, "tail", "-f", "/dev/null",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "failed to start solve container")
        self.container_id = completed.stdout.strip()
        for source, target in (
            (Path(__file__).with_name("file_bridge.py"), "/tmp/mycodeagent-file-bridge.py"),
            (Path(__file__).with_name("file_engine.py"), "/tmp/file_engine.py"),
            (Path(__file__).with_name("grep_engine.py"), "/tmp/grep_engine.py"),
        ):
            copied = subprocess.run(
                ["docker", "cp", str(source), f"{self.container_id}:{target}"],
                capture_output=True, text=True, check=False,
            )
            if copied.returncode != 0:
                self.close()
                raise RuntimeError(copied.stderr.strip() or "failed to install file bridge")
        self._verify_testbed()
        check = self.file_backend.grep(
            GrepRequest(pattern="__mycodeagent_bridge_self_check_7f4d2b9a__", path=".")
        )
        if check.startswith("Error:"):
            self.close()
            raise RuntimeError(f"grep bridge self-check failed: {check}")
        return self

    def _verify_testbed(self) -> None:
        commands = (
            ["test", "-w", "/testbed"],
            ["git", "cat-file", "-e", f"{self.base_commit}^{{commit}}"],
        )
        for command in commands:
            result = self.command_runner.run(command, cwd="/testbed")
            if result.exit_code != 0:
                self.close()
                raise RuntimeError(f"image /testbed validation failed: {' '.join(command)}")
        baseline = self.file_backend.call(
            "create_baseline", {"base_commit": self.base_commit},
        )
        if baseline.startswith("Error:"):
            self.close()
            raise RuntimeError(f"image /testbed baseline failed: {baseline}")

    @staticmethod
    def _remove_orphans(sandbox_id: str) -> None:
        listed = subprocess.run(
            [
                "docker", "ps", "--all", "--quiet", "--filter",
                f"label=mycodeagent.sandbox={sandbox_id}",
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

    @staticmethod
    def remove_run_orphans(run_id: str) -> None:
        """Remove solve containers left by an interrupted invocation of one run."""
        listed = subprocess.run(
            [
                "docker", "ps", "--all", "--quiet",
                "--filter", "label=mycodeagent.role=swebench-solve",
                "--filter", f"label=mycodeagent.run={run_id}",
            ],
            capture_output=True, text=True, check=False,
        )
        for container_id in listed.stdout.split():
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True, text=True, check=False,
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

    def export_patch(self) -> str:
        output = self.file_backend.call("git_patch", {"base_commit": "BASELINE"})
        if output.startswith("Error:"):
            raise RuntimeError(output)
        return output

    def __enter__(self) -> "DockerSandbox":
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
