"""Stdlib-only JSON bridge for file operations inside solve containers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from file_engine import FileEngine
from grep_engine import GrepRequest, search

ROOT = Path("/testbed")
BASELINE_PATH = Path("/tmp/mycodeagent-baseline")


def git_patch(base: str) -> str:
    if base == "BASELINE":
        baseline_path = BASELINE_PATH
        if not baseline_path.is_file():
            raise RuntimeError("container workspace baseline is missing")
        base = baseline_path.read_text(encoding="utf-8").strip()
    env = os.environ.copy()
    descriptor, index_path = tempfile.mkstemp(prefix="mycodeagent-index-")
    os.close(descriptor)
    Path(index_path).unlink()
    env["GIT_INDEX_FILE"] = index_path
    try:
        for argv in (["git", "read-tree", base], ["git", "add", "-A"]):
            result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
        result = subprocess.run(["git", "diff", "--cached", "--binary", "--full-index", base, "--"], cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        patch = result.stdout
        if patch:
            validate_patch(base, patch)
        return patch
    finally:
        Path(index_path).unlink(missing_ok=True)


def validate_patch(base: str, patch: str) -> None:
    """Check a generated patch against the exact image baseline index."""
    descriptor, index_path = tempfile.mkstemp(prefix="mycodeagent-check-index-")
    os.close(descriptor)
    Path(index_path).unlink()
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = index_path
    try:
        subprocess.run(
            ["git", "read-tree", base], cwd=ROOT, env=env,
            check=True, capture_output=True, text=True,
        )
        checked = subprocess.run(
            ["git", "apply", "--cached", "--check", "--whitespace=nowarn", "-"],
            cwd=ROOT, env=env, input=patch, capture_output=True, text=True,
        )
        if checked.returncode != 0:
            detail = checked.stderr.strip() or checked.stdout.strip()
            raise RuntimeError(f"patch does not apply to image baseline: {detail}")
    finally:
        Path(index_path).unlink(missing_ok=True)


def create_baseline(base_commit: str) -> str:
    """Snapshot the prepared image tree without changing refs or the real index."""
    descriptor, index_path = tempfile.mkstemp(prefix="mycodeagent-baseline-index-")
    os.close(descriptor)
    Path(index_path).unlink()
    env = os.environ.copy()
    env.update({
        "GIT_INDEX_FILE": index_path,
        "GIT_AUTHOR_NAME": "myCodeAgent",
        "GIT_AUTHOR_EMAIL": "eval@localhost",
        "GIT_COMMITTER_NAME": "myCodeAgent",
        "GIT_COMMITTER_EMAIL": "eval@localhost",
    })
    try:
        subprocess.run(["git", "read-tree", base_commit], cwd=ROOT, env=env, check=True)
        changed = subprocess.run(
            ["git", "diff", "--name-only", "-z", base_commit, "--"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout.split(b"\0")
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout.split(b"\0")
        paths = sorted({
            raw.decode("utf-8", "surrogateescape")
            for raw in [*changed, *untracked]
            if raw
        })
        if paths:
            for start in range(0, len(paths), 500):
                subprocess.run(
                    ["git", "add", "-A", "--", *paths[start:start + 500]],
                    cwd=ROOT, env=env, check=True,
                )
        tree = subprocess.run(
            ["git", "write-tree"], cwd=ROOT, env=env, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "commit-tree", tree, "-p", base_commit, "-m", "myCodeAgent image baseline"],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True,
        ).stdout.strip()
        BASELINE_PATH.write_text(commit, encoding="utf-8")
        return commit
    finally:
        Path(index_path).unlink(missing_ok=True)


def dispatch(operation: str, args: dict, *, root: Path = ROOT) -> str:
    files = FileEngine(root, restrict_to_root=True)
    if operation == "grep":
        return search(root, GrepRequest(**args), restrict_to_root=True).output
    if operation == "read": return files.read(**args)
    if operation == "write": return files.write(**args)
    if operation == "edit": return files.edit(**args)
    if operation == "glob": return files.glob(**args)
    if operation == "git_patch": return git_patch(args["base_commit"])
    if operation == "create_baseline": return create_baseline(args["base_commit"])
    raise ValueError(f"unknown operation: {operation}")


def main() -> int:
    payload = json.load(sys.stdin)
    try:
        output = dispatch(payload["operation"], payload.get("args", {}))
        json.dump({"output": output}, sys.stdout)
        return 0
    except Exception as exc:
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
