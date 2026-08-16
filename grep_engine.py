"""Portable grep implementation shared by local and container backends."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_DIRS = {
    ".git", ".svn", ".hg", ".venv", "node_modules", "__pycache__",
    ".pytest_cache", ".sessions", ".transcripts", ".ssh",
}
SENSITIVE_NAMES = {
    ".env", ".netrc", ".npmrc", ".pypirc", "credentials",
    "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa",
}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
MAX_CONTENT_LINE_CHARS = 500
CONTENT_LINE_TRUNCATION_MARKER = " [...line truncated]"


@dataclass(frozen=True)
class GrepRequest:
    pattern: str
    path: str = "."
    glob: str | None = None
    output_mode: str = "files_with_matches"
    ignore_case: bool = False
    line_number: bool = True
    head_limit: int = 250


@dataclass(frozen=True)
class GrepResult:
    output: str
    implementation: str


def _is_sensitive(path: Path) -> bool:
    lowered = tuple(part.lower() for part in path.parts)
    name = path.name.lower()
    return (
        name in SENSITIVE_NAMES
        or name.startswith(".env.")
        or path.suffix.lower() in SENSITIVE_SUFFIXES
        or any(part in {".sessions", ".transcripts", ".ssh"} for part in lowered)
        or ".git" in lowered
    )


def _resolve(root: Path, value: str, *, restrict_to_root: bool) -> Path:
    root = root.resolve()
    raw = Path(value).expanduser()
    target = (root / raw).resolve()
    if not target.is_relative_to(root) and (restrict_to_root or not raw.is_absolute()):
        raise ValueError(f"Path escapes workspace: {value}")
    return target


def _relative_name(root: Path, search_path: Path, file_path: Path) -> str:
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        if search_path.is_file():
            return file_path.name
        return file_path.relative_to(search_path).as_posix()


def _matches_glob(relative_path: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    normalized = pattern.replace("\\", "/")
    if "/" in normalized:
        return fnmatch.fnmatch(relative_path, normalized) or fnmatch.fnmatch(
            relative_path, f"**/{normalized}"
        )
    return fnmatch.fnmatch(Path(relative_path).name, normalized)


def _walk_files(search_path: Path, pattern: str | None) -> list[Path]:
    if search_path.is_file():
        return [] if _is_sensitive(search_path) else [search_path]
    files = _git_files(search_path, pattern)
    if files is not None:
        return files
    files = []
    for current, dirs, names in os.walk(search_path):
        current_path = Path(current)
        dirs[:] = sorted(
            directory for directory in dirs
            if directory not in EXCLUDED_DIRS
            and not _is_sensitive((current_path / directory).resolve())
        )
        for name in sorted(names):
            candidate = (current_path / name).resolve()
            if not candidate.is_relative_to(search_path):
                continue
            if _is_sensitive(candidate):
                continue
            relative = candidate.relative_to(search_path).as_posix()
            if _matches_glob(relative, pattern):
                files.append(candidate)
    return files


def _git_files(search_path: Path, pattern: str | None) -> list[Path] | None:
    """Use Git's ignore engine when the search directory belongs to a worktree."""
    try:
        top = subprocess.run(
            ["git", "-C", str(search_path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if top.returncode != 0:
            return None
        repo_root = Path(top.stdout.strip()).resolve()
        listed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True, timeout=10, check=False,
        )
        if listed.returncode != 0:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None

    files: list[Path] = []
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        candidate = (repo_root / raw.decode("utf-8", "surrogateescape")).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(search_path):
            continue
        if _is_sensitive(candidate):
            continue
        relative = candidate.relative_to(search_path).as_posix()
        if _matches_glob(relative, pattern):
            files.append(candidate)
    return sorted(files)


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(4096)
    except OSError:
        return True


def _limit(lines: list[str], limit: int) -> str:
    if limit and len(lines) > limit:
        hidden = len(lines) - limit
        return "\n".join([*lines[:limit], f"... ({hidden} more lines)"])
    return "\n".join(lines)


def _truncate_content_line(line: str) -> str:
    if len(line) <= MAX_CONTENT_LINE_CHARS:
        return line
    return line[:MAX_CONTENT_LINE_CHARS] + CONTENT_LINE_TRUNCATION_MARKER


def _python_search(root: Path, search_path: Path, request: GrepRequest) -> str:
    flags = re.IGNORECASE if request.ignore_case else 0
    try:
        expression = re.compile(request.pattern, flags)
    except re.error as exc:
        return f"Error: invalid grep pattern: {exc}"

    output: list[str] = []
    for file_path in _walk_files(search_path, request.glob):
        if _looks_binary(file_path):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        relative = _relative_name(root, search_path, file_path)
        matched_lines: list[tuple[int, str, int]] = []
        for number, line in enumerate(lines, start=1):
            matches = list(expression.finditer(line))
            if matches:
                matched_lines.append((number, line, len(matches)))
        if not matched_lines:
            continue
        if request.output_mode == "files_with_matches":
            output.append(relative)
        elif request.output_mode == "count_matches":
            output.append(f"{relative}:{sum(item[2] for item in matched_lines)}")
        else:
            for number, line, _count in matched_lines:
                prefix = f"{relative}:{number}:" if request.line_number else f"{relative}:"
                output.append(f"{prefix}{_truncate_content_line(line)}")

    return _limit(output, request.head_limit) if output else "No matches found."


def _rg_search(root: Path, search_path: Path, request: GrepRequest, rg_path: str) -> str:
    args = [rg_path, "--json", "--hidden", "--color=never"]
    if request.ignore_case:
        args.append("--ignore-case")
    if request.glob:
        args.extend(["--glob", request.glob])
    # ripgrep gives the last matching glob precedence, so mandatory sensitive
    # exclusions must remain after any caller-supplied include glob.
    args.extend([
        "--glob", "!.git", "--glob", "!.svn", "--glob", "!.hg",
        "--glob", "!.env", "--glob", "!.env.*", "--glob", "!.sessions/**",
        "--glob", "!.transcripts/**", "--glob", "!.ssh/**", "--glob", "!*.key",
        "--glob", "!*.pem", "--glob", "!*.p12", "--glob", "!*.pfx",
        "--glob", "!.netrc", "--glob", "!.npmrc", "--glob", "!.pypirc",
        "--glob", "!credentials", "--glob", "!id_dsa", "--glob", "!id_ecdsa",
        "--glob", "!id_ed25519", "--glob", "!id_rsa",
    ])
    args.extend(["--", request.pattern, str(search_path)])
    try:
        completed = subprocess.run(
            args, cwd=root, capture_output=True, text=True, timeout=20, check=False,
        )
    except subprocess.TimeoutExpired:
        return "Error: grep timed out after 20s. Try a more specific path or pattern."
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip()
        return f"Error: grep failed. {detail}".strip()

    files: dict[str, int] = {}
    content: list[str] = []
    for raw in completed.stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        raw_path = data.get("path", {}).get("text")
        if not isinstance(raw_path, str):
            continue
        file_path = Path(raw_path)
        if not file_path.is_absolute():
            file_path = root / file_path
        relative = _relative_name(root, search_path, file_path.resolve())
        files[relative] = files.get(relative, 0) + len(data.get("submatches", []))
        if request.output_mode == "content":
            line = str(data.get("lines", {}).get("text", "")).rstrip("\r\n")
            number = data.get("line_number")
            prefix = f"{relative}:{number}:" if request.line_number else f"{relative}:"
            content.append(f"{prefix}{_truncate_content_line(line)}")

    if request.output_mode == "content":
        lines = content
    elif request.output_mode == "count_matches":
        lines = [f"{name}:{count}" for name, count in sorted(files.items())]
    else:
        lines = sorted(files)
    return _limit(lines, request.head_limit) if lines else "No matches found."


def search(
    root: Path,
    request: GrepRequest,
    *,
    restrict_to_root: bool = False,
) -> GrepResult:
    try:
        search_path = _resolve(root, request.path, restrict_to_root=restrict_to_root)
    except ValueError as exc:
        return GrepResult(f"Error: {exc}", "validation")
    if not search_path.exists():
        return GrepResult(f"Error: Path not found: {request.path}", "validation")
    if _is_sensitive(search_path):
        return GrepResult(
            f"Error: Access to sensitive path is blocked: {request.path}", "validation"
        )
    rg_path = shutil.which("rg")
    if rg_path:
        return GrepResult(_rg_search(Path(root).resolve(), search_path, request, rg_path), "ripgrep")
    return GrepResult(_python_search(Path(root).resolve(), search_path, request), "python_fallback")


def _main() -> int:
    payload = json.load(sys.stdin)
    request = GrepRequest(**payload["request"])
    result = search(Path(payload["root"]), request, restrict_to_root=True)
    json.dump({"output": result.output, "implementation": result.implementation}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
