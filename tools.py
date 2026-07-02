import asyncio
import fnmatch
import json
import os
import shutil
import subprocess
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


WORKDIR = Path.cwd()
TOOL_OUTPUT_PREVIEW_CHARS = 500
READ_ONLY_TOOL_NAMES = {"read_file", "glob", "grep"}
# Noisy/sensitive directories pruned from glob traversal and results, keeping
# `glob` consistent with `run_grep`'s excludes (`.git/.svn/.hg`).
EXCLUDE_DIRS = {
    ".git",
    ".svn",
    ".hg",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".claude",
}

# Budget for a single tool output handed back to the model. Char-based (chars
# ~= 4x tokens); a token-aware policy can replace it later without touching callers.
TOOL_OUTPUT_MAX_CHARS = 48000
TOOL_OUTPUT_MAX_LINE_CHARS = 2000
DEFAULT_MAX_PARALLEL_TOOL_CALLS = 4

# read_file bounds. The reader self-bounds and is exempt from truncate_middle
# (see run_tool_call), so these are the sole authority over read output size.
MAX_READ_LINES = 1000              # max lines returned; also the `limit` default and hard max
MAX_LINE_CHARS = TOOL_OUTPUT_MAX_LINE_CHARS  # per-line cap (reuse the existing constant)
MAX_READ_BYTES = 40_000            # cap on returned content bytes; stays under TOOL_OUTPUT_MAX_CHARS
MAX_READ_FILE_BYTES = 5 * 1024 * 1024  # st_size pre-check: stream to EOF for an exact total
                                        # only below this; larger files skip the full scan


def _resolve_max_parallel_tool_calls() -> int:
    raw = os.getenv("MAX_PARALLEL_TOOL_CALLS", str(DEFAULT_MAX_PARALLEL_TOOL_CALLS))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_PARALLEL_TOOL_CALLS
    return max(1, value)


MAX_PARALLEL_TOOL_CALLS = _resolve_max_parallel_tool_calls()


def truncate_middle(
    text: str,
    *,
    max_chars: int = TOOL_OUTPUT_MAX_CHARS,
    max_line_chars: int = TOOL_OUTPUT_MAX_LINE_CHARS,
) -> str:
    """Bound a single tool output, preserving both ends.

    Two defenses, applied in order:
      1. Per-line cap: clip any line longer than `max_line_chars` so one minified
         file or base64 blob can't eat the whole budget.
      2. Middle-truncation: if still over `max_chars`, keep the head and the tail
         and elide the middle. The tail is kept on purpose -- command errors
         (stderr is appended last) and end-of-file content live there, which the
         old head-only `[:50000]` cap silently dropped.

    No-op when the text already fits.
    """
    if not text:
        return text

    if max_line_chars and max_line_chars > 0:
        capped: list[str] = []
        clipped_any = False
        for line in text.split("\n"):
            if len(line) > max_line_chars:
                line = line[:max_line_chars] + " [...line truncated]"
                clipped_any = True
            capped.append(line)
        if clipped_any:
            text = "\n".join(capped)

    if len(text) <= max_chars:
        return text

    total_lines = text.count("\n") + 1
    head_chars = max_chars * 6 // 10
    tail_chars = max_chars - head_chars
    elided = len(text) - head_chars - tail_chars
    marker = f"\n\n[... {elided} chars elided from the middle ...]\n\n"
    return f"Total output lines: {total_lines}\n\n{text[:head_chars]}{marker}{text[-tail_chars:]}"


TOOLS = [
    {
        "type": "function",
        "name": "bash",
        "description": "Run a shell command in the current workspace.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read text file contents from the workspace. Output is line-numbered "
            "(line_number<TAB>content) and ends with a summary (total lines, and a "
            "'use offset=N to continue' hint when the file is longer than what was "
            "returned). To page through a large file, pass offset/limit instead of "
            "re-reading from the top."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from. Defaults to 1.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Number of lines to read (max 1000). Defaults to 1000. Pair with "
                        "offset to page through large files."
                    ),
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file in workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": "Replace exact text in a workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "glob",
        "description": (
            "Find files or directories in the workspace by glob pattern. "
            "Recursive patterns like '**/config.json' are supported and search the "
            "whole tree (noisy dirs such as .venv/node_modules are skipped); avoid "
            "the bare '**/*' which is too broad. "
            "When the user names a search directory, pass it as directory instead of "
            "prefixing it into pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern relative to directory, e.g. '*.py' or '**/*.py'. "
                        "Do not include the directory prefix here."
                    ),
                },
                "directory": {
                    "type": "string",
                    "description": (
                        "Directory to search within, relative to the workspace or absolute "
                        "inside the workspace. Defaults to the workspace."
                    ),
                },
                "include_dirs": {
                    "type": "boolean",
                    "description": "Whether directory matches should be included in results.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return. Defaults to 1000.",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "grep",
        "description": (
            "Search workspace file contents with ripgrep. "
            "Put the search location in path, and use glob only to filter file names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for in file contents.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search, relative to the workspace or absolute "
                        "inside the workspace. Defaults to the workspace."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": "Optional file filter such as '*.py' or 'tests/*.py'.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count_matches"],
                    "description": (
                        "files_with_matches returns paths, content returns matching lines, "
                        "count_matches returns per-file match counts."
                    ),
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Whether matching should ignore case.",
                },
                "line_number": {
                    "type": "boolean",
                    "description": "Whether content mode should include line numbers.",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Maximum number of output lines to return. Defaults to 250.",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "todo",
        "description": "Rewrite the current session plan for multi-step work.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label.",
                            },
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    },
]

TASK_TOOL = {
    "type": "function",
    "name": "task",
    "description": (
        "Spawn a read-only exploration subagent with fresh context. It can inspect "
        "the workspace with glob, grep, and read_file, then returns a summary of findings.\n\n"
        "When to use:\n"
        "- Your task will clearly require more than 3 search queries\n"
        "- You need to understand how a module, feature, or code path works\n"
        "- You need to investigate multiple files and patterns\n"
        "- You want to gather context before planning or making changes\n\n"
        "When NOT to use:\n"
        "- Reading a known file path\n"
        "- Searching a small number of known files\n"
        "- Tasks completable in 1-2 direct tool calls\n"
        "- You already have the information you need"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "description": {
                "type": "string",
                "description": "Short description of the delegated exploration task.",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}

TOOLS.append(TASK_TOOL)


class PlanItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    content: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "completed"] = "pending"
    active_form: str = Field(default="", alias="activeForm")

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value) -> str:
        return str(value).strip()

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value) -> str:
        return str(value).strip().lower()

    @field_validator("active_form", mode="before")
    @classmethod
    def normalize_active_form(cls, value) -> str:
        return str(value).strip()


class TodoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PlanItem] = Field(max_length=12)

    @model_validator(mode="after")
    def validate_single_in_progress(self):
        in_progress_count = sum(1 for item in self.items if item.status == "in_progress")
        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")
        return self


class PlanningState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PlanItem] = Field(default_factory=list)


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, params: TodoParams) -> str:
        self.state = PlanningState(items=params.items)
        return self.render_with_reminder()

    def render_with_reminder(self) -> str:
        # Event-driven echo: every write returns the latest list to the model,
        # so it stays aware of its plan without a round-based nudge.
        if not self.state.items:
            return (
                "Todo list cleared.\n\n"
                "<system-reminder>\n"
                "Your todo list is now empty. You have no tracked tasks.\n"
                "</system-reminder>"
            )
        if self.all_items_completed():
            # When the plan is finished, stop nudging for more todo calls --
            # that reflex makes the model emit a gratuitous trailing todo call
            # right after its deliverable, which the loop then has to surface
            # specially. Steer it to deliver the result instead.
            return (
                "Todos updated. All tracked items are complete.\n\n"
                "<system-reminder>\n"
                "Your todo list is now fully complete:\n\n"
                f"{self.render()}\n\n"
                "Provide the final result the user asked for in your next message. "
                "Do not call the todo tool again unless the plan changes.\n"
                "</system-reminder>"
            )
        return (
            "Todos updated. Keep using the todo tool to track your progress.\n\n"
            "<system-reminder>\n"
            "Your todo list has changed. Here are the latest contents of your todo list:\n\n"
            f"{self.render()}\n\n"
            "Continue on with the tasks at hand if applicable.\n"
            "</system-reminder>"
        )

    def has_active_plan(self) -> bool:
        return len(self.state.items) > 0

    def all_items_completed(self) -> bool:
        return self.has_active_plan() and all(item.status == "completed" for item in self.state.items)

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."

        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)

        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


class BashParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    offset: StrictInt = Field(default=1, ge=1)
    limit: StrictInt = Field(default=MAX_READ_LINES, ge=1, le=MAX_READ_LINES)


class WriteFileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    content: str


class EditFileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str


class GlobParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1)
    directory: str | None = None
    include_dirs: bool = True
    limit: StrictInt = Field(default=1000, ge=1)


class GrepParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1)
    path: str = "."
    glob: str | None = None
    output_mode: Literal["content", "files_with_matches", "count_matches"] = "files_with_matches"
    ignore_case: bool = False
    line_number: bool = True
    head_limit: StrictInt = Field(default=250, ge=0)


class TaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    description: str = "exploration"


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: dangerous command!"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)."
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}."
    output = (result.stdout + result.stderr).strip()
    return output if output else "(no output)"


def safe_path(path: str) -> Path:
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


# Common non-text extensions, rejected early with a clear message. The null-byte
# sniff below catches the rest; this list just avoids opening obvious binaries.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".pdf",
    ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class", ".jar",
    ".pyc", ".pyo", ".wasm", ".node",
    ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".sqlite", ".sqlite3", ".db", ".woff", ".woff2", ".ttf", ".otf",
}


def _looks_binary(fp: Path) -> bool:
    """True for obvious non-text files: known binary extension or a NUL byte in the head."""
    if fp.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(fp, "rb") as handle:
            return b"\x00" in handle.read(4096)
    except OSError:
        return False


def run_read(path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
    """Read a slice of a text file, streaming and bounded.

    Returns `cat -n`-style numbered lines for the window [offset, offset+limit)
    plus a trailing summary (total lines, end-of-file/cap notes, and a forward
    `offset=` hint when more remains) so the model can page instead of re-reading.
    """
    try:
        fp = safe_path(path)
    except Exception as e:
        return f"Error: {e}"
    if not fp.exists():
        return f"Error: File not found: {path}"
    if not fp.is_file():
        return f"Error: Not a file: {path}"
    if _looks_binary(fp):
        return (
            f"Error: '{path}' appears to be a binary or non-text file. "
            "Use appropriate tools to inspect it."
        )

    try:
        # st_size pre-check: only stream all the way to EOF (for an exact total
        # line count) when the file is small enough that the scan is cheap.
        count_to_eof = fp.stat().st_size <= MAX_READ_FILE_BYTES

        rendered: list[str] = []
        rendered_bytes = 0
        truncated_lines: list[int] = []
        total_lines = 0
        last_line = offset - 1
        max_lines_reached = False
        max_bytes_reached = False
        reached_eof = True

        with open(fp, encoding="utf-8", errors="replace") as handle:
            for i, raw in enumerate(handle, start=1):
                total_lines = i
                if i < offset:
                    continue
                line_limit_reached = len(rendered) >= limit
                byte_limit_reached = rendered_bytes >= MAX_READ_BYTES
                if line_limit_reached or byte_limit_reached:
                    max_lines_reached = max_lines_reached or line_limit_reached
                    max_bytes_reached = max_bytes_reached or byte_limit_reached
                    if count_to_eof:
                        continue  # window full; keep counting only
                    reached_eof = False
                    break
                content = raw.rstrip("\n")
                if len(content) > MAX_LINE_CHARS:
                    content = content[:MAX_LINE_CHARS] + " [...line truncated]"
                    truncated_lines.append(i)
                line = f"{i}\t{content}"
                rendered.append(line)
                rendered_bytes += len(line)
                last_line = i
    except Exception as e:
        return f"Error: {e}"

    if not rendered:
        if total_lines == 0:
            return "<system-reminder>File exists but is empty.</system-reminder>"
        return (
            f"<system-reminder>File has {total_lines} lines; offset {offset} "
            "is past the end of the file.</system-reminder>"
        )

    body = "\n".join(rendered)
    footer = _read_footer(
        offset=offset,
        last_line=last_line,
        n=len(rendered),
        limit=limit,
        total_lines=total_lines,
        reached_eof=reached_eof,
        is_large=not count_to_eof,
        max_lines_reached=max_lines_reached,
        max_bytes_reached=max_bytes_reached,
        truncated_lines=truncated_lines,
    )
    return f"{body}\n\n{footer}"


def _read_footer(
    *,
    offset: int,
    last_line: int,
    n: int,
    limit: int,
    total_lines: int,
    reached_eof: bool,
    is_large: bool,
    max_lines_reached: bool,
    max_bytes_reached: bool,
    truncated_lines: list[int],
) -> str:
    head = f"Read {n} lines (lines {offset}-{last_line})."

    if reached_eof:
        total_part = f" Total lines: {total_lines}."
    else:
        total_part = f" Total lines: {total_lines}+ (not fully counted)."

    notes: list[str] = []
    if max_lines_reached:
        notes.append(f"Stopped at the {limit}-line limit")
    elif max_bytes_reached:
        notes.append(f"Stopped at the {MAX_READ_BYTES}-byte limit")
    elif reached_eof and 0 < n < limit:
        notes.append("End of file")
    if truncated_lines:
        notes.append(f"lines {truncated_lines} truncated to {MAX_LINE_CHARS} chars")
    if is_large:
        notes.append("large file -- use grep or a targeted offset/limit to narrow the read")
    note_part = (" " + "; ".join(notes) + ".") if notes else ""

    more_remains = max_lines_reached or max_bytes_reached or not reached_eof
    hint = f" Use offset={last_line + 1} to continue." if more_remains else ""

    return f"[{head}{total_part}{note_part}{hint}]"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def _limit_lines(lines: list[str], limit: int) -> tuple[list[str], str]:
    if limit and len(lines) > limit:
        return lines[:limit], f"\n... ({len(lines) - limit} more lines)"
    return lines, ""


# Match-everything patterns: too broad to be useful, answered with a listing.
BROAD_GLOB_PATTERNS = {"**", "**/", "**/*"}


def _glob_excluded(rel: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in rel.parts)


def _glob_listing(pattern: str, base: Path) -> str:
    entries = [
        f"{child.name}/" if child.is_dir() else child.name
        for child in sorted(base.iterdir())
        if child.name not in EXCLUDE_DIRS
    ]
    body = "\n".join(entries) if entries else "(empty)"
    return (
        f"Error: pattern `{pattern}` matches everything and is too broad. "
        "Anchor the search to a subdirectory (e.g. `subdir/**/name`). "
        f"Top-level entries of `{base}` (directories marked with `/`):\n{body}"
    )


def _glob_walk(base: Path, tail: str, include_dirs: bool) -> list[Path]:
    """Recursive `**/<tail>` search that prunes noisy dirs during traversal.

    Matches the basename against `tail` (not the relative path) so `fnmatch`'s
    `*`-spans-`/` behavior cannot leak, and depth-0 entries under `base` match.
    """
    matches: list[Path] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        names = files + dirs if include_dirs else files
        for name in names:
            if fnmatch.fnmatch(name, tail):
                matches.append(Path(root) / name)
    return matches


def _format_glob_matches(pattern: str, base: Path, matches: list[Path], limit: int) -> str:
    matches = [path for path in matches if not _glob_excluded(path.relative_to(base))]
    matches.sort()
    total = len(matches)
    if total == 0:
        return f"No matches found for pattern `{pattern}`."

    lines = [str(path.relative_to(base)) for path in matches[:limit]]
    message = f"Found {total} matches for pattern `{pattern}`."
    if total > limit:
        message += f" Showing first {limit}."
    return "\n".join([message, *lines])


def run_glob(
    pattern: str,
    directory: str | None = None,
    include_dirs: bool = True,
    limit: int = 1000,
) -> str:
    try:
        base = safe_path(directory or ".")
        if not base.exists():
            return f"Error: Directory not found: {directory or '.'}"
        if not base.is_dir():
            return f"Error: Not a directory: {directory or '.'}"

        if pattern in BROAD_GLOB_PATTERNS:
            return _glob_listing(pattern, base)

        if pattern.startswith("**/") and "/" not in pattern[3:]:
            # Precise recursive search: fast, pruned os.walk over the tree.
            matches = _glob_walk(base, pattern[3:], include_dirs)
        else:
            # Non-recursive or exotic `**` shapes: pathlib handles the pattern;
            # noisy dirs are dropped from the results by the formatter.
            matches = list(base.glob(pattern))
            if not include_dirs:
                matches = [path for path in matches if path.is_file()]

        return _format_glob_matches(pattern, base, matches, limit)
    except Exception as e:
        return f"Error: {e}"


def _strip_workdir_prefix(output: str) -> str:
    prefix = str(WORKDIR) + "/"
    return "\n".join(
        line[len(prefix) :] if line.startswith(prefix) else line
        for line in output.splitlines()
    )


def run_grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    output_mode: str = "files_with_matches",
    ignore_case: bool = False,
    line_number: bool = True,
    head_limit: int = 250,
) -> str:
    rg_path = shutil.which("rg")
    if not rg_path:
        return "Error: ripgrep (`rg`) is not installed. Install it with `brew install ripgrep`."

    try:
        search_path = safe_path(path)
        if not search_path.exists():
            return f"Error: Path not found: {path}"

        args = [
            rg_path,
            "--hidden",
            "--max-columns",
            "500",
            "--glob",
            "!.git",
            "--glob",
            "!.svn",
            "--glob",
            "!.hg",
            "--glob",
            "!.env",
            "--glob",
            "!.env.*",
        ]
        if ignore_case:
            args.append("--ignore-case")
        if glob:
            args.extend(["--glob", glob])
        if output_mode == "files_with_matches":
            args.append("--files-with-matches")
        elif output_mode == "count_matches":
            args.append("--count-matches")
        elif output_mode == "content" and line_number:
            args.append("--line-number")

        args.extend(["--", pattern, str(search_path)])

        result = subprocess.run(
            args,
            shell=False,
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return "Error: grep timed out after 20s. Try a more specific path or pattern."
    except Exception as e:
        return f"Error: {e}"

    output = _strip_workdir_prefix(result.stdout.strip())
    stderr = result.stderr.strip()
    if result.returncode == 1:
        return "No matches found."
    if result.returncode not in (0, 1):
        return f"Error: grep failed. {stderr}".strip()

    lines = output.splitlines()
    lines, suffix = _limit_lines(lines, head_limit)
    output = "\n".join(lines) + suffix
    return output if output else "No matches found."


@dataclass
class ToolRuntimeSpec:
    name: str
    params_model: type[BaseModel]
    sanitize_args: Callable[[dict], dict]
    execute: Callable[[BaseModel], Awaitable[str]]
    # True when the tool only reads state and is safe to run concurrently with
    # other tool calls in the same turn. Consumed by the threaded executor; the
    # read-only tools and the explore-only `task` subagent qualify, while tools
    # that mutate the workspace or the global TODO do not.
    concurrency_safe: bool = False


def async_tool(fn: Callable[[BaseModel], str]) -> Callable[[BaseModel], Awaitable[str]]:
    async def wrapper(params: BaseModel) -> str:
        return await asyncio.to_thread(fn, params)

    return wrapper


def sanitize_common_string(value: str) -> str:
    cleaned = value.lstrip()
    while cleaned and cleaned[0] in {">", "$", "#"}:
        cleaned = cleaned[1:].lstrip()
    return cleaned


def sanitize_bash_args(args: dict) -> dict:
    clean = dict(args)
    command = clean.get("command")
    if isinstance(command, str):
        clean["command"] = sanitize_common_string(command)
    return clean


def sanitize_file_args(args: dict) -> dict:
    clean = dict(args)
    path = clean.get("path")
    if isinstance(path, str):
        clean["path"] = sanitize_common_string(path)
    return clean


def sanitize_search_args(args: dict) -> dict:
    clean = dict(args)
    for key in ("pattern", "path", "directory"):
        value = clean.get(key)
        if isinstance(value, str):
            clean[key] = sanitize_common_string(value)
    return clean


def sanitize_passthrough(args: dict) -> dict:
    return dict(args)


def sanitize_task_args(args: dict) -> dict:
    clean = dict(args)
    for key in ("prompt", "description"):
        value = clean.get(key)
        if isinstance(value, str):
            clean[key] = sanitize_common_string(value)
    return clean


async def unavailable_task_runner(prompt: str, description: str) -> str:
    return "Error: task runner is not configured."


def build_tool_registry(
    tool_names: set[str] | None = None,
    *,
    task_runner: Callable[[str, str], Awaitable[str]] | None = None,
) -> dict[str, ToolRuntimeSpec]:
    runner = task_runner or unavailable_task_runner
    registry = {
        "bash": ToolRuntimeSpec(
            name="bash",
            params_model=BashParams,
            sanitize_args=sanitize_bash_args,
            execute=async_tool(lambda params: run_bash(params.command)),
        ),
        "read_file": ToolRuntimeSpec(
            name="read_file",
            params_model=ReadFileParams,
            sanitize_args=sanitize_file_args,
            execute=async_tool(lambda params: run_read(params.path, params.offset, params.limit)),
            concurrency_safe=True,
        ),
        "write_file": ToolRuntimeSpec(
            name="write_file",
            params_model=WriteFileParams,
            sanitize_args=sanitize_file_args,
            execute=async_tool(lambda params: run_write(params.path, params.content)),
        ),
        "edit_file": ToolRuntimeSpec(
            name="edit_file",
            params_model=EditFileParams,
            sanitize_args=sanitize_file_args,
            execute=async_tool(lambda params: run_edit(params.path, params.old_text, params.new_text)),
        ),
        "glob": ToolRuntimeSpec(
            name="glob",
            params_model=GlobParams,
            sanitize_args=sanitize_search_args,
            execute=async_tool(lambda params: run_glob(
                params.pattern,
                params.directory,
                params.include_dirs,
                params.limit,
            )),
            concurrency_safe=True,
        ),
        "grep": ToolRuntimeSpec(
            name="grep",
            params_model=GrepParams,
            sanitize_args=sanitize_search_args,
            execute=async_tool(lambda params: run_grep(
                params.pattern,
                params.path,
                params.glob,
                params.output_mode,
                params.ignore_case,
                params.line_number,
                params.head_limit,
            )),
            concurrency_safe=True,
        ),
        "todo": ToolRuntimeSpec(
            name="todo",
            params_model=TodoParams,
            sanitize_args=sanitize_passthrough,
            execute=async_tool(lambda params: TODO.update(params)),
        ),
        "task": ToolRuntimeSpec(
            name="task",
            params_model=TaskParams,
            sanitize_args=sanitize_task_args,
            execute=lambda params: runner(params.prompt, params.description),
            concurrency_safe=True,
        ),
    }
    if tool_names is not None:
        return {name: spec for name, spec in registry.items() if name in tool_names}
    return registry


TOOL_REGISTRY = build_tool_registry()
EXPLORE_TOOL_REGISTRY = build_tool_registry(READ_ONLY_TOOL_NAMES)
EXPLORE_TOOLS = [tool for tool in TOOLS if tool["name"] in READ_ONLY_TOOL_NAMES]


def configure_task_runner(task_runner: Callable[[str, str], Awaitable[str]]) -> None:
    TOOL_REGISTRY["task"] = build_tool_registry(task_runner=task_runner)["task"]


def parse_tool_args(raw_arguments) -> tuple[dict, str | None]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments: {e}"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, None


async def run_tool_call_async(
    item,
    registry: dict[str, ToolRuntimeSpec] | None = None,
) -> tuple[dict, bool]:
    used_todo = item.name == "todo"
    args, parse_error = parse_tool_args(item.arguments)
    active_registry = registry or TOOL_REGISTRY
    spec = active_registry.get(item.name)

    if item.name == "bash":
        preview = args.get("command", "") if isinstance(args, dict) else ""
        print(f"\033[33m$ {preview}\033[0m")
    else:
        print(f"\033[33m# {item.name} {args}\033[0m")

    if parse_error:
        output = f"Error: invalid arguments for tool '{item.name}': {parse_error}"
    elif spec is None:
        output = f"Error: unknown tool '{item.name}'"
    else:
        clean_args = spec.sanitize_args(args)
        try:
            params = spec.params_model.model_validate(clean_args)
        except Exception as e:
            output = f"Error: invalid arguments for tool '{item.name}': {e}"
        else:
            try:
                output = await spec.execute(params)
            except Exception as e:
                output = f"Error: tool '{item.name}' failed: {e}"

    # Bound the output for the model's context at the one chokepoint every tool
    # flows through. `todo` is control-plane state (small, structured); `read_file`
    # self-bounds (line caps + a footer) and middle-truncation would break its line
    # numbering -- leave both verbatim; everything else gets middle-truncated if oversized.
    if item.name not in ("todo", "read_file"):
        output = truncate_middle(output)

    if item.name == "todo":
        print(output)
    else:
        print(output[:TOOL_OUTPUT_PREVIEW_CHARS])

    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": output,
    }, used_todo


def run_tool_call(
    item,
    registry: dict[str, ToolRuntimeSpec] | None = None,
) -> tuple[dict, bool]:
    return _run_async_from_sync(run_tool_call_async(item, registry))


def _run_async_from_sync(awaitable):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("Use the async tool execution API inside a running event loop.")


def _partition_tool_calls(
    tool_calls,
    registry: dict[str, ToolRuntimeSpec],
) -> list[tuple[bool, list]]:
    batches: list[tuple[bool, list]] = []
    for item in tool_calls:
        if item.type != "function_call":
            continue
        spec = registry.get(item.name)
        is_safe = bool(spec and spec.concurrency_safe)
        if is_safe and batches and batches[-1][0]:
            batches[-1][1].append(item)
        else:
            batches.append((is_safe, [item]))
    return batches


async def execute_tool_calls_async(
    tool_calls,
    registry: dict[str, ToolRuntimeSpec] | None = None,
) -> tuple[list[dict], bool]:
    active_registry = registry or TOOL_REGISTRY
    results = []
    used_todo = False

    for is_safe, batch in _partition_tool_calls(tool_calls, active_registry):
        if is_safe and len(batch) > 1:
            for start in range(0, len(batch), MAX_PARALLEL_TOOL_CALLS):
                chunk = batch[start:start + MAX_PARALLEL_TOOL_CALLS]
                chunk_results = await asyncio.gather(
                    *(run_tool_call_async(item, active_registry) for item in chunk)
                )
                for tool_result, called_todo in chunk_results:
                    if called_todo:
                        used_todo = True
                    results.append(tool_result)
        else:
            for item in batch:
                tool_result, called_todo = await run_tool_call_async(item, active_registry)
                if called_todo:
                    used_todo = True
                results.append(tool_result)
    return results, used_todo


def execute_tool_calls(
    tool_calls,
    registry: dict[str, ToolRuntimeSpec] | None = None,
) -> tuple[list[dict], bool]:
    return _run_async_from_sync(execute_tool_calls_async(tool_calls, registry))
