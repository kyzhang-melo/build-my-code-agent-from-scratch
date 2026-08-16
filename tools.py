import asyncio
import codecs
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from collections import deque
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from file_engine import (
    MAX_LINE_CHARS,
    MAX_READ_BYTES,
    MAX_READ_FILE_BYTES,
    FileEngine,
)
from permissions import (
    PermissionBehavior,
    PermissionDecision,
    PermissionService,
    bash_hard_deny_reason,
    is_sensitive_path,
    permission_denied_output,
)
from trace import TraceContext, emit_trace
from workspace import Workspace
from grep_engine import GrepRequest
from sandbox import CommandResult, CommandRunner, FileBackend, LocalFileBackend


TOOL_OUTPUT_PREVIEW_CHARS = 500
READ_ONLY_TOOL_NAMES = {"read_file", "glob", "grep"}
# Budget for a single tool output handed back to the model. Char-based (chars
# ~= 4x tokens); a token-aware policy can replace it later without touching callers.
TOOL_OUTPUT_MAX_CHARS = 48000
TOOL_OUTPUT_MAX_LINE_CHARS = MAX_LINE_CHARS
DEFAULT_MAX_PARALLEL_TOOL_CALLS = 4

# `run_bash` bounds its own output before returning a structured result.  Leave
# room for the metadata block and a truncation marker so the dispatcher-level
# truncation chokepoint never has to rewrite that result.
BASH_TIMEOUT_SECONDS = 120
BASH_TERMINATE_GRACE_SECONDS = 0.5
BASH_POST_EXIT_DRAIN_SECONDS = 0.5
BASH_PROCESS_POLL_SECONDS = 0.01
BASH_READ_CHUNK_BYTES = 8192
BASH_OUTPUT_MAX_CHARS = 46_000
BASH_ERROR_MAX_CHARS = 500
BASH_CANDIDATE_PATHS = (
    "/bin/bash",
    "/usr/bin/bash",
    "/usr/local/bin/bash",
    "/opt/homebrew/bin/bash",
)

# read_file bounds. The reader self-bounds and is exempt from truncate_middle
# (see run_tool_call), so these are the sole authority over read output size.
MAX_READ_LINES = 1000              # max lines returned; also the `limit` default and hard max
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
        "description": "Run a non-interactive Bash command in the current workspace.",
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
            "Read text file contents. Relative paths resolve within the workspace; "
            "a path outside the workspace must be absolute. Output is line-numbered "
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
        "description": (
            "Create or completely rewrite a workspace file, or append exact content to it. "
            "Parent directories are created automatically. Use overwrite for new files and "
            "complete rewrites; use edit_file for precise changes to an existing file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": (
                        "overwrite replaces the whole file; append adds content exactly as "
                        "provided without inserting a newline. Defaults to overwrite."
                    ),
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": (
            "Make one or more precise replacements in an existing workspace file. Each "
            "old_text must identify one unique, non-overlapping region of the original file. "
            "Put separate changes to the same file in one edits array. Keep old_text as small "
            "as possible while including enough surrounding context to make it unique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "description": (
                        "Targeted replacements, all matched against the original file rather "
                        "than against the result of earlier edits."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "description": "Unique text to replace; may span multiple lines.",
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Replacement text; may span multiple lines.",
                            },
                        },
                        "required": ["old_text", "new_text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "glob",
        "description": (
            "Find files or directories by glob pattern. Relative search directories "
            "resolve within the workspace; a directory outside it must be absolute. "
            "Recursive patterns like '**/config.json' are supported and search the "
            "whole tree (noisy dirs such as .venv/node_modules are skipped). "
            "When the user names a search directory, pass it as directory instead of "
            "prefixing it into pattern. "
            "CRITICAL GLOB RULE: never use broad recursive patterns like '**/*', "
            "'**', '**/', or '**/**'; they match too many irrelevant files and "
            "produce truncated, low-value output. Prefer specific patterns like "
            "'**/*.py', '**/*.md', 'src/**/*.ts', or 'tests/**/*_test.py'. "
            "Use pattern='*' only for a shallow top-level listing in a specific "
            "directory."
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
                        "Directory to search within. Relative paths resolve within the "
                        "workspace; an external directory must be absolute. Defaults to "
                        "the workspace."
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
            "Search file contents. Relative search paths resolve within the "
            "workspace; a path outside it must be absolute. "
            "Put the search location in path, and use glob only to filter file names. "
            "The valid JSON arguments are pattern, path, glob, output_mode, ignore_case, "
            "line_number, and head_limit. Do not pass command-line flags such as -n, -A, "
            "-B, or -C. Use line_number=true for line numbers; use read_file with offset "
            "and limit to inspect surrounding lines."
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
                        "File or directory to search. Relative paths resolve within the "
                        "workspace; an external path must be absolute. Defaults to the workspace."
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

GIT_DIFF_TOOL = {
    "type": "function",
    "name": "git_diff",
    "description": (
        "Review all current workspace changes relative to HEAD as a unified Git diff, "
        "including untracked and binary files. This tool accepts no paths, revisions, "
        "flags, or commands and cannot inspect Git history."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

ALL_TOOL_SCHEMAS = [*TOOLS, GIT_DIFF_TOOL]


def select_tool_schemas(tool_names: set[str] | frozenset[str]) -> list[dict]:
    """Return schemas for one session, rejecting unknown tool names."""
    available = {tool["name"] for tool in ALL_TOOL_SCHEMAS}
    unknown = sorted(tool_names - available)
    if unknown:
        raise ValueError(f"unknown tool names: {', '.join(unknown)}")
    return [tool for tool in ALL_TOOL_SCHEMAS if tool["name"] in tool_names]


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
    mode: Literal["overwrite", "append"] = "overwrite"


class EditParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(min_length=1)
    new_text: str


class EditFileParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    edits: list[EditParams] = Field(min_length=1)


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


class GitDiffParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    description: str = "exploration"


@dataclass
class BashResult:
    """The bounded, model-facing outcome of one foreground shell command."""

    status: Literal["completed", "failed", "timed_out", "execution_error", "blocked"]
    exit_code: int | None
    output: str
    truncated: bool
    duration_ms: int
    timed_out: bool = False
    post_exit_cleanup: bool = False

    def render(self) -> str:
        output = self.output or "(no output)"
        return (
            f"[status] {self.status}\n"
            f"[exit_code] {self.exit_code if self.exit_code is not None else 'null'}\n"
            f"[timed_out] {str(self.timed_out).lower()}\n"
            f"[post_exit_cleanup] {str(self.post_exit_cleanup).lower()}\n"
            f"[truncated] {str(self.truncated).lower()}\n"
            f"[duration_ms] {self.duration_ms}\n\n"
            f"{output}"
        )


def render_remote_bash_result(result: CommandResult, duration_ms: int) -> str:
    """Render a remote command with the same status contract as local Bash."""
    raw_output = (result.stdout + result.stderr).strip()
    output = truncate_middle(raw_output, max_chars=BASH_OUTPUT_MAX_CHARS)
    return BashResult(
        status=(
            "timed_out" if result.timed_out
            else "completed" if result.exit_code == 0
            else "failed"
        ),
        exit_code=None if result.timed_out else result.exit_code,
        output=output,
        truncated=output != raw_output,
        duration_ms=duration_ms,
        timed_out=result.timed_out,
    ).render()


class _BashOutputBuffer:
    """Keep the useful tail of a command's combined output within a char budget."""

    def __init__(self, max_chars: int | None = None) -> None:
        self.max_chars = BASH_OUTPUT_MAX_CHARS if max_chars is None else max_chars
        self._chunks: deque[str] = deque()
        self._chars = 0
        self._discarded_chars = 0

    @property
    def truncated(self) -> bool:
        return self._discarded_chars > 0

    def append(self, text: str) -> None:
        if not text:
            return
        self._chunks.append(text)
        self._chars += len(text)

        while self._chars > self.max_chars and self._chunks:
            excess = self._chars - self.max_chars
            first = self._chunks.popleft()
            if len(first) <= excess:
                self._discarded_chars += len(first)
                self._chars -= len(first)
                continue
            self._discarded_chars += excess
            self._chunks.appendleft(first[excess:])
            self._chars -= excess

    def render(self) -> str:
        content = "".join(self._chunks)
        if not self.truncated:
            return content

        discarded = self._discarded_chars
        # The marker is part of the bounded result too. Its digit count depends
        # on the final discarded count, so recompute until both fit together.
        while True:
            marker = f"[... {discarded} chars of earlier command output discarded ...]\n"
            available = max(0, self.max_chars - len(marker))
            extra_discarded = max(0, len(content) - available)
            if extra_discarded == 0:
                return marker + content
            content = content[extra_discarded:]
            discarded += extra_discarded


async def _pump_bash_stream(
    stream: asyncio.StreamReader,
    queue: asyncio.Queue[str | None],
) -> None:
    """Decode one pipe incrementally and forward chunks in observed arrival order."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := await stream.read(BASH_READ_CHUNK_BYTES):
            text = decoder.decode(chunk)
            if text:
                await queue.put(text)
        tail = decoder.decode(b"", final=True)
        if tail:
            await queue.put(tail)
    finally:
        await queue.put(None)


async def _collect_bash_output(
    queue: asyncio.Queue[str | None],
    buffer: _BashOutputBuffer,
) -> None:
    finished_streams = 0
    while finished_streams < 2:
        item = await queue.get()
        if item is None:
            finished_streams += 1
        else:
            buffer.append(item)


def _signal_bash_process_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    if proc.pid is None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


async def _terminate_bash_process_group(proc: asyncio.subprocess.Process) -> None:
    """Stop the shell and descendants, escalating from TERM to KILL when needed."""
    _signal_bash_process_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(_wait_for_bash_exit(proc), timeout=BASH_TERMINATE_GRACE_SECONDS)
    except TimeoutError:
        _signal_bash_process_group(proc, signal.SIGKILL)
        await _wait_for_bash_exit(proc)


async def _wait_for_bash_exit(proc: asyncio.subprocess.Process) -> int:
    """Wait for the shell process itself, not for inherited output pipes to close."""
    # asyncio.Process.wait() may wait for pipe transports to close when a
    # descendant inherits stdout/stderr. `returncode` changes when the shell
    # exits, which lets the caller apply a separate post-exit drain policy.
    while proc.returncode is None:
        await asyncio.sleep(BASH_PROCESS_POLL_SECONDS)
    return proc.returncode


def _limit_bash_error(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    if len(text) > BASH_ERROR_MAX_CHARS:
        return text[:BASH_ERROR_MAX_CHARS] + " [...error truncated]"
    return text


def _resolve_bash_executable() -> str:
    """Find Bash from fixed absolute paths without consulting $SHELL or PATH."""
    for raw_path in BASH_CANDIDATE_PATHS:
        path = Path(raw_path)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    checked = ", ".join(BASH_CANDIDATE_PATHS)
    raise FileNotFoundError(f"Bash executable not found. Checked: {checked}")


async def run_bash(workspace: Workspace, command: str) -> str:
    """Run one foreground Bash command with bounded, merged output."""
    started_at = time.monotonic()
    deny_reason = bash_hard_deny_reason(command, workspace.root)
    if deny_reason:
        return BashResult(
            status="blocked",
            exit_code=None,
            output=deny_reason,
            truncated=False,
            duration_ms=0,
        ).render()

    buffer = _BashOutputBuffer()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)
    proc: asyncio.subprocess.Process | None = None
    reader_tasks: tuple[asyncio.Task[None], asyncio.Task[None]] | None = None
    collector_task: asyncio.Task[None] | None = None

    try:
        bash_path = _resolve_bash_executable()
        proc = await asyncio.create_subprocess_exec(
            bash_path,
            "-c",
            command,
            cwd=str(workspace.root),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert proc.stdout is not None and proc.stderr is not None
        reader_tasks = (
            asyncio.create_task(_pump_bash_stream(proc.stdout, queue)),
            asyncio.create_task(_pump_bash_stream(proc.stderr, queue)),
        )
        collector_task = asyncio.create_task(_collect_bash_output(queue, buffer))

        try:
            exit_code = await asyncio.wait_for(
                _wait_for_bash_exit(proc),
                timeout=BASH_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await _terminate_bash_process_group(proc)
            await asyncio.gather(*reader_tasks)
            await collector_task
            buffer.append(f"\n\n[Command killed by timeout ({BASH_TIMEOUT_SECONDS}s)]")
            return BashResult(
                status="timed_out",
                exit_code=None,
                output=buffer.render(),
                truncated=buffer.truncated,
                duration_ms=round((time.monotonic() - started_at) * 1000),
                timed_out=True,
            ).render()

        try:
            await asyncio.wait_for(
                asyncio.shield(collector_task),
                timeout=BASH_POST_EXIT_DRAIN_SECONDS,
            )
        except TimeoutError:
            # A completed shell whose streams stay open has left descendants
            # behind. Foreground bash has no background-task contract, so clean
            # the group rather than hanging the agent indefinitely.
            await _terminate_bash_process_group(proc)
            await asyncio.gather(*reader_tasks)
            await collector_task
            buffer.append("\n\n[Command descendants were terminated after the shell exited.]")
            return BashResult(
                status="failed",
                exit_code=exit_code,
                output=buffer.render(),
                truncated=buffer.truncated,
                duration_ms=round((time.monotonic() - started_at) * 1000),
                post_exit_cleanup=True,
            ).render()

        status: Literal["completed", "failed"] = "completed" if exit_code == 0 else "failed"
        return BashResult(
            status=status,
            exit_code=exit_code,
            output=buffer.render(),
            truncated=buffer.truncated,
            duration_ms=round((time.monotonic() - started_at) * 1000),
        ).render()
    except asyncio.CancelledError:
        if proc is not None:
            await _terminate_bash_process_group(proc)
        raise
    except (FileNotFoundError, OSError, RuntimeError) as error:
        if proc is not None:
            await _terminate_bash_process_group(proc)
        buffer.append(f"Command execution failed: {_limit_bash_error(error)}")
        return BashResult(
            status="execution_error",
            exit_code=None,
            output=buffer.render(),
            truncated=buffer.truncated,
            duration_ms=round((time.monotonic() - started_at) * 1000),
        ).render()
    finally:
        if reader_tasks is not None:
            for task in reader_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*reader_tasks, return_exceptions=True)
        if collector_task is not None and not collector_task.done():
            collector_task.cancel()
            await asyncio.gather(collector_task, return_exceptions=True)


def run_read(workspace: Workspace, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
    return FileEngine(
        workspace.root,
        max_line_chars=MAX_LINE_CHARS,
        max_read_bytes=MAX_READ_BYTES,
        max_read_file_bytes=MAX_READ_FILE_BYTES,
    ).read(path, offset, limit)


def run_write(workspace: Workspace, path: str, content: str, mode: str = "overwrite") -> str:
    return FileEngine(workspace.root).write(path, content, mode)


def run_edit(
    workspace: Workspace,
    path: str,
    edits_or_old_text: list[EditParams] | list[dict] | str,
    new_text: str | None = None,
) -> str:
    try:
        if isinstance(edits_or_old_text, str):
            if new_text is None:
                raise ValueError("new_text is required with a legacy old_text argument")
            edits = [EditParams(old_text=edits_or_old_text, new_text=new_text)]
        else:
            if new_text is not None:
                raise ValueError("new_text cannot be combined with an edits list")
            edits = [
                edit if isinstance(edit, EditParams) else EditParams.model_validate(edit)
                for edit in edits_or_old_text
            ]
    except Exception as exc:
        return f"Error: {exc}"
    return FileEngine(workspace.root).edit(
        path, [edit.model_dump() for edit in edits],
    )


def run_glob(
    workspace: Workspace,
    pattern: str,
    directory: str | None = None,
    include_dirs: bool = True,
    limit: int = 1000,
) -> str:
    return FileEngine(workspace.root).glob(pattern, directory, include_dirs, limit)


def run_grep(
    workspace: Workspace,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    output_mode: str = "files_with_matches",
    ignore_case: bool = False,
    line_number: bool = True,
    head_limit: int = 250,
) -> str:
    return LocalFileBackend(workspace).grep(GrepRequest(
        pattern=pattern,
        path=path,
        glob=glob,
        output_mode=output_mode,
        ignore_case=ignore_case,
        line_number=line_number,
        head_limit=head_limit,
    ))


def run_git_diff(workspace: Workspace) -> str:
    """Render every workspace change without exposing arbitrary Git commands."""
    fd, index_name = tempfile.mkstemp(prefix="agent-git-diff-index-")
    os.close(fd)
    index_path = Path(index_name)
    index_path.unlink()
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_INDEX_FILE": str(index_path),
        "GIT_OPTIONAL_LOCKS": "0",
    })

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            shell=False,
            cwd=str(workspace.root),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )

    try:
        for args in (("read-tree", "HEAD"), ("add", "-A")):
            result = git(*args)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                return f"Error: git_diff failed. {detail}".strip()
        result = git(
            "diff",
            "--cached",
            "--no-ext-diff",
            "--binary",
            "--full-index",
            "HEAD",
            "--",
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            return f"Error: git_diff failed. {detail}".strip()
        output = result.stdout.strip()
        return truncate_middle(output) if output else "No changes found."
    except subprocess.TimeoutExpired:
        return "Error: git_diff timed out after 20s."
    except Exception as exc:
        return f"Error: git_diff failed. {exc}"
    finally:
        index_path.unlink(missing_ok=True)


@dataclass
class ToolRuntimeSpec:
    name: str
    params_model: type[BaseModel]
    sanitize_args: Callable[[dict], dict]
    execute: Callable[[BaseModel], Awaitable[str]]
    # True when the tool only reads state and is safe to run concurrently with
    # other tool calls in the same turn. Consumed by the threaded executor; the
    # read-only tools and the explore-only `task` subagent qualify, while tools
        # that mutate the workspace or session TODO do not.
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


def sanitize_edit_args(args: dict) -> dict:
    """Normalize common model mistakes before strict Pydantic validation."""
    clean = sanitize_file_args(args)
    edits = clean.get("edits")
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list):
                clean["edits"] = parsed

    old_text = clean.get("old_text")
    new_text = clean.get("new_text")
    if isinstance(old_text, str) and isinstance(new_text, str):
        current_edits = clean.get("edits")
        if isinstance(current_edits, list):
            clean["edits"] = [*current_edits, {"old_text": old_text, "new_text": new_text}]
        elif current_edits is None:
            clean["edits"] = [{"old_text": old_text, "new_text": new_text}]
        clean.pop("old_text", None)
        clean.pop("new_text", None)
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
    workspace: Workspace,
    todo: TodoManager,
    tool_names: set[str] | None = None,
    *,
    task_runner: Callable[[str, str], Awaitable[str]] | None = None,
    file_backend: FileBackend | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, ToolRuntimeSpec]:
    """Build a tool registry bound to one workspace.

    The workspace is captured here rather than looked up per call, so a tool can
    never operate on a directory other than the one its registry was built for.
    """
    runner = task_runner or unavailable_task_runner
    files = file_backend or LocalFileBackend(workspace)
    location = files.execution_location
    if location not in {"local", "remote"}:
        raise ValueError(f"Unsupported file backend location: {location!r}")
    if location == "local" and not workspace.root.is_dir():
        raise ValueError(f"Local workspace directory does not exist: {workspace.root}")
    remote_call = getattr(files, "call", None)
    if location == "remote" and not callable(remote_call):
        raise TypeError("Remote file backend must implement call(operation, args)")
    if command_runner is not None and command_runner.execution_location != location:
        raise ValueError(
            "File backend and command runner execution locations do not match: "
            f"{location!r} != {command_runner.execution_location!r}"
        )

    def file_call(operation: str, args: dict, local: Callable[[], str]) -> str:
        if location == "local":
            return local()
        assert callable(remote_call)
        return remote_call(operation, args)

    def remote_bash_call(command: str) -> str:
        if command_runner is None:
            return "Error: remote bash runner is not configured."
        deny_reason = bash_hard_deny_reason(command, workspace.root)
        if deny_reason:
            return f"Error: Command blocked by safety policy: {deny_reason}"
        started_at = time.monotonic()
        result = command_runner.run(
            ["bash", "-lc", command], cwd=command_runner.default_cwd, timeout=120,
        )
        return render_remote_bash_result(
            result, round((time.monotonic() - started_at) * 1000),
        )

    async def execute_bash(params: BashParams) -> str:
        if location == "local":
            return await run_bash(workspace, params.command)
        return await asyncio.to_thread(remote_bash_call, params.command)

    def git_diff_call() -> str:
        output = file_call(
            "git_patch", {"base_commit": "BASELINE"}, lambda: run_git_diff(workspace),
        )
        if output.startswith("Error:"):
            return output
        return truncate_middle(output) if output else "No changes found."
    registry = {
        "bash": ToolRuntimeSpec(
            name="bash",
            params_model=BashParams,
            sanitize_args=sanitize_bash_args,
            execute=execute_bash,
        ),
        "read_file": ToolRuntimeSpec(
            name="read_file",
            params_model=ReadFileParams,
            sanitize_args=sanitize_file_args,
            execute=async_tool(lambda params: file_call("read", {
                "path": params.path, "offset": params.offset, "limit": params.limit,
            }, lambda: run_read(workspace, params.path, params.offset, params.limit))),
            concurrency_safe=True,
        ),
        "write_file": ToolRuntimeSpec(
            name="write_file",
            params_model=WriteFileParams,
            sanitize_args=sanitize_file_args,
            execute=async_tool(lambda params: file_call("write", {
                "path": params.path, "content": params.content, "mode": params.mode,
            }, lambda: run_write(workspace, params.path, params.content, params.mode))),
        ),
        "edit_file": ToolRuntimeSpec(
            name="edit_file",
            params_model=EditFileParams,
            sanitize_args=sanitize_edit_args,
            execute=async_tool(lambda params: file_call("edit", {
                "path": params.path,
                "edits": [edit.model_dump() for edit in params.edits],
            }, lambda: run_edit(workspace, params.path, params.edits))),
        ),
        "glob": ToolRuntimeSpec(
            name="glob",
            params_model=GlobParams,
            sanitize_args=sanitize_search_args,
            execute=async_tool(lambda params: file_call("glob", {
                "pattern": params.pattern, "directory": params.directory,
                "include_dirs": params.include_dirs, "limit": params.limit,
            }, lambda: run_glob(workspace, params.pattern, params.directory, params.include_dirs, params.limit))),
            concurrency_safe=True,
        ),
        "grep": ToolRuntimeSpec(
            name="grep",
            params_model=GrepParams,
            sanitize_args=sanitize_search_args,
            execute=async_tool(lambda params: files.grep(GrepRequest(
                pattern=params.pattern, path=params.path, glob=params.glob,
                output_mode=params.output_mode, ignore_case=params.ignore_case,
                line_number=params.line_number, head_limit=params.head_limit,
            ))),
            concurrency_safe=True,
        ),
        "git_diff": ToolRuntimeSpec(
            name="git_diff",
            params_model=GitDiffParams,
            sanitize_args=sanitize_passthrough,
            execute=async_tool(lambda params: git_diff_call()),
            concurrency_safe=True,
        ),
        "todo": ToolRuntimeSpec(
            name="todo",
            params_model=TodoParams,
            sanitize_args=sanitize_passthrough,
            execute=async_tool(lambda params: todo.update(params)),
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


EXPLORE_TOOLS = [tool for tool in TOOLS if tool["name"] in READ_ONLY_TOOL_NAMES]


def parse_tool_args(raw_arguments) -> tuple[dict, str | None]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments: {e}"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, None


def _tool_call_preview(tool_name: str, args: dict) -> str:
    if tool_name == "bash":
        return f"$ {args.get('command', '')}"
    if tool_name == "write_file":
        content = args.get("content", "")
        size = len(content) if isinstance(content, str) else "?"
        return (
            f"# write_file path={args.get('path', '')!r} "
            f"mode={args.get('mode', 'overwrite')!r} chars={size}"
        )
    if tool_name == "edit_file":
        edits = args.get("edits")
        count = len(edits) if isinstance(edits, list) else "?"
        return f"# edit_file path={args.get('path', '')!r} edits={count}"
    return f"# {tool_name} {args}"


def _trace_arguments(tool_name: str, args: dict) -> dict:
    """Project validated tool arguments into non-sensitive trace metadata."""
    if tool_name == "write_file":
        content = args.get("content", "")
        return {
            "path": args.get("path", ""),
            "mode": args.get("mode", "overwrite"),
            "content_chars": len(content) if isinstance(content, str) else 0,
        }
    if tool_name == "edit_file":
        edits = args.get("edits", [])
        return {
            "path": args.get("path", ""),
            "edits_count": len(edits) if isinstance(edits, list) else 0,
        }
    if tool_name == "bash":
        command = args.get("command", "")
        return {"command_chars": len(command) if isinstance(command, str) else 0}
    if tool_name == "task":
        prompt = args.get("prompt", "")
        return {
            "description": args.get("description", "exploration"),
            "prompt_chars": len(prompt) if isinstance(prompt, str) else 0,
        }
    if tool_name == "todo":
        items = args.get("items", [])
        counts = {"pending": 0, "in_progress": 0, "completed": 0}
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("status", "pending") in counts:
                    counts[item.get("status", "pending")] += 1
        return {"item_count": len(items) if isinstance(items, list) else 0,
                "status_counts": counts}
    if tool_name == "grep":
        pattern = args.get("pattern", "")
        return {
            key: value
            for key, value in {
                "path": args.get("path", "."),
                "glob": args.get("glob"),
                "output_mode": args.get("output_mode", "files_with_matches"),
                "ignore_case": args.get("ignore_case", False),
                "line_number": args.get("line_number", True),
                "head_limit": args.get("head_limit", 250),
                "pattern_chars": len(pattern) if isinstance(pattern, str) else 0,
            }.items()
            if value is not None
        }
    if tool_name == "glob":
        pattern = args.get("pattern", "")
        return {
            "directory": args.get("directory", "."),
            "include_dirs": args.get("include_dirs", False),
            "limit": args.get("limit", 1000),
            "pattern_chars": len(pattern) if isinstance(pattern, str) else 0,
        }
    if tool_name == "read_file":
        return {
            "path": args.get("path", ""),
            "offset": args.get("offset", 1),
            "limit": args.get("limit", MAX_READ_LINES),
        }
    return {"argument_keys": sorted(str(key) for key in args)}


def _safe_trace_arguments(tool_name: str, args: dict) -> dict:
    try:
        return _trace_arguments(tool_name, args)
    except Exception:
        return {"argument_keys": sorted(str(key) for key in args)}


def _todo_items_snapshot(todo: TodoManager) -> list[dict]:
    return [
        item.model_dump(by_alias=True)
        for item in todo.state.items
    ]


def _todo_transitions(before: list[dict], after: list[dict]) -> list[dict]:
    before_by_content = {item["content"]: item.get("status") for item in before}
    after_by_content = {item["content"]: item.get("status") for item in after}
    transitions = []
    for content in dict.fromkeys([*before_by_content, *after_by_content]):
        old = before_by_content.get(content)
        new = after_by_content.get(content)
        if old != new:
            transitions.append({"content": content, "from": old, "to": new})
    return transitions


def _tool_reported_error(tool_name: str, output: str) -> bool:
    if output.startswith("Error:"):
        return True
    if tool_name == "bash":
        return output.startswith("[status] failed") or output.startswith(
            ("[status] timed_out", "[status] execution_error", "[status] blocked")
        )
    return False


def _extract_validation_issues(exc: Exception) -> list[dict]:
    """Extract non-sensitive validation issues from a Pydantic ValidationError.

    Only ``loc`` (as ``path``) and ``type`` are kept; ``input``, ``msg``, and
    ``ctx`` are dropped because they may contain user content or sensitive
    values.
    """
    errors_fn = getattr(exc, "errors", None)
    if not callable(errors_fn):
        return []
    try:
        raw_errors = errors_fn()
    except Exception:
        return []
    issues: list[dict] = []
    for err in raw_errors:
        loc = err.get("loc", ())
        loc_path = ".".join(str(part) for part in loc) if loc else ""
        issues.append({"path": loc_path, "type": err.get("type", "")})
    return issues


def _raw_arguments_fingerprint(raw_arguments) -> dict:
    """Return a non-sensitive fingerprint of the raw tool-call arguments.

    Stores only the character count and a truncated SHA-256 hash so the
    offline analyzer can detect exact-repeat calls without exposing the
    original content.
    """
    if not isinstance(raw_arguments, str):
        return {"raw_arguments_chars": 0, "raw_arguments_sha256": ""}
    chars = len(raw_arguments)
    if chars == 0:
        return {"raw_arguments_chars": 0, "raw_arguments_sha256": ""}
    digest = hashlib.sha256(raw_arguments.encode("utf-8", errors="replace")).hexdigest()[:16]
    return {"raw_arguments_chars": chars, "raw_arguments_sha256": digest}


def _detect_internal_truncation(tool_name: str, output: str) -> bool:
    """Detect whether a self-bounding tool truncated its own output.

    ``bash`` renders a ``[truncated] true/false`` metadata line in its
    header block; ``read_file``'s footer notes "Stopped at the" or
    "truncated" when line/byte/per-line limits are hit.  Other tools do
    not self-bound, so they always return ``False``.
    """
    if tool_name == "bash":
        header = output.split("\n\n", 1)[0] if "\n\n" in output else output
        return "[truncated] true" in header
    if tool_name == "read_file":
        for line in reversed(output.splitlines()):
            if line.startswith("[Read ") and line.endswith("]"):
                return "Stopped at the" in line or "truncated" in line
        return False
    return False


async def run_tool_call_async(
    item,
    registry: dict[str, ToolRuntimeSpec],
    todo: TodoManager,
    permission_service: PermissionService | None = None,
    permission_source: str = "parent",
    trace_context: TraceContext | None = None,
    api_call: int | None = None,
    step_index: int | None = None,
) -> tuple[dict, bool]:
    started_at = time.monotonic()
    used_todo = item.name == "todo"
    args, parse_error = parse_tool_args(item.arguments)
    spec = registry.get(item.name)
    status = "success"
    error_type = None
    normalized_args = args
    todo_before = None

    def emit_requested(
        argument_error: str | None = None,
        validation_issues: list[dict] | None = None,
    ) -> None:
        emit_trace(
            trace_context,
            "tool.requested",
            call_id=item.call_id,
            source=permission_source,
            tool_name=item.name,
            arguments=_safe_trace_arguments(item.name, normalized_args),
            argument_error=argument_error,
            validation_issues=validation_issues or [],
            api_call=api_call,
            step_index=step_index,
            **_raw_arguments_fingerprint(item.arguments),
        )

    print(f"\033[33m{_tool_call_preview(item.name, args)}\033[0m")

    if parse_error:
        emit_requested(parse_error)
        status = "invalid_arguments"
        error_type = "json_parse"
        output = f"Error: invalid arguments for tool '{item.name}': {parse_error}"
    elif spec is None:
        emit_requested()
        status = "unknown_tool"
        error_type = "unknown_tool"
        output = f"Error: unknown tool '{item.name}'"
    else:
        clean_args = spec.sanitize_args(args)
        normalized_args = clean_args
        try:
            params = spec.params_model.model_validate(clean_args)
        except Exception as e:
            emit_requested(validation_issues=_extract_validation_issues(e))
            status = "invalid_arguments"
            error_type = "validation"
            output = f"Error: invalid arguments for tool '{item.name}': {e}"
            if item.name == "grep":
                output += (
                    "\nValid grep arguments: pattern, path, glob, output_mode, "
                    "ignore_case, line_number, head_limit. CLI flags such as -n, -A, "
                    "-B, and -C are not supported. Use line_number=true for line "
                    "numbers; use read_file with offset and limit for surrounding lines."
                )
        else:
            if hasattr(params, "model_dump"):
                normalized_args = params.model_dump(by_alias=True)
            emit_requested()
            if item.name == "todo":
                try:
                    todo_before = _todo_items_snapshot(todo)
                except Exception:
                    todo_before = None
            decision = None
            if permission_service is not None:
                try:
                    decision = await permission_service.authorize(
                        item.name,
                        normalized_args,
                        source=permission_source,
                        trace_context=trace_context,
                        call_id=item.call_id,
                    )
                except Exception:
                    decision = PermissionDecision(
                        PermissionBehavior.DENY,
                        "Permission check failed; the tool was not executed.",
                    )
            if decision is not None and decision.behavior is PermissionBehavior.DENY:
                status = "permission_denied"
                error_type = "permission_denied"
                output = permission_denied_output(item.name, decision)
            else:
                try:
                    output = await spec.execute(params)
                except asyncio.CancelledError:
                    emit_trace(
                        trace_context,
                        "tool.completed",
                        call_id=item.call_id,
                        source=permission_source,
                        tool_name=item.name,
                        arguments=_safe_trace_arguments(item.name, normalized_args),
                        duration_ms=round((time.monotonic() - started_at) * 1000),
                        success=False,
                        status="cancelled",
                        error_type="cancelled",
                        output_chars=0,
                        output_truncated=False,
                        runtime_output_truncated=False,
                        tool_internal_truncated=False,
                        truncated_chars=0,
                        api_call=api_call,
                        step_index=step_index,
                    )
                    raise
                except Exception as e:
                    status = "execution_error"
                    error_type = type(e).__name__
                    output = f"Error: tool '{item.name}' failed: {e}"
                else:
                    if _tool_reported_error(item.name, output):
                        status = "execution_error"
                        error_type = "tool_reported_error"
                    if item.name == "todo" and status == "success":
                        try:
                            todo_after = _todo_items_snapshot(todo)
                            transitions = _todo_transitions(todo_before or [], todo_after)
                        except Exception:
                            pass
                        else:
                            emit_trace(
                                trace_context,
                                "todo.changed",
                                call_id=item.call_id,
                                source=permission_source,
                                before=todo_before or [],
                                after=todo_after,
                                transitions=transitions,
                            )

    trace_arguments = _safe_trace_arguments(item.name, normalized_args)
    # Bound the output for the model's context at the one chokepoint every tool
    # flows through. `todo` is control-plane state (small, structured); `read_file`
    # self-bounds (line caps + a footer) and `bash` self-bounds before rendering
    # structured metadata -- leave these verbatim; everything else gets
    # middle-truncated if oversized.
    original_output_length = len(output)
    tool_internal_truncated = _detect_internal_truncation(item.name, output)
    if item.name not in ("todo", "read_file", "bash"):
        output = truncate_middle(output)
    runtime_output_truncated = len(output) < original_output_length
    truncated_chars = original_output_length - len(output) if runtime_output_truncated else 0
    output_truncated = runtime_output_truncated

    if item.name == "todo":
        print(output)
    else:
        print(output[:TOOL_OUTPUT_PREVIEW_CHARS])

    emit_trace(
        trace_context,
        "tool.completed",
        call_id=item.call_id,
        source=permission_source,
        tool_name=item.name,
        arguments=trace_arguments,
        duration_ms=round((time.monotonic() - started_at) * 1000),
        success=status == "success",
        status=status,
        error_type=error_type,
        output_chars=len(output),
        output_truncated=output_truncated,
        runtime_output_truncated=runtime_output_truncated,
        tool_internal_truncated=tool_internal_truncated,
        truncated_chars=truncated_chars,
        api_call=api_call,
        step_index=step_index,
    )

    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": output,
    }, used_todo


def run_tool_call(
    item,
    registry: dict[str, ToolRuntimeSpec],
    todo: TodoManager,
    permission_service: PermissionService | None = None,
    permission_source: str = "parent",
    trace_context: TraceContext | None = None,
    api_call: int | None = None,
    step_index: int | None = None,
) -> tuple[dict, bool]:
    return _run_async_from_sync(
        run_tool_call_async(
            item,
            registry,
            todo,
            permission_service,
            permission_source,
            trace_context,
            api_call,
            step_index,
        )
    )


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
    registry: dict[str, ToolRuntimeSpec],
    todo: TodoManager,
    permission_service: PermissionService | None = None,
    permission_source: str = "parent",
    trace_context: TraceContext | None = None,
    api_call: int | None = None,
) -> tuple[list[dict], bool]:
    results = []
    used_todo = False
    step = 0

    for is_safe, batch in _partition_tool_calls(tool_calls, registry):
        if is_safe and len(batch) > 1:
            for start in range(0, len(batch), MAX_PARALLEL_TOOL_CALLS):
                chunk = batch[start:start + MAX_PARALLEL_TOOL_CALLS]
                chunk_results = await asyncio.gather(
                    *(
                        run_tool_call_async(
                            item,
                            registry,
                            todo,
                            permission_service,
                            permission_source,
                            trace_context,
                            api_call,
                            step + i,
                        )
                        for i, item in enumerate(chunk)
                    )
                )
                step += len(chunk)
                for tool_result, called_todo in chunk_results:
                    if called_todo:
                        used_todo = True
                    results.append(tool_result)
        else:
            for item in batch:
                tool_result, called_todo = await run_tool_call_async(
                    item,
                    registry,
                    todo,
                    permission_service,
                    permission_source,
                    trace_context,
                    api_call,
                    step,
                )
                step += 1
                if called_todo:
                    used_todo = True
                results.append(tool_result)
    return results, used_todo


def execute_tool_calls(
    tool_calls,
    registry: dict[str, ToolRuntimeSpec],
    todo: TodoManager,
    permission_service: PermissionService | None = None,
    permission_source: str = "parent",
    trace_context: TraceContext | None = None,
    api_call: int | None = None,
) -> tuple[list[dict], bool]:
    return _run_async_from_sync(
        execute_tool_calls_async(
            tool_calls,
            registry,
            todo,
            permission_service,
            permission_source,
            trace_context,
            api_call,
        )
    )
