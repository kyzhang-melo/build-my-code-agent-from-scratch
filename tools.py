import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


WORKDIR = Path.cwd()
PLAN_REMINDER_INTERVAL = 3
TOOL_OUTPUT_PREVIEW_CHARS = 500
READ_ONLY_TOOL_NAMES = {"read_file", "glob", "grep"}


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
        "description": "Read file contents from workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
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
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, params: TodoParams) -> str:
        self.state = PlanningState(items=params.items, rounds_since_update=0)
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def has_active_plan(self) -> bool:
        return len(self.state.items) > 0

    def all_items_completed(self) -> bool:
        return self.has_active_plan() and all(item.status == "completed" for item in self.state.items)

    def snapshot_signature(self) -> tuple[tuple[str, str, str], ...]:
        return tuple((item.content, item.status, item.active_form) for item in self.state.items)

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
    limit: StrictInt | None = None


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
    return output[:50000] if output else "(no output)"


def safe_path(path: str) -> Path:
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def run_read(path: str, limit: int | None = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


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


def run_glob(
    pattern: str,
    directory: str | None = None,
    include_dirs: bool = True,
    limit: int = 1000,
) -> str:
    try:
        base = safe_path(directory or ".")
        if pattern.startswith("**") and base == WORKDIR:
            return (
                "Error: unsafe glob pattern from workspace root. "
                "Pass a more specific directory when using '**'."
            )
        if not base.exists():
            return f"Error: Directory not found: {directory or '.'}"
        if not base.is_dir():
            return f"Error: Not a directory: {directory or '.'}"

        matches = list(base.glob(pattern))
        if not include_dirs:
            matches = [path for path in matches if path.is_file()]
        matches.sort()

        total = len(matches)
        returned = matches[:limit]
        lines = [str(path.relative_to(base)) for path in returned]
        if total == 0:
            return f"No matches found for pattern `{pattern}`."

        message = f"Found {total} matches for pattern `{pattern}`."
        if total > limit:
            message += f" Showing first {limit}."
        return "\n".join([message, *lines])[:50000]
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
    return output[:50000] if output else "No matches found."


@dataclass
class ToolRuntimeSpec:
    name: str
    params_model: type[BaseModel]
    sanitize_args: Callable[[dict], dict]
    execute: Callable[[BaseModel], str]


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


def unavailable_task_runner(prompt: str, description: str) -> str:
    return "Error: task runner is not configured."


def build_tool_registry(
    tool_names: set[str] | None = None,
    *,
    task_runner: Callable[[str, str], str] | None = None,
) -> dict[str, ToolRuntimeSpec]:
    runner = task_runner or unavailable_task_runner
    registry = {
        "bash": ToolRuntimeSpec(
            name="bash",
            params_model=BashParams,
            sanitize_args=sanitize_bash_args,
            execute=lambda params: run_bash(params.command),
        ),
        "read_file": ToolRuntimeSpec(
            name="read_file",
            params_model=ReadFileParams,
            sanitize_args=sanitize_file_args,
            execute=lambda params: run_read(params.path, params.limit),
        ),
        "write_file": ToolRuntimeSpec(
            name="write_file",
            params_model=WriteFileParams,
            sanitize_args=sanitize_file_args,
            execute=lambda params: run_write(params.path, params.content),
        ),
        "edit_file": ToolRuntimeSpec(
            name="edit_file",
            params_model=EditFileParams,
            sanitize_args=sanitize_file_args,
            execute=lambda params: run_edit(params.path, params.old_text, params.new_text),
        ),
        "glob": ToolRuntimeSpec(
            name="glob",
            params_model=GlobParams,
            sanitize_args=sanitize_search_args,
            execute=lambda params: run_glob(
                params.pattern,
                params.directory,
                params.include_dirs,
                params.limit,
            ),
        ),
        "grep": ToolRuntimeSpec(
            name="grep",
            params_model=GrepParams,
            sanitize_args=sanitize_search_args,
            execute=lambda params: run_grep(
                params.pattern,
                params.path,
                params.glob,
                params.output_mode,
                params.ignore_case,
                params.line_number,
                params.head_limit,
            ),
        ),
        "todo": ToolRuntimeSpec(
            name="todo",
            params_model=TodoParams,
            sanitize_args=sanitize_passthrough,
            execute=lambda params: TODO.update(params),
        ),
        "task": ToolRuntimeSpec(
            name="task",
            params_model=TaskParams,
            sanitize_args=sanitize_task_args,
            execute=lambda params: runner(params.prompt, params.description),
        ),
    }
    if tool_names is not None:
        return {name: spec for name, spec in registry.items() if name in tool_names}
    return registry


TOOL_REGISTRY = build_tool_registry()
EXPLORE_TOOL_REGISTRY = build_tool_registry(READ_ONLY_TOOL_NAMES)
EXPLORE_TOOLS = [tool for tool in TOOLS if tool["name"] in READ_ONLY_TOOL_NAMES]


def configure_task_runner(task_runner: Callable[[str, str], str]) -> None:
    TOOL_REGISTRY["task"] = build_tool_registry(task_runner=task_runner)["task"]


def parse_tool_args(raw_arguments) -> tuple[dict, str | None]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments: {e}"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, None


def run_tool_call(
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
                output = spec.execute(params)
            except Exception as e:
                output = f"Error: tool '{item.name}' failed: {e}"

    if item.name == "todo":
        print(output)
    else:
        print(output[:TOOL_OUTPUT_PREVIEW_CHARS])

    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": output,
    }, used_todo


def execute_tool_calls(
    tool_calls,
    registry: dict[str, ToolRuntimeSpec] | None = None,
) -> tuple[list[dict], bool]:
    results = []
    used_todo = False
    for item in tool_calls:
        if item.type != "function_call":
            continue
        tool_result, called_todo = run_tool_call(item, registry)
        if called_todo:
            used_todo = True
        results.append(tool_result)
    return results, used_todo
