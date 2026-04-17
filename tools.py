import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


WORKDIR = Path.cwd()
PLAN_REMINDER_INTERVAL = 3


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


@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(PlanItem(
                content=content,
                status=status,
                active_form=active_form,
            ))

        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.items = normalized
        self.state.rounds_since_update = 0
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


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: dangerous command!"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
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


@dataclass
class ToolRuntimeSpec:
    name: str
    required_fields: set[str]
    optional_fields: set[str]
    string_fields: set[str]
    int_fields: set[str]
    sanitize_args: Callable[[dict], dict]
    execute: Callable[[dict], str]


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


def sanitize_passthrough(args: dict) -> dict:
    return dict(args)


def validate_tool_args(spec: ToolRuntimeSpec, args: dict) -> list[str]:
    errors = []
    allowed = spec.required_fields | spec.optional_fields

    missing = spec.required_fields - set(args.keys())
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")

    unknown = set(args.keys()) - allowed
    if unknown:
        errors.append(f"unknown fields: {sorted(unknown)}")

    for field in spec.string_fields:
        if field in args and not isinstance(args[field], str):
            errors.append(f"field '{field}' must be a string")

    for field in spec.int_fields:
        value = args.get(field)
        if field in args and value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"field '{field}' must be an integer")

    return errors


def build_tool_registry() -> dict[str, ToolRuntimeSpec]:
    return {
        "bash": ToolRuntimeSpec(
            name="bash",
            required_fields={"command"},
            optional_fields=set(),
            string_fields={"command"},
            int_fields=set(),
            sanitize_args=sanitize_bash_args,
            execute=lambda args: run_bash(args["command"]),
        ),
        "read_file": ToolRuntimeSpec(
            name="read_file",
            required_fields={"path"},
            optional_fields={"limit"},
            string_fields={"path"},
            int_fields={"limit"},
            sanitize_args=sanitize_file_args,
            execute=lambda args: run_read(args["path"], args.get("limit")),
        ),
        "write_file": ToolRuntimeSpec(
            name="write_file",
            required_fields={"path", "content"},
            optional_fields=set(),
            string_fields={"path", "content"},
            int_fields=set(),
            sanitize_args=sanitize_file_args,
            execute=lambda args: run_write(args["path"], args["content"]),
        ),
        "edit_file": ToolRuntimeSpec(
            name="edit_file",
            required_fields={"path", "old_text", "new_text"},
            optional_fields=set(),
            string_fields={"path", "old_text", "new_text"},
            int_fields=set(),
            sanitize_args=sanitize_file_args,
            execute=lambda args: run_edit(args["path"], args["old_text"], args["new_text"]),
        ),
        "todo": ToolRuntimeSpec(
            name="todo",
            required_fields={"items"},
            optional_fields=set(),
            string_fields=set(),
            int_fields=set(),
            sanitize_args=sanitize_passthrough,
            execute=lambda args: TODO.update(args["items"]),
        ),
    }


TOOL_REGISTRY = build_tool_registry()


def parse_tool_args(raw_arguments) -> tuple[dict, str | None]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments: {e}"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, None


def run_tool_call(item) -> tuple[dict, bool]:
    used_todo = item.name == "todo"
    args, parse_error = parse_tool_args(item.arguments)
    spec = TOOL_REGISTRY.get(item.name)

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
        errors = validate_tool_args(spec, clean_args)
        if errors:
            output = f"Error: invalid arguments for tool '{item.name}': {'; '.join(errors)}"
        else:
            try:
                output = spec.execute(clean_args)
            except Exception as e:
                output = f"Error: tool '{item.name}' failed: {e}"

    if item.name == "todo":
        print(output)
    else:
        print(output[0:200])

    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": output,
    }, used_todo


def execute_tool_calls(tool_calls) -> tuple[list[dict], bool]:
    results = []
    used_todo = False
    for item in tool_calls:
        if item.type != "function_call":
            continue
        tool_result, called_todo = run_tool_call(item)
        if called_todo:
            used_todo = True
        results.append(tool_result)
    return results, used_todo
