import json
import os
import subprocess
from pathlib import Path


WORKDIR = Path.cwd()


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
    }
]


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


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


def execute_tool_calls(response_output) -> list[dict]:
    results = []
    for item in response_output:
        if item.type != "function_call":
            continue

        try:
            args = json.loads(item.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        handler = TOOL_HANDLERS.get(item.name)
        if item.name == "bash":
            print(f"\033[33m$ {args.get('command', '')}\033[0m")
        else:
            print(f"\033[33m# {item.name} {args}\033[0m")

        # unknown tool
        if handler is None:
            output = f"Error: unknown tool '{item.name}'"
        else:
            try:
                output = handler(**args)
            # bad arguments 
            except TypeError as e:
                output = f"Error: invalid arguments for tool '{item.name}': {e}"
            # runtime failure
            except Exception as e:
                output = f"Error: tool '{item.name}' failed: {e}"
        print(output[0:200])

        results.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": output,
        })
    return results
