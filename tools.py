import json
import os
import subprocess


TOOLS = [{
    "type": "function",
    "name": "bash",
    "description": "Run a shell command in the current workspace.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    }
}]


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


def execute_tool_calls(response_output) -> list[dict]:
    results = []
    for item in response_output:
        if item.type != "function_call":
            continue

        try:
            args = json.loads(item.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        command = args.get("command", "")
        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        print(output[0:200])

        results.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": output,
        })
    return results
