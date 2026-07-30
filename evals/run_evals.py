#!/usr/bin/env python3
"""Standalone mini-fixture eval runner.

Runs the real parent agent (in-process) against small, self-contained scenarios
and checks its behavior. Each scenario is a declarative directory:

    evals/scenarios/<name>/
        config.json      # prompt + expectations
        template/        # optional starting files copied into the workspace

Unlike the pytest suite in tests/, this driver calls a live model and therefore
costs real tokens. It is a standalone script (not collected by pytest), so an
ordinary `pytest` run never triggers it.

Usage:
    python evals/run_evals.py [--scenario NAME] [--model ID]
                              [--reasoning-effort LEVEL]
                              [--max-output-tokens TOKENS]
                              [--keep-workspaces] [--list]
"""

import argparse
import asyncio
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# The runner lives in evals/; make the project root importable so `import main`
# works regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402  (path setup must happen first)
from trace import MemoryTraceSink, TraceContext  # noqa: E402
from permissions import (  # noqa: E402
    ApprovalRequest,
    ApprovalResponse,
)

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
RUNS_DIR = Path(__file__).resolve().parent / ".runs"

class AutoApproveHandler:
    """Headless approval handler that approves every ASK request.

    The default TerminalApprovalHandler rejects all ASK decisions when stdin is
    not a TTY, which would deny every write_file/edit_file/bash call. Evals need
    to observe the agent actually doing the work, so this mirrors a "yolo" mode.
    Hard denials in PermissionManager._core_guard still apply.
    """

    async def request(self, request: ApprovalRequest) -> ApprovalResponse:
        if request.allow_for_session:
            return ApprovalResponse("approve_for_session")
        return ApprovalResponse("approve")


@dataclass
class Check:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    error: str = ""
    final_text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    api_calls: int = 0
    workspace: str = ""
    trace_event_count: int = 0


def discover_scenarios(only: str | None) -> list[Path]:
    if not SCENARIOS_DIR.is_dir():
        return []
    dirs = sorted(p for p in SCENARIOS_DIR.iterdir() if (p / "config.json").is_file())
    if only:
        dirs = [p for p in dirs if p.name == only]
    return dirs


def load_config(scenario_dir: Path) -> dict:
    with (scenario_dir / "config.json").open() as fh:
        config = json.load(fh)
    if "prompt" not in config:
        raise ValueError(f"{scenario_dir.name}/config.json is missing 'prompt'")
    return config


def prepare_workspace(scenario_dir: Path, run_root: Path) -> Path:
    workspace = run_root / scenario_dir.name / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    template = scenario_dir / "template"
    if template.is_dir():
        shutil.copytree(template, workspace, dirs_exist_ok=True)
    return workspace


# --- Assertion helpers ------------------------------------------------------

def _check_files_exist(workspace: Path, rels: list[str]) -> list[Check]:
    checks = []
    for rel in rels:
        exists = (workspace / rel).exists()
        checks.append(Check(f"file exists: {rel}", exists,
                            "" if exists else "not found"))
    return checks


def _check_file_contains(workspace: Path, specs: list[dict]) -> list[Check]:
    checks = []
    for spec in specs:
        rel, needle = spec["file"], spec["contains"]
        target = workspace / rel
        if not target.exists():
            checks.append(Check(f"{rel} contains {needle!r}", False, "file not found"))
            continue
        text = target.read_text(errors="replace")
        found = needle in text
        checks.append(Check(f"{rel} contains {needle!r}", found,
                            "" if found else "substring not present"))
    return checks


def _check_tools_used(events: list[dict], specs: list[dict]) -> list[Check]:
    checks = []
    completed = [event for event in events if event.get("event") == "tool.completed"]
    for spec in specs:
        name = spec["name"]
        args_contains = spec.get("args_contains", {})
        match = None
        for event in completed:
            if event.get("tool_name") != name:
                continue
            arguments = event.get("arguments", {})
            if all(str(sub) in str(arguments.get(key, "")) for key, sub in args_contains.items()):
                match = event
                break
        label = f"tool used: {name}"
        if args_contains:
            label += f" with {args_contains}"
        checks.append(Check(label, match is not None,
                            "" if match else "no matching tool call"))
    return checks


def _check_tools_not_used(events: list[dict], names: list[str]) -> list[Check]:
    used = {
        event.get("tool_name")
        for event in events
        if event.get("event") == "tool.requested"
    }
    return [Check(f"tool NOT used: {name}", name not in used,
                  "" if name not in used else "tool was called")
            for name in names]


def _event_matches(event: dict, spec: dict, aliases: dict[str, str]) -> bool:
    for key, expected in spec.items():
        actual_key = aliases.get(key, key)
        if actual_key == "args_contains":
            arguments = event.get("arguments", {})
            if not all(
                str(value) in str(arguments.get(arg_key, ""))
                for arg_key, value in expected.items()
            ):
                return False
        elif event.get(actual_key) != expected:
            return False
    return True


def _check_event_specs(
    events: list[dict],
    event_name: str,
    specs: list[dict],
    *,
    label: str,
    aliases: dict[str, str] | None = None,
) -> list[Check]:
    candidates = [event for event in events if event.get("event") == event_name]
    aliases = aliases or {}
    checks = []
    for spec in specs:
        matched = any(_event_matches(event, spec, aliases) for event in candidates)
        checks.append(Check(
            f"{label}: {spec}",
            matched,
            "" if matched else "no matching trace event",
        ))
    return checks


def _check_todo_transitions(events: list[dict], specs: list[dict]) -> list[Check]:
    transitions = [
        transition
        for event in events
        if event.get("event") == "todo.changed"
        for transition in event.get("transitions", [])
    ]
    return [
        Check(
            f"todo transition: {spec}",
            any(all(transition.get(key) == value for key, value in spec.items())
                for transition in transitions),
            "" if any(all(transition.get(key) == value for key, value in spec.items())
                      for transition in transitions) else "no matching transition",
        )
        for spec in specs
    ]


def _check_final_answer(final_text: str, needles: list[str]) -> list[Check]:
    checks = []
    for needle in needles:
        found = needle in final_text
        checks.append(Check(f"final answer contains {needle!r}", found,
                            "" if found else "substring not present"))
    return checks


def evaluate(expect: dict, workspace: Path, events: list[dict],
             final_text: str) -> list[Check]:
    checks: list[Check] = []
    checks += _check_files_exist(workspace, expect.get("files_exist", []))
    checks += _check_file_contains(workspace, expect.get("file_contains", []))
    checks += _check_tools_used(events, expect.get("tools_used", []))
    checks += _check_tools_not_used(events, expect.get("tools_not_used", []))
    checks += _check_event_specs(
        events,
        "tool.completed",
        expect.get("tool_completed", []),
        label="tool completed",
        aliases={"name": "tool_name"},
    )
    checks += _check_event_specs(
        events,
        "permission.decided",
        expect.get("permission_decisions", []),
        label="permission decision",
        aliases={"tool": "tool_name"},
    )
    checks += _check_todo_transitions(events, expect.get("todo_transitions", []))
    checks += _check_event_specs(
        events,
        "stop_gate.checked",
        expect.get("stop_gate_decisions", []),
        label="stop gate decision",
    )
    checks += _check_final_answer(final_text, expect.get("final_answer_contains", []))
    return checks


# --- Scenario execution -----------------------------------------------------

async def run_scenario(
    scenario_dir: Path,
    run_root: Path,
    *,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
) -> ScenarioResult:
    config = load_config(scenario_dir)
    name = config.get("name", scenario_dir.name)
    workspace = prepare_workspace(scenario_dir, run_root)

    # Every scenario receives a fresh session, including its workspace, todo,
    # permission service, registry, prompt, and trace context.
    workspace_root = workspace.resolve()
    trace_sink = MemoryTraceSink()
    trace_context = TraceContext(
        sink=trace_sink,
        run_id=f"{run_root.name}:{scenario_dir.name}",
        agent_id="parent",
    )

    session = main.create_parent_session(
        workspace_root,
        approval_handler=AutoApproveHandler(),
        trace_context=trace_context,
        on_text=None,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )

    state = main.LoopState(messages=[{"role": "user", "content": config["prompt"]}])
    result = ScenarioResult(name=name, passed=False, workspace=str(workspace))

    try:
        timeout = config.get("timeout")
        coro = main.agent_loop(state, session)
        outcome = await (asyncio.wait_for(coro, timeout) if timeout else coro)
    except asyncio.TimeoutError:
        result.error = f"timed out after {config.get('timeout')}s"
        return result
    except Exception as exc:  # noqa: BLE001 - surface any harness/model error per scenario
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    events = trace_sink.events
    result.final_text = outcome.final_text
    result.tool_calls = [
        event["tool_name"]
        for event in events
        if event.get("event") == "tool.completed"
    ]
    result.trace_event_count = len(events)
    result.api_calls = outcome.api_calls
    result.checks = evaluate(config.get("expect", {}), workspace, events, outcome.final_text)
    result.passed = all(check.passed for check in result.checks) and bool(result.checks)
    if not result.checks:
        result.error = "no expectations defined in config.json 'expect'"
    return result


# --- Reporting --------------------------------------------------------------

def print_scenario(result: ScenarioResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    color = "\033[32m" if result.passed else "\033[31m"
    print(f"\n{color}[{status}]\033[0m {result.name}  "
          f"(api_calls={result.api_calls}, tools={result.tool_calls})")
    if result.error:
        print(f"    error: {result.error}")
    for check in result.checks:
        mark = "\033[32m✓\033[0m" if check.passed else "\033[31m✗\033[0m"
        line = f"    {mark} {check.label}"
        if not check.passed and check.detail:
            line += f"  ({check.detail})"
        print(line)


def write_reports(
    results: list[ScenarioResult],
    run_root: Path,
    *,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
) -> None:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": main.MODEL_ID,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "scenarios": [
            {
                "name": r.name,
                "passed": r.passed,
                "error": r.error,
                "api_calls": r.api_calls,
                "tool_calls": r.tool_calls,
                "trace_event_count": r.trace_event_count,
                "checks": [{"label": c.label, "passed": c.passed, "detail": c.detail}
                           for c in r.checks],
            }
            for r in results
        ],
    }
    (run_root / "report.json").write_text(json.dumps(report, indent=2))

    lines = [
        f"# Eval Report ({report['generated_at']})",
        "",
        f"- Model: `{report['model']}`",
        f"- Reasoning effort: `{report['reasoning_effort']}`",
        f"- Max output tokens: `{report['max_output_tokens']}`",
        f"- Passed: **{report['passed']}/{report['total']}**",
        "",
        "| Scenario | Result | api_calls |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.api_calls} |")
    (run_root / "summary.md").write_text("\n".join(lines) + "\n")


def cleanup_workspaces(results: list[ScenarioResult], keep_all: bool) -> None:
    # Keep workspaces for failures (for debugging) and drop them for passes,
    # unless --keep-workspaces asks to retain everything.
    if keep_all:
        return
    for r in results:
        if r.passed and r.workspace:
            shutil.rmtree(Path(r.workspace).parent, ignore_errors=True)


async def main_async(args: argparse.Namespace) -> int:
    scenario_dirs = discover_scenarios(args.scenario)
    if not scenario_dirs:
        target = args.scenario or "<any>"
        print(f"No scenarios found (looked for '{target}' in {SCENARIOS_DIR}).")
        return 1

    if args.list:
        print("Available scenarios:")
        for d in scenario_dirs:
            print(f"  {d.name}")
        return 0

    if args.model:
        main.MODEL_ID = args.model

    run_root = RUNS_DIR / datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[evals] model={main.MODEL_ID!r} scenarios={len(scenario_dirs)} "
          f"run_dir={run_root}")

    results: list[ScenarioResult] = []
    for scenario_dir in scenario_dirs:
        results.append(
            await run_scenario(
                scenario_dir,
                run_root,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )
        print_scenario(results[-1])

    write_reports(
        results,
        run_root,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
    )
    cleanup_workspaces(results, args.keep_workspaces)

    passed = sum(1 for r in results if r.passed)
    print(f"\n[evals] {passed}/{len(results)} passed. Report: {run_root}/summary.md")
    return 0 if passed == len(results) else 1


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return parsed

    parser = argparse.ArgumentParser(description="Run mini-fixture behavioral evals.")
    parser.add_argument("--scenario", help="run only the named scenario directory")
    parser.add_argument("--model", help="override MODEL_ID for this run")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        help="reasoning effort sent to the Responses API",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        help="maximum output tokens per agent call (default: provider limit)",
    )
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--keep-workspaces", action="store_true",
                        help="retain workspaces for passing scenarios too")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
