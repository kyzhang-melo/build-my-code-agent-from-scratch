"""Live-model paired micro-evals for falsifiable harness hypotheses.

The deterministic cases in :mod:`evals.micro.cases` characterize one tool
decision.  This module adds the behavioral half of Phase 2: the same short
scenario is run against baseline and candidate harness variants, with repeated
trials and model-scoped acceptance criteria.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import tools
from permissions import ApprovalRequest, ApprovalResponse
from trace import MemoryTraceSink, TraceContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "evals" / ".runs" / "micro-paired"
VARIANTS = ("baseline", "candidate")
HYPOTHESIS_ID = "grep-n-alias-v1"


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    evidence_key: str
    candidate_change: str
    scenario_names: tuple[str, ...]
    preregistered_criteria: tuple[str, ...]


GREP_ALIAS_HYPOTHESIS = Hypothesis(
    hypothesis_id=HYPOTHESIS_ID,
    statement=(
        "Accepting the common grep JSON alias -n reduces validation friction "
        "and recovery calls without reducing task success."
    ),
    evidence_key="failure_matrix:grep/validation + param_friction:grep/-n",
    candidate_change=(
        "Expose -n in the grep tool schema and normalize it to line_number "
        "before strict runtime validation."
    ),
    scenario_names=(
        "grep-n-find-definition",
        "grep-n-find-config",
        "grep-n-find-test",
    ),
    preregistered_criteria=(
        "candidate success rate is not lower than baseline for either model",
        "candidate produces fewer -n validation failures for both models",
        "candidate median API calls are not higher than baseline for either model",
        "candidate introduces no tool error signature absent from baseline",
    ),
)


class AutoApproveHandler:
    async def request(self, request: ApprovalRequest) -> ApprovalResponse:
        if request.allow_for_session:
            return ApprovalResponse("approve_for_session")
        return ApprovalResponse("approve")


@dataclass
class PairedTrialResult:
    model: str
    provider: str
    scenario: str
    trial: int
    variant: str
    passed: bool
    api_calls: int = 0
    alias_validation_failures: int = 0
    tool_errors: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class VariantMetrics:
    model: str
    variant: str
    attempts: int
    passed: int
    success_rate: float
    median_api_calls: float
    alias_validation_failures: int
    tool_errors: list[str]


@dataclass
class CriterionResult:
    label: str
    passed: bool
    detail: str


def discover_scenarios(hypothesis: Hypothesis = GREP_ALIAS_HYPOTHESIS) -> list[Path]:
    paths = [SCENARIOS_DIR / name for name in hypothesis.scenario_names]
    missing = [str(path) for path in paths if not (path / "config.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing paired micro-eval scenarios: {missing}")
    return paths


def load_scenario(path: Path) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if not config.get("prompt") or not config.get("expect", {}).get("final_contains"):
        raise ValueError(f"{path}/config.json needs prompt and expect.final_contains")
    return config


def prepare_workspace(scenario_dir: Path, destination: Path) -> Path:
    workspace = destination / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    template = scenario_dir / "template"
    if template.is_dir():
        shutil.copytree(template, workspace, dirs_exist_ok=True)
    return workspace


def _candidate_grep_schema(schemas: list[dict]) -> list[dict]:
    candidate = copy.deepcopy(schemas)
    grep_schema = next(schema for schema in candidate if schema.get("name") == "grep")
    grep_schema["description"] = (
        grep_schema["description"].replace(
            "Do not pass command-line flags such as -n, -A, -B, or -C. ",
            "The common boolean alias -n is accepted for line_number. "
            "Other command-line flags such as -A, -B, or -C are not supported. ",
        )
    )
    grep_schema["parameters"]["properties"]["-n"] = {
        "type": "boolean",
        "description": "Alias for line_number. Prefer line_number in new calls.",
    }
    return candidate


def normalize_grep_n_alias(args: dict) -> dict:
    """Candidate-only normalization kept outside the production harness."""
    clean = tools.sanitize_search_args(args)
    if "-n" not in clean:
        return clean
    alias_value = clean.pop("-n")
    if "line_number" in clean and clean["line_number"] != alias_value:
        # Preserve a deterministic validation failure for contradictory input.
        clean["-n"] = alias_value
        return clean
    clean["line_number"] = alias_value
    return clean


def apply_harness_variant(session, variant: str):
    """Return a session with an eval-only baseline or candidate tool boundary."""
    if variant == "baseline":
        return session
    if variant != "candidate":
        raise ValueError(f"unknown harness variant: {variant}")
    registry = dict(session.registry)
    registry["grep"] = replace(
        registry["grep"],
        sanitize_args=normalize_grep_n_alias,
    )
    return replace(
        session,
        tools=_candidate_grep_schema(session.tools),
        registry=registry,
    )


def _count_alias_validation_failures(events: list[dict]) -> int:
    return sum(
        1
        for event in events
        if event.get("event") == "tool.requested"
        and any(
            issue.get("path") == "-n"
            for issue in event.get("validation_issues", [])
        )
    )


def _tool_errors(events: list[dict]) -> list[str]:
    return sorted({
        f"{event.get('tool_name', 'unknown')}/{event.get('error_type') or 'unknown'}"
        for event in events
        if event.get("event") == "tool.completed" and event.get("success") is False
    })


def _evaluate_scenario(config: dict, events: list[dict], final_text: str) -> list[str]:
    failed: list[str] = []
    for needle in config["expect"].get("final_contains", []):
        if needle not in final_text:
            failed.append(f"final answer missing {needle!r}")
    completed = [
        event for event in events
        if event.get("event") == "tool.completed"
        and event.get("tool_name") == "grep"
        and event.get("success") is True
    ]
    if config["expect"].get("successful_grep") and not completed:
        failed.append("no successful grep call")
    return failed


async def run_trial(
    *,
    model: str,
    provider: str,
    scenario_dir: Path,
    trial: int,
    variant: str,
    run_root: Path,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    max_api_calls: int,
) -> PairedTrialResult:
    # Lazy import keeps deterministic pytest independent of live-model env vars.
    import main

    config = load_scenario(scenario_dir)
    trial_root = (
        run_root / model.replace("/", "__").replace(":", "_")
        / f"trial-{trial}" / variant / scenario_dir.name
    )
    workspace = prepare_workspace(scenario_dir, trial_root)
    sink = MemoryTraceSink()
    trace = TraceContext(
        sink=sink,
        run_id=f"{run_root.name}:{model}:{trial}:{variant}:{scenario_dir.name}",
        agent_id="parent",
    )
    main.MODEL_ID = model
    session = main.create_parent_session(
        workspace,
        approval_handler=AutoApproveHandler(),
        trace_context=trace,
        on_text=None,
        max_api_calls=max_api_calls,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        tool_names={"grep", "read_file"},
    )
    session = apply_harness_variant(session, variant)
    result = PairedTrialResult(
        model=model,
        provider=provider,
        scenario=scenario_dir.name,
        trial=trial,
        variant=variant,
        passed=False,
    )
    state = main.LoopState(messages=[{"role": "user", "content": config["prompt"]}])
    try:
        timeout = int(config.get("timeout", 180))
        outcome = await asyncio.wait_for(main.agent_loop(state, session), timeout=timeout)
        result.api_calls = outcome.api_calls
        result.failed_checks = _evaluate_scenario(config, sink.events, outcome.final_text)
        result.passed = not result.failed_checks
    except Exception as exc:  # noqa: BLE001 - persist per-trial live failures
        result.error = f"{type(exc).__name__}: {exc}"
        result.api_calls = state.api_call_count
    result.alias_validation_failures = _count_alias_validation_failures(sink.events)
    result.tool_errors = _tool_errors(sink.events)
    if result.passed:
        shutil.rmtree(trial_root, ignore_errors=True)
    return result


def aggregate(results: list[PairedTrialResult]) -> list[VariantMetrics]:
    groups: dict[tuple[str, str], list[PairedTrialResult]] = {}
    for result in results:
        groups.setdefault((result.model, result.variant), []).append(result)
    metrics: list[VariantMetrics] = []
    for (model, variant), rows in sorted(groups.items()):
        api_calls = [row.api_calls for row in rows]
        metrics.append(VariantMetrics(
            model=model,
            variant=variant,
            attempts=len(rows),
            passed=sum(row.passed for row in rows),
            success_rate=sum(row.passed for row in rows) / len(rows),
            median_api_calls=float(statistics.median(api_calls)),
            alias_validation_failures=sum(row.alias_validation_failures for row in rows),
            tool_errors=sorted({error for row in rows for error in row.tool_errors}),
        ))
    return metrics


def evaluate_acceptance(
    metrics: list[VariantMetrics],
    *,
    minimum_models: int = 2,
) -> list[CriterionResult]:
    by_model = {
        (metric.model, metric.variant): metric
        for metric in metrics
    }
    models = sorted({metric.model for metric in metrics})
    complete_models = [
        model for model in models
        if (model, "baseline") in by_model and (model, "candidate") in by_model
    ]
    criteria = [CriterionResult(
        "at least two models",
        len(complete_models) >= minimum_models,
        f"paired models={len(complete_models)}",
    )]
    if len(complete_models) < minimum_models:
        return criteria

    success_ok = all(
        by_model[(model, "candidate")].success_rate
        >= by_model[(model, "baseline")].success_rate
        for model in complete_models
    )
    criteria.append(CriterionResult(
        "no per-model success-rate regression",
        success_ok,
        "; ".join(
            f"{model}: {by_model[(model, 'baseline')].success_rate:.1%} -> "
            f"{by_model[(model, 'candidate')].success_rate:.1%}"
            for model in complete_models
        ),
    ))
    friction_ok = all(
        by_model[(model, "candidate")].alias_validation_failures
        < by_model[(model, "baseline")].alias_validation_failures
        for model in complete_models
    )
    criteria.append(CriterionResult(
        "fewer -n validation failures for every model",
        friction_ok,
        "; ".join(
            f"{model}: {by_model[(model, 'baseline')].alias_validation_failures} -> "
            f"{by_model[(model, 'candidate')].alias_validation_failures}"
            for model in complete_models
        ),
    ))
    calls_ok = all(
        by_model[(model, "candidate")].median_api_calls
        <= by_model[(model, "baseline")].median_api_calls
        for model in complete_models
    )
    criteria.append(CriterionResult(
        "no per-model median API-call increase",
        calls_ok,
        "; ".join(
            f"{model}: {by_model[(model, 'baseline')].median_api_calls:g} -> "
            f"{by_model[(model, 'candidate')].median_api_calls:g}"
            for model in complete_models
        ),
    ))
    new_errors_by_model = {
        model: sorted(
            set(by_model[(model, "candidate")].tool_errors)
            - set(by_model[(model, "baseline")].tool_errors)
        )
        for model in complete_models
    }
    new_errors_by_model = {
        model: errors for model, errors in new_errors_by_model.items() if errors
    }
    criteria.append(CriterionResult(
        "no per-model candidate-only tool error signature",
        not new_errors_by_model,
        f"new errors={new_errors_by_model or 'none'}",
    ))
    return criteria


async def run_experiment(
    *,
    models: list[str],
    k: int,
    run_root: Path,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    max_api_calls: int = 8,
) -> list[PairedTrialResult]:
    provider = os.getenv("OPENROUTER_PROVIDER", "")
    scenarios = discover_scenarios()
    results: list[PairedTrialResult] = []
    for model in models:
        for trial in range(1, k + 1):
            # Alternate order to reduce temporal/provider-order bias.
            variants = VARIANTS if trial % 2 else tuple(reversed(VARIANTS))
            for scenario in scenarios:
                for variant in variants:
                    result = await run_trial(
                        model=model,
                        provider=provider,
                        scenario_dir=scenario,
                        trial=trial,
                        variant=variant,
                        run_root=run_root,
                        reasoning_effort=reasoning_effort,
                        max_output_tokens=max_output_tokens,
                        max_api_calls=max_api_calls,
                    )
                    results.append(result)
                    print(
                        f"[{model}] {scenario.name} k={trial} {variant}: "
                        f"{'PASS' if result.passed else 'FAIL'} "
                        f"api_calls={result.api_calls} -n_errors={result.alias_validation_failures}"
                    )
    return results


def write_report(
    run_root: Path,
    results: list[PairedTrialResult],
    metrics: list[VariantMetrics],
    criteria: list[CriterionResult],
    *,
    k: int,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    max_api_calls: int,
) -> None:
    accepted = bool(criteria) and all(item.passed for item in criteria)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip())
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": GREP_ALIAS_HYPOTHESIS.__dict__,
        "provider": os.getenv("OPENROUTER_PROVIDER", ""),
        "base_url": os.getenv("OPENROUTER_BASE_URL", ""),
        "harness_commit": commit,
        "harness_worktree_dirty": dirty,
        "k": k,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "max_api_calls": max_api_calls,
        "accepted": accepted,
        "criteria": [item.__dict__ for item in criteria],
        "metrics": [item.__dict__ for item in metrics],
        "trials": [item.__dict__ for item in results],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Paired Harness Micro-Eval",
        "",
        f"- Hypothesis: `{HYPOTHESIS_ID}`",
        f"- Verdict: **{'ACCEPT' if accepted else 'REJECT'}**",
        f"- Provider pin: `{payload['provider'] or 'un-pinned'}`",
        f"- Harness: `{commit or 'unavailable'}` ({'dirty' if dirty else 'clean'})",
        f"- Repeats: **k={k}**",
        f"- Max API calls per attempt: **{max_api_calls}**",
        "",
        "## Preregistered criteria",
        "",
    ]
    for item in criteria:
        lines.append(f"- [{'x' if item.passed else ' '}] {item.label}: {item.detail}")
    lines += [
        "",
        "## Model × variant metrics",
        "",
        "| Model | Variant | Passed | Success | Median API calls | -n validation failures | Tool errors |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        lines.append(
            f"| {item.model} | {item.variant} | {item.passed}/{item.attempts} | "
            f"{item.success_rate:.1%} | {item.median_api_calls:g} | "
            f"{item.alias_validation_failures} | {', '.join(item.tool_errors) or 'none'} |"
        )
    lines += [
        "",
        "## Next gate",
        "",
        (
            "If every preregistered criterion passes, implement the candidate in the production "
            "harness and run SWE-bench `small_10.json` once to check for side effects."
        ),
    ]
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
