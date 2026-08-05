from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evals.judge.core import CompareConfig, run_comparison, validate_compatible_runs
from evals.judge.models import AggregateJudgment, DimensionJudgment, PairJudgment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(*, run_id: str, reasoning: str | None, warm: bool = False) -> dict:
    value = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": "dataset",
        "split": "test",
        "instance_ids": ["repo__issue-1"],
        "model": "contestant/model",
        "provider": "contestant-provider",
        "max_api_calls": 30,
        "reasoning_effort": reasoning,
        "max_output_tokens": None,
        "instance_timeout_seconds": None,
        "harness_commit": "abc",
        "swebench_commit": "def",
        "auto_compact": True,
    }
    if warm:
        value["eval_mode"] = "warm_context_sequence"
    return value


def _make_run(root: Path, *, run_id: str, reasoning: str | None, warm: bool = False) -> Path:
    run = root / run_id
    _write_json(run / "manifest.json", _manifest(run_id=run_id, reasoning=reasoning, warm=warm))
    _write_json(run / "tasks.json", [{
        "instance_id": "repo__issue-1",
        "repo": "org/repo",
        "base_commit": "base",
        "problem_statement": "Fix the broken behavior.",
    }])
    _write_json(run / "summary.json", {
        "instances": [{
            "instance_id": "repo__issue-1",
            "official_status": "resolved" if reasoning else "unresolved",
            "agent_status": "completed",
            "api_calls": 4 if reasoning else 7,
            "duration_seconds": 2.5,
            "patch_status": "produced",
        }]
    })
    attempt = run / "instances" / "repo__issue-1" / "attempt-1"
    attempt.mkdir(parents=True)
    _write_json(attempt / "result.json", {
        "instance_id": "repo__issue-1",
        "attempt": 1,
        "agent_status": "completed",
        "patch_status": "produced",
        "api_calls": 4,
    })
    (attempt / "agent.log").write_text(
        f"Read {run}/instances/repo__issue-1/attempt-1/workspace/file.py\n"
        "Located the failing branch.\nVerified the diff.\n",
        encoding="utf-8",
    )
    (attempt / "trace.jsonl").write_text("\n".join([
        json.dumps({
            "event": "tool.requested", "run_id": run_id, "timestamp": "now",
            "sequence": 1, "agent_id": "parent", "tool_name": "read_file",
            "arguments": {"path": str(attempt / "workspace" / "file.py")},
        }),
        json.dumps({
            "event": "llm.usage", "run_id": run_id, "timestamp": "now",
            "sequence": 2, "model": "contestant/model", "cost": 9.99,
            "reasoning_tokens": 999,
        }),
    ]) + "\n", encoding="utf-8")
    (attempt / "patch.diff").write_text("diff --git a/file.py b/file.py\n", encoding="utf-8")
    (attempt / "final_text.txt").write_text("Implemented and verified.", encoding="utf-8")
    return run


def _dimension(preference: str = "A") -> DimensionJudgment:
    return DimensionJudgment(
        preference=preference,
        explanation="A is better supported.",
        evidence=[{"ref": "A:L2", "claim": "It localized the branch."}],
    )


def _pair_json() -> str:
    return PairJudgment(
        localization_quality=_dimension(),
        exploration_discipline=_dimension(),
        evidence_grounded_reasoning=_dimension(),
        editing_discipline=_dimension(),
        verification_and_recovery=_dimension(),
        overall_preference="A",
        overall_explanation="A follows a more disciplined path.",
        confidence=0.8,
    ).model_dump_json()


def _aggregate_json() -> str:
    return AggregateJudgment(
        observed_effect="mixed",
        summary="The observed process effect is mixed.",
        supported_mechanisms=["One trajectory localized earlier."],
        counterexamples=[],
        interaction_effects=[],
        alternative_explanations=["Sampling variance."],
        limitations=["One instance."],
        confidence=0.4,
    ).model_dump_json()


def _config(tmp_path: Path, run_a: Path, run_b: Path, **overrides) -> CompareConfig:
    values = dict(
        judge_run_id="judge-1",
        factor="reasoning",
        run_a=run_a,
        run_b=run_b,
        output_dir=tmp_path / "judge-1",
        judge_model="independent/judge",
        judge_provider="judge-provider",
    )
    values.update(overrides)
    return CompareConfig(**values)


def test_validate_reasoning_and_context_pairs() -> None:
    low = _manifest(run_id="low", reasoning="low")
    xhigh = _manifest(run_id="xhigh", reasoning="xhigh")
    assert validate_compatible_runs(low, xhigh, "reasoning") == ("low", "xhigh")

    warm = _manifest(run_id="warm", reasoning="low", warm=True)
    assert validate_compatible_runs(low, warm, "context") == ("cold", "warm")

    with pytest.raises(ValueError, match="same context mode"):
        validate_compatible_runs(low, _manifest(run_id="warm-x", reasoning="xhigh", warm=True), "reasoning")
    with pytest.raises(ValueError, match="same reasoning effort"):
        validate_compatible_runs(low, _manifest(run_id="warm-x", reasoning="xhigh", warm=True), "context")


def test_compare_blinds_inputs_and_reveals_outcomes_afterward(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, run_id="secret-low-run", reasoning=None)
    run_b = _make_run(tmp_path, run_id="secret-xhigh-run", reasoning="xhigh")
    prompts: list[str] = []

    async def create(prompt: str) -> str:
        prompts.append(prompt)
        return _aggregate_json() if "after experimental labels were revealed" in prompt else _pair_json()

    summary = asyncio.run(run_comparison(_config(tmp_path, run_a, run_b), create))

    assert len(prompts) == 2
    blind_prompt = prompts[0]
    assert "secret-low-run" not in blind_prompt
    assert "secret-xhigh-run" not in blind_prompt
    assert "xhigh" not in blind_prompt
    assert "official_status" not in blind_prompt
    assert "9.99" not in blind_prompt
    assert "reasoning_tokens" not in blind_prompt
    assert "<WORKSPACE>" in blind_prompt
    assert "T1:" in blind_prompt
    assert summary["instances"][0]["run_a_outcome"]["official_status"] == "unresolved"
    assert summary["instances"][0]["run_b_outcome"]["official_status"] == "resolved"
    assert summary["aggregate"]["status"] == "completed"
    assert (tmp_path / "judge-1" / "summary.md").is_file()


def test_input_too_large_skips_all_judge_calls(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, run_id="low", reasoning="low")
    run_b = _make_run(tmp_path, run_id="xhigh", reasoning="xhigh")
    calls = 0

    async def create(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return _pair_json()

    config = _config(tmp_path, run_a, run_b, max_input_chars=10)
    summary = asyncio.run(run_comparison(config, create))

    assert calls == 0
    assert summary["status_counts"] == {"input_too_large": 1}
    assert summary["aggregate"]["status"] == "no_completed_instances"


def test_completed_pair_is_reused_and_failed_pair_requires_flag(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, run_id="low", reasoning="low")
    run_b = _make_run(tmp_path, run_id="xhigh", reasoning="xhigh")
    calls: list[str] = []

    async def create(prompt: str) -> str:
        calls.append(prompt)
        return _aggregate_json() if "after experimental labels were revealed" in prompt else _pair_json()

    config = _config(tmp_path, run_a, run_b)
    asyncio.run(run_comparison(config, create))
    assert len(calls) == 2
    calls.clear()

    asyncio.run(run_comparison(config, create))
    assert len(calls) == 1, "completed pair is reused; aggregate is regenerated"

    judgment_path = config.output_dir / "instances" / "repo__issue-1" / "judgment.json"
    _write_json(judgment_path, {"status": "judge_error", "instance_id": "repo__issue-1"})
    calls.clear()
    asyncio.run(run_comparison(config, create))
    assert len(calls) == 0, "existing failure is retained without --rerun-failed"

    calls.clear()
    rerun = _config(tmp_path, run_a, run_b, rerun_failed=True)
    asyncio.run(run_comparison(rerun, create))
    assert len(calls) == 2
