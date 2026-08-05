from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import evals.judge.core as judge_core
from evals.judge.core import CompareConfig, run_comparison, validate_compatible_runs
from evals.judge.models import (
    AggregateJudgment,
    DimensionJudgment,
    EvidenceRef,
    PairJudgment,
)


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


@pytest.mark.parametrize("ref", ["A:L10", "B:L10-L18", "A:T10", "B:T10-T18", "A:T10-18"])
def test_evidence_reference_accepts_log_and_trace_ranges(ref: str) -> None:
    assert EvidenceRef(ref=ref, claim="Evidence.").ref == ref


@pytest.mark.parametrize("ref", ["A:L", "A:T1-L2", "C:T1", "A:T1-"])
def test_evidence_reference_rejects_invalid_ranges(ref: str) -> None:
    with pytest.raises(ValueError):
        EvidenceRef(ref=ref, claim="Evidence.")


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
    row = summary["instances"][0]
    assert row["outcomes_by_factor_value"]["default"]["official_status"] == "unresolved"
    assert row["outcomes_by_factor_value"]["xhigh"]["official_status"] == "resolved"
    for side in ("A", "B"):
        factor_value = row["trajectories"][side]["factor_value"]
        assert (
            row["trajectories"][side]["outcome"]
            == row["outcomes_by_factor_value"][factor_value]
        )
    aggregate_prompt = prompts[1]
    assert '"run_a_outcome"' not in aggregate_prompt
    assert '"run_b_outcome"' not in aggregate_prompt
    assert '"trajectories"' in aggregate_prompt
    assert summary["aggregate"]["status"] == "completed"
    assert (tmp_path / "judge-1" / "summary.md").is_file()


def test_existing_v1_manifest_is_upgraded_without_rerunning_completed_pair(
    tmp_path: Path,
) -> None:
    run_a = _make_run(tmp_path, run_id="low", reasoning=None)
    run_b = _make_run(tmp_path, run_id="xhigh", reasoning="xhigh")
    config = _config(tmp_path, run_a, run_b)
    output = config.output_dir
    _write_json(output / "manifest.json", {
        "schema_version": 1,
        "factor": "reasoning",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "judge_model": "independent/judge",
        "judge_provider": "judge-provider",
        "seed": 0,
        "max_input_chars": 400_000,
        "rubric_version": "process-pair-v1",
        "aggregate_prompt_version": "factor-review-v1",
    })
    result_path = output / "instances" / "repo__issue-1" / "judgment.json"
    _write_json(result_path, {
        "schema_version": 1,
        "instance_id": "repo__issue-1",
        "status": "completed",
        "mapping": {
            "A": {"run": "low", "factor_value": "default"},
            "B": {"run": "xhigh", "factor_value": "xhigh"},
        },
        "judgment": json.loads(_pair_json()),
    })
    calls: list[str] = []

    async def create(prompt: str) -> str:
        calls.append(prompt)
        return _aggregate_json()

    summary = asyncio.run(run_comparison(config, create))

    assert len(calls) == 1
    upgraded = json.loads((output / "manifest.json").read_text())
    assert upgraded["schema_version"] == 2
    assert upgraded["aggregate_prompt_version"] == "factor-review-v2"
    assert upgraded["repair_prompt_version"] == "schema-repair-v1"
    assert upgraded["generation_settings"]["provider_defaults"] is True
    assert upgraded["protocol_history"] == [{
        "schema_version": 1,
        "aggregate_prompt_version": "factor-review-v1",
    }]
    row = summary["instances"][0]
    assert row["trajectories"]["A"]["outcome"]["official_status"] == "unresolved"
    assert row["trajectories"]["B"]["outcome"]["official_status"] == "resolved"


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


def test_invalid_pair_response_gets_one_small_schema_repair(
    tmp_path: Path, capsys
) -> None:
    run_a = _make_run(tmp_path, run_id="low", reasoning="low")
    run_b = _make_run(tmp_path, run_id="xhigh", reasoning="xhigh")
    prompts: list[str] = []

    async def create(prompt: str) -> str:
        prompts.append(prompt)
        if "Repair the structure" in prompt:
            return _pair_json()
        if "after experimental labels were revealed" in prompt:
            return _aggregate_json()
        return '{"wrong_field": true}'

    config = _config(tmp_path, run_a, run_b)
    summary = asyncio.run(run_comparison(config, create))

    assert len(prompts) == 3
    repair_prompt = prompts[1]
    assert "wrong_field" in repair_prompt
    assert "Fix the broken behavior" not in repair_prompt
    result_dir = config.output_dir / "instances" / "repo__issue-1"
    result = json.loads((result_dir / "judgment.json").read_text())
    assert result["status"] == "completed"
    assert result["response_repaired"] is True
    assert (result_dir / "repair-prompt.txt").is_file()
    assert (result_dir / "repair-response.txt").is_file()
    assert summary["repaired_responses"] == 1
    output = capsys.readouterr().out
    assert "[judge] repo__issue-1 started" in output
    assert "[judge] repo__issue-1 repaired" in output
    assert "[judge] aggregate completed" in output


def test_invalid_repair_remains_invalid_and_does_not_aggregate(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, run_id="low", reasoning="low")
    run_b = _make_run(tmp_path, run_id="xhigh", reasoning="xhigh")
    calls = 0

    async def create(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "{}"

    config = _config(tmp_path, run_a, run_b)
    summary = asyncio.run(run_comparison(config, create))

    assert calls == 2
    assert summary["status_counts"] == {"invalid_response": 1}
    assert summary["aggregate"]["status"] == "no_completed_instances"


def test_repair_provider_error_is_recorded_per_instance(
    tmp_path: Path, monkeypatch
) -> None:
    run_a = _make_run(tmp_path, run_id="low", reasoning="low")
    run_b = _make_run(tmp_path, run_id="xhigh", reasoning="xhigh")

    async def fake_judge_call(_create, prompt: str) -> str:
        if "Repair the structure" in prompt:
            raise RuntimeError("repair unavailable")
        return "{}"

    monkeypatch.setattr(judge_core, "_judge_call", fake_judge_call)

    async def unused_create(_prompt: str) -> str:
        raise AssertionError("fake_judge_call owns this test")

    summary = asyncio.run(run_comparison(
        _config(tmp_path, run_a, run_b), unused_create
    ))

    assert summary["status_counts"] == {"repair_provider_error": 1}
    assert summary["aggregate"]["status"] == "no_completed_instances"


def test_invalid_aggregate_response_gets_one_schema_repair(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, run_id="low", reasoning="low")
    run_b = _make_run(tmp_path, run_id="xhigh", reasoning="xhigh")
    aggregate_calls = 0

    async def create(prompt: str) -> str:
        nonlocal aggregate_calls
        if "Repair the structure" in prompt:
            assert "observed_effect" in prompt
            return _aggregate_json()
        if "after experimental labels were revealed" in prompt:
            aggregate_calls += 1
            return '{"summary": "incomplete"}'
        return _pair_json()

    config = _config(tmp_path, run_a, run_b)
    summary = asyncio.run(run_comparison(config, create))

    assert aggregate_calls == 1
    assert summary["aggregate"]["status"] == "completed"
    assert summary["aggregate"]["response_repaired"] is True
    assert summary["repaired_responses"] == 1


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
