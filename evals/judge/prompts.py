from __future__ import annotations

import json

from .models import AggregateJudgment, PairJudgment


RUBRIC_VERSION = "process-pair-v1"
AGGREGATE_PROMPT_VERSION = "factor-review-v2"
REPAIR_PROMPT_VERSION = "schema-repair-v1"


def build_pair_prompt(*, problem: str, trajectory_a: str, trajectory_b: str) -> str:
    schema = json.dumps(PairJudgment.model_json_schema(), ensure_ascii=False)
    return f"""You are a strict, neutral process judge comparing two code-agent trajectories.

Judge only the observable problem-solving process. Do not guess hidden chain-of-thought.
Do not determine patch correctness; an official evaluator handles correctness separately.
The trajectories are anonymized. Never infer or speculate which configuration produced A or B.

Evaluate these dimensions:
1. localization_quality: speed and accuracy in finding relevant files, symbols, and failure paths.
2. exploration_discipline: relevance, non-repetition, and appropriate stopping of exploration.
3. evidence_grounded_reasoning: whether decisions follow from observed code, tests, and tool results.
4. editing_discipline: timing, scope, and consistency of edits with the established diagnosis.
5. verification_and_recovery: validation, diff review, response to failures, and correction of bad hypotheses.

For every important claim, cite evidence as A:L10-L18 or B:L20 for numbered log lines,
and A:T42 or B:T17 for trace sequence numbers. Prefer equivalent or unclear when evidence
does not support a meaningful difference. A shorter trajectory is not automatically better.

Issue:
<issue>
{problem}
</issue>

Trajectory A:
<trajectory-a>
{trajectory_a}
</trajectory-a>

Trajectory B:
<trajectory-b>
{trajectory_b}
</trajectory-b>

Return ONLY one JSON object matching this schema, without markdown fences or commentary:
{schema}
"""


def build_aggregate_prompt(
    *, factor: str, first_value: str, second_value: str, rows: list[dict]
) -> str:
    schema = json.dumps(AggregateJudgment.model_json_schema(), ensure_ascii=False)
    evidence = json.dumps(rows, ensure_ascii=False, indent=2)
    return f"""You are reviewing blind, per-instance process judgments after experimental labels were revealed.

Factor: {factor}
First value: {first_value}
Second value: {second_value}

Each trajectory's revealed factor value, source run, and official outcome are aligned in the
`trajectories` object; do not reinterpret `A` or `B` as the caller's original run order.
Use only the supplied judgments, mappings, operational metrics, and official outcomes. Describe the
observed effect in these runs, possible process mechanisms, counterexamples, interactions, alternative
explanations, and limitations. Do not claim a universal or causal effect. Official outcome is contextual
evidence and must not overwrite the blind process judgment.

Per-instance evidence:
{evidence}

Return ONLY one JSON object matching this schema, without markdown fences or commentary:
{schema}
"""


def build_repair_prompt(*, raw_response: str, validation_error: str, schema: dict) -> str:
    rendered_schema = json.dumps(schema, ensure_ascii=False)
    return f"""Repair the structure of an LLM judge response so it matches the required JSON schema.

Make only the minimum structural corrections required by the validation errors. Preserve all
preferences, explanations, confidence values, claims, and evidence meanings. Do not add new
trajectory evidence or reassess the original comparison.

Validation errors:
<validation-errors>
{validation_error}
</validation-errors>

Original response:
<original-response>
{raw_response}
</original-response>

Return ONLY one JSON object matching this schema, without markdown fences or commentary:
{rendered_schema}
"""
