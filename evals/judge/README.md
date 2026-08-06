# Process-based LLM Judge

This optional post-eval tool compares how two agents worked on the same
SWE-bench instances. Official SWE-bench results remain the correctness signal;
the judge evaluates localization, exploration, evidence use, editing,
verification, and recovery.

The first pass anonymizes and randomly orders each pair. It includes the issue,
agent log, structured tool trace, patch, and final text, but hides the target
configuration, official status, token usage, and cost. A second pass receives
the blind judgments after labels and official outcomes are revealed and writes
an experimental factor-level review.

## Run a comparison

Configure a judge model that is independent of the contestant model:

```bash
export JUDGE_MODEL_ID="<independent-judge-model>"
export JUDGE_PROVIDER=""
```

The CLI loads `.env` from the repository root. `--judge-model` takes precedence
over `JUDGE_MODEL_ID`, and `--judge-provider` takes precedence over
`JUDGE_PROVIDER`. The judge client currently uses `OPENROUTER_API_KEY` and
`OPENROUTER_BASE_URL`; a non-empty judge provider pins that provider and
disables fallback routing.

Compare reasoning settings in two cold runs:

```bash
./.venv/bin/python evals/judge/run_judge.py compare \
  --judge-run-id django-4.1-cold-reasoning-001 \
  --factor reasoning \
  --run-a evals/.runs/swebench/<default-run> \
  --run-b evals/.runs/swebench/<xhigh-run>
```

Compare cold and persisted-warm runs with the same reasoning setting:

```bash
./.venv/bin/python evals/judge/run_judge.py compare \
  --judge-run-id django-4.1-default-context-001 \
  --factor context \
  --run-a evals/.runs/swebench/<cold-run> \
  --run-b evals/.runs/swebench-sequence/<warm-run>
```

Artifacts are written under `evals/.runs/judge/<judge-run-id>/`. Successful
per-instance judgments are reused. Pass `--rerun-failed` to retry only failed,
invalid, missing-artifact, or oversized entries. Inputs larger than
`--max-input-chars` are marked `input_too_large`; the runner never silently
truncates or summarizes a trajectory.

`--judge-run-id` is a user-chosen experiment identifier and output-directory
name. Use a new ID when changing the judge model, compared runs, factor, or
other judge settings. An existing successful judgment is intentionally reused,
and the manifest rejects incompatible configuration rather than silently
mixing results. A descriptive convention is:

```text
<suite>-<factor>-<contestant-setting>-<judge-model>-<sequence>
django-4.1-context-default-qwen38-001
```

If a judge response fails schema validation, the runner makes one small repair
call containing only the invalid JSON, validation errors, and target schema. It
does not resend or reassess the trajectories. The original and repaired
responses remain available in the instance artifacts. Progress is printed as
each pair starts, completes, is reused, or fails, followed by the aggregate
stage.

The report's `Repaired responses` count therefore means that the initial model
output was not schema-valid, not that the trajectory was judged again. Common
causes include Markdown JSON fences, a misspelled or extra field, and an
evidence reference that is not in the required `A:L...` / `B:T...` form. A
successful repair preserves the initial response, validation error, repair
prompt, and repair response for audit. A high repair rate is still useful when
comparing judge models because it indicates weaker structured-output
reliability.

Judge generation currently uses provider defaults: the runner does not force a
temperature or an output-token cap. This is recorded in the judge manifest.

The A/B ordering prevents presentation-order bias; reversing `--run-a` and
`--run-b` does not create an independent experiment or remove run-level
confounding. Claims about a reasoning or context factor require independently
repeated eval runs for each factor value.

Judge output is marked `uncalibrated` until it has been checked against human
labels. It describes effects observed in the supplied runs and must not be
treated as proof of a universal or causal effect.

## Reading the report

- `completed` means the instance judgment passed validation, possibly after one
  schema repair.
- `repaired` counts schema corrections; it is not a process preference or an
  outcome change.
- `cold`, `warm`, or a reasoning value is the decoded factor preference after
  the blind A/B mapping is revealed.
- `equivalent` means the visible process evidence did not support a meaningful
  preference.
- Official outcomes are displayed alongside process preferences after judging;
  disagreement between them is expected and should be discussed, not hidden.
- `uncalibrated` means the complete judge configuration has not yet been
  validated against human labels. Confidence values are model assessments, not
  calibrated probabilities.

Calibration applies to the whole setup: judge model, rubric and prompt
versions, generation settings, and input construction. Useful checks include
human labels on representative pairs, repeated judgments, A/B position swaps,
and comparison of preference stability across independent judge models.
