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

If a judge response fails schema validation, the runner makes one small repair
call containing only the invalid JSON, validation errors, and target schema. It
does not resend or reassess the trajectories. The original and repaired
responses remain available in the instance artifacts. Progress is printed as
each pair starts, completes, is reused, or fails, followed by the aggregate
stage.

Judge generation currently uses provider defaults: the runner does not force a
temperature or an output-token cap. This is recorded in the judge manifest.

The A/B ordering prevents presentation-order bias; reversing `--run-a` and
`--run-b` does not create an independent experiment or remove run-level
confounding. Claims about a reasoning or context factor require independently
repeated eval runs for each factor value.

Judge output is marked `uncalibrated` until it has been checked against human
labels. It describes effects observed in the supplied runs and must not be
treated as proof of a universal or causal effect.
