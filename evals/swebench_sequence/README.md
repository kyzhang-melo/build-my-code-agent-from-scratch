# Warm-context SWE-bench sequence eval

This eval runs a chronological list of issues from one repository. Each issue
starts from its own clean SWE-bench base commit, so code changes never leak
between instances. The agent session is closed and reopened from an external
JSONL store between episodes, allowing conversation and todo context to carry
forward through the same persistence path used by the product.

Completed and API-budget-exhausted episodes commit their history. A timeout or
agent error rolls back that episode's prompt, messages, and todo state; the next
episode resumes from the last committed boundary. There is no test-result or
grader feedback in the history.

Run the curated Django sequence:

```bash
./.venv/bin/python evals/swebench_sequence/run_swebench_sequence.py run \
  --run-id django-db-models-warm-001 \
  --subset evals/swebench/subsets/django_db_models_6.json
```

Generation runs are immutable. If generation stops midway, retain its artifacts
for diagnosis and start again with a new run ID. The `evaluate` and `report`
phases may be run separately after generation has completed.

For a cold-start comparison, pass the same ordered subset to the original
runner. That runner still creates an independent session for every issue:

```bash
./.venv/bin/python evals/swebench/run_swebench.py run \
  --run-id django-db-models-cold-001 \
  --subset evals/swebench/subsets/django_db_models_6.json
```

Warm artifacts live under `evals/.runs/swebench-sequence/<run-id>/`. The stable
workspace path is recreated for every episode and then archived under the
corresponding `instances/.../attempt-1/workspace`. Session persistence remains
separate at `<run-id>/session/warm-context-sequence.jsonl`.
