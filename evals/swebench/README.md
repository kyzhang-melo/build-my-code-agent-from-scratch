# SWE-bench eval

This adapter runs the real `myCodeAgent-v0` parent agent in host-side Git
worktrees, exports the resulting repository changes, and delegates grading to
the unmodified official SWE-bench Docker harness.

Live runs call a model, clone GitHub repositories, and may consume substantial
time, tokens, disk, and Docker resources. They are separate from pytest.

## Prerequisites

- The official SWE-bench checkout and its own virtual environment.
- Docker Desktop running.
- The normal `myCodeAgent-v0` model environment variables.

Defaults expect sibling checkouts:

```text
/Users/xixi/myProject/
  myCodeAgent-v0/
  SWE-bench/
```

Override them with `--swebench-repo`, `--swebench-python`,
`SWEBENCH_REPO`, or `SWEBENCH_PYTHON`.

If Hugging Face is accessed through a SOCKS proxy, install SOCKS support in
the SWE-bench environment:

```bash
/Users/xixi/myProject/SWE-bench/.venv/bin/pip install "httpx[socks]"
```

## Run the curated smoke set

Run the complete pipeline:

```bash
./.venv/bin/python evals/swebench/run_swebench.py run \
  --run-id verified-smoke-001 \
  --subset evals/swebench/subsets/smoke.json
```

The command generates predictions, starts the official Docker evaluator only
when at least one prediction exists, and then writes the merged report. It
stops at the failed stage while preserving all artifacts, so the same command
can be run again after correcting an infrastructure problem.

Official evaluation runs with four workers by default. Override the concurrency
for a particular machine with `--max-workers`.

The individual phase commands remain available for recovery and debugging.

Generate a prediction:

```bash
./.venv/bin/python evals/swebench/run_swebench.py generate \
  --run-id verified-smoke-001 \
  --subset evals/swebench/subsets/smoke.json
```

Run the official Docker evaluator:

```bash
./.venv/bin/python evals/swebench/run_swebench.py evaluate \
  --run-id verified-smoke-001
```

Merge the agent and official results:

```bash
./.venv/bin/python evals/swebench/run_swebench.py report \
  --run-id verified-smoke-001
```

The smoke subset is curated only to validate the pipeline. Its result is not a
representative benchmark score.

## Artifacts and recovery

Runs are stored under `evals/.runs/swebench/<run-id>/`. Repository mirrors are
cached under `evals/.cache/swebench/repos/`; both locations are ignored by Git.

Each task attempt retains its workspace, trace, final text, patch, log, and
structured result. Existing results are skipped. To retry only failed,
timed-out, budget-exhausted, or empty-patch tasks while preserving the previous
attempt:

```bash
./.venv/bin/python evals/swebench/run_swebench.py generate \
  --run-id verified-smoke-001 \
  --subset evals/swebench/subsets/smoke.json \
  --rerun-failed
```

Use a new run ID when changing the model, budget, subset, harness commit, or
SWE-bench commit. A run manifest rejects incompatible resume attempts.
