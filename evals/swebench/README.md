# SWE-bench eval

This adapter runs the real `myCodeAgent-v0` parent agent against Git
repositories whose history ends at each task's base commit, exports the
resulting repository changes, and delegates grading to the unmodified official
SWE-bench Docker harness. Generation remains host-local by default. Pass
`--generate-environment docker` to opt into the experimental per-attempt solve
container backend.

The eval agent uses a restricted tool profile: workspace-bound file tools,
`todo`, the read-only exploration subagent, and a parameter-free `git_diff`.
It does not receive the general-purpose `bash` tool. In Docker generation mode,
the workspace is mounted at `/testbed`, networking is disabled, and `grep`
executes inside the solve container. The remaining file tools still operate on
the same bind-mounted workspace from the host and will migrate incrementally.

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

Agent calls leave reasoning effort and the output-token limit unspecified by
default. Set either field explicitly when a run needs a fixed generation
configuration:

```bash
./.venv/bin/python evals/swebench/run_swebench.py run \
  --run-id verified-small-glm-5.2-high-16k-001 \
  --subset evals/swebench/subsets/small.json \
  --reasoning-effort high \
  --max-output-tokens 16000
```

The effective settings, including resolved model context/input/output limits,
are recorded in the run manifest. Resuming the same run ID with different
settings is rejected. A compatible older manifest is upgraded once by recording
the limits resolved for the resumed run and an auditable old/new harness-commit
migration record. Later resumes again require the recorded harness commit.

Use `--reasoning-effort none` to explicitly send
`reasoning: {"effort": "none"}`. Omitting the option leaves the API field
unset and lets the provider choose its default.

Omitting `--max-output-tokens` omits the API field and records `null` in the
manifest. Passing `--max-output-tokens 8000` sends and records an explicit
8,000-token limit. Provider-default mode keeps a 32,768-token context
reservation when the model context permits, capped at half of smaller context
windows.

Agent instances have no overall timeout by default. Use
`--instance-timeout <seconds>` only when a run needs an explicit safety limit.
This setting applies to the agent loop and solve container, not the official
Docker test timeout.

Generation uses staged steering by default. After model calls 45 and 60, the
eval injects a session-local instruction to converge and deliver a patch; the
second instruction distinguishes workspaces with and without changes. The hard
limit defaults to 90 calls. Use `--steering-policy none` for an unsteered
control run, or `--max-api-calls` to override the hard limit. Steering applies
only to the parent SWE-bench agent, not ordinary CLI sessions, exploration
subagents, or sequence evals. The policy and thresholds are recorded in the run
manifest.

The command generates predictions, starts the official Docker evaluator only
when at least one prediction exists, and then writes the merged report. It
preserves per-instance failures and continues the batch, so the same command
can be run again with `--rerun-failed` after correcting a problem.

Patch generation runs in five isolated agent processes by default. Override it
with `--agent-workers`. Official Docker evaluation also uses five workers by
default and can be changed independently with `--max-workers`. The two pools
run in separate pipeline phases, so their resource usage does not overlap.

Instance images are retained by default so later evaluations can reuse them
without downloading them again. This can consume substantial Docker disk
space. To discard newly downloaded instance images after an evaluation, use
`--cache-level env` with either `run` or `evaluate`.

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

### Pre-pulling evaluation images

The `evaluate` phase downloads a Docker image for each instance on demand. When
running the full 500-instance set, this can be slow and may hit Docker Hub rate
limits (`429 Too Many Requests`). The `prepare_remote_images.py` helper pulls all
required images before the evaluator starts and re-tags them to the names the
SWE-bench harness expects.

Dry-run to see what would be pulled:

```bash
./.venv/bin/python evals/swebench/prepare_remote_images.py --dry-run
```

Pull from Docker Hub (default, four concurrent workers):

```bash
./.venv/bin/python evals/swebench/prepare_remote_images.py
```

Pull from Epoch AI's GHCR mirror to avoid Docker Hub rate limits. Reduce
`--max-workers` if pulling to a slow disk, such as an SMR HDD:

```bash
./.venv/bin/python evals/swebench/prepare_remote_images.py \
  --source ghcr \
  --max-workers 1
```

`--source ghcr` pulls from `ghcr.io/epoch-research/swe-bench.eval.*` and re-tags
to `swebench/sweb.eval.*` so the official harness can use them unchanged.
`--subset` defaults to `evals/swebench/subsets/verified_500.json`.
`--max-workers` controls the number of concurrent `docker pull` operations, and
`--force` re-pulls images that are already present locally.

Merge the agent and official results:

```bash
./.venv/bin/python evals/swebench/run_swebench.py report \
  --run-id verified-smoke-001
```

The `run` pipeline and the standalone `report` command automatically generate
Phase-3 harness diagnostics after the ordinary SWE-bench summary succeeds:

```text
harness-diagnostic.json       # structured facts for this run
harness-diagnostic.md         # human-readable per-run diagnostic report
harness-diagnostic-diff.json  # structured comparison with the previous run
harness-diagnostic-diff.md    # new, disappeared, and changed diagnostic signals
```

A previous run is used only when dataset, split, exact instance set, model,
provider, resolved model limits, API-call budget, reasoning effort, and
output-token configuration all match; timeout and auto-compaction settings must
match as well. Harness commits
are deliberately allowed to differ so a harness change can be measured. If no
like-for-like baseline exists, the diff says so instead of comparing unrelated
runs. Dirty runs remain visible with a warning and should not independently be
used to announce a regression.

Automatic diagnostic failures are printed as warnings and do not replace or
invalidate the official SWE-bench result. The original trace and report remain
available for repair and rerunning the `report` phase.

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
