# Mini-Fixture Evals

Behavioral evaluations that run the **real parent agent** against small,
self-contained scenarios and check what it actually did. They complement the
`tests/` suite: `tests/` verify component correctness (no LLM), while these
verify end-to-end agent *behavior* (with a live model).

Because each run calls a real model, these evals cost tokens and are
non-deterministic. The runner is a standalone script and is **not** collected by
pytest, so a normal `pytest` run never triggers it.

## Layout

```
evals/
  run_evals.py            # standalone runner
  scenarios/
    <name>/
      config.json         # prompt + expectations
      template/           # optional starting files copied into the workspace
  .runs/                  # per-run workspaces + reports (git-ignored)
```

## Running

Requires the same environment as `main.py` (`OPENROUTER_API_KEY`,
`OPENROUTER_BASE_URL`, `MODEL_ID`; see the project README).

```bash
# List scenarios
./.venv/bin/python evals/run_evals.py --list

# Run all scenarios
./.venv/bin/python evals/run_evals.py

# Run one scenario, overriding the model
./.venv/bin/python evals/run_evals.py --scenario 01-create-file --model moonshotai/kimi-k2.5:exacto

# Keep workspaces for passing scenarios too (failures are always kept)
./.venv/bin/python evals/run_evals.py --keep-workspaces
```

The process exits non-zero if any scenario fails. Reports are written to
`evals/.runs/<timestamp>/` (`report.json` + `summary.md`).

## How a scenario runs

For each scenario the runner:

1. Creates a fresh workspace under `evals/.runs/<timestamp>/<name>/workspace`
   and copies in `template/` (if present).
2. Points the tools' `WORKDIR` at that workspace, resets the shared `todo`
   state, and installs a scenario-scoped permission service with an
   **auto-approve** handler (so writes/shell run headlessly). Hard denials in
   the permission layer still apply.
3. Runs the parent agent loop in-process on the scenario `prompt`.
4. Extracts the tool-call trajectory from the loop history and the agent's
   final message, then evaluates the `expect` block.

A scenario passes only if **all** of its expectations hold.

## config.json format

```json
{
  "name": "human-readable name",
  "prompt": "the user message sent to the agent",
  "timeout": 180,
  "expect": {
    "files_exist": ["path/relative/to/workspace"],
    "file_contains": [{ "file": "hello.txt", "contains": "substring" }],
    "tools_used": [{ "name": "write_file", "args_contains": { "path": "hello.txt" } }],
    "tools_not_used": ["edit_file"],
    "final_answer_contains": ["substring in the agent's closing message"]
  }
}
```

- `prompt` is required; everything under `expect` is optional, but a scenario
  with no expectations is reported as an error (nothing to assert).
- `args_contains` is a substring match against the stringified tool argument.
- `timeout` is in seconds; omit for no timeout.

## Adding a scenario

1. `mkdir evals/scenarios/<NN-name>`.
2. Write `config.json` with a `prompt` and an `expect` block.
3. Add any starting files under `template/`.
4. Verify: `./.venv/bin/python evals/run_evals.py --scenario <NN-name>`.

## Notes / limitations (by design)

- Single run per scenario (no `pass@k` / multi-trial metrics yet).
- No LLM-as-judge; assertions are deterministic (files, tool trajectory, text).
- No external-codebase / Docker layer. These can be added as the project
  matures.
