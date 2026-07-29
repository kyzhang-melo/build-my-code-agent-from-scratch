# myCodeAgent-v0

A learning project that refactors a monolithic code-agent loop into a multi-file structure.

## Files

- `main.py`: agent loop, OpenAI client initialization, CLI entrypoint.
- `tools.py`: tool schema and shell tool execution logic.
- `permissions.py`: workspace safety policy and approval handling.
- `trace.py`: lightweight runtime trace events and in-memory/JSONL sinks.
- `prompts.py`: system prompt definition.
- `message_utils.py`: message protocol adapter helpers and resume sanitization.
- `session.py`: `AgentSession` dataclass and stop-gate implementations.
- `session_store.py`: append-only JSONL session persistence and resume.
- `workspace.py`: `Workspace` value object with path-escape check.
- `context_compact.py`: token estimation and conversation-history compaction.
- `evals/`: live-model behavioral scenarios, assertions, and reports.

## Requirements

- Python 3.10+
- `openai`
- `python-dotenv`
- `pytest`

Install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv
```

Or install from file:

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```env
OPENROUTER_API_KEY="your_key"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
MODEL_ID="moonshotai/kimi-k2.5:exacto"
OPENROUTER_PROVIDER="moonshotai/int4"
```

`OPENROUTER_PROVIDER` is optional, but recommended when testing context-window
behavior through OpenRouter. When it is set, the agent pins requests to that
provider and disables fallbacks, so the effective context window stays
consistent across turns and compaction side calls.

When using a pinned OpenRouter provider, add the `:exacto` suffix to `MODEL_ID`.
This tells OpenRouter to route the model request exactly as specified instead of
silently falling back to another route. For example:

```env
MODEL_ID="moonshotai/kimi-k2.5:exacto"
OPENROUTER_PROVIDER="moonshotai/int4"
```

Leave `OPENROUTER_PROVIDER` empty if you want OpenRouter's default routing.

### Model Choice and Context Window

For debugging this agent harness, prefer `kimi-k2.5`, `tencent/hy3`, or another
model with strong agentic capability. Models with weaker tool-use and
long-context behavior may fail on prompt cases that require reading several
files and summarizing the results.

Also check the context-window detection logic in `main.py` when changing
`MODEL_ID`. The project resolves known model IDs to their configured context
window sizes there. If a model is not recognized, the agent falls back to the
default context window of `32k`, which can trigger compaction much earlier than
expected.

## Run

```bash
python main.py
```

Type your request at `s01 >>`.

- `q`, `exit`, or empty input will quit.

### Session Persistence

By default, each session is persisted as a logically append-only JSONL log in
`.sessions/<session_id>.jsonl` within the workspace. The file records every
completed turn's history and todo state; physical writes use atomic file
replacement so compaction and crashes cannot leave a half-rewritten file.

CLI flags:

```bash
python main.py                    # new session (default)
python main.py --name kevin       # new session with a human-readable name
python main.py --continue         # resume the most recent session
python main.py --resume <target>  # resume by name, id, or path
python main.py --list-sessions    # list saved sessions and exit
python main.py --no-session       # disable persistence for this run
```

In-session command:

- `/sessions` — list saved sessions in the current workspace.

Resume behavior:

- The session's cwd must match the current working directory. A mismatch is
  rejected (the session header's cwd is not adopted, since that would let a
  disk file determine the workspace and permission boundary).
- Permission mode and session-level approvals are not restored.
- When the model or pinned provider has changed, `reasoning` items and
  provider-assigned `function_call.id` values are dropped during load.
  `call_id` is retained so tool-call/output pairing stays intact.
- Unpaired function calls and outputs, including partially completed parallel
  tool batches, are removed before replay. A safe diagnostic count is printed
  whenever resume sanitization changes the loaded history.
- Todo state, including an explicitly cleared plan, is restored so
  `TodoStopGate` remains consistent after resume.
- A turn interrupted before `agent_loop` completes is discarded; resume starts
  from the last completed turn, avoiding accidental replay of tool side
  effects.
- A per-session lock enforces one CLI writer at a time. A second process trying
  to resume the same active session is rejected.

Session names are optional, case-insensitively unique within the workspace, and
do not replace the immutable session id. `last`, `continue`, and `new` are
reserved names.

Session and pre-compaction transcript files are treated as sensitive runtime
state: file reads/writes and shell access are blocked, and search tools exclude
their directories.

## Testing

Run the default fast suite:

```bash
pytest
```

Run all tests including marked ones:

```bash
pytest -m "integration or slow or not (integration or slow)"
```

### Runtime Trace and Behavioral Evals

The runtime trace records structured facts at the agent's execution boundaries.
It captures session start/end, requested and completed tool calls, permission
decisions, todo transitions, and stop-gate decisions. Trace events contain safe
metadata such as paths, modes, counts, durations, and statuses; they do not
retain full file contents, shell commands, edit text, task prompts, or tool
output.

Normal CLI runs use a no-op trace sink and do not create trace files. The eval
runner executes the real parent agent in-process and injects an in-memory trace
sink for each scenario. It combines those trace events with workspace and final
answer assertions to verify end-to-end agent behavior. Each scenario receives a
fresh conversation state, isolated workspace, todo state, and permission
service.

Live-model evals require the same OpenRouter environment variables as
`main.py`. They cost tokens and may be non-deterministic; the regular `pytest`
suite never runs them.

List available scenarios:

```bash
./.venv/bin/python evals/run_evals.py --list
```

Run all scenarios:

```bash
./.venv/bin/python evals/run_evals.py
```

Run one scenario or override the model:

```bash
./.venv/bin/python evals/run_evals.py --scenario 01-create-file
./.venv/bin/python evals/run_evals.py \
  --scenario 01-create-file \
  --model moonshotai/kimi-k2.5:exacto
```

Passing scenario workspaces are deleted by default, while failed workspaces are
kept for debugging. Preserve every workspace with:

```bash
./.venv/bin/python evals/run_evals.py --keep-workspaces
```

Reports are written to `evals/.runs/<timestamp>/report.json` and
`summary.md`. See [`evals/README.md`](evals/README.md) for the scenario format,
trace-backed assertions, and implementation notes.

### SWE-bench

The SWE-bench adapter runs the real parent agent in isolated host-side Git
worktrees, exports its changes as official prediction patches, and delegates
grading to an unmodified SWE-bench Docker harness:

```bash
./.venv/bin/python evals/swebench/run_swebench.py generate \
  --run-id verified-smoke-001 \
  --subset evals/swebench/subsets/smoke.json
./.venv/bin/python evals/swebench/run_swebench.py evaluate \
  --run-id verified-smoke-001
./.venv/bin/python evals/swebench/run_swebench.py report \
  --run-id verified-smoke-001
```

This workflow calls a live model and may clone large repositories or run
resource-intensive Docker evaluations. It is never included in pytest. See
[`evals/swebench/README.md`](evals/swebench/README.md) for prerequisites,
artifacts, budgets, and recovery behavior.

### Testing Conventions

- Write tests with `pytest` only; do not add standalone `if __name__ == "__main__"` test scripts.
- Group tests by behavior (`dispatcher`, `loop`, `path_safety`, `message_protocol`) instead of one-file-per-scenario.
- Reuse shared setup in `tests/conftest.py` instead of duplicating import/env bootstrapping.
- Mark expensive tests with `@pytest.mark.integration` or `@pytest.mark.slow`.
