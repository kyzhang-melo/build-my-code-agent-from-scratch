# myCodeAgent-v0

This project is a code agent built from scratch. I built it to understand how agents like Claude Code and Codex actually work under the hood. Actually the core of an LLM agent is just a loop: read a message, decide whether to call a tool, run it, feed the result back, repeat — plus the file/search/shell tools, session handling, permission checks, and persistence needed to make that loop usable on a real project.

I kept the core (agent loop, tool registry, session wiring) decoupled from any specific tool set or prompt, so it's not just a code agent. This project can be extended into other types of agents (a data agent, a work agent, an assistant agent etc.) by swapping in different tools and system prompts.

The funniest part of building an LLM agent from scratch is that, since LLMs are very good at writing code, you can build a code agent first and then let it modify itself (a.k.a. self-hosting, or self-evolving). This agent has been used to build and debug itself, some of the commits in this repo were made by the agent running on its own code.

## Files

**Core agent** — the loop itself: read a message, decide on a tool call, run it, feed the result back.

- `main.py`: agent loop, OpenAI client initialization, CLI entrypoint.
- `tools.py`: tool schema and shell tool execution logic.
- `prompts.py`: system prompt definition.
- `message_utils.py`: message protocol adapter helpers and resume sanitization.

**Supporting infrastructure** — plumbing the core loop depends on but that isn't the loop itself.

- `workspace.py`: `Workspace` value object with path-escape check.
- `context_compact.py`: token estimation and conversation-history compaction.
- `trace.py`: lightweight runtime trace events and in-memory/JSONL sinks.

**User experience** — permissions and session handling that make the agent safe and pleasant to run interactively.

- `permissions.py`: workspace safety policy and approval handling.
- `session.py`: `AgentSession` dataclass and stop-gate implementations.
- `session_store.py`: append-only JSONL session persistence and resume.

**Eval-system support** — infrastructure that exists mainly to run the evaluation harness described below.

- `grep_engine.py`: portable ripgrep/Python search engine for local and container backends.
- `file_engine.py`: shared stdlib-only read/write/edit/glob semantics for all backends.
- `file_bridge.py`: stdlib-only JSON bridge for solve-container file operations.
- `sandbox.py`: session-scoped local/Docker execution backends and solve-container lifecycle.

**Eval system**

- `evals/`: mini-fixture, SWE-bench, warm-context, analysis, and process-judge
  evaluation workflows.

Read-only access outside the workdir requires an absolute path; write operations
remain restricted to the workdir.

## Requirements

- Python 3.10+
- `openai`
- `prompt-toolkit==3.0.52`
- `python-dotenv`
- `pytest`

Install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai "prompt-toolkit==3.0.52" python-dotenv
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

`OPENROUTER_PROVIDER` is optional, but recommended when testing context-window behavior through OpenRouter. When it is set, the agent pins requests to that provider and disables fallbacks, so the effective context window stays consistent across turns and compaction side calls.

When using a pinned OpenRouter provider, add the `:exacto` suffix to `MODEL_ID`. This tells OpenRouter to route the model request exactly as specified instead of silently falling back to another route. For example:

```env
MODEL_ID="moonshotai/kimi-k2.5:exacto"
OPENROUTER_PROVIDER="moonshotai/int4"
```

Leave `OPENROUTER_PROVIDER` empty if you want OpenRouter's default routing.

### Model Choice and Context Limits

For debugging this agent harness, prefer `kimi-k2.5`, `hy3`, `deepseek-v4-flash` or another model with strong agentic capability. Models with weaker tool-use and long-context behavior may fail on prompt cases that require reading several files and summarizing the results.

Also check the model-limit detection logic in `main.py` when changing `MODEL_ID`. The project resolves known model IDs to a total context window, maximum input, and (when known) maximum output. For example, hy3 has a 256k total context window but accepts at most 192k input tokens, so compaction uses the lower input limit. Unknown models fall back to a conservative 32k limit.

For one run, `MODEL_LIMITS_OVERRIDE` can replace the catalog with a complete JSON value. All three fields are required so the input/output budget remains consistent:

```env
MODEL_LIMITS_OVERRIDE='{"context_window_tokens":262144,"max_input_tokens":192000,"max_output_tokens":128000}'
```

## Run

```bash
python main.py
```

On macOS, iTerm2 is the recommended terminal for running `python main.py`.

Type your request at `s01 >>`.

- `q`, `exit`, or empty input will quit.

Agent generation leaves both reasoning effort and the output-token limit unspecified by default, so the provider chooses them. Set either field explicitly for one CLI run:

```bash
python main.py --reasoning-effort high --max-output-tokens 16000
```

Omit `--reasoning-effort` to leave the reasoning effort unspecified. Supported explicit values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`; the selected model and provider must support the chosen value. `none` sends `reasoning: {"effort": "none"}`, whereas omitting the option leaves the API field unset.

Omit `--max-output-tokens` to leave the API field unset. Passing `--max-output-tokens 8000` sends an explicit limit of 8,000. When the field is unset, context compaction reserves up to 32,768 output tokens, capped at half of the context window for smaller or unknown models. If a model has a known maximum output limit, an explicit value above that limit is rejected locally.

### Session Persistence

By default, each session is persisted as a logically append-only JSONL log in `~/.mycodeagent/sessions/--<encoded-workspace>--/<session_id>.jsonl`, outside the workspace. The file records every completed turn's history and todo state; physical writes use atomic file replacement so compaction and crashes cannot leave a half-rewritten file.

Set `MYCODEAGENT_SESSION_DIR` or pass `--session-dir <dir>` to use an explicit session directory. The CLI flag takes precedence over the environment variable; either override is the final directory and is not extended with an encoded workspace name.

CLI flags:

```bash
python main.py                    # new session (default)
python main.py --name kevin       # new session with a human-readable name
python main.py --continue         # resume the most recent session
python main.py --resume <target>  # resume by name, id, or path
python main.py --list-sessions    # list saved sessions and exit
python main.py --no-session       # disable persistence for this run
python main.py --session-dir /tmp/my-sessions
```

In-session commands:

- `/copy` — copy the most recent model response to the system clipboard.
- `/sessions` — list saved sessions in the current workspace.
- `/approval` — toggle human approval on/off. Enter it once to disable approval
  prompts (the agent runs autonomously); enter it again to restore interactive
  approval. Core safety guards (workspace escapes, hard-denied shell commands,
  sensitive files) still apply when auto-approval is on.

Resume behavior:

- The session's cwd must match the current working directory. A mismatch is rejected (the session header's cwd is not adopted, since that would let a
  disk file determine the workspace and permission boundary).
- Permission mode and session-level approvals are not restored.
- When the model or pinned provider has changed, `reasoning` items and provider-assigned `function_call.id` values are dropped during load. `call_id` is retained so tool-call/output pairing stays intact.
- Unpaired function calls and outputs, including partially completed parallel tool batches, are removed before replay. A safe diagnostic count is printed whenever resume sanitization changes the loaded history.
- Todo state, including an explicitly cleared plan, is restored so `TodoStopGate` remains consistent after resume.
- A turn interrupted before `agent_loop` completes is discarded; resume starts from the last completed turn, avoiding accidental replay of tool side effects.
- A per-session lock enforces one CLI writer at a time. A second process trying to resume the same active session is rejected.
- Legacy workspace-local `.sessions` directories are not searched automatically. Resume one explicitly with `python main.py --resume .sessions/<session_id>.jsonl`; it continues writing to that legacy file.

Session names are optional, case-insensitively unique within the workspace, and do not replace the immutable session id. `last`, `continue`, and `new` are reserved names.

External session files are outside the workspace permission boundary. Legacy `.sessions` and workspace-local pre-compaction transcript files remain treated as sensitive runtime state: file reads/writes and shell access are blocked, and search tools exclude their directories.

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

The runtime trace records structured facts at the agent's execution boundaries. It captures session start/end, requested and completed tool calls, permission decisions, todo transitions, stop-gate decisions, and per-model-call usage. `llm.usage` events retain provider-reported cost and token details when available, including cached, cache-write, and reasoning tokens. Calls are attributed to the parent agent or the explore subagent; missing provider fields remain `null` rather than being reported as zero. Other trace events contain safe metadata such as paths, modes, counts, durations, and statuses; they do not retain full file contents, shell commands, edit text, task prompts, or tool output.

Normal CLI runs use a no-op trace sink and do not create trace files. The eval runner executes the real parent agent in-process and injects an in-memory trace sink for each scenario. It combines those trace events with workspace and final answer assertions to verify end-to-end agent behavior. Each scenario receives a fresh conversation state, isolated workspace, todo state, and permission service.

Live-model evals require the same OpenRouter environment variables as `main.py`. They cost tokens and may be non-deterministic; the regular `pytest` suite never runs them.

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

Passing scenario workspaces are deleted by default, while failed workspaces are kept for debugging. Preserve every workspace with:

```bash
./.venv/bin/python evals/run_evals.py --keep-workspaces
```

Reports are written to `evals/.runs/<timestamp>/report.json` and `summary.md`. See [`evals/README.md`](evals/README.md) for the scenario format, trace-backed assertions, and implementation notes.

### Eval System

The repository has several deliberately separate evaluation layers:

- Mini-fixture evals exercise end-to-end agent behavior in small local workspaces.
- Micro evals isolate deterministic or paired harness decisions.
- Cold SWE-bench runs use Docker-based generation with a fresh session and disposable instance-image `/testbed` workspace per instance.
- Warm-context SWE-bench sequences preserve conversation context across an ordered set of instances while resetting a host-side Git workspace each time. This runner has not been migrated to Docker-based generation.
- Official SWE-bench grading remains the correctness signal. The optional LLM judge compares the recorded processes after eval completion; it does not replace outcome grading.

Start with [`evals/README.md`](evals/README.md), then see the focused guides for [`evals/swebench/`](evals/swebench/README.md), [`evals/swebench_sequence/`](evals/swebench_sequence/README.md), and [`evals/judge/`](evals/judge/README.md).

### SWE-bench

The cold SWE-bench adapter now uses Docker for both generation and official grading. During generation, the real parent agent runs inside each official instance image's prepared `/testbed`; the solve container has networking disabled, is removed after the attempt, and exports only the prediction patch and diagnostic artifacts. Cold runs no longer clone host-side Git mirrors or retain a repository workspace per attempt.

The cold eval tool profile includes workspace-bound file/search tools and `bash`, so the agent can inspect `/testbed` and review changes with `git diff`. The standalone `git_diff` tool is not included in this profile.

Run the complete generate, official-evaluate, and report pipeline with:

```bash
./.venv/bin/python evals/swebench/run_swebench.py run \
  --run-id verified-small-11-001 \
  --subset evals/swebench/subsets/small_11.json \
  --runs-dir /mnt/docker-data/swebench/runs \
  --reasoning-effort high \
  --agent-workers 2 \
  --max-workers 2
```

The former cold-runner options `--generate-environment` and `--repo-cache` have been removed. Docker generation is no longer optional, and cold runs no longer download or use host repository mirrors. Existing command lines must drop both options; Docker may still pull missing instance images, and an uncached SWE-bench dataset may still require a download.

`evals/swebench_sequence` is deliberately unchanged. Its warm-context runner still uses a host repository mirror and recreates/archives a local workspace for each episode, so its own `--repo-cache` option remains valid. Migration of that workflow to Docker-based generation is deferred.

These workflows call a live model and can run resource-intensive Docker evaluations. They are never included in pytest. See [`evals/swebench/README.md`](evals/swebench/README.md) for prerequisites, artifacts, budgets, and recovery behavior, and [`evals/swebench_sequence/README.md`](evals/swebench_sequence/README.md) for the separate warm-context workflow.

### Testing Conventions

- Write tests with `pytest` only; do not add standalone `if __name__ == "__main__"` test scripts.
- Group tests by behavior (`dispatcher`, `loop`, `path_safety`, `message_protocol`) instead of one-file-per-scenario.
- Reuse shared setup in `tests/conftest.py` instead of duplicating import/env bootstrapping.
- Mark expensive tests with `@pytest.mark.integration` or `@pytest.mark.slow`.
