# AGENTS.md

Guidelines for contributors and coding agents working in this repository.

## Project Goal

This project is a personal code-agent learning project. Its current core modules are:

- `main.py`: agent loop, CLI entrypoint, Agent configuration, and tool dispatch wiring
- `tools.py`: tool schemas, runtime validation/sanitization, file/search/shell tools, and todo manager
- `permissions.py`: workspace safety checks, permission decisions, and terminal approval handling
- `message_utils.py`: Responses API message normalization and response-item adapters
- `prompts.py`: parent-agent and exploration-subagent system prompts
- `context_compact.py`: token estimation, transcript snapshots, and conversation-history compaction
- `trace.py`: lightweight runtime events and trace sinks used to observe tool, permission, todo, and stop-gate behavior
- `evals/`: live-model behavioral evaluations that run isolated scenarios, consume runtime traces, assert workspace and agent outcomes, and generate JSON/Markdown reports
- `tests/`: behavior-focused pytest coverage for the loop, tools, permissions, paths, and message protocol

The eval module complements the deterministic pytest suite with manually invoked,
end-to-end checks against a live model. Each scenario uses a fresh conversation,
workspace, todo state, permission service, and in-memory trace sink so behavioral
regressions can be attributed to one isolated task. Because these evaluations
consume real model tokens and are non-deterministic, they must remain separate
from the default test suite.

Keep changes simple and educational.

## Design Rules

1. Preserve module boundaries.
- Keep orchestration in `main.py`.
- Keep tool definitions/runtime execution in `tools.py`.
- With the evolution of this project, the `tools.py` could be divided into multiple tool scripts and placed in a folder named `tools`.
- Keep message protocol adapters in `message_utils.py`.
- Keep prompt text in `prompts.py`.

2. Prefer small, reversible changes.
- One behavior change per commit.
- Avoid large refactors unless explicitly requested.

3. Use OpenAI-compatible API style consistently.
- Keep request/response handling in one protocol style at a time.

## Safety Rules

1. Treat model tool input as untrusted.
- Validate parsed JSON arguments.
- Validate required/optional fields and primitive types.
- Apply only conservative sanitization (leading prompt markers like `>`, `$`, `#`).
- Handle malformed input gracefully.

2. Keep shell execution guarded.
- Maintain dangerous-command checks.
- Keep timeout and output truncation.

3. Never expose secrets.
- Do not print full API keys.
- Use `.env` for local credentials.

4. Keep todo execution contract lightweight.
- If an active todo plan exists, do not silently finalize unresolved work.
- Allow completion after all todo items are done or explicit todo rewrite.
- Keep contract nudges bounded to avoid infinite loops.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv
python main.py
```

## Expected Environment Variables

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL`
- `MODEL_ID`

## Testing Checklist

Before committing:

1. `./.venv/bin/python -m py_compile main.py tools.py prompts.py message_utils.py`
2. `./.venv/bin/python -m pytest -q`
3. Run `python main.py`
4. Verify:
- one normal tool path (e.g. `bash`/`read_file`)
- one todo path (plan creation + stdout visibility)
- one no-tool finalization path

## Git Workflow

1. Check status: `git status`
2. Stage intended files only
3. Commit with clear message
4. Push to your branch

## Out of Scope (Unless Requested)

- Switching provider protocols mid-change
- Introducing new frameworks
- Rewriting architecture beyond this learning split
- Expanding runtime policy into a heavy framework/plugin system

DO NOT send optional commentary
