# AGENTS.md

Guidelines for contributors and coding agents working in this repository.

## Project Goal

This project is a personal code-agent learning project. Its current core modules are:

- `main.py`: agent loop, CLI entrypoint, session factories (`create_parent_session` / `create_explore_session`), and tool dispatch wiring
- `session.py`: `AgentSession` dataclass and stop-gate implementations; one `AgentSession` owns every dependency a single agent run needs (workspace, todo, registry, permission service, trace context, stop gate, session store)
- `session_store.py`: append-only JSONL session persistence (`SessionStore` / `NullSessionStore`), resume sanitization delegation, `todo_state` entry support, and session listing
- `tools.py`: tool schemas, runtime validation/sanitization, file/search/shell tools, `TodoManager`, and `build_tool_registry` (a factory that binds a tool registry to one `Workspace` + `TodoManager`)
- `workspace.py`: `Workspace` value object — a resolved root directory plus the workspace-escape check, frozen so it cannot change under a running session
- `permissions.py`: workspace safety checks, permission decisions, and terminal approval handling
- `message_utils.py`: Responses API message normalization, response-item adapters, and resume-time sanitization (`sanitize_resumed_message`, `drop_orphan_tool_calls`)
- `prompts.py`: system-prompt factory functions (`build_parent_system`, `build_explore_system`) that render per workspace
- `context_compact.py`: token estimation, transcript snapshots, and conversation-history compaction
- `trace.py`: lightweight runtime events and trace sinks used to observe tool, permission, todo, and stop-gate behavior
- `evals/`: live-model behavioral evaluations that run isolated scenarios, consume runtime traces, assert workspace and agent outcomes, and generate JSON/Markdown reports
- `tests/`: behavior-focused pytest coverage for the loop, tools, permissions, paths, message protocol, session isolation, and session persistence

The eval module complements the deterministic pytest suite with manually invoked,
end-to-end checks against a live model. Each scenario builds a fresh `AgentSession`
via `create_parent_session` so it gets its own conversation, workspace, todo state,
permission service, registry, and in-memory trace sink; behavioral regressions can
be attributed to one isolated task. Because these evaluations consume real model
tokens and are non-deterministic, they must remain separate from the default test
suite.

Keep changes simple and educational.

## Design Rules

1. Preserve module boundaries.
- Keep orchestration in `main.py`.
- Keep tool definitions/runtime execution in `tools.py`.
- With the evolution of this project, the `tools.py` could be divided into multiple tool scripts and placed in a folder named `tools`.
- Keep message protocol adapters and resume sanitization in `message_utils.py`.
- Keep prompt text in `prompts.py`.
- Keep session-scoped dependency assembly in `session.py`.
- Keep session persistence (file format, entry model, resume loading) in `session_store.py`.

2. No module-level mutable globals for session state.
- A session's workspace, todo, tool registry, permission service, trace context, stop gate, and session store live on one `AgentSession` instance, never on module-level singletons.
- Tool registries are built per session by `build_tool_registry(workspace, todo)`; do not reintroduce a module-level `TOOL_REGISTRY`.
- File/search/shell tools take an explicit `Workspace` parameter; do not reintroduce a module-level `WORKDIR` or `safe_path`.
- System prompts are built per workspace by factory functions in `prompts.py`; do not reintroduce module-level prompt constants that callers string-replace.
- `context_compact.compact_history_async` receives the `TodoManager` as a parameter; do not import a global `TODO`.
- Session stores are built per session (`SessionStore.create` / `NullSessionStore`); tests and evals use `NullSessionStore` to avoid filesystem side effects.

3. Prefer small, reversible changes.
- One behavior change per commit.
- Avoid large refactors unless explicitly requested.

4. Use OpenAI-compatible API style consistently.
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

1. `./.venv/bin/python -m py_compile main.py tools.py prompts.py message_utils.py session.py session_store.py workspace.py`
2. `./.venv/bin/python -m pytest -q`
3. Run `python main.py`
4. Verify:
- one normal tool path (e.g. `bash`/`read_file`)
- one todo path (plan creation + stdout visibility)
- one no-tool finalization path
- one session persistence path (run a turn, exit, `--continue`, verify history is restored)

`tests/test_session_isolation.py` is the regression guard for the no-globals
rule: it verifies two `AgentSession` instances share no mutable workspace, todo,
registry, permission service, or trace context. If a change reintroduces a
module-level singleton captured by `create_parent_session`, one of those tests
should fail.

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
