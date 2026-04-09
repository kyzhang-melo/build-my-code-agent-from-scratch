# AGENTS.md

Guidelines for contributors and coding agents working in this repository.

## Project Goal

This project is a learning-focused code agent split into multiple modules:

- `main.py`: loop control and app entrypoint
- `tools.py`: tool schema and execution helpers
- `prompts.py`: system prompt definitions

Keep changes simple and educational.

## Design Rules

1. Preserve module boundaries.
- Keep orchestration in `main.py`.
- Keep tool definitions/execution in `tools.py`.
- Keep prompt text in `prompts.py`.

2. Prefer small, reversible changes.
- One behavior change per commit.
- Avoid large refactors unless explicitly requested.

3. Use OpenAI-compatible API style consistently.
- Keep request/response handling in one protocol style at a time.

## Safety Rules

1. Treat model tool input as untrusted.
- Validate parsed JSON arguments.
- Handle malformed input gracefully.

2. Keep shell execution guarded.
- Maintain dangerous-command checks.
- Keep timeout and output truncation.

3. Never expose secrets.
- Do not print full API keys.
- Use `.env` for local credentials.

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

1. `python -m py_compile main.py tools.py prompts.py`
2. Run `python main.py`
3. Verify at least one query with no tool call and one with a tool call

## Git Workflow

1. Check status: `git status`
2. Stage intended files only
3. Commit with clear message
4. Push to your branch

## Out of Scope (Unless Requested)

- Switching provider protocols mid-change
- Introducing new frameworks
- Rewriting architecture beyond this learning split
