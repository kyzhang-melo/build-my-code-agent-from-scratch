# myCodeAgent-v0

A learning project that refactors a monolithic code-agent loop into a multi-file structure.

## Files

- `main.py`: agent loop, OpenAI client initialization, CLI entrypoint.
- `tools.py`: tool schema and shell tool execution logic.
- `prompts.py`: system prompt definition.
- `message_utils.py`: message protocol adapter helpers.

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

For debugging this agent harness, prefer `kimi-k2.5` or another model with
strong agentic capability. Models with weaker tool-use and long-context behavior
may fail on prompt cases that require reading several files and summarizing the
results.

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

## Testing

Run the default fast suite:

```bash
pytest
```

Run all tests including marked ones:

```bash
pytest -m "integration or slow or not (integration or slow)"
```

### Testing Conventions

- Write tests with `pytest` only; do not add standalone `if __name__ == "__main__"` test scripts.
- Group tests by behavior (`dispatcher`, `loop`, `path_safety`, `message_protocol`) instead of one-file-per-scenario.
- Reuse shared setup in `tests/conftest.py` instead of duplicating import/env bootstrapping.
- Mark expensive tests with `@pytest.mark.integration` or `@pytest.mark.slow`.
