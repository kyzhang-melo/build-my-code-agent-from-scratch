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

Create a `.env` file in this directory:

```env
OPENROUTER_API_KEY="your_key"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
MODEL_ID="minimax/minimax-m2.5"
```

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
