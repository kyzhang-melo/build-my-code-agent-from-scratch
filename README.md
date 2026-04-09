# myCodeAgent-v0

A learning project that refactors a monolithic code-agent loop into a multi-file structure.

## Files

- `main.py`: agent loop, OpenAI client initialization, CLI entrypoint.
- `tools.py`: tool schema and shell tool execution logic.
- `prompts.py`: system prompt definition.

## Requirements

- Python 3.10+
- `openai`
- `python-dotenv`

Install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv
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
