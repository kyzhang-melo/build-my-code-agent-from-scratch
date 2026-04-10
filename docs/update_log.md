# update_log

## Apr 9

### What I've done

- Build up the baseline, a simple code agent loop with one single tool (run_bash)
- Organize the project file structure into `main.py`, `prompts.py`, and `tools.py`
- Use `run_one_turn()` function control the agent step
- The loop trajectory is `messages -> model -> tool_result -> next_turn`

### Why

- Build a minimal code agent and the init version of the project structure. It's a good start for evolution.

## Apr 10

### What I've done

- Add features: multi-tools, tool_dispatcher, message adapter before API call
- Add pytest toolkits
- The two utility functions, `extract_text()` and `normalize_messages()`, which act as adapters, have been separated from main.py and made into a separate file `message_utils.py`
- Pytest: 9 passed

### Why

- Evolve this agent from single tool to multiple tools
- Normalize messages to reduce malformed-history API errors
- Develop with tests
- Keep the project structure