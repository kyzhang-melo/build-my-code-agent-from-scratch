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

## Apr 24

### What I've done

- Add features: todo write.
- Upgrade the tool dispatcher to tool runtime.
- Using the `TodoManager` as a minimal orchestrator. The agent orchestrator is responsible for updating and displaying the todo items list, and reminding the agent to refresh the todo items list.
- Upgrade the logic of the `run_one_turn` function, made checking a snapshot of the "todo items" list an essential operation.

### Why

- Enhance the code agent's ability to execute long-step tasks, plan before execution.
- The complexity of the code agent has stepped up a notch, and the harness engineering is beginning to come into play.

## May 15

### What I've done

- Refactor the definition of class `PlanItem`, `PlanningState`, `TodoManager` from dataclass decorator to Pydantic, and add the definition of the class `TodoParams`.

### Why

- Since the todo items are made by the llm model, and the output of the llm model is nondeterminism. So the validation of the todo items is needed before update the list of todo items. Using Pydantic is more suitable than dataclass decorator.

## May 22

### What I've done

- Refactor the code structure of `main.py`, use `TodoPlanningPolicy` to encapsulate `handle_no_tool_calls`, `before_tool_calls`, and `after_tool_calls`. This makes the logic of the `run_one_turn` function clearer.

### Why

- Before refactoring, the function `run_one_turn` had more than 100 lines in total, and the logic was mixed together, making it difficult to read.

## May 25

### What I've done

- Change the turn_count field of the LoopState class to api_call_count
- Add a global variable MAX_API_CALLS_PER_USER_TURN
- Change the policy of handle_no_tool_calls method

### Why
- `turn_count` is not a true turn count. Better naming: `api_call_count`, and increment it once per `client.responses.create(...)` call
- no `MAX_API_CALLS_PER_USER_TURN`, so the agent can loop forever
- `todo_rewrite_ack_pending` can let the agent stop with unfinished todo items

## May 26

### What I've done

- Upgrade the parameter validation for the bash, read, write, edit tools to Pydantic Params.
- Add dispatcher tests for Pydantic validation failures on basic tools.
- Add `TOOL_OUTPUT_PREVIEW_CHARS = 500` to make tool output previews more useful during prompt-case testing.
- Add prompt cases for manually testing single-tool, multi-tool, and todowrite behavior.

### Why

- Leveraging Pydantic models for parameter validation eliminates the complexity of writing validation logic manually.
- Enhance the extensibility of parameter validation logic.
- Preparing to integrate the grep tool.
- Improve manual prompt-case testing by showing more useful command/test output in the terminal preview.

## May 27

### What I've done

- Add `grep` and `glob` tools as structured retrieval tools.
- Keep the integration style consistent with existing tools: OpenAI tool schema, Pydantic Params, `run_*` functions, and `TOOL_REGISTRY` dispatch.
- Add workspace path checks for `glob` and `grep`; `grep` uses ripgrep with `shell=False`.
- Tune `glob` safety: block unsafe recursive search from workspace root, but allow `**/*.py` inside an explicit subdirectory.
- Add schema and system prompt guidance so search locations go into `directory` / `path` instead of being embedded in patterns.
- Add `tests/test_search_tools.py` and prompt cases for manual grep/glob testing.
- Fix the colored input prompt so long input deletion works correctly in the terminal.

### Why

- Structured retrieval tools make codebase exploration safer and more predictable than relying on raw `bash`.
- `grep` and `glob` prepare the project for explore-mode subagents, where retrieval and reading can be separated from editing.
- Runtime validation and workspace checks are necessary because model tool inputs are untrusted.
- The prompt UI fix improves manual evaluation during long prompt-case testing.