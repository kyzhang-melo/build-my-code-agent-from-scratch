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