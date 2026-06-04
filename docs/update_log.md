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

## June 01

### What I've done

- Add explore-only subagent feature, when the code agent need to explore/read many files, the parent-agent can use the `task` tool to create explore-only subagent to explore and read files. 
- Add the `task` tool for parent-agent delegation.
- Add `AgentConfig` to separate parent-agent and explore-subagent runtime configuration.
- Add read-only explore subagent config with only `glob`, `grep`, and `read_file`.
- Split prompts into `PARENT_SYSTEM` and `EXPLORE_SUBAGENT_SYSTEM`, while keeping `SYSTEM = PARENT_SYSTEM` for compatibility.
- Adjust the parent/subagent contract: the subagent returns a natural-language exploration report, and the parent agent judges whether the task goal was completed.
- Add debug scaffold to print the user query, normalized message count, and the last 3 messages sent to the LLM.
- Validate the feature with prompt-case logs covering missing files, read-only write attempts, serial subagent delegation, and broad architecture exploration.
- Find a bug, which caused by the <reminder> mechanism. 

### Why
- The code agent can benefit from the subagent feature, since the subagent has its own independent context.
- Keeping subagents explore-only avoids write/edit concurrency problems while still giving the parent agent fresh-context codebase exploration.
- The agent will always inject the <reminder> prompt, after the todo list has been completed totally. This bug will be fixed in the future.
- The reminder bug is now understood as a todo lifecycle issue, not a subagent contract issue, so it can be fixed in a separate branch.

## June 04

### What I've done

- Fix the reminder mechanism bug by making the todo reminder from counter-driven to event-driven.
- Remove the round-counter variables and methods: `PLAN_REMINDER_INTERVAL`, `PlanningState.rounds_since_update`, `TodoManager.note_round_without_update()`, and `TodoManager.reminder()`.
- Add `TodoManager.render_with_reminder()`: every `todo` write now returns the latest list wrapped in a `<system-reminder>` block (with a separate empty-list message), so the model stays aware of its plan only when the plan actually changes.
- Simplify `TodoPlanningPolicy.after_tool_calls()` to drop the counter reset and the periodic reminder injection; keep the contract early-stop gate (`handle_no_tool_calls`) unchanged.
- Replace the obsolete interval-reminder test with tests for the new write-time echo (non-empty and cleared cases).

- Fix "stale answers from a previous turn", carry the turn's answer on `LoopState.final_text` instead of recovering it via `extract_text()` over shared history.
- Set `final_text` only when a turn ends on a model message with no tool calls; also set it for the `max_api_calls` warning. Leave it `None` when a turn ends right after a tool call.
- Read `state.final_text` in the REPL loop and in `run_subagent` instead of `extract_text(state.messages)`.
- Pass the current turn's text into `handle_subagent_no_tool_calls()` (as a `final_text` argument) so the subagent completion check evaluates only this turn's no-tool response, never a stale history scan.
- Add a regression test ensuring a new turn that produces no text does not surface a prior turn's assistant message, a positive test for capturing the current turn's text, and a test that a long prior assistant message is not mistaken for the current (empty) subagent summary.

### Why

- Counter-driven `reminder()` only guarded against "no plan" and never checked `all_items_completed()`, so the counter kept injecting `<reminder>Refresh your current plan...>` even after every todo was done.
- Keeping the contract gate preserves our runtime protection against ending with unfinished todos, which the product agents do not enforce.

- `extract_text()` scanned the entire accumulated history in reverse and returned the most recent assistant text. When a user turn ended on a tool call without a closing reply, it surfaced an unrelated previous turn's answer.
- This was pre-existing but masked by the old reminder bug, whose spurious nudge forced an extra model turn that usually produced text; removing the bogus reminder exposed it.