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

## June 17
### What I've Done

- Refactored this project, remove dead todo rewrite-ack machinery, unify stop policies, unify loop outcome.

### Why

- Since the reminder mechanism has been changed, the agent harness does not need the `transition_reason` field and the `todo_rewrite_ack_pending` field of the `LoopState` class.
- Before the refactor, the todo tool and the subagent tool have two different stop policies, but they share the same `run_one_turn()` function. So the `StopGate` protocol has been introduced to unify the stop policies.
- Before the refactor, `run_one_turn()` function returns a boolean type value, `agent_loop()` function returns nothing. When these two functions are used by different agents (parent agent or subagent), the boolean value is not enough. So the Literal `StopReason` and the class `StepOutcome` and `TurnOutcome` have been introduced to unify the loop outcome.

## June 18

### What I've done

- Fix the swallowed-deliverable bug: the model's real answer (e.g. an evolution summary) was kept in history but never shown to the user when it appeared on a turn that continues the loop, so the user only saw a terse recap.
- Add an `on_text` sink on `AgentConfig` and surface assistant text on every loop continuation in `run_one_turn()`: both when the text accompanies a tool call (Door 1), and when a stop-gate nudge rejects a no-tool answer (Door 2). The parent prints via `emit_assistant_text`; the explore subagent stays silent (`on_text=None`). Stop paths still deliver via `final_text`, so nothing prints twice.
- Make the todo reminder state-aware: when the plan is fully complete, `render_with_reminder()` steers the model to deliver its result instead of nudging for more todo calls, removing the gratuitous trailing `todo` call that buried summaries.
- Tighten `PARENT_SYSTEM`: the closing message must contain the actual result (the findings/report), not just a list of completed steps.
- Add regression tests for surfacing on both continuation paths and the no-double-print boundary on the give-up path.

### Why

- `run_one_turn()` only displayed the final no-tool message (`final_text`), tying display to loop termination. Any assistant text on a continuing turn (riding a tool call, or rejected by the todo contract nudge because bookkeeping lagged) was retained in history but never printed.
- The implementation streams all message parts regardless of tool calls, and separates streaming output from final-answer detection. This fix mirrors that approach: surface text on every continuation, while treating only tool-free text as the final answer.
- Verified with prompt-case logs: deliverables now print in full across todo-driven summarize, build, and plan-rewrite tasks.

## June 22

### What I've done

- Upgrade the `glob` tool to allow recursive searches from the workspace, such as `**/config.json`, `**/pyproject.toml`, and `**/README.md`.
- Replace the old root-level `**` blanket block with a narrower broad-pattern guard for match-everything patterns like `**`, `**/`, and `**/*`.
- Add a helpful broad-pattern error response that includes the top-level workspace listing and marks directories with a trailing `/`, so the model can choose a more specific search location on the next call.
- Add noisy directory pruning for recursive glob walking, skipping directories such as `.git`, `.svn`, `.hg`, `.venv`, `node_modules`, `__pycache__`, `.pytest_cache`, and `.claude`.
- Add shared glob formatting helpers so results are sorted, limited, and filtered consistently after traversal.
- Update the `glob` tool description to explain that recursive filename searches are supported and noisy directories are skipped.
- Extend `tests/test_search_tools.py` with coverage for broad-pattern guidance, precise recursive filename search, excluded-directory pruning, and the existing escape/limit behavior.
- Validate the behavior with prompt-case logs in `/Users/xixi/myProject/tmp_folder_for_test`, including `**/config.json`, `**/pyproject.toml`, `**/README.md`, `**/__main__.py`, root-level `*.log`, and the broad `**/*` guidance path.

### Why

- The old guard treated every `pattern.startswith("**") and base == WORKDIR` call as unsafe. That blocked common, legitimate file discovery tasks like finding `config.json` anywhere in the workspace.
- When `**/filename` was rejected, the model had to fall back to shallow patterns like `*` and `*/*`, which was token-expensive and could miss deeper files.
- The real risk is broad, noisy traversal, not simply using `**` from the workspace root. The upgraded behavior separates precise recursive lookup from match-everything traversal.
- Returning the top-level listing on broad-pattern rejection makes the error actionable instead of a dead end.
- Directory pruning keeps recursive glob useful while avoiding common large or noisy folders.

## June 26

### What I've done

- Add token tracking and context window budget management to track model `input_tokens` usage in `LoopState`.
- Implement `compact_history()` core functionality to summarize and compress old conversation history (tool calls, model responses) while preserving user messages.
- Add a JSONL snapshot backup mechanism before any history mutation to ensure failure-safe ordering.
- Add a `templates/compact.md` prompt template specifically designed for the compaction task.
- Add a minimal slash-command dispatcher to the REPL, including `/compact [focus]` for manual context compaction and `/help`.
- Integrate an auto-compaction trigger in the `agent_loop` that fires at clean turn boundaries when the token load reaches 85% of the input budget.
- Introduce a thrash guard that drops the oldest non-summary user messages if a fresh summary still exceeds the context budget, preventing infinite re-summarization loops.

### Why

- The code agent needs a context compaction mechanism to keep going past the context-window limit during long-running tasks.
- Using a side-call summarization ensures we only keep the work that would otherwise be lost and don't orphan tool calls.
- Auto-compaction prevents context-length API errors transparently, while the manual `/compact` command gives users explicit control over when and how the context is summarized.
- The thrash guard protects the agent from wasting API calls on repeated summarizations when user messages take up too much space.

## June 30

### What I've done

- Updated `run_read` to read files line by line instead of loading the entire file into memory.

### Why

- The old `run_read` implementation loaded whole files at once, which could increase latency and memory usage for large files.

## July 02

### What I've done

- Converted `main.py` to `AsyncOpenAI`, async `run_one_turn`, `agent_loop`, `run_subagent`, and CLI command handling.
- Added async compaction helpers in `context_compact.py`.
- Converted `tools.py` to awaitable `ToolRuntimeSpec.execute`, added `async_tool(...)`, and partitioned `execute_tool_calls_async(...)`.
- Safe tools now run concurrently with `asyncio.gather`; unsafe tools stay sequential. `MAX_PARALLEL_TOOL_CALLS` defaults to `4`.
- `task` is now async-native, so multiple subagents can run concurrently (currently only explore-subagent has been implemented).

### Why

- Although the synchronous mechanism is simple and intuitive, it introduces excessive latency when the agent needs to explore the code repository. Therefore, an asynchronous mechanism was introduced.

### Items for improvement

- The current implementation employs a chunking mechanism to control the execution of asynchronous tasks, not the semaphore mechanism.
- The `glob` tool still needs to add more mechanism to discourage the LLM from emitting the broad pattern `**/*`.
- In debug stage, the ID number for subagents could be introduced.
- When the agent is asked about issues that occurred prior to the context compaction, its behavior comes across as quite stale.

## July 07

### What I've done

- Fixed the context compaction history rebuild: compacted history is now `[summary checkpoint] + [recent tail]`, instead of preserving every old user request before the summary.
- Removed same-role message merging from `normalize_messages()`, so compact summaries and new user queries keep clear turn boundaries.
- Added previous-summary fold-forward to the compaction flow, so repeated `/compact` calls update the existing checkpoint instead of letting summaries decay.
- Reworked `templates/compact.md` into a checkpoint-style handoff prompt that tells the model not to answer, continue, or re-execute old tasks during summarization.
- Updated the auto-compaction trimming path to protect summary messages and drop complete retained user turns rather than deleting isolated user messages.
- Validated the compact fix manually with `kimi-k2.5`: after `/compact`, follow-up questions used the summary checkpoint instead of replaying old user requests.

### Why

- The old compact design kept all previous user requests as live user messages. After `/compact`, the model could treat those old imperatives as active tasks and do the pre-compaction work again.
- Merging adjacent user messages made the bug worse: old requests, the compact summary, and the new user query could collapse into one large user message with unclear boundaries.
- A checkpoint plus recent-tail design better matches how compaction should work: summarized history becomes historical context, while only the newest tail remains verbatim.
- Fold-forward summaries are needed because long sessions may compact multiple times; each new compact must preserve the accumulated summary instead of summarizing only the latest slice.

### Items for improvement

- Sometimes, when the LLM responds with an empty message, the agent loop ends the turn without producing any output.
- Some models with weaker agentic capabilities perform poorly with this agent harness when the user asks the model to read many files and summarize them.
