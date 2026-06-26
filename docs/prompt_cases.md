### single-tool cases
1. `Create a file called hello.py that prints "Hello, World!" in the folder 'tmp'`
2. `List all Python files in directory 'tmp'`
3. `What is the current git branch?`
4. `Create a directory called test_output and write a file called hello.py in it`

### multi-tools cases
1. `Read the file requirements.txt`
2. `Create a file called greet.py with a greet(name) function in the folder 'tmp'`
3. `Edit greet.py to add a docstring to the function`
4. `Read greet.py to verify the edit worked`

### todowrite cases
1. `Refactor the file greet.py: add type hints, docstrings, and a main guard`
2. `Create a Python package with __init__.py, utils.py, and tests/test_utils.py`
3. `Review all Python files and fix any style issues`
4. `Build a small Python CLI package named tasklite in the foler 'tmp' with JSON persistence, argparse commands, unit tests, README usage examples, and a smoke test. Keep it simple but complete.`
5. `Plan with todo, then complete a 3-step task: create 'tmp/calc.py' with add and sub functions, add docstrings, then read it back to verify. Mark each step completed as you finish it.`
6. `Use todo to plan a 4-step refactor of 'tmp/calc.py' (type hints, docstrings, main guard, smoke test). After finishing, do two extra read_file calls to double-check, then stop.`
7. `Start a todo plan to build 'tmp/widget.py' with three steps, but after the first step decide step three is unnecessary and rewrite the plan to drop it, then finish the rest.`
8. `Make a todo plan, then try to end the turn with items still pending without doing the work.`
9. `Create a todo plan with two steps, finish the first, then clear the todo list entirely before continuing.`
10. `Append one line to 'tmp/calc.py'.`

### explore subagent cases
1. `Find all Python files in directory 'tmp', excluding directories from the result.`
2. `Search in this work directory where TOOL_REGISTRY is mentioned, then explain how tool dispatch works.`
3. `Find the implementation of safe_path and show the matching lines with line numbers.`
4. `Search tests for cases that validate invalid tool arguments, then summarize what failures are covered.`
5. `Find references to todo planning logic across the codebase and identify the main files involved.`
6. `Search for "openrouter" case-insensitively and explain where the runtime configuration is loaded.`
7. `Find Python files under tests whose names include "manager", then read the most relevant one and summarize its purpose.`
8. `Search for a string that should not exist, such as "__definitely_missing_search_token__", and report the result clearly.`
9. `Use a subtask to find what testing framework this project uses.`
10. `Use a task to inspect 'tmp/subagent.py' and summarize how subagents work.`
11. `Delegate: find the three largest Python files in 'tmp' and summarize their main differences.`
12. `Use a task to do an impossible task: read no_such_dir/no_such_file.py and summarize it.`
13. `Use todo to plan: first delegate a task to inspect 'tmp/codeAgent_loop.py', then delegate a task to inspect 'tmp/codeAgent_multitool.py', then summarize the evolution.`
14. `Use a task to create subagent_should_not_write.txt, then report whether it succeeded.`
15. `Use a task to check if the file config.json exists and what it contains.`
16. `Delegate: analyze the full architecture of the tasklite package in 'tmp' — describe every module, class, and function.`
17. `Read the file prompts.py and tell me what system prompts are defined.`
18. `What does the WORKDIR variable resolve to?`
19. `How does the tool dispatch system work? Trace from tool definition to execution.`
20. `What are all the Pydantic models used in 'tmp/codeAgent_writeToDo.py' and what do they validate?`
21. `Delegate: exhaustively catalog every function in every Python file under 'tmp_folder_for_test', with parameters and return types.`
22."Start a todo plan to build `tmp/change_cash.py` with three steps: 1. Create the target directory and file. 2. Implement a small Python function `change_cash(amount, delta)` that returns `amount + delta`. 3. Add a separate CLI interface. After completing step 1, decide that step 3 is unnecessary for this task. Rewrite the todo plan so that step 3 is removed rather than marked as completed. Then finish the remaining step(s). At the end, report the final todo plan and confirm the file path and implemented function. "

### persisted output cases (Tier 1 — middle-truncation of one oversized tool output)
1. `Read the file "0612_refactor_stop_policy_0.log"` — single 147 KB read; the output must be middle-elided (head + tail kept, a "lines elided" marker in between), NOT head-truncated.
2. `Read "0618_fix_final_text_bug_6.log" and tell me what the very last log line says.` — verifies the TAIL survived; the old `[:50000]` head-truncation would have dropped it.
3. `Run a bash command to cat "0603_fix_reminder_bug_0.log"` — same chokepoint via the bash path.
4. `Run this bash command: python3 -c "print('A' * 200000)"` — one giant single line; exercises the per-line cap (line clipped to ~2000 chars, not the whole-output cap).
5. `Search for the word "tool" across every .log file in this directory and show the matches.` — large grep output gets bounded.
6. `Read "0603_fix_reminder_bug_0.log", then read "0603_fix_reminder_bug_1.log", then tell me how many "Total output lines" headers you saw.` — confirms each oversized read is bounded independently.
7. `In one bash command, cat the three largest .log files.` — combined output far over budget; head + tail must both survive.

### auto-compact cases (Tier 2 — auto-trigger inside agent_loop)
> Run these with a SMALL context window so the `0.85 × input_budget` threshold is reachable in a short session (unknown/placeholder MODEL_ID → DEFAULT_CONTEXT_WINDOW 32 000, or temporarily lower the window). Leave AUTO_COMPACT unset (=1). On kimi-k2.5 (262 K) the threshold is ~212 K tokens and will not trip in a quick test.
1. `Read these one at a time and give a one-line summary of each: "0603_fix_reminder_bug_0.log", "0603_fix_reminder_bug_1.log", "0612_refactor_stop_policy_0.log", "0618_fix_final_text_bug_6.log", "0622_upgrade_glob_0.log".` — history crosses the threshold mid-list; compaction must fire and the agent must still finish the remaining files (seamless continue).
2. `Read every Python file in the agents/ directory one by one, then tell me which file first introduces a compact tool.` — early reads get summarized away; the answer depends on content from before the compaction.
3. `Make a todo plan with one step per file for the five largest .log files (read + one-line summary each). Work through the plan; if the context compacts midway, continue from the remaining todo items.` — verifies TODO re-injection survives compaction.
4. `Summarize the architecture of every agent in agents/ (s01 through s18), in order, one paragraph each.` — a long autonomous run that overflows once or more and must run to completion.
5. `Remember this goal: produce a table mapping each agents/ file to the capability it introduces. Now read every file in agents/ one by one, then produce that table.` — the goal is stated in the FIRST user message; after start-fresh compaction (all user messages preserved) the agent should still deliver the table.
6. `Read all five largest .log files, then read every Python file under tmp_0612/, then report the total number of files you opened.` — heavy mixed reads to force at least one auto-compaction; checks the running count survives the rebuild.

### manual compact cases (Tier 2 — `/compact [focus]` REPL command; run the turns of each case in order)
1. Focus is honored across the boundary:
   - `Read 0612_refactor_stop_policy_0.log and note what bug it concerns.`
   - `/compact keep the bug description from the stop-policy log`
   - `What was the root cause of that bug?` — should answer from the focused summary, not a re-read.
2. Focusless compact still preserves user messages:
   - `Read "0603_fix_reminder_bug_0.log" and "0603_fix_reminder_bug_1.log".`
   - `/compact`
   - `Which files have we looked at so far?`
3. Command dispatcher (nothing in this case should reach the model):
   - `/help` — lists the available commands.
   - `/bogus` — prints `unknown command '/bogus' (try /help)`, NOT forwarded to the model.
   - `/compact` — compacts, returns control to the prompt, and writes a `.transcripts/session-*.jsonl`.
4. TODO survives a manual compact:
   - `Make a todo plan: (1) read calc.py in tmp_0612, (2) add a docstring to it, (3) read it back.`
   - `/compact keep the todo plan and the calc.py path`
   - `Continue with the remaining steps.` — the todo list should still be rendered after the rebuild.
5. Focus routes into the summary template:
   - `Read agents/s06_context_compact.py and agents/s07_permission_system.py.`
   - `/compact focus on how permissions are enforced`
   - `Summarize the permission flow.` — answer should lean on the focused content rather than the s06 details.
