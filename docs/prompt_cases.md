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
