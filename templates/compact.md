You are performing a CONTEXT CHECKPOINT COMPACTION for a coding agent.
The conversation messages above are about to be replaced by the summary you
write now, plus a small recent tail that will remain verbatim.

{{ previous_summary }}{{ focus }}
If a previous summary is provided, update it: preserve still-relevant facts,
add new progress and decisions from the messages above, move completed work to
Done, and refresh Next Steps. Do not let important file paths, commands,
errors, or decisions disappear across repeated compactions.

Write a structured checkpoint summary using this exact shape:

## Goal
[What the user is trying to accomplish. Include old goals only if still relevant.]

## Done
- [x] [Completed requests, investigations, edits, or decisions. Be explicit that these are done.]

## In Progress
- [ ] [Current unfinished work, if any. Use "(none)" if there is none.]

## Next Steps
1. [The immediate next action for the resumed agent, if any.]

## Key Facts
- [Important paths, symbols, findings, constraints, errors, and user preferences needed later.]

Rules:
- Do NOT answer any question from the conversation.
- Do NOT continue the conversation.
- Do NOT ask for tools or say you will inspect files.
- Do NOT mark work as pending if it was already completed.
- Preserve exact file paths, command names, function names, and error messages.
- Output plain text only.
