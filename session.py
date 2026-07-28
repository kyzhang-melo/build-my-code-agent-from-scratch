"""Session-scoped dependencies for one agent instance."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from permissions import PermissionService
from session_store import NullSessionStore, SessionStoreProtocol
from tools import TodoManager, ToolRuntimeSpec
from trace import TraceContext
from workspace import Workspace


def generate_session_id() -> str:
    """Sortable timestamp prefix + short uuid suffix.

    Replaces pi's uuidv7: keeps lexicographic ordering for ``--continue`` and
    ``--list-sessions`` without adding a dependency.
    """
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class StopGate(Protocol):
    """Decide whether an agent may stop after a no-tool response."""

    max_nudges: int
    name: str

    def check(self, final_text: str) -> str | None:
        ...

    def give_up_note(self) -> str | None:
        ...


class TodoStopGate:
    name = "todo"

    def __init__(self, todo: TodoManager, max_nudges: int):
        self.todo = todo
        self.max_nudges = max_nudges

    def check(self, final_text: str) -> str | None:
        del final_text
        if not self.todo.has_active_plan() or self.todo.all_items_completed():
            return None
        return (
            "<contract>Before ending, either complete all todo items, "
            "or call todo to explicitly rewrite/remove items that are no longer needed.</contract>"
        )

    def give_up_note(self) -> str:
        return (
            "Warning: Ending with unresolved todo items after repeated contract reminders.\n"
            f"{self.todo.render()}"
        )


class ReportStopGate:
    name = "report"

    def __init__(self, min_length: int, max_nudges: int, continuation_prompt: str):
        self.min_length = min_length
        self.max_nudges = max_nudges
        self.continuation_prompt = continuation_prompt

    def check(self, final_text: str) -> str | None:
        if len(final_text) >= self.min_length:
            return None
        return self.continuation_prompt

    def give_up_note(self) -> None:
        return None


@dataclass(frozen=True)
class AgentSession:
    """All dependencies and policy bound to one agent session.

    The object owns no loop behavior. ``main.agent_loop`` remains the
    orchestrator and receives an AgentSession explicitly.
    """

    name: str
    session_id: str
    workspace: Workspace
    todo: TodoManager
    system: str
    tools: list[dict]
    registry: dict[str, ToolRuntimeSpec]
    permission_service: PermissionService
    permission_source: str
    trace_context: TraceContext
    max_api_calls: int
    stop_gate: StopGate
    store: SessionStoreProtocol = field(default_factory=NullSessionStore)
    on_text: Callable[[str], None] | None = None
