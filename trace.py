"""Lightweight, read-only runtime tracing for tools and agent control flow."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


MAX_TRACE_STRING_CHARS = 500
SENSITIVE_KEYS = {
    "command",
    "new_text",
    "old_text",
    "output",
    "prompt",
    "raw_arguments",
}


class TraceSink(Protocol):
    def emit(self, event: dict[str, Any]) -> None:
        """Consume one event. Implementations may add transport metadata."""


class NullTraceSink:
    def emit(self, event: dict[str, Any]) -> None:
        del event


class MemoryTraceSink:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            record = dict(event)
            record["sequence"] = self._sequence
            self._events.append(record)

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def by_type(self, event_name: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("event") == event_name]


class JsonlTraceSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._sequence = 0
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            record = dict(event)
            record["sequence"] = self._sequence
            safe_record = sanitize_trace_value(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe_record, ensure_ascii=True) + "\n")


@dataclass(frozen=True)
class TraceContext:
    sink: TraceSink = field(default_factory=NullTraceSink)
    run_id: str = ""
    agent_id: str = "parent"


def sanitize_trace_value(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe, bounded copy with sensitive values removed."""
    if key.lower() in SENSITIVE_KEYS:
        if isinstance(value, str):
            return f"<redacted:{len(value)} chars>"
        return "<redacted>"
    if isinstance(value, str):
        if len(value) <= MAX_TRACE_STRING_CHARS:
            return value
        return value[:MAX_TRACE_STRING_CHARS] + f"...<{len(value) - MAX_TRACE_STRING_CHARS} chars omitted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {
            str(child_key): sanitize_trace_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_trace_value(item) for item in value]
    return str(value)[:MAX_TRACE_STRING_CHARS]


def emit_trace(
    context: TraceContext | None,
    event_name: str,
    *,
    call_id: str | None = None,
    **payload: Any,
) -> None:
    """Best-effort event emission. Trace failures never affect the caller."""
    if context is None:
        return
    event = {
        "event": event_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": context.run_id,
        "agent_id": context.agent_id,
        **sanitize_trace_value(payload),
    }
    if call_id is not None:
        event["call_id"] = call_id
    try:
        context.sink.emit(event)
    except Exception:
        # Tracing is observational. A broken sink must never alter runtime
        # permission decisions, tool execution, or agent completion.
        return
