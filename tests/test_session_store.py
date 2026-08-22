"""Tests for session persistence (session_store.py) and resume sanitization."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from session_store import (
    NullSessionStore,
    SessionStore,
    encode_workspace_path,
    ensure_session_name_available,
    find_most_recent_session,
    get_default_session_dir,
    list_session_headers,
    validate_session_name,
)
from workspace import Workspace
from message_utils import drop_orphan_tool_calls, sanitize_resumed_message


def _session_dir(workspace: Workspace) -> Path:
    return workspace.root.parent / f"{workspace.root.name}-sessions"


def _create_store(workspace: Workspace, *args, **kwargs) -> SessionStore:
    kwargs.setdefault("session_dir", _session_dir(workspace))
    return SessionStore.create(workspace, *args, **kwargs)


def _assistant(*blocks, model="model-a", provider="") -> dict:
    return {
        "role": "assistant",
        "content": list(blocks),
        "runtime": {"model_id": model, "provider": provider, "protocol": "responses"},
    }


def _text(text: str) -> dict:
    return {"type": "text", "text": text, "source": "test"}


def _tool_call(call_id="c1", item_id="fc1") -> dict:
    return {
        "type": "tool_call", "name": "bash", "arguments": "{}",
        "pairing": {"call_id": call_id, "item_id": item_id},
    }


def _reasoning() -> dict:
    return {"type": "reasoning", "provider_item": {
        "type": "reasoning", "id": "r1", "summary": "thinking...",
    }}


def _tool_result(call_id="c1", content="done") -> dict:
    return {"role": "tool", "call_id": call_id, "content": content, "is_error": False}


# ---------------------------------------------------------------------------
# SessionStore: create + sync + messages round-trip
# ---------------------------------------------------------------------------


def test_default_session_dir_is_external_and_workspace_scoped(tmp_path) -> None:
    workspace_a = tmp_path / "projects" / "alpha"
    workspace_b = tmp_path / "projects" / "beta"
    agent_dir = tmp_path / "agent-home"

    dir_a = get_default_session_dir(workspace_a, agent_dir=agent_dir)
    dir_b = get_default_session_dir(workspace_b, agent_dir=agent_dir)

    assert dir_a == agent_dir / "sessions" / encode_workspace_path(workspace_a)
    assert dir_b == agent_dir / "sessions" / encode_workspace_path(workspace_b)
    assert dir_a != dir_b
    assert not dir_a.is_relative_to(workspace_a.resolve())


def test_no_file_created_until_sync(workspace) -> None:
    store = _create_store(workspace, "test-id-1", "model-a")
    assert not store.path.exists()
    store.sync([{"role": "user", "content": "hello"}])
    assert store.path.exists()
    assert store.path.parent == _session_dir(workspace).resolve()
    assert not (workspace.root / ".sessions").exists()


def test_incremental_append_no_duplicates(workspace) -> None:
    store = _create_store(workspace, "test-id-2", "model-a")
    h1 = [{"role": "user", "content": "hello"}]
    store.sync(h1)
    count_after_1 = store.entry_count

    h2 = h1 + [{"role": "assistant", "content": "hi there"}]
    store.sync(h2)
    count_after_2 = store.entry_count

    assert count_after_2 == count_after_1 + 1
    # Re-syncing the same history should not add entries.
    store.sync(h2)
    assert store.entry_count == count_after_2


def test_messages_projection_matches_history(workspace) -> None:
    store = _create_store(workspace, "test-id-3", "model-a")
    history = [
        {"role": "user", "content": "hello"},
        _assistant(_text("hi")),
        {"role": "user", "content": "do something"},
    ]
    store.sync(history)
    projected = store.messages()
    assert projected == history


def test_compaction_triggers_history_reset(workspace) -> None:
    store = _create_store(workspace, "test-id-4", "model-a")
    h1 = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
    ]
    store.sync(h1)

    # Simulate compaction: history is rewritten in place.
    h2 = [
        {"role": "user", "content": "[CONTEXT SUMMARY] ..."},
        {"role": "user", "content": "new request"},
    ]
    store.sync(h2)

    projected = store.messages()
    # Projection should be the post-compaction history only.
    assert projected == h2
    # The file should contain a history_reset entry.
    with store.path.open("r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    reset_entries = [e for e in lines if e.get("type") == "history_reset"]
    assert len(reset_entries) == 1
    assert reset_entries[0]["reason"] == "compaction"


def test_create_sync_open_roundtrip(workspace) -> None:
    store = _create_store(workspace, "test-id-5", "model-a")
    history = [
        {"role": "user", "content": "hello"},
        _assistant(_text("world")),
    ]
    store.sync(history)

    # Reopen the same file.
    reopened = SessionStore.open(store.path, workspace, "model-a")
    assert reopened.session_id == "test-id-5"
    assert reopened.messages() == history


def test_v1_history_migrates_and_first_sync_writes_v2(workspace) -> None:
    store = _create_store(workspace, "legacy", "model-a")
    store.sync([
        {"role": "user", "content": "run"},
        {"type": "function_call", "id": "fc1", "call_id": "c1",
         "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
    ])
    lines = store.path.read_text().splitlines()
    header = json.loads(lines[0])
    header["version"] = 1
    store.path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")

    reopened = SessionStore.open(store.path, workspace, "model-a")
    migrated = reopened.messages()
    assert [message.get("role") for message in migrated] == ["user", "assistant", "tool"]

    reopened.sync(migrated)
    assert json.loads(store.path.read_text().splitlines()[0])["version"] == 2


def test_resume_different_model_drops_reasoning(workspace) -> None:
    store = _create_store(workspace, "test-id-6", "model-a")
    history = [
        {"role": "user", "content": "hello"},
        _assistant(_reasoning(), _tool_call()),
        _tool_result(),
        _assistant(_text("result")),
    ]
    store.sync(history)

    reopened = SessionStore.open(store.path, workspace, "model-b")
    projected = reopened.messages()

    # Reasoning should be dropped.
    assert not any(m.get("type") == "reasoning" for m in projected)
    assert not any(m.get("role") == "tool" for m in projected)
    assert any("Historical tool call" in str(m.get("content")) for m in projected)


def test_resume_same_model_preserves_reasoning_and_id(workspace) -> None:
    store = _create_store(workspace, "test-id-7", "model-a")
    history = [
        _assistant(_reasoning(), _tool_call()),
        _tool_result(),
    ]
    store.sync(history)

    reopened = SessionStore.open(store.path, workspace, "model-a")
    projected = reopened.messages()
    assistant = projected[0]
    assert assistant["content"][0]["provider_item"]["id"] == "r1"
    assert assistant["content"][1]["pairing"]["item_id"] == "fc1"


def test_resume_same_model_different_provider_sanitizes(workspace) -> None:
    store = _create_store(
        workspace, "test-provider", "model-a", "provider-a",
    )
    store.sync([
        _assistant(_reasoning(), _tool_call(), provider="provider-a"),
        _tool_result(),
    ])

    reopened = SessionStore.open(
        store.path, workspace, "model-a", "provider-b",
    )
    projected = reopened.messages()

    assert not any(item.get("type") == "reasoning" for item in projected)
    assert not any(item.get("role") == "tool" for item in projected)
    assert reopened.resume_diagnostics.dropped_reasoning == 1
    assert reopened.resume_diagnostics.textualized_cross_runtime_exchanges == 1


def test_resume_drops_orphan_trailing_function_call(workspace) -> None:
    store = _create_store(workspace, "test-id-8", "model-a")
    # History ends with a function_call that has no output (process died mid-tool).
    history = [
        {"role": "user", "content": "run something"},
        {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "bash", "arguments": "{}"},
    ]
    store.sync(history)

    reopened = SessionStore.open(store.path, workspace, "model-a")
    projected = reopened.messages()
    assert not any(m.get("type") == "function_call" for m in projected)
    assert len(projected) == 1
    assert projected[0]["role"] == "user"


def test_resume_reports_invalid_jsonl_lines(workspace) -> None:
    store = _create_store(workspace, "test-invalid-line", "model-a")
    store.sync([{"role": "user", "content": "hello"}])
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    reopened = SessionStore.open(store.path, workspace, "model-a")
    assert reopened.messages() == [{"role": "user", "content": "hello"}]
    assert reopened.resume_diagnostics.ignored_invalid_lines == 1


def test_future_session_version_is_rejected(workspace) -> None:
    path = workspace.root / ".sessions" / "future.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "type": "session_header",
        "version": 999,
        "session_id": "future",
        "created_at": "2026-01-01T00:00:00Z",
        "cwd": str(workspace.root),
        "model_id": "model-a",
    }) + "\n")

    with pytest.raises(ValueError, match="newer than supported"):
        SessionStore.open(path, workspace, "model-a")


def test_resume_cwd_mismatch_rejected(tmp_path) -> None:
    ws_a = Workspace(tmp_path / "a")
    ws_a.root.mkdir()
    store = _create_store(ws_a, "test-id-9", "model-a")
    store.sync([{"role": "user", "content": "hello"}])

    ws_b = Workspace(tmp_path / "b")
    ws_b.root.mkdir()
    with pytest.raises(ValueError, match="cwd mismatch"):
        SessionStore.open(store.path, ws_b, "model-a")


def test_resume_missing_file_rejected(workspace) -> None:
    with pytest.raises(ValueError, match="not found"):
        SessionStore.open(workspace.root / ".sessions" / "nonexistent.jsonl", workspace, "model-a")


def test_legacy_workspace_session_can_be_opened_explicitly(workspace) -> None:
    legacy_dir = workspace.root / ".sessions"
    store = _create_store(
        workspace, "legacy", "model-a", session_dir=legacy_dir,
    )
    store.sync([{"role": "user", "content": "hello"}])

    reopened = SessionStore.open(store.path, workspace, "model-a")

    assert reopened.messages() == [{"role": "user", "content": "hello"}]
    assert find_most_recent_session(_session_dir(workspace), workspace.root) is None


def test_null_store_writes_nothing(workspace) -> None:
    null = NullSessionStore()
    null.sync([{"role": "user", "content": "hello"}])
    assert null.messages() == []
    assert null.entry_count == 0
    assert null.path is None
    assert null.header() is None


# ---------------------------------------------------------------------------
# find_most_recent_session + list_session_headers
# ---------------------------------------------------------------------------


def test_find_most_recent_session(workspace) -> None:
    sessions_dir = _session_dir(workspace)
    sessions_dir.mkdir()

    # Create two session files with different mtimes.
    old_path = sessions_dir / "old.jsonl"
    old_path.write_text(json.dumps({
        "type": "session_header", "version": 1, "session_id": "old",
        "created_at": "2026-01-01T00:00:00Z", "cwd": str(workspace.root),
        "model_id": "model-a",
    }) + "\n")

    import time
    time.sleep(0.05)

    new_path = sessions_dir / "new.jsonl"
    new_path.write_text(json.dumps({
        "type": "session_header", "version": 1, "session_id": "new",
        "created_at": "2026-01-02T00:00:00Z", "cwd": str(workspace.root),
        "model_id": "model-a",
    }) + "\n")

    result = find_most_recent_session(sessions_dir, workspace.root)
    assert result is not None
    assert result.name == "new.jsonl"


def test_find_most_recent_session_filters_other_cwd(workspace, tmp_path) -> None:
    sessions_dir = _session_dir(workspace)
    sessions_dir.mkdir()
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()

    (sessions_dir / "wrong.jsonl").write_text(json.dumps({
        "type": "session_header", "version": 1, "session_id": "wrong",
        "created_at": "2026-01-01T00:00:00Z", "cwd": str(other_cwd),
        "model_id": "model-a",
    }) + "\n")

    result = find_most_recent_session(sessions_dir, workspace.root)
    assert result is None


def test_list_session_headers(workspace) -> None:
    sessions_dir = _session_dir(workspace)
    sessions_dir.mkdir()
    (sessions_dir / "a.jsonl").write_text(json.dumps({
        "type": "session_header", "version": 1, "session_id": "id-a",
        "created_at": "2026-01-01T00:00:00Z", "cwd": str(workspace.root),
        "model_id": "model-a",
    }) + "\n")
    (sessions_dir / "b.jsonl").write_text(json.dumps({
        "type": "session_header", "version": 1, "session_id": "id-b",
        "created_at": "2026-01-02T00:00:00Z", "cwd": str(workspace.root),
        "model_id": "model-a",
    }) + "\n")

    headers = list_session_headers(sessions_dir)
    assert len(headers) == 2
    ids = {h["session_id"] for h in headers}
    assert ids == {"id-a", "id-b"}


def test_list_session_headers_filters_shared_directory_by_cwd(tmp_path) -> None:
    sessions_dir = tmp_path / "shared-sessions"
    workspace_a = Workspace(tmp_path / "a")
    workspace_b = Workspace(tmp_path / "b")
    workspace_a.root.mkdir()
    workspace_b.root.mkdir()
    store_a = _create_store(
        workspace_a, "shared-a", "model-a", session_dir=sessions_dir,
    )
    store_b = _create_store(
        workspace_b, "shared-b", "model-a", session_dir=sessions_dir,
    )
    store_a.sync([{"role": "user", "content": "a"}])
    store_b.sync([{"role": "user", "content": "b"}])

    headers = list_session_headers(sessions_dir, cwd=workspace_a.root)

    assert [header["session_id"] for header in headers] == ["shared-a"]


def test_session_names_are_scoped_by_cwd_in_shared_directory(tmp_path) -> None:
    sessions_dir = tmp_path / "shared-sessions"
    workspace_a = Workspace(tmp_path / "a")
    workspace_b = Workspace(tmp_path / "b")
    workspace_a.root.mkdir()
    workspace_b.root.mkdir()
    store_a = _create_store(
        workspace_a,
        "named-a",
        "model-a",
        session_name="same-name",
        session_dir=sessions_dir,
    )
    store_a.sync([{"role": "user", "content": "a"}])

    store_b = _create_store(
        workspace_b,
        "named-b",
        "model-a",
        session_name="same-name",
        session_dir=sessions_dir,
    )
    store_b.sync([{"role": "user", "content": "b"}])

    assert store_b.path.exists()


def test_session_name_is_persisted_and_must_be_unique(workspace) -> None:
    store = _create_store(
        workspace, "named-id", "model-a", session_name="kevin",
    )
    store.sync([{"role": "user", "content": "hello"}])

    header = list_session_headers(_session_dir(workspace))[0]
    assert header["session_name"] == "kevin"
    with pytest.raises(ValueError, match="already exists"):
        ensure_session_name_available(
            _session_dir(workspace), "KEVIN", cwd=workspace.root,
        )


@pytest.mark.parametrize("name", ["last", "continue", "new", "../kevin", "two words"])
def test_invalid_or_reserved_session_names_are_rejected(name) -> None:
    with pytest.raises(ValueError):
        validate_session_name(name)


def test_session_lock_rejects_a_second_writer(workspace) -> None:
    store = _create_store(
        workspace, "locked", "model-a", acquire_lock=True,
    )
    try:
        with pytest.raises(ValueError, match="already open"):
            _create_store(
                workspace, "locked", "model-a", acquire_lock=True,
            )
    finally:
        store.close()

    replacement = _create_store(
        workspace, "locked", "model-a", acquire_lock=True,
    )
    replacement.close()


def test_session_lock_reclaims_stale_owner(workspace) -> None:
    store = _create_store(workspace, "stale-lock", "model-a")
    store.sync([{"role": "user", "content": "hello"}])
    lock_path = store.path.with_suffix(".lock")
    lock_path.write_text(json.dumps({
        "pid": 999_999_999,
        "token": "stale",
        "acquired_at": "2026-01-01T00:00:00Z",
    }))

    reopened = SessionStore.open(
        store.path, workspace, "model-a", acquire_lock=True,
    )
    try:
        assert lock_path.exists()
        lock_data = json.loads(lock_path.read_text())
        assert lock_data["pid"] == os.getpid()
        assert lock_data["token"] != "stale"
    finally:
        reopened.close()

    assert not lock_path.exists()


def test_invalid_session_lock_is_rejected_without_deleting_it(workspace) -> None:
    store = _create_store(workspace, "invalid-lock", "model-a")
    store.sync([{"role": "user", "content": "hello"}])
    lock_path = store.path.with_suffix(".lock")
    lock_path.write_text("{not-json}")

    with pytest.raises(ValueError, match="lock is invalid"):
        SessionStore.open(
            store.path, workspace, "model-a", acquire_lock=True,
        )

    assert lock_path.read_text() == "{not-json}"


# ---------------------------------------------------------------------------
# message_utils sanitizers
# ---------------------------------------------------------------------------


def test_sanitize_same_model_returns_copy() -> None:
    msg = {"type": "reasoning", "id": "r1", "summary": "thinking"}
    result = sanitize_resumed_message(msg, same_model=True)
    assert result == msg
    assert result is not msg  # should be a copy


def test_sanitize_different_model_drops_reasoning() -> None:
    msg = {"type": "reasoning", "id": "r1", "summary": "thinking"}
    assert sanitize_resumed_message(msg, same_model=False) == {}


def test_sanitize_different_model_strips_function_call_id() -> None:
    msg = {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "bash", "arguments": "{}"}
    result = sanitize_resumed_message(msg, same_model=False)
    assert "id" not in result
    assert result["call_id"] == "c1"
    assert result["name"] == "bash"


def test_sanitize_preserves_regular_messages() -> None:
    msg = {"role": "user", "content": "hello"}
    assert sanitize_resumed_message(msg, same_model=False) == msg


def test_drop_orphan_trailing_function_call() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
    ]
    result = drop_orphan_tool_calls(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_drop_orphan_preserves_paired_calls() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
        {"role": "assistant", "content": "result"},
    ]
    result = drop_orphan_tool_calls(messages)
    assert len(result) == 4


def test_drop_orphan_output_without_call() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"type": "function_call_output", "call_id": "orphan", "output": "stray"},
    ]
    result = drop_orphan_tool_calls(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_drop_orphan_empty_list() -> None:
    assert drop_orphan_tool_calls([]) == []


def test_drop_orphan_multiple_trailing_calls() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "read_file", "arguments": "{}"},
    ]
    result = drop_orphan_tool_calls(messages)
    assert len(result) == 1


def test_drop_orphan_handles_partially_completed_parallel_batch() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
    ]
    result = drop_orphan_tool_calls(messages)

    assert [item.get("call_id") for item in result if item.get("type") == "function_call"] == ["c1"]
    assert not any(item.get("call_id") == "c2" for item in result)


def test_drop_orphan_rejects_output_that_precedes_call() -> None:
    messages = [
        {"type": "function_call_output", "call_id": "c1", "output": "early"},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
    ]
    assert drop_orphan_tool_calls(messages) == []


# ---------------------------------------------------------------------------
# todo_state persistence
# ---------------------------------------------------------------------------


def test_todo_state_persisted_and_restored(workspace) -> None:
    store = _create_store(workspace, "test-todo-1", "model-a")
    store.sync([{"role": "user", "content": "do something"}])
    todo_items = [
        {"content": "step 1", "status": "completed", "activeForm": ""},
        {"content": "step 2", "status": "in_progress", "activeForm": "working on step 2"},
    ]
    store.sync_todo(todo_items)

    reopened = SessionStore.open(store.path, workspace, "model-a")
    restored = reopened.last_todo_items()
    assert restored is not None
    assert len(restored) == 2
    assert restored[0]["content"] == "step 1"
    assert restored[1]["status"] == "in_progress"


def test_todo_state_empty_clears_previous_plan(workspace) -> None:
    store = _create_store(workspace, "test-todo-2", "model-a")
    store.sync([{"role": "user", "content": "hello"}])
    store.sync_todo([{"content": "old step", "status": "pending"}])
    store.sync_todo([])
    store.sync_todo(None)

    assert store.last_todo_items() == []
    reopened = SessionStore.open(store.path, workspace, "model-a")
    assert reopened.last_todo_items() == []


def test_todo_state_not_in_message_projection(workspace) -> None:
    store = _create_store(workspace, "test-todo-3", "model-a")
    store.sync([{"role": "user", "content": "hello"}])
    store.sync_todo([{"content": "step 1", "status": "pending"}])

    projected = store.messages()
    assert len(projected) == 1
    assert projected[0]["role"] == "user"


def test_resume_with_todo_does_not_create_spurious_history_reset(workspace) -> None:
    store = _create_store(workspace, "test-todo-prefix", "model-a")
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    store.sync(history)
    store.sync_todo([{"content": "step 1", "status": "pending"}])

    reopened = SessionStore.open(store.path, workspace, "model-a")
    count_before = reopened.entry_count
    reopened.sync(history)

    assert reopened.entry_count == count_before
    lines = [
        json.loads(line)
        for line in reopened.path.read_text().splitlines()
        if line.strip()
    ]
    assert not any(entry.get("type") == "history_reset" for entry in lines)


def test_history_prefix_change_is_detected_even_when_tail_is_unchanged(workspace) -> None:
    store = _create_store(workspace, "test-prefix", "model-a")
    original = [
        {"role": "user", "content": "old"},
        _assistant(_text("same tail")),
    ]
    store.sync(original)
    changed = [
        {"role": "user", "content": "new"},
        _assistant(_text("same tail")),
    ]
    store.sync(changed)

    assert store.messages() == changed
    lines = [
        json.loads(line)
        for line in store.path.read_text().splitlines()
        if line.strip()
    ]
    assert any(entry.get("type") == "history_reset" for entry in lines)
