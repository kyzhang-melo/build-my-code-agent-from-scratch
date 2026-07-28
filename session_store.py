"""Session persistence as append-only JSONL.

Each session is one ``.sessions/<session_id>.jsonl`` file. The first line is a
``session_header``; subsequent lines are ``message`` entries (one per history
dict) or ``history_reset`` markers.

Because ``context_compact.py`` still rewrites ``state.messages`` in place, the
store detects compaction by comparing the live history against the persisted
prefix. When they diverge, a ``history_reset`` entry is appended and the full
current history is re-written. The projection (``messages()``) returns only
what follows the last ``history_reset`` (or everything if there is none).

Resume-time sanitization (drop reasoning on model change, drop orphan tool
calls) is delegated to ``message_utils``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from message_utils import ResumeSanitizeDiagnostics, sanitize_resumed_history
from workspace import Workspace

CURRENT_SESSION_VERSION = 1
SESSIONS_DIR_NAME = ".sessions"
SESSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RESERVED_SESSION_NAMES = {"last", "continue", "new"}
SESSION_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(existing: set[str]) -> str:
    for _ in range(100):
        sid = uuid.uuid4().hex[:8]
        if sid not in existing:
            return sid
    return uuid.uuid4().hex


def _message_hash(msg: dict) -> str:
    return hashlib.sha256(
        json.dumps(msg, default=str, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def validate_session_id(session_id: str) -> str:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(
            "Session id must contain only letters, digits, '.', '_', or '-', "
            "and must start and end with a letter or digit"
        )
    return session_id


def validate_session_name(session_name: str | None) -> str:
    if session_name is None or not session_name.strip():
        return ""
    name = session_name.strip()
    if not SESSION_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Session name must be 1-64 characters and contain only letters, "
            "digits, '-', or '_'"
        )
    if name.casefold() in RESERVED_SESSION_NAMES:
        raise ValueError(f"Session name is reserved: {name}")
    return name


def ensure_session_name_available(
    session_dir: Path,
    session_name: str,
) -> None:
    folded = session_name.casefold()
    for header in list_session_headers(session_dir):
        if str(header.get("session_name", "")).casefold() == folded:
            raise ValueError(f"Session name already exists: {session_name}")


class SessionStoreProtocol(Protocol):
    session_id: str
    resume_diagnostics: ResumeSanitizeDiagnostics

    def sync(self, history: list[dict]) -> None:
        ...

    def messages(self) -> list[dict]:
        ...

    def sync_todo(self, todo_items: list | None) -> None:
        ...

    def last_todo_items(self) -> list | None:
        ...

    def close(self) -> None:
        ...

    @property
    def is_persistent(self) -> bool:
        ...


@dataclass(frozen=True)
class SessionHeader:
    type: str = "session_header"
    version: int = CURRENT_SESSION_VERSION
    session_id: str = ""
    created_at: str = ""
    cwd: str = ""
    model_id: str = ""
    provider: str = ""
    session_name: str = ""
    updated_at: str = ""


@dataclass
class SessionEntry:
    type: str
    id: str
    timestamp: str
    # For ``message`` entries: the history dict and the model that produced it.
    # For ``history_reset`` entries: reason (e.g. "compaction").
    # For ``todo_state`` entries: the serialized todo items.
    message: dict | None = None
    model_id: str = ""
    provider: str = ""
    reason: str = ""
    todo_items: list | None = None


@dataclass
class SessionStore:
    """Append-only JSONL session log.

    Created via ``SessionStore.create(...)`` for new sessions or
    ``SessionStore.open(...)`` for resume. Use ``NullSessionStore`` for tests
    and evals that must not touch the filesystem.
    """

    workspace: Workspace
    session_id: str
    model_id: str
    provider: str
    path: Path
    session_name: str = ""
    created_at: str = ""
    updated_at: str = ""
    resume_diagnostics: ResumeSanitizeDiagnostics = field(
        default_factory=ResumeSanitizeDiagnostics
    )
    _entries: list[SessionEntry] = field(default_factory=list)
    _entry_ids: set[str] = field(default_factory=set)
    _persisted_message_hashes: list[str] = field(default_factory=list)
    _file_created: bool = False
    _invalid_lines: int = 0
    _lock_path: Path | None = None
    _lock_token: str = ""

    # -- constructors --

    @classmethod
    def create(
        cls,
        workspace: Workspace,
        session_id: str,
        model_id: str,
        provider: str = "",
        session_name: str | None = None,
        *,
        acquire_lock: bool = False,
    ) -> SessionStore:
        validate_session_id(session_id)
        normalized_name = validate_session_name(session_name)
        store = cls(
            workspace=workspace,
            session_id=session_id,
            model_id=model_id,
            provider=provider,
            path=cls._resolve_path(workspace, session_id),
            session_name=normalized_name,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        if normalized_name:
            ensure_session_name_available(
                workspace.root / SESSIONS_DIR_NAME,
                normalized_name,
            )
        if acquire_lock:
            store._acquire_lock()
        return store

    @classmethod
    def open(
        cls,
        path: str | Path,
        workspace: Workspace,
        current_model_id: str,
        current_provider: str = "",
        *,
        acquire_lock: bool = False,
    ) -> SessionStore:
        """Load an existing session file for resume.

        Raises ``ValueError`` if the file is missing, has no valid header, or
        the header cwd does not match the workspace root.
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise ValueError(f"Session file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Session path is not a file: {resolved}")

        entries_raw, invalid_lines = cls._load_file(resolved)
        if not entries_raw:
            raise ValueError(f"Session file is empty: {resolved}")

        header = cls._parse_header(entries_raw[0])
        if header is None:
            raise ValueError(f"Session file has no valid header: {resolved}")
        if header.version != CURRENT_SESSION_VERSION:
            if header.version > CURRENT_SESSION_VERSION:
                raise ValueError(
                    f"Session version {header.version} is newer than supported "
                    f"version {CURRENT_SESSION_VERSION}"
                )
            raise ValueError(
                f"Session version {header.version} requires a migration"
            )
        validate_session_id(header.session_id)

        header_cwd = Path(header.cwd).resolve()
        if header_cwd != workspace.root:
            raise ValueError(
                f"Session cwd mismatch: header says {header.cwd}, "
                f"current workspace is {workspace.root}"
            )

        store = cls(
            workspace=workspace,
            session_id=header.session_id,
            model_id=current_model_id,
            provider=current_provider,
            path=resolved,
            session_name=header.session_name,
            created_at=header.created_at,
            updated_at=header.updated_at,
            _invalid_lines=invalid_lines,
        )
        store._load_entries(entries_raw[1:])
        store._file_created = True
        store._rebuild_persisted_state()
        if acquire_lock:
            store._acquire_lock()
        return store

    # -- public API --

    def sync(self, history: list[dict]) -> None:
        """Persist the current history at a turn boundary.

        Appends new entries for messages not yet persisted. If the history has
        diverged from the persisted prefix (e.g. after compaction), appends a
        ``history_reset`` and re-writes the full current history.
        """
        if not self._persisted_message_hashes:
            if not history:
                return
            # No persisted prefix yet (first sync or after a reset was already
            # written). Just append all messages.
            self._append_message_entries(history)
        else:
            same_prefix = self._history_matches_prefix(history)
            if same_prefix:
                new_messages = history[len(self._persisted_message_hashes):]
                if not new_messages:
                    return
                self._append_message_entries(new_messages)
            else:
                self._append_reset("compaction")
                self._append_message_entries(history)
        self._flush()

    def messages(self) -> list[dict]:
        """Project persisted entries back into a history list (sanitized)."""
        projected: list[dict] = []
        reset_index = self._last_reset_index()
        for entry in self._entries[reset_index:]:
            if entry.type == "message" and entry.message is not None:
                projected.append(entry.message)
        same_runtime = all(
            entry.type != "message"
            or (
                entry.model_id == self.model_id
                and entry.provider == self.provider
            )
            for entry in self._entries[reset_index:]
        )
        messages, diagnostics = sanitize_resumed_history(
            projected,
            same_runtime=same_runtime,
            invalid_lines=self._invalid_lines,
        )
        self.resume_diagnostics = diagnostics
        return messages

    def sync_todo(self, todo_items: list | None) -> None:
        """Persist the current todo state as a ``todo_state`` entry.

        Empty items are a tombstone: they prevent a previously active plan
        from being resurrected on resume.
        """
        normalized = list(todo_items or [])
        previous = self.last_todo_items()
        if previous is not None and previous == normalized:
            return
        entry = SessionEntry(
            type="todo_state",
            id=_short_id(self._entry_ids),
            timestamp=_now_iso(),
            todo_items=normalized,
        )
        self._entries.append(entry)
        self._entry_ids.add(entry.id)
        self._flush()

    def last_todo_items(self) -> list | None:
        """Return the todo items from the most recent ``todo_state`` entry."""
        for entry in reversed(self._entries):
            if entry.type == "todo_state" and entry.todo_items is not None:
                return entry.todo_items
        return None

    def header(self) -> SessionHeader | None:
        if not self._entries and not self._file_created:
            return None
        return SessionHeader(
            session_id=self.session_id,
            created_at=self.created_at or self._first_timestamp() or _now_iso(),
            cwd=str(self.workspace.root),
            model_id=self.model_id,
            provider=self.provider,
            session_name=self.session_name,
            updated_at=self.updated_at,
        )

    def close(self) -> None:
        self._release_lock()

    @property
    def is_persistent(self) -> bool:
        return True

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    # -- internal: loading --

    @staticmethod
    def _load_file(path: Path) -> tuple[list[dict], int]:
        entries: list[dict] = []
        invalid_lines = 0
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    if line_number == 1:
                        raise ValueError(f"Invalid session header JSON: {path}")
                    invalid_lines += 1
                    continue
                if not isinstance(value, dict):
                    invalid_lines += 1
                    continue
                entries.append(value)
        return entries, invalid_lines

    @staticmethod
    def _parse_header(raw: dict) -> SessionHeader | None:
        if not isinstance(raw, dict) or raw.get("type") != "session_header":
            return None
        try:
            return SessionHeader(
                session_id=str(raw.get("session_id", "")),
                created_at=str(raw.get("created_at", "")),
                cwd=str(raw.get("cwd", "")),
                model_id=str(raw.get("model_id", "")),
                provider=str(raw.get("provider", "")),
                session_name=str(raw.get("session_name", "")),
                updated_at=str(raw.get("updated_at", "")),
                version=int(raw.get("version", 1)),
            )
        except (TypeError, ValueError):
            return None

    def _load_entries(self, raw_entries: list[dict]) -> None:
        for raw in raw_entries:
            entry_type = raw.get("type", "")
            if entry_type == "message":
                msg = raw.get("message")
                if not isinstance(msg, dict):
                    self._invalid_lines += 1
                    continue
                entry = SessionEntry(
                    type="message",
                    id=str(raw.get("id", "")),
                    timestamp=str(raw.get("timestamp", "")),
                    message=msg,
                    model_id=str(raw.get("model_id", "")),
                    provider=str(raw.get("provider", "")),
                )
            elif entry_type == "history_reset":
                entry = SessionEntry(
                    type="history_reset",
                    id=str(raw.get("id", "")),
                    timestamp=str(raw.get("timestamp", "")),
                    reason=str(raw.get("reason", "")),
                )
            elif entry_type == "todo_state":
                items = raw.get("todo_items")
                if not isinstance(items, list):
                    self._invalid_lines += 1
                    continue
                entry = SessionEntry(
                    type="todo_state",
                    id=str(raw.get("id", "")),
                    timestamp=str(raw.get("timestamp", "")),
                    todo_items=items,
                )
            else:
                self._invalid_lines += 1
                continue
            self._entries.append(entry)
            if entry.id:
                self._entry_ids.add(entry.id)

    def _rebuild_persisted_state(self) -> None:
        """Recompute persisted_count and tail hash from loaded entries."""
        reset_index = self._last_reset_index()
        self._persisted_message_hashes = [
            _message_hash(entry.message)
            for entry in self._entries[reset_index:]
            if entry.type == "message" and entry.message is not None
        ]

    # -- internal: writing --

    @staticmethod
    def _resolve_path(workspace: Workspace, session_id: str) -> Path:
        validate_session_id(session_id)
        return workspace.root / SESSIONS_DIR_NAME / f"{session_id}.jsonl"

    def _history_matches_prefix(self, history: list[dict]) -> bool:
        if not self._persisted_message_hashes:
            return False
        if len(history) < len(self._persisted_message_hashes):
            return False
        return [
            _message_hash(msg)
            for msg in history[:len(self._persisted_message_hashes)]
        ] == self._persisted_message_hashes

    def _append_message_entries(self, messages: list[dict]) -> None:
        for msg in messages:
            entry = SessionEntry(
                type="message",
                id=_short_id(self._entry_ids),
                timestamp=_now_iso(),
                message=dict(msg),
                model_id=self.model_id,
                provider=self.provider,
            )
            self._entries.append(entry)
            self._entry_ids.add(entry.id)
            self._persisted_message_hashes.append(_message_hash(msg))

    def _append_reset(self, reason: str) -> None:
        entry = SessionEntry(
            type="history_reset",
            id=_short_id(self._entry_ids),
            timestamp=_now_iso(),
            reason=reason,
        )
        self._entries.append(entry)
        self._entry_ids.add(entry.id)
        self._persisted_message_hashes = []

    def _flush(self) -> None:
        """Write the file. First write uses exclusive create (``x``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "type": "session_header",
            "version": CURRENT_SESSION_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at or self._first_timestamp() or _now_iso(),
            "updated_at": _now_iso(),
            "cwd": str(self.workspace.root),
            "model_id": self.model_id,
            "provider": self.provider,
            "session_name": self.session_name,
        }
        self.updated_at = header["updated_at"]
        lines = [json.dumps(header, ensure_ascii=False)]
        for entry in self._entries:
            lines.append(json.dumps(self._entry_to_dict(entry), ensure_ascii=False))
        content = "\n".join(lines) + "\n"

        tmp = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            tmp.write_text(content, encoding="utf-8")
            if not self._file_created:
                # Hard-linking a complete same-directory temp file gives the
                # initial write atomic, exclusive-create semantics.
                try:
                    os.link(tmp, self.path)
                except FileExistsError as exc:
                    raise ValueError(
                        f"Session file already exists: {self.path}"
                    ) from exc
                self._file_created = True
            else:
                os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _entry_to_dict(entry: SessionEntry) -> dict:
        if entry.type == "message":
            return {
                "type": "message",
                "id": entry.id,
                "timestamp": entry.timestamp,
                "model_id": entry.model_id,
                "provider": entry.provider,
                "message": entry.message,
            }
        if entry.type == "todo_state":
            return {
                "type": "todo_state",
                "id": entry.id,
                "timestamp": entry.timestamp,
                "todo_items": entry.todo_items,
            }
        return {
            "type": "history_reset",
            "id": entry.id,
            "timestamp": entry.timestamp,
            "reason": entry.reason,
        }

    # -- internal: queries --

    def _last_reset_index(self) -> int:
        for i in range(len(self._entries) - 1, -1, -1):
            if self._entries[i].type == "history_reset":
                return i + 1
        return 0

    def _first_timestamp(self) -> str:
        for entry in self._entries:
            if entry.timestamp:
                return entry.timestamp
        return ""

    def _acquire_lock(self) -> None:
        lock_path = self.path.with_suffix(".lock")
        token = uuid.uuid4().hex
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "pid": os.getpid(),
            "token": token,
            "acquired_at": _now_iso(),
        })
        for _ in range(2):
            try:
                fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                try:
                    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                    owner_pid = int(lock_data["pid"])
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    raise ValueError(f"Session lock is invalid: {lock_path}")
                try:
                    os.kill(owner_pid, 0)
                except ProcessLookupError:
                    lock_path.unlink(missing_ok=True)
                    continue
                except PermissionError:
                    pass
                raise ValueError(
                    f"Session is already open by process {owner_pid}: {self.session_id}"
                )
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                self._lock_path = lock_path
                self._lock_token = token
                return
        raise ValueError(f"Could not acquire session lock: {self.session_id}")

    def _release_lock(self) -> None:
        if self._lock_path is None:
            return
        try:
            data = json.loads(self._lock_path.read_text(encoding="utf-8"))
            if data.get("token") == self._lock_token:
                self._lock_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
        finally:
            self._lock_path = None
            self._lock_token = ""


class NullSessionStore:
    """No-op store for tests and evals. Writes nothing, reads nothing."""

    session_id: str = ""
    resume_diagnostics = ResumeSanitizeDiagnostics()

    def sync(self, history: list[dict]) -> None:
        pass

    def messages(self) -> list[dict]:
        return []

    def sync_todo(self, todo_items: list | None) -> None:
        del todo_items

    def last_todo_items(self) -> list | None:
        return None

    def close(self) -> None:
        pass

    @property
    def is_persistent(self) -> bool:
        return False

    def header(self) -> None:
        return None

    @property
    def entry_count(self) -> int:
        return 0

    @property
    def path(self) -> Path | None:
        return None


def find_most_recent_session(session_dir: Path, cwd: Path) -> Path | None:
    """Return the most recently modified session file whose header cwd matches.

    Only reads the first line of each file (header peek), not the full file.
    """
    if not session_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for f in session_dir.glob("*.jsonl"):
        try:
            with f.open("r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()
            if not first_line:
                continue
            header = json.loads(first_line)
            if header.get("type") != "session_header":
                continue
            header_cwd = Path(str(header.get("cwd", ""))).resolve()
            if header_cwd != cwd:
                continue
            mtime = f.stat().st_mtime
            candidates.append((mtime, f))
        except (json.JSONDecodeError, OSError):
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def list_session_headers(session_dir: Path) -> list[dict[str, Any]]:
    """List session headers (first line only) for ``--list-sessions``."""
    if not session_dir.exists():
        return []
    headers: list[dict[str, Any]] = []
    for f in sorted(session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with f.open("r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()
            if not first_line:
                continue
            header = json.loads(first_line)
            if header.get("type") != "session_header":
                continue
            header["_path"] = str(f)
            header["_mtime"] = f.stat().st_mtime
            headers.append(header)
        except (json.JSONDecodeError, OSError):
            continue
    return headers
