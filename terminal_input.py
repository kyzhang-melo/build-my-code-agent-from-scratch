"""Session-scoped terminal input lifecycle management."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import BaseStyle


class InputState(str, Enum):
    IDLE = "idle"
    PROMPTING = "prompting"
    PROCESSING = "processing"
    CLOSING = "closing"
    CLOSED = "closed"


class TerminalInput:
    """Own stdin and one prompt_toolkit session for a single CLI run."""

    def __init__(
        self,
        *,
        session: PromptSession[str] | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        style: BaseStyle | None = None,
    ) -> None:
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._is_tty = self._stdin.isatty()
        self._session = session
        if self._is_tty and self._session is None:
            self._session = PromptSession(multiline=False, style=style)
        self._lock = asyncio.Lock()
        self._state = InputState.IDLE

    @property
    def state(self) -> InputState:
        return self._state

    async def prompt(self, message: AnyFormattedText = "") -> str:
        """Read one input while serializing all access to the terminal."""
        async with self._lock:
            if self._state in {InputState.CLOSING, InputState.CLOSED}:
                raise EOFError

            previous = self._state
            self._state = InputState.PROMPTING
            try:
                if self._is_tty:
                    assert self._session is not None
                    with patch_stdout(raw=True):
                        return await self._session.prompt_async(message)
                return await self._read_non_tty(message)
            finally:
                if self._state == InputState.PROMPTING:
                    self._state = (
                        InputState.PROCESSING
                        if previous == InputState.PROCESSING
                        else InputState.IDLE
                    )

    @contextmanager
    def processing(self) -> Iterator[None]:
        """Mark agent work while allowing a nested approval prompt."""
        if self._state != InputState.IDLE:
            raise RuntimeError(f"Cannot start processing while input state is {self._state.value}")
        self._state = InputState.PROCESSING
        try:
            yield
        finally:
            if self._state == InputState.PROCESSING:
                self._state = InputState.IDLE

    async def close(self) -> None:
        """Finish the lifecycle after any active prompt releases stdin."""
        async with self._lock:
            if self._state == InputState.CLOSED:
                return
            self._state = InputState.CLOSING
            self._state = InputState.CLOSED

    async def _read_non_tty(self, message: AnyFormattedText) -> str:
        prompt_text = "".join(fragment[1] for fragment in to_formatted_text(message))
        self._stdout.write(prompt_text)
        self._stdout.flush()
        line = await asyncio.to_thread(self._stdin.readline)
        if line == "":
            raise EOFError
        if line.endswith("\n"):
            line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
        return line
