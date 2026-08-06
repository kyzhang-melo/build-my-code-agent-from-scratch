from __future__ import annotations

import asyncio
import io

import pytest

from terminal_input import InputState, TerminalInput


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_non_tty_prompt_preserves_text_and_line_semantics() -> None:
    stdin = io.StringIO("hello中文world\r\n")
    stdout = io.StringIO()
    terminal = TerminalInput(stdin=stdin, stdout=stdout)

    result = asyncio.run(terminal.prompt("s01 >> "))

    assert result == "hello中文world"
    assert stdout.getvalue() == "s01 >> "
    assert terminal.state == InputState.IDLE


def test_non_tty_eof_restores_idle_state() -> None:
    terminal = TerminalInput(stdin=io.StringIO(), stdout=io.StringIO())

    with pytest.raises(EOFError):
        asyncio.run(terminal.prompt("s01 >> "))

    assert terminal.state == InputState.IDLE


def test_prompt_calls_are_serialized() -> None:
    async def scenario() -> None:
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        calls: list[str] = []

        class Session:
            async def prompt_async(self, message):
                calls.append(str(message))
                if len(calls) == 1:
                    first_entered.set()
                    await release_first.wait()
                return f"answer-{len(calls)}"

        terminal = TerminalInput(session=Session(), stdin=_TTY())
        first = asyncio.create_task(terminal.prompt("first"))
        await first_entered.wait()
        assert terminal.state == InputState.PROMPTING

        second = asyncio.create_task(terminal.prompt("second"))
        await asyncio.sleep(0)
        assert len(calls) == 1

        release_first.set()
        assert await first == "answer-1"
        assert await second == "answer-2"
        assert terminal.state == InputState.IDLE

    asyncio.run(scenario())


def test_processing_prompt_restores_processing_state() -> None:
    class Session:
        async def prompt_async(self, _message):
            return "y"

    terminal = TerminalInput(session=Session(), stdin=_TTY())

    async def scenario() -> None:
        with terminal.processing():
            assert terminal.state == InputState.PROCESSING
            assert await terminal.prompt("approve?") == "y"
            assert terminal.state == InputState.PROCESSING
        assert terminal.state == InputState.IDLE

    asyncio.run(scenario())


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt(), asyncio.CancelledError()])
def test_prompt_failure_restores_idle_state(error: BaseException) -> None:
    class Session:
        async def prompt_async(self, _message):
            raise error

    terminal = TerminalInput(session=Session(), stdin=_TTY())

    with pytest.raises(type(error)):
        asyncio.run(terminal.prompt("s01 >> "))

    assert terminal.state == InputState.IDLE


def test_close_is_idempotent_and_blocks_future_prompts() -> None:
    terminal = TerminalInput(stdin=io.StringIO("unused\n"), stdout=io.StringIO())

    async def scenario() -> None:
        await terminal.close()
        await terminal.close()
        assert terminal.state == InputState.CLOSED
        with pytest.raises(EOFError):
            await terminal.prompt("s01 >> ")

    asyncio.run(scenario())


def test_real_prompt_session_handles_output_while_waiting() -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    async def scenario() -> str:
        with create_pipe_input() as pipe_input:
            session = PromptSession[str](input=pipe_input, output=DummyOutput())
            terminal = TerminalInput(session=session, stdin=_TTY())
            task = asyncio.create_task(terminal.prompt("s01 >> "))
            await asyncio.sleep(0)
            print("background output")
            pipe_input.send_text("mixed中英\n")
            return await task

    assert asyncio.run(scenario()) == "mixed中英"
