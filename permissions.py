"""Permission decisions and terminal approval handling for tool execution."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


READ_ONLY_TOOLS = {"read_file", "glob", "grep", "task", "todo"}
CONTROLLED_TOOLS = {"write_file", "edit_file", "bash"}

SENSITIVE_BASENAMES = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


ApprovalKind = Literal["approve", "approve_for_session", "reject"]


@dataclass(frozen=True)
class PermissionDecision:
    behavior: PermissionBehavior
    reason: str
    action: str | None = None
    allow_for_session: bool = False


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    action: str
    description: str
    allow_for_session: bool
    source: str = "parent"


@dataclass(frozen=True)
class ApprovalResponse:
    kind: ApprovalKind
    feedback: str = ""


@dataclass
class PermissionManager:
    workdir: Path
    mode: PermissionMode = PermissionMode.DEFAULT
    auto_approve_actions: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.workdir = self.workdir.resolve()

    def set_mode(self, mode: str | PermissionMode) -> None:
        self.mode = PermissionMode(mode)

    def remember(self, action: str) -> None:
        self.auto_approve_actions.add(action)

    def check(self, tool_name: str, tool_input: dict) -> PermissionDecision:
        guarded = self._core_guard(tool_name, tool_input)
        if guarded is not None:
            return guarded

        if tool_name in CONTROLLED_TOOLS and self.mode is PermissionMode.PLAN:
            return PermissionDecision(
                behavior=PermissionBehavior.DENY,
                reason=f"Plan mode blocks {tool_name} because it can modify state.",
            )

        if tool_name in {"write_file", "edit_file"}:
            target = self._resolve_path(str(tool_input.get("path", "")))
            action = f"file_mutation:{target}"
            if action in self.auto_approve_actions:
                return PermissionDecision(
                    behavior=PermissionBehavior.ALLOW,
                    reason=f"Session approval covers {target}.",
                    action=action,
                    allow_for_session=True,
                )
            return PermissionDecision(
                behavior=PermissionBehavior.ASK,
                reason=f"{tool_name} will modify {target}.",
                action=action,
                allow_for_session=True,
            )

        if tool_name == "bash":
            command = str(tool_input.get("command", ""))
            return PermissionDecision(
                behavior=PermissionBehavior.ASK,
                reason="Shell commands require approval.",
                action=f"bash:{command}",
                allow_for_session=False,
            )

        if tool_name in READ_ONLY_TOOLS:
            return PermissionDecision(
                behavior=PermissionBehavior.ALLOW,
                reason=f"{tool_name} is an allowed read-only or session-control tool.",
            )

        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            reason=f"No permission policy is configured for {tool_name}.",
            action=f"tool:{tool_name}",
            allow_for_session=False,
        )

    def describe_request(self, tool_name: str, tool_input: dict, decision: PermissionDecision) -> str:
        if tool_name == "bash":
            return f"Run shell command:\n{tool_input.get('command', '')}"
        if tool_name in {"write_file", "edit_file"}:
            return f"Allow {tool_name} on {self._resolve_path(str(tool_input.get('path', '')))}?"
        return decision.reason

    def _core_guard(self, tool_name: str, tool_input: dict) -> PermissionDecision | None:
        if tool_name in {"read_file", "write_file", "edit_file"}:
            raw_path = str(tool_input.get("path", ""))
            try:
                target = self._resolve_path(raw_path)
            except ValueError as exc:
                return PermissionDecision(PermissionBehavior.DENY, str(exc))

            if tool_name == "read_file" and is_sensitive_path(target, self.workdir):
                return PermissionDecision(
                    PermissionBehavior.DENY,
                    f"Reading sensitive file '{raw_path}' is blocked.",
                )

            if tool_name in {"write_file", "edit_file"}:
                if is_protected_write_path(target, self.workdir):
                    return PermissionDecision(
                        PermissionBehavior.DENY,
                        f"Writing protected path '{raw_path}' is blocked.",
                    )

        if tool_name == "bash":
            reason = bash_hard_deny_reason(str(tool_input.get("command", "")), self.workdir)
            if reason:
                return PermissionDecision(PermissionBehavior.DENY, reason)

        return None

    def _resolve_path(self, value: str) -> Path:
        target = (self.workdir / os.path.expanduser(value)).resolve()
        if not target.is_relative_to(self.workdir):
            raise ValueError(f"Path escapes workspace: {value}")
        return target


class TerminalApprovalHandler:
    def __init__(
        self,
        *,
        interactive: bool,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.interactive = interactive
        self._input_fn = input_fn
        self._lock = asyncio.Lock()

    async def request(self, request: ApprovalRequest) -> ApprovalResponse:
        if not self.interactive:
            return ApprovalResponse(
                "reject",
                "Interactive approval is unavailable in headless mode.",
            )

        options = "[y]es / [n]o"
        if request.allow_for_session:
            options += " / [a]llow this path for the session"
        prompt = f"\n[permission] {request.description}\n{options}: "

        async with self._lock:
            for _ in range(3):
                try:
                    answer = (await asyncio.to_thread(self._input_fn, prompt)).strip()
                except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
                    return ApprovalResponse("reject", "Approval was cancelled.")

                lowered = answer.lower()
                if lowered in {"y", "yes"}:
                    return ApprovalResponse("approve")
                if request.allow_for_session and lowered in {"a", "always", "session"}:
                    return ApprovalResponse("approve_for_session")
                if lowered in {"n", "no"}:
                    return ApprovalResponse("reject")
                if lowered.startswith("n:"):
                    return ApprovalResponse("reject", answer[2:].strip())
                if lowered.startswith("no "):
                    return ApprovalResponse("reject", answer[3:].strip())

            return ApprovalResponse("reject", "No valid approval response was provided.")


@dataclass
class PermissionService:
    manager: PermissionManager
    handler: TerminalApprovalHandler | None = None

    async def authorize(
        self,
        tool_name: str,
        tool_input: dict,
        *,
        source: str = "parent",
    ) -> PermissionDecision:
        decision = self.manager.check(tool_name, tool_input)
        if decision.behavior is not PermissionBehavior.ASK:
            return decision

        if self.handler is None:
            return PermissionDecision(
                PermissionBehavior.DENY,
                "Interactive approval is unavailable in headless mode.",
                action=decision.action,
            )

        request = ApprovalRequest(
            tool_name=tool_name,
            action=decision.action or f"tool:{tool_name}",
            description=self.manager.describe_request(tool_name, tool_input, decision),
            allow_for_session=decision.allow_for_session,
            source=source,
        )
        response = await self.handler.request(request)
        if response.kind == "approve_for_session" and request.allow_for_session:
            self.manager.remember(request.action)
            return PermissionDecision(
                PermissionBehavior.ALLOW,
                "Approved for this session.",
                action=request.action,
                allow_for_session=True,
            )
        if response.kind == "approve":
            return PermissionDecision(
                PermissionBehavior.ALLOW,
                "Approved by user.",
                action=request.action,
            )

        reason = "Permission denied by user."
        if response.feedback:
            reason += f" Feedback: {response.feedback}"
        if source != "parent":
            reason += " Do not retry or attempt to bypass this restriction indirectly."
        return PermissionDecision(PermissionBehavior.DENY, reason, action=request.action)


def permission_denied_output(tool_name: str, decision: PermissionDecision) -> str:
    return json.dumps(
        {
            "error": "permission_denied",
            "tool": tool_name,
            "reason": decision.reason,
            "retryable": False,
        },
        ensure_ascii=True,
    )


def is_sensitive_path(path: Path, workdir: Path) -> bool:
    name = path.name.lower()
    if name in SENSITIVE_BASENAMES or name.startswith(".env."):
        return True
    if path.suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    try:
        parts = tuple(part.lower() for part in path.relative_to(workdir).parts)
    except ValueError:
        return True
    return len(parts) >= 2 and parts[-2:] == (".git", "config")


def is_protected_write_path(path: Path, workdir: Path) -> bool:
    if is_sensitive_path(path, workdir):
        return True
    try:
        parts = tuple(part.lower() for part in path.relative_to(workdir).parts)
    except ValueError:
        return True
    return ".git" in parts or ".ssh" in parts


def bash_hard_deny_reason(command: str, workdir: Path) -> str | None:
    if re.search(r"\bsudo\b", command):
        return "Privilege escalation with sudo is blocked."
    if re.search(r"\b(?:shutdown|reboot|halt|poweroff)\b", command):
        return "System shutdown commands are blocked."
    if re.search(r"(?:>|>>|1>|2>)\s*/dev/", command):
        return "Writing to device files is blocked."

    for match in re.finditer(r"(?:^|[;&|]\s*)cd\s+([^\s;&|]+)", command):
        target = _shell_path(match.group(1), workdir)
        if target is not None and not target.is_relative_to(workdir):
            return f"Changing directory outside the workspace is blocked: {match.group(1)}"

    for match in re.finditer(r"(?:^|[\s;&|])(?:[12])?>>?\s*([^\s;&|]+)", command):
        target = _shell_path(match.group(1), workdir)
        if target is not None and not target.is_relative_to(workdir):
            return f"Redirecting output outside the workspace is blocked: {match.group(1)}"

    for segment in re.split(r"&&|\|\||;|\|", command):
        try:
            argv = shlex.split(segment.strip())
        except ValueError:
            continue
        if not argv or Path(argv[0]).name != "rm":
            continue
        recursive = any(arg.startswith("-") and "r" in arg.lower() for arg in argv[1:])
        if not recursive:
            continue
        for arg in argv[1:]:
            if arg.startswith("-"):
                continue
            target = _shell_path(arg, workdir)
            if target is not None and not target.is_relative_to(workdir):
                return f"Recursive deletion outside the workspace is blocked: {arg}"

    return None


def _shell_path(raw: str, workdir: Path) -> Path | None:
    value = raw.strip("'\"")
    if not value or any(char in value for char in "$`*?{}[]"):
        return None
    return (workdir / os.path.expanduser(value)).resolve()
