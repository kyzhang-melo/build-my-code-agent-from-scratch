"""The workspace root a session's tools are confined to."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """A resolved root directory plus the workspace-escape check.

    Resolution is a method rather than a free function reading a module global,
    so a tool can only ever resolve paths against the workspace it was handed.
    Frozen because the root must not change under a running session.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    def resolve(self, path: str) -> Path:
        """Resolve a workspace-relative path, rejecting anything that escapes."""
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved
