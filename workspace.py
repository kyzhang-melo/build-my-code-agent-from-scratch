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

    def resolve(self, path: str, *, allow_external_absolute: bool = False) -> Path:
        """Resolve a path, optionally allowing explicit absolute external paths."""
        expanded = Path(path).expanduser()
        is_absolute = expanded.is_absolute()
        resolved = (self.root / expanded).resolve()
        if not resolved.is_relative_to(self.root):
            if allow_external_absolute and is_absolute:
                return resolved
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved
