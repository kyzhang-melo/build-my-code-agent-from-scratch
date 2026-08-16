from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str = ""
    instance_image_key: str = ""
    platform: str = "linux/x86_64"

    @classmethod
    def from_dict(cls, raw: dict) -> "Task":
        return cls(**{
            field: raw[field]
            for field in cls.__dataclass_fields__
            if field in raw
        })

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    instance_id: str
    attempt: int
    agent_status: str
    patch_status: str
    stop_reason: str = ""
    api_calls: int = 0
    duration_seconds: float = 0.0
    patch_bytes: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
