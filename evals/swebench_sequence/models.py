from __future__ import annotations

from dataclasses import asdict, dataclass, field

from evals.swebench.models import AgentResult


@dataclass(frozen=True)
class SequenceSpec:
    dataset: str
    split: str
    selection: str
    sequence_id: str
    repo: str
    subsystem: str
    instance_ids: list[str]


@dataclass
class SequenceAgentResult(AgentResult):
    sequence_position: int = 0
    session_id: str = ""
    history_committed: bool = False
    resumed_message_count: int = 0
    history_tokens_before: int = 0
    history_tokens_after: int = 0
    resume_diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
