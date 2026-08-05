from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Preference = Literal["A", "B", "equivalent", "unclear"]


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str = Field(pattern=r"^[AB]:(?:L\d+(?:-L?\d+)?|T\d+(?:-T?\d+)?)$")
    claim: str


class DimensionJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preference: Preference
    explanation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class PairJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    localization_quality: DimensionJudgment
    exploration_discipline: DimensionJudgment
    evidence_grounded_reasoning: DimensionJudgment
    editing_discipline: DimensionJudgment
    verification_and_recovery: DimensionJudgment
    overall_preference: Preference
    overall_explanation: str
    confidence: float = Field(ge=0, le=1)


class AggregateJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_effect: Literal["first_value_better", "second_value_better", "mixed", "none", "unclear"]
    summary: str
    supported_mechanisms: list[str]
    counterexamples: list[str]
    interaction_effects: list[str]
    alternative_explanations: list[str]
    limitations: list[str]
    confidence: float = Field(ge=0, le=1)
