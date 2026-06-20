"""Judge result models: JudgeVerdict, JudgeResult, SelfConsistencyBundle."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from text_features_detector.models.features import Feature


class JudgeVerdict(StrEnum):
    TRUE = "true"
    FALSE = "false"
    ABSTAIN = "abstain"


class JudgeResult(BaseModel):
    """Raw output produced by a judge for a single (sample, model, strategy) triple."""

    sample_id: str
    feature: Feature
    model_id: str
    strategy: str  # 'simple_binary' | 'geval'
    predicted_label: bool | None = Field(
        None,
        description="Parsed boolean prediction; None when the judge abstained.",
    )
    verdict: JudgeVerdict = JudgeVerdict.ABSTAIN
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Self-reported confidence [0, 1].")
    rationale: str | None = None
    raw_response: str | None = None

    # Cost / latency telemetry
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    api_calls: int = 1
    retries: int = 0
    failed: bool = False
    error_message: str | None = None

    # Provenance
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_used: str | None = None


class SelfConsistencyBundle(BaseModel):
    """Aggregated result of n repeated judge calls for the same sample+model+strategy."""

    sample_id: str
    feature: Feature
    model_id: str
    strategy: str
    n_runs: int
    results: list[JudgeResult]

    majority_label: bool | None = None
    agreement_rate: float = Field(0.0, description="Fraction of results matching majority label.")
    entropy: float = Field(0.0, ge=0.0, le=1.0, description="Binary entropy of response distribution.")

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_api_calls: int = 0
    total_retries: int = 0
    failed_runs: int = 0

    def compute_majority(self) -> None:
        """Fill majority_label, agreement_rate, entropy from self.results in place."""
        valid = [r for r in self.results if r.predicted_label is not None]
        self.failed_runs = len(self.results) - len(valid)
        if not valid:
            return
        trues = sum(1 for r in valid if r.predicted_label)
        falses = len(valid) - trues
        self.majority_label = trues >= falses
        self.agreement_rate = max(trues, falses) / len(valid)
        p = trues / len(valid)
        if 0 < p < 1:
            self.entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        else:
            self.entropy = 0.0

        self.total_input_tokens = sum(r.input_tokens for r in self.results)
        self.total_output_tokens = sum(r.output_tokens for r in self.results)
        self.total_latency_ms = sum(r.latency_ms for r in self.results)
        self.total_cost_usd = sum(r.estimated_cost_usd for r in self.results)
        self.total_api_calls = sum(r.api_calls for r in self.results)
        self.total_retries = sum(r.retries for r in self.results)
