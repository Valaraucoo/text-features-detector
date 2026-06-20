"""Experiment configuration models: DatasetConfig and RunConfig."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from text_features_detector.models.features import Feature


class DatasetConfig(BaseModel):
    name: str
    max_samples: int | None = None
    balance_classes: bool = True
    features: list[Feature] = Field(default_factory=list)


class RunConfig(BaseModel):
    """Full configuration for one evaluation run."""

    run_id: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    datasets: list[DatasetConfig]
    model_ids: list[str] = Field(..., description="Keys into the model registry.")
    strategies: list[str] = Field(default=["simple_binary"])

    self_consistency_n: int = Field(
        default=1,
        ge=1,
        description="Number of repeated results per sample for self-consistency (1 = disabled).",
    )
    self_consistency_model_ids: list[str] = Field(
        default_factory=list,
        description="Subset of model_ids to run self-consistency on. Empty = all.",
    )

    max_concurrent: int = Field(default=5, description="Max parallel API calls.")
    timeout: float = Field(default=60.0, description="Per-call timeout in seconds.")
    temperature: float = Field(default=0.0)
    seed: int | None = None
    output_dir: str = "results"
