"""Evaluation sample model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from text_features_detector.models.features import Feature


class EvalSample(BaseModel):
    """A single evaluation item: text + feature + gold label."""

    id: str = Field(..., description="Globally unique identifier, e.g. 'sst2-val-000042'.")
    dataset: str = Field(..., description="Source dataset identifier, e.g. 'sst2'.")
    text: str = Field(..., description="The text to be judged.")
    feature: Feature = Field(..., description="Feature to detect.")
    gold_label: bool = Field(..., description="Ground-truth: True = feature is present.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v
