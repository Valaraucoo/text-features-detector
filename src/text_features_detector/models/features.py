"""Feature domain models: Feature enum and FeatureSpec."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Feature(StrEnum):
    SENTIMENT_POSITIVE = "sentiment_positive"
    FORMALITY = "formality"
    GRAMMATICAL_ACCEPTABILITY = "grammatical_acceptability"


class FeatureSpec(BaseModel):
    """Defines a single binary feature that a judge should detect.

    The `criteria` and optional `evaluation_steps` are forwarded verbatim to
    the judge prompt / GEval rubric, so they should be written from the
    judge's perspective: describe *what makes the feature present*.
    """

    name: Feature = Field(..., description="Unique identifier matching the Feature enum.")
    display_name: str = Field(..., description="Human-readable label for reports.")
    criteria: str = Field(
        ...,
        description=(
            "One-sentence criterion for the feature being PRESENT (label=True). Used verbatim in judge prompts."
        ),
    )
    negative_criteria: str = Field(
        ...,
        description=(
            "One-sentence criterion for the opposite class (label=False). Used to disambiguate absent/opposite cases."
        ),
    )
    evaluation_steps: list[str] | None = Field(
        default=None,
        description=("Optional ordered rubric steps for GEval. Auto-generated from criteria when None."),
    )
    positive_label_description: str = Field(
        default="present",
        description="Short word used in prompts for label=True (e.g. 'positive', 'formal').",
    )
    negative_label_description: str = Field(
        default="absent",
        description="Short word used in prompts for label=False.",
    )
