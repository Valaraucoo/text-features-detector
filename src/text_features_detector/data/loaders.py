"""Dataset loaders: download public HuggingFace datasets and return List[EvalSample].

Supported datasets
------------------
sst2               : stanfordnlp/sst2 — sentiment (positive / negative)
pavlick_formality  : osyvokon/pavlick-formality-scores — formality (formal / informal)
cola               : nyu-mll/glue (cola) — grammatical acceptability

All loaders share the same signature:
    load_*(split, max_samples, balance_classes, seed) -> list[EvalSample]

Binarisation for pavlick_formality
-----------------------------------
avg_score >= +FORMALITY_THRESHOLD  → formal   (gold_label=True)
avg_score <= -FORMALITY_THRESHOLD  → informal  (gold_label=False)
|avg_score| < FORMALITY_THRESHOLD  → skipped  (ambiguous neutral zone)
"""

from __future__ import annotations

import logging
import random
from typing import Any

from text_features_detector.data.helpers import balance_and_sample, load_from_hf
from text_features_detector.models import EvalSample, Feature

logger = logging.getLogger(__name__)


def load_sst2(
    max_samples: int | None = None,
    balance_classes: bool = True,
    seed: int = 42,
) -> list[EvalSample]:
    """stanfordnlp/sst2 · validation split · label 1 = positive, 0 = negative."""
    rng = random.Random(seed)
    raw = load_from_hf("stanfordnlp/sst2", split="validation")
    samples = [
        EvalSample(
            id=f"sst2-{i:06d}",
            dataset="sst2",
            text=row["sentence"].strip(),
            feature=Feature.SENTIMENT_POSITIVE,
            gold_label=bool(row["label"] == 1),
            metadata={},
        )
        for i, row in enumerate(raw)
        if row["label"] in (0, 1) and row["sentence"].strip()
    ]
    logger.info("sst2 (validation): %d samples loaded", len(samples))
    return balance_and_sample(samples, max_samples, balance_classes, rng)


def load_pavlick_formality(
    max_samples: int | None = None,
    balance_classes: bool = True,
    seed: int = 42,
) -> list[EvalSample]:
    """osyvokon/pavlick-formality-scores · train split · avg_score ∈ [-3, 3].

    Binarisation (threshold = 0.5):
      avg_score >= +0.5 → formal   (gold_label=True)
      avg_score <= -0.5 → informal  (gold_label=False)
      |avg_score| < 0.5 → skipped  (neutral)
    """
    _THRESHOLD = 0.5
    rng = random.Random(seed)
    raw = load_from_hf("osyvokon/pavlick-formality-scores", split="train")
    samples: list[EvalSample] = []
    skipped = 0
    for i, row in enumerate(raw):
        score = float(row["avg_score"])
        text = (row.get("sentence") or "").strip()
        if not text:
            continue
        if score >= _THRESHOLD:
            label = True
        elif score <= -_THRESHOLD:
            label = False
        else:
            skipped += 1
            continue
        samples.append(
            EvalSample(
                id=f"pavlick-{i:06d}",
                dataset="pavlick_formality",
                text=text,
                feature=Feature.FORMALITY,
                gold_label=label,
                metadata={"avg_score": score, "domain": row.get("domain", "")},
            )
        )
    logger.info(
        "pavlick_formality (train): %d samples loaded (%d skipped as neutral)",
        len(samples),
        skipped,
    )
    return balance_and_sample(samples, max_samples, balance_classes, rng)


def load_cola(
    max_samples: int | None = None,
    balance_classes: bool = True,
    seed: int = 42,
) -> list[EvalSample]:
    """nyu-mll/glue (cola) · validation split · label 1 = acceptable, 0 = unacceptable."""
    rng = random.Random(seed)
    raw = load_from_hf("nyu-mll/glue", "cola", split="validation")
    samples = [
        EvalSample(
            id=f"cola-{i:06d}",
            dataset="cola",
            text=row["sentence"].strip(),
            feature=Feature.GRAMMATICAL_ACCEPTABILITY,
            gold_label=bool(row["label"] == 1),
            metadata={},
        )
        for i, row in enumerate(raw)
        if row["label"] in (0, 1) and row["sentence"].strip()
    ]
    logger.info("cola (validation): %d samples loaded", len(samples))
    return balance_and_sample(samples, max_samples, balance_classes, rng)


DATASET_LOADERS: dict[str, Any] = {
    "sst2": load_sst2,
    "pavlick_formality": load_pavlick_formality,
    "cola": load_cola,
}


def load_dataset_samples(
    dataset_name: str,
    max_samples: int | None = None,
    balance_classes: bool = True,
    seed: int = 42,
) -> list[EvalSample]:
    """Unified entry-point: load and normalise a named dataset."""
    if dataset_name not in DATASET_LOADERS:
        available = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {available}")
    return DATASET_LOADERS[dataset_name](
        max_samples=max_samples,
        balance_classes=balance_classes,
        seed=seed,
    )
