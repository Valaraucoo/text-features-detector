from __future__ import annotations

import random
from typing import Any

from text_features_detector.models import EvalSample


def load_from_hf(path: str, name: str | None = None, split: str = "train") -> Any:
    """Load a HuggingFace dataset split; raises ImportError / DatasetNotFoundError on failure."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    return load_dataset(path, name, split=split, trust_remote_code=False)


def balance_and_sample(
    samples: list[EvalSample],
    max_samples: int | None,
    balance: bool,
    rng: random.Random,
) -> list[EvalSample]:
    """Optionally balance classes then truncate to max_samples."""
    if not samples:
        return samples
    if balance:
        pos = [s for s in samples if s.gold_label]
        neg = [s for s in samples if not s.gold_label]
        n = min(len(pos), len(neg))
        if max_samples is not None:
            n = min(n, max_samples // 2)
        rng.shuffle(pos)
        rng.shuffle(neg)
        samples = pos[:n] + neg[:n]
    elif max_samples is not None:
        rng.shuffle(samples)
        samples = samples[:max_samples]
    rng.shuffle(samples)
    return samples
