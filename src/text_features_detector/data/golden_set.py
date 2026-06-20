"""Golden set builder: compose multiple datasets into a unified corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from text_features_detector.data.loaders import load_dataset_samples
from text_features_detector.models import DatasetConfig, EvalSample

logger = logging.getLogger(__name__)


def build_golden_set(dataset_configs: list[DatasetConfig], seed: int = 42) -> list[EvalSample]:
    """Build the golden evaluation set from a list of DatasetConfig objects."""
    all_samples: list[EvalSample] = []
    for cfg in dataset_configs:
        logger.info("Loading dataset=%s max=%s", cfg.name, cfg.max_samples)
        samples = load_dataset_samples(
            dataset_name=cfg.name,
            max_samples=cfg.max_samples,
            balance_classes=cfg.balance_classes,
            seed=seed,
        )
        if cfg.features:
            samples = [s for s in samples if s.feature in cfg.features]
        logger.info("  → %d samples (features: %s)", len(samples), {s.feature for s in samples})
        all_samples.extend(samples)
    logger.info("Golden set total: %d samples", len(all_samples))
    return all_samples


def save_golden_set(samples: list[EvalSample], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.model_dump_json() + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


def load_golden_set(path: Path | str) -> list[EvalSample]:
    path = Path(path)
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                samples.append(EvalSample.model_validate(json.loads(line)))
    return samples


def summarise_golden_set(samples: list[EvalSample]) -> dict[str, dict]:
    from collections import defaultdict

    summary: dict[str, dict] = defaultdict(lambda: {"total": 0, "positive": 0, "negative": 0})
    for s in samples:
        summary[s.feature]["total"] += 1
        key = "positive" if s.gold_label else "negative"
        summary[s.feature][key] += 1
    return dict(summary)
