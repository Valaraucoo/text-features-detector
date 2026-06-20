"""Tests for golden set builder and serialisation (no network calls)."""

import pytest

from text_features_detector.data.golden_set import (
    build_golden_set,
    load_golden_set,
    save_golden_set,
    summarise_golden_set,
)
from text_features_detector.models import DatasetConfig, EvalSample


def _make_sample(i: int, feature: str = "sentiment_positive", gold: bool = True) -> EvalSample:
    return EvalSample(
        id=f"fake-{i:04d}",
        dataset="fake",
        text=f"Sample text number {i}.",
        feature=feature,
        gold_label=gold,
    )


def test_save_and_load_golden_set(tmp_path):
    samples = [_make_sample(i, gold=(i % 2 == 0)) for i in range(20)]
    path = tmp_path / "golden.jsonl"
    save_golden_set(samples, path)

    loaded = load_golden_set(path)
    assert len(loaded) == 20
    assert loaded[0].id == samples[0].id
    assert loaded[0].gold_label == samples[0].gold_label


def test_summarise_golden_set():
    samples = (
        [_make_sample(i, feature="sentiment_positive", gold=True) for i in range(5)]
        + [_make_sample(i + 5, feature="sentiment_positive", gold=False) for i in range(3)]
        + [_make_sample(i + 10, feature="formality", gold=True) for i in range(4)]
    )
    summary = summarise_golden_set(samples)
    assert summary["sentiment_positive"]["total"] == 8
    assert summary["sentiment_positive"]["positive"] == 5
    assert summary["sentiment_positive"]["negative"] == 3
    assert summary["formality"]["total"] == 4


def test_save_creates_parent_dirs(tmp_path):
    samples = [_make_sample(0)]
    nested = tmp_path / "a" / "b" / "c" / "golden.jsonl"
    save_golden_set(samples, nested)
    assert nested.exists()


def test_build_golden_set_unknown_dataset_raises():
    configs = [DatasetConfig(name="nonexistent_dataset_xyz", split="train")]
    with pytest.raises((ValueError, Exception)):
        build_golden_set(configs, seed=42)
