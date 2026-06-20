"""Tests for the model registry."""

from pathlib import Path

import pytest

from text_features_detector.judges.registry import ModelRegistry


@pytest.fixture
def registry() -> ModelRegistry:
    # Use the real configs/models.yaml
    cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    return ModelRegistry(cfg_path)


def test_registry_loads_models(registry):
    ids = registry.list_ids()
    assert len(ids) == 5


def test_registry_get_known_model(registry):
    m = registry.get("gpt-4.1-nano")
    assert m.provider == "openai"
    assert m.tier == "legacy_cheap"
    assert m.cost_input_per_1m > 0


def test_registry_unknown_model_raises(registry):
    with pytest.raises(KeyError, match="nonexistent-model"):
        registry.get("nonexistent-model")


def test_cost_estimate_zero_tokens(registry):
    m = registry.get("gpt-4.1-nano")
    assert m.estimate_cost(0, 0) == pytest.approx(0.0)


def test_cost_estimate_one_million_tokens(registry):
    m = registry.get("gpt-4.1-nano")
    cost = m.estimate_cost(1_000_000, 1_000_000)
    expected = m.cost_input_per_1m + m.cost_output_per_1m
    assert cost == pytest.approx(expected)


def test_by_tier_cheap(registry):
    cheap = registry.by_tier("cheap")
    assert len(cheap) >= 1
    assert all(m.tier == "cheap" for m in cheap)


def test_openai_only_registry(registry):
    assert all(registry.get(model_id).provider == "openai" for model_id in registry.list_ids())
