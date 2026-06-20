"""Tests for feature registry."""

import pytest

from text_features_detector.features import FEATURE_REGISTRY, get_feature_spec, list_features


def test_list_features_returns_all():
    features = list_features()
    assert "sentiment_positive" in features
    assert "formality" in features
    assert "grammatical_acceptability" in features


def test_get_feature_valid():
    spec = get_feature_spec("sentiment_positive")
    assert spec.name == "sentiment_positive"
    assert spec.criteria
    assert spec.positive_label_description == "positive"
    assert spec.negative_label_description == "negative"


def test_get_feature_unknown_raises():
    with pytest.raises(KeyError, match="unknown_xyz"):
        get_feature_spec("unknown_xyz")


def test_all_features_have_criteria():
    for name, spec in FEATURE_REGISTRY.items():
        assert spec.criteria, f"{name} has empty criteria"
        assert spec.negative_criteria, f"{name} has empty negative_criteria"
        assert spec.display_name, f"{name} has empty display_name"


def test_sentiment_positive_negative_criteria_mentions_missing_positive_qualities():
    spec = get_feature_spec("sentiment_positive")
    assert "missing" in spec.negative_criteria.lower()
    assert "lacking" in spec.negative_criteria.lower()
