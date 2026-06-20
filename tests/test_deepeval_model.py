"""Offline tests for DeepEval model bridge and GEval metric factory."""

from deepeval.models import DeepEvalBaseLLM

from text_features_detector.judges.deepeval_model import PydanticAIDeepEvalModel, get_judge_model
from text_features_detector.judges.registry import ModelConfig
from text_features_detector.strategies.geval_strategy import (
    _THRESHOLD,
    build_text_feature_criteria,
    get_text_feature_metric,
)


def _make_cfg(model_id: str = "gpt-4.1-nano", provider: str = "openai") -> ModelConfig:
    return ModelConfig(model_id=model_id, provider=provider, display_name=model_id, tier="cheap")


# ---------------------------------------------------------------------------
# get_judge_model
# ---------------------------------------------------------------------------


def test_get_judge_model_returns_deepeval_base_llm():
    model = get_judge_model(_make_cfg())
    assert isinstance(model, DeepEvalBaseLLM)


def test_get_judge_model_returns_pydantic_ai_model():
    model = get_judge_model(_make_cfg())
    assert isinstance(model, PydanticAIDeepEvalModel)


def test_get_judge_model_name_matches_model_id():
    cfg = _make_cfg("claude-sonnet-4-5", "anthropic")
    model = get_judge_model(cfg)
    assert model.get_model_name() == "claude-sonnet-4-5"


def test_get_judge_model_telemetry_starts_at_zero():
    model = get_judge_model(_make_cfg())
    assert model.total_input_tokens == 0
    assert model.total_output_tokens == 0
    assert model.total_latency_ms == 0.0
    assert model.call_count == 0


def test_get_judge_model_temperature_and_timeout_passed():
    model = get_judge_model(_make_cfg(), temperature=0.5, timeout=30.0)
    assert model.temperature == 0.5
    assert model.judge.timeout == 30.0


# ---------------------------------------------------------------------------
# get_text_feature_metric
# ---------------------------------------------------------------------------


def test_get_text_feature_metric_threshold():
    model = get_judge_model(_make_cfg())
    metric = get_text_feature_metric(name="Test", criteria="The text is positive.", model=model)
    assert metric.threshold == _THRESHOLD
    assert metric.threshold == 0.8


def test_get_text_feature_metric_has_rubric():
    model = get_judge_model(_make_cfg())
    metric = get_text_feature_metric(name="Test", criteria="The text is positive.", model=model)
    assert metric.rubric is not None
    assert len(metric.rubric) == 2


def test_get_text_feature_metric_rubric_ranges():
    model = get_judge_model(_make_cfg())
    metric = get_text_feature_metric(name="Test", criteria="The text is positive.", model=model)
    ranges = {tuple(r.score_range) for r in metric.rubric}
    assert (0, 4) in ranges
    assert (8, 10) in ranges


def test_get_text_feature_metric_evaluation_model():
    cfg = _make_cfg("gemini-2.5-flash", "google")
    model = get_judge_model(cfg)
    metric = get_text_feature_metric(name="Test", criteria="Some criteria.", model=model)
    assert metric.evaluation_model == "gemini-2.5-flash"


def test_get_text_feature_metric_evaluation_steps_none():
    model = get_judge_model(_make_cfg())
    metric = get_text_feature_metric(name="Test", criteria="...", model=model)
    # When evaluation_steps=None, GEval auto-generates them from criteria
    # We just verify the metric object is created without raising
    assert metric is not None


def test_get_text_feature_metric_evaluation_steps_provided():
    model = get_judge_model(_make_cfg())
    steps = ["Step 1: Check X.", "Step 2: Check Y.", "Return True if both."]
    metric = get_text_feature_metric(name="Custom", criteria="...", model=model, evaluation_steps=steps)
    assert metric.evaluation_steps == steps


# ---------------------------------------------------------------------------
# build_text_feature_criteria
# ---------------------------------------------------------------------------


def test_build_text_feature_criteria_contains_display_name():
    from text_features_detector.features import get_feature_spec
    from text_features_detector.models import Feature

    spec = get_feature_spec(Feature.SENTIMENT_POSITIVE)
    criteria = build_text_feature_criteria(spec)
    assert spec.display_name in criteria


def test_build_text_feature_criteria_contains_spec_criteria():
    from text_features_detector.features import get_feature_spec
    from text_features_detector.models import Feature

    spec = get_feature_spec(Feature.FORMALITY)
    criteria = build_text_feature_criteria(spec)
    assert spec.criteria in criteria


def test_build_text_feature_criteria_contains_negative_criteria():
    from text_features_detector.features import get_feature_spec
    from text_features_detector.models import Feature

    spec = get_feature_spec(Feature.FORMALITY)
    criteria = build_text_feature_criteria(spec)
    assert spec.negative_criteria in criteria
    assert spec.negative_label_description in criteria
