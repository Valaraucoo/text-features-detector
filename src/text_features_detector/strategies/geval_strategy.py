"""GEvalJudge: DeepEval GEval metric backed by Pydantic AI."""

from __future__ import annotations

import logging

from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from text_features_detector.judges.deepeval_model import PydanticAIDeepEvalModel, get_judge_model
from text_features_detector.judges.registry import ModelConfig
from text_features_detector.models import (
    EvalSample,
    FeatureSpec,
    JudgeResult,
    JudgeVerdict,
)

logger = logging.getLogger(__name__)

_THRESHOLD = 0.8

_RUBRIC = [
    Rubric(
        score_range=(0, 4),
        expected_outcome=(
            "FAIL: the feature is absent, contradicted by the text, or not clearly supported. "
            "Ambiguous, weak, or merely possible evidence should fail."
        ),
    ),
    Rubric(
        score_range=(8, 10),
        expected_outcome=(
            "PASS: the feature is clearly present and directly supported by the text. "
            "The judgment should not rely on unstated assumptions."
        ),
    ),
]


def get_text_feature_metric(
    *,
    name: str,
    criteria: str,
    model: PydanticAIDeepEvalModel,
    evaluation_steps: list[str] | None = None,
) -> GEval:
    """Build a binary GEval metric for a single text feature.

    The metric is pre-configured with:
    - threshold=0.8 (binary pass/fail at score 8/10)
    - Binary FAIL/PASS rubrics to guide the judge toward clear decisions
    - evaluation_params restricted to ACTUAL_OUTPUT only
    """
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        rubric=_RUBRIC,
        model=model,
        threshold=_THRESHOLD,
        evaluation_steps=evaluation_steps,
        strict_mode=False,
        async_mode=True,
    )


def build_text_feature_criteria(feature_spec: FeatureSpec) -> str:
    """Build the GEval criteria string from a FeatureSpec."""
    return (
        f"Evaluate whether the text matches the positive label for '{feature_spec.display_name}'.\n\n"
        f"Positive label (score high): {feature_spec.positive_label_description}\n"
        f"Positive criterion: {feature_spec.criteria}\n\n"
        f"Negative label (score low): {feature_spec.negative_label_description}\n"
        f"Negative criterion: {feature_spec.negative_criteria}\n\n"
        "Score high only when the positive label is clearly supported. "
        "Score low when the negative label fits better or when the evidence is ambiguous."
    )


async def run_geval(
    sample: EvalSample,
    feature_spec: FeatureSpec,
    model_config: ModelConfig,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> JudgeResult:
    """Run GEval for a single sample and return a JudgeResult."""
    model = get_judge_model(model_config, temperature=temperature, timeout=timeout)
    criteria = build_text_feature_criteria(feature_spec)
    metric = get_text_feature_metric(
        name=feature_spec.display_name,
        criteria=criteria,
        model=model,
    )
    test_case = LLMTestCase(
        input=f"Classify the text for feature: {feature_spec.display_name}",
        actual_output=sample.text,
    )

    try:
        await metric.a_measure(test_case, _show_indicator=False)
    except Exception as exc:
        logger.warning("GEval failed for sample=%s model=%s: %s", sample.id, model_config.model_id, exc)
        return _failed_result(sample, model_config, model, str(exc))

    score = metric.score
    if score is None:
        return _failed_result(sample, model_config, model, "GEval returned no score")

    return _success_result(sample, model_config, model, score, metric.reason)


def _success_result(
    sample: EvalSample,
    model_config: ModelConfig,
    model: PydanticAIDeepEvalModel,
    score: float,
    reason: str | None,
) -> JudgeResult:
    label = score >= _THRESHOLD
    return JudgeResult(
        sample_id=sample.id,
        feature=sample.feature,
        model_id=model_config.model_id,
        strategy="geval",
        predicted_label=label,
        verdict=JudgeVerdict.TRUE if label else JudgeVerdict.FALSE,
        confidence=float(score),
        rationale=reason,
        raw_response=f"geval_score={score:.4f}",
        input_tokens=model.total_input_tokens,
        output_tokens=model.total_output_tokens,
        latency_ms=model.total_latency_ms,
        estimated_cost_usd=model_config.estimate_cost(
            model.total_input_tokens,
            model.total_output_tokens,
        ),
        api_calls=model.call_count,
        failed=False,
    )


def _failed_result(
    sample: EvalSample,
    model_config: ModelConfig,
    model: PydanticAIDeepEvalModel,
    error_message: str,
) -> JudgeResult:
    cost = model_config.estimate_cost(model.total_input_tokens, model.total_output_tokens)
    return JudgeResult(
        sample_id=sample.id,
        feature=sample.feature,
        model_id=model_config.model_id,
        strategy="geval",
        verdict=JudgeVerdict.ABSTAIN,
        input_tokens=model.total_input_tokens,
        output_tokens=model.total_output_tokens,
        latency_ms=model.total_latency_ms,
        estimated_cost_usd=cost,
        api_calls=model.call_count,
        failed=True,
        error_message=error_message,
    )
