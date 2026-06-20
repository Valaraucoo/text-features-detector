"""SimpleBinaryJudge: one prompt → Pydantic structured output."""

from __future__ import annotations

import logging

from text_features_detector.judges.judge import PydanticAIJudge
from text_features_detector.judges.registry import ModelConfig
from text_features_detector.models import (
    EvalSample,
    FeatureSpec,
    JudgeResult,
    JudgeVerdict,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise text analysis system. Your task is to determine whether a \
given text exhibits a specific linguistic or stylistic feature.

Return a structured answer with:
- label: true if the feature is PRESENT, false if ABSENT
- confidence: a float between 0.0 and 1.0
- rationale: one or two concise sentences explaining your decision.

Use the positive and negative criteria literally. Return false when the negative
class applies, even if the text mentions words or concepts related to the
positive class."""

USER_TEMPLATE = """\
TEXT:
\"\"\"
{text}
\"\"\"

FEATURE TO DETECT:
{display_name}

POSITIVE LABEL (return label=true):
{positive_label}

POSITIVE CRITERION:
{criteria}

NEGATIVE LABEL (return label=false):
{negative_label}

NEGATIVE CRITERION:
{negative_criteria}

Is the POSITIVE LABEL or NEGATIVE LABEL a better fit for the text above?
Respond with the JSON object."""


async def run_simple_binary(
    sample: EvalSample,
    feature_spec: FeatureSpec,
    model_config: ModelConfig,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 60.0,
) -> JudgeResult:
    """Run a single simple-binary judge call and return a JudgeResult."""
    prompt_user = USER_TEMPLATE.format(
        text=sample.text,
        display_name=feature_spec.display_name,
        positive_label=feature_spec.positive_label_description,
        criteria=feature_spec.criteria,
        negative_label=feature_spec.negative_label_description,
        negative_criteria=feature_spec.negative_criteria,
    )
    judge = PydanticAIJudge(model_config, timeout=timeout)

    try:
        output, in_tok, out_tok, api_calls, latency_ms = await judge.run_binary(
            prompt=prompt_user,
            instructions=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("Judge call failed for sample=%s model=%s: %s", sample.id, model_config.model_id, exc)
        return JudgeResult(
            sample_id=sample.id,
            feature=sample.feature,
            model_id=model_config.model_id,
            strategy="simple_binary",
            verdict=JudgeVerdict.ABSTAIN,
            failed=True,
            error_message=str(exc),
        )

    verdict = JudgeVerdict.TRUE if output.label else JudgeVerdict.FALSE

    cost = model_config.estimate_cost(in_tok, out_tok)

    return JudgeResult(
        sample_id=sample.id,
        feature=sample.feature,
        model_id=model_config.model_id,
        strategy="simple_binary",
        predicted_label=output.label,
        verdict=verdict,
        confidence=output.confidence,
        rationale=output.rationale,
        raw_response=output.model_dump_json(),
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        estimated_cost_usd=cost,
        api_calls=api_calls,
        retries=0,
        failed=False,
        prompt_used=f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{prompt_user}",
    )
