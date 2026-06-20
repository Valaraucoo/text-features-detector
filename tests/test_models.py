"""Tests for Pydantic models and self-consistency bundle."""

import pytest

from text_features_detector.models import (
    DatasetConfig,
    EvalSample,
    Feature,
    JudgeResult,
    JudgeVerdict,
    RunConfig,
    SelfConsistencyBundle,
)

_F = Feature.SENTIMENT_POSITIVE


def test_eval_sample_rejects_blank_text():
    with pytest.raises(ValueError, match="blank"):
        EvalSample(id="x", dataset="d", text="   ", feature=_F, gold_label=True)


def test_eval_sample_valid():
    s = EvalSample(id="a", dataset="sst2", text="Great!", feature=_F, gold_label=True)
    assert s.gold_label is True


def test_eval_sample_accepts_feature_string():
    # StrEnum: string value is also accepted by Pydantic
    s = EvalSample(id="a", dataset="sst2", text="Great!", feature="sentiment_positive", gold_label=True)
    assert s.feature is Feature.SENTIMENT_POSITIVE


def test_judge_result_defaults():
    r = JudgeResult(sample_id="x", feature=_F, model_id="m", strategy="s")
    assert r.verdict == JudgeVerdict.ABSTAIN
    assert r.predicted_label is None
    assert r.api_calls == 1


def _make_result(label: bool | None) -> JudgeResult:
    return JudgeResult(
        sample_id="s1",
        feature=Feature.SENTIMENT_POSITIVE,
        model_id="gpt-4.1-nano",
        strategy="simple_binary",
        predicted_label=label,
        verdict=JudgeVerdict.TRUE if label else (JudgeVerdict.FALSE if label is not None else JudgeVerdict.ABSTAIN),
        input_tokens=100,
        output_tokens=40,
        latency_ms=200.0,
        estimated_cost_usd=0.001,
    )


def test_self_consistency_majority_all_true():
    results = [_make_result(True), _make_result(True), _make_result(True)]
    bundle = SelfConsistencyBundle(
        sample_id="s1",
        feature=_F,
        model_id="m",
        strategy="s",
        n_runs=3,
        results=results,
    )
    bundle.compute_majority()
    assert bundle.majority_label is True
    assert bundle.agreement_rate == 1.0
    assert bundle.entropy == pytest.approx(0.0)


def test_self_consistency_majority_mixed():
    results = [_make_result(True), _make_result(False), _make_result(True)]
    bundle = SelfConsistencyBundle(
        sample_id="s1",
        feature=_F,
        model_id="m",
        strategy="s",
        n_runs=3,
        results=results,
    )
    bundle.compute_majority()
    assert bundle.majority_label is True
    assert bundle.agreement_rate == pytest.approx(2 / 3)
    assert bundle.entropy > 0.0


def test_self_consistency_all_abstained():
    results = [_make_result(None), _make_result(None)]
    bundle = SelfConsistencyBundle(
        sample_id="s1",
        feature=_F,
        model_id="m",
        strategy="s",
        n_runs=2,
        results=results,
    )
    bundle.compute_majority()
    assert bundle.majority_label is None
    assert bundle.failed_runs == 2


def test_run_config_valid():
    cfg = RunConfig(
        run_id="test_run",
        datasets=[DatasetConfig(name="sst2", max_samples=50)],
        model_ids=["gpt-4.1-nano"],
        strategies=["simple_binary"],
    )
    assert cfg.self_consistency_n == 1
    assert cfg.temperature == 0.0
