"""Tests for experiment runner checkpoint/resume behavior."""

from pathlib import Path

from text_features_detector.eval.runner import ExperimentRunner
from text_features_detector.models import (
    DatasetConfig,
    EvalSample,
    Feature,
    JudgeResult,
    JudgeVerdict,
    RunConfig,
)


def _run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_id="resume_test",
        datasets=[DatasetConfig(name="sst2", max_samples=1)],
        model_ids=["gpt-4.1-nano"],
        strategies=["simple_binary"],
        output_dir=str(tmp_path),
    )


def _sample() -> EvalSample:
    return EvalSample(
        id="s1",
        dataset="sst2",
        text="A wonderful film.",
        feature=Feature.SENTIMENT_POSITIVE,
        gold_label=True,
    )


def test_failed_checkpoint_result_is_not_done(tmp_path):
    run_cfg = _run_config(tmp_path)
    runner = ExperimentRunner(run_cfg, [_sample()])
    runner._append_result(
        JudgeResult(
            sample_id="s1",
            feature=Feature.SENTIMENT_POSITIVE,
            model_id="gpt-4.1-nano",
            strategy="simple_binary",
            verdict=JudgeVerdict.ABSTAIN,
            failed=True,
            error_message="rate limit",
        )
    )

    resumed = ExperimentRunner(run_cfg, [_sample()])
    assert not resumed._is_done("s1", "gpt-4.1-nano", "simple_binary")


def test_successful_checkpoint_result_is_done(tmp_path):
    run_cfg = _run_config(tmp_path)
    runner = ExperimentRunner(run_cfg, [_sample()])
    runner._append_result(
        JudgeResult(
            sample_id="s1",
            feature=Feature.SENTIMENT_POSITIVE,
            model_id="gpt-4.1-nano",
            strategy="simple_binary",
            predicted_label=True,
            verdict=JudgeVerdict.TRUE,
            failed=False,
        )
    )

    resumed = ExperimentRunner(run_cfg, [_sample()])
    assert resumed._is_done("s1", "gpt-4.1-nano", "simple_binary")
