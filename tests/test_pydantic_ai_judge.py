"""Offline tests for the Pydantic AI judge wrapper."""

from text_features_detector.judges.judge import BinaryJudgeOutput, PydanticAIJudge
from text_features_detector.judges.registry import ModelConfig


def test_model_config_pydantic_ai_model_string():
    cfg = ModelConfig(
        model_id="gpt-4.1-nano",
        provider="openai",
        display_name="GPT-4.1 nano",
        tier="cheap",
    )
    assert cfg.pydantic_ai_model == "openai:gpt-4.1-nano"


def test_binary_judge_output_schema():
    output = BinaryJudgeOutput(label=True, confidence=0.9, rationale="The text is clearly positive.")
    assert output.label is True
    assert output.confidence == 0.9
    assert "positive" in output.rationale


def test_pydantic_ai_judge_init_only_no_api_call():
    cfg = ModelConfig(
        model_id="claude-haiku-4-5",
        provider="anthropic",
        display_name="Claude Haiku 4.5",
        tier="cheap",
    )
    judge = PydanticAIJudge(cfg, timeout=30)
    assert judge.model_config.model_id == "claude-haiku-4-5"
    assert judge.timeout == 30
