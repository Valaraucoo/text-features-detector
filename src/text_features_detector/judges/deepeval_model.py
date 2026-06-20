"""DeepEval model bridge backed by Pydantic AI.

Provides the adapter that DeepEval's GEval metric requires (DeepEvalBaseLLM)
while routing actual inference through PydanticAIJudge.

Usage:
    model = get_judge_model(model_config)
    metric = GEval(..., model=model)
"""

from __future__ import annotations

import asyncio

from deepeval.models import DeepEvalBaseLLM

from text_features_detector.judges.judge import PydanticAIJudge
from text_features_detector.judges.registry import ModelConfig


class PydanticAIDeepEvalModel(DeepEvalBaseLLM):
    """Minimal DeepEval model bridge backed by Pydantic AI.

    Tracks token usage and latency across all GEval calls so that
    the caller can report telemetry in JudgeResult.
    """

    def __init__(self, model_config: ModelConfig, temperature: float = 0.0, timeout: float = 60.0) -> None:
        super().__init__(model=model_config.model_id)
        self.judge = PydanticAIJudge(model_config, timeout=timeout)
        self.model_config = model_config
        self.temperature = temperature
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_ms = 0.0
        self.call_count = 0

    def get_model_name(self) -> str:
        return self.model_config.model_id

    def load_model(self) -> None:
        return None

    def generate(self, prompt: str, schema=None) -> str:  # noqa: ANN001
        return asyncio.run(self.a_generate(prompt, schema=schema))

    async def a_generate(self, prompt: str, schema=None) -> str:  # noqa: ANN001
        text, in_tok, out_tok, api_calls, latency_ms = await self.judge.run_text(
            prompt=prompt,
            max_tokens=self.model_config.max_tokens_output,
            temperature=self.temperature,
        )
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_latency_ms += latency_ms
        self.call_count += api_calls
        return text


def get_judge_model(
    model_config: ModelConfig,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> PydanticAIDeepEvalModel:
    """Return a DeepEvalBaseLLM instance backed by Pydantic AI for the given model."""
    return PydanticAIDeepEvalModel(model_config, temperature=temperature, timeout=timeout)
