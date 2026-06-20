"""Pydantic AI based LLM judge client.

This module is the single inference layer used by judge strategies.  It replaces
provider-specific adapters with Pydantic AI's provider-qualified model strings,
for example:

    openai:gpt-4.1-mini
    anthropic:claude-sonnet-4-5
    google:gemini-2.5-flash
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings, UsageLimits

from text_features_detector.judges.registry import ModelConfig

logger = logging.getLogger(__name__)

_MAX_RETRIES = 6
_BASE_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 20.0


class BinaryJudgeOutput(BaseModel):
    """Structured output returned by the simple binary judge."""

    label: bool = Field(description="True if the feature is present, false otherwise.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence from 0 to 1.")
    rationale: str = Field(description="Brief justification for the decision.")


class TextJudgeOutput(BaseModel):
    """Structured text wrapper for GEval's custom LLM bridge."""

    text: str


class PydanticAIJudge:
    """Small wrapper around Pydantic AI Agent with usage/cost extraction."""

    def __init__(self, model_config: ModelConfig, timeout: float = 60.0) -> None:
        self.model_config = model_config
        self.timeout = timeout

    async def run_binary(
        self,
        prompt: str,
        instructions: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[BinaryJudgeOutput, int, int, int, float]:
        """Run a structured binary judgment.

        Returns:
            (output, input_tokens, output_tokens, api_calls, latency_ms)
        """
        max_tokens = max_tokens or self.model_config.max_tokens_output
        agent = Agent(
            self.model_config.pydantic_ai_model,
            instructions=instructions,
            output_type=BinaryJudgeOutput,
        )
        t0 = time.perf_counter()
        result = await _run_with_backoff(
            lambda: agent.run(
                prompt,
                model_settings=_model_settings(temperature, max_tokens, self.timeout),
                usage_limits=UsageLimits(response_tokens_limit=max_tokens),
            ),
            model_id=self.model_config.model_id,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = result.usage
        return (
            result.output,
            usage.input_tokens or 0,
            usage.output_tokens or 0,
            usage.requests or 1,
            latency_ms,
        )

    async def run_text(
        self,
        prompt: str,
        instructions: str = "",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[str, int, int, int, float]:
        """Run an unstructured text request and return text with usage."""
        max_tokens = max_tokens or self.model_config.max_tokens_output
        agent = Agent(
            self.model_config.pydantic_ai_model,
            instructions=instructions,
            output_type=str,
        )
        t0 = time.perf_counter()
        result = await _run_with_backoff(
            lambda: agent.run(
                prompt,
                model_settings=_model_settings(temperature, max_tokens, self.timeout),
                usage_limits=UsageLimits(response_tokens_limit=max_tokens),
            ),
            model_id=self.model_config.model_id,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = result.usage
        return (
            result.output,
            usage.input_tokens or 0,
            usage.output_tokens or 0,
            usage.requests or 1,
            latency_ms,
        )


def _model_settings(temperature: float, max_tokens: int, timeout: float) -> ModelSettings:
    return ModelSettings(temperature=temperature, max_tokens=max_tokens, timeout=timeout)


async def _run_with_backoff(call, model_id: str):  # noqa: ANN001, ANN202
    """Run an async model call with exponential backoff for transient provider errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await call()
        except Exception as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES - 1 or not _is_retryable_error(exc):
                raise
            delay = _backoff_delay(attempt)
            logger.warning(
                "LLM call failed for model=%s (attempt %d/%d): %s. Retrying in %.2fs",
                model_id,
                attempt + 1,
                _MAX_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("LLM call failed without exception") from last_exc


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "rate_limit",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection",
            "503",
            "502",
            "500",
        )
    )


def _backoff_delay(attempt: int) -> float:
    base = min(_BASE_BACKOFF_S * (2**attempt), _MAX_BACKOFF_S)
    jitter = random.uniform(0, min(0.25 * base, 1.0))
    return base + jitter
