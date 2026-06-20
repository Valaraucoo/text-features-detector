"""Model registry: loads models.yaml and provides pricing / config lookups."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    model_id: str
    provider: str  # openai | anthropic | google
    display_name: str
    tier: str  # cheap | mid | expensive
    cost_input_per_1m: float = 0.0
    cost_output_per_1m: float = 0.0
    max_concurrent: int = 5
    max_tokens_output: int = 512
    extra: dict[str, Any] = field(default_factory=dict)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1_000_000 * self.cost_input_per_1m + output_tokens / 1_000_000 * self.cost_output_per_1m

    @property
    def pydantic_ai_model(self) -> str:
        """Provider-qualified model string accepted by Pydantic AI."""
        return f"{self.provider}:{self.model_id}"


_DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "models.yaml"


class ModelRegistry:
    def __init__(self, path: Path | str | None = None) -> None:
        resolved = Path(path) if path else _DEFAULT_REGISTRY_PATH
        with resolved.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        self._models: dict[str, ModelConfig] = {}
        for model_id, cfg in raw.get("models", {}).items():
            self._models[model_id] = ModelConfig(
                model_id=model_id,
                provider=cfg["provider"],
                display_name=cfg.get("display_name", model_id),
                tier=cfg.get("tier", "mid"),
                cost_input_per_1m=float(cfg.get("cost_input_per_1m", 0.0)),
                cost_output_per_1m=float(cfg.get("cost_output_per_1m", 0.0)),
                max_concurrent=int(cfg.get("max_concurrent", 5)),
                max_tokens_output=int(cfg.get("max_tokens_output", 512)),
                extra={
                    k: v
                    for k, v in cfg.items()
                    if k
                    not in {
                        "provider",
                        "display_name",
                        "tier",
                        "cost_input_per_1m",
                        "cost_output_per_1m",
                        "max_concurrent",
                        "max_tokens_output",
                    }
                },
            )

    def get(self, model_id: str) -> ModelConfig:
        try:
            return self._models[model_id]
        except KeyError:
            available = ", ".join(sorted(self._models))
            raise KeyError(f"Model {model_id!r} not in registry. Available: {available}") from None

    def list_ids(self) -> list[str]:
        return sorted(self._models)

    def by_tier(self, tier: str) -> list[ModelConfig]:
        return [m for m in self._models.values() if m.tier == tier]


# Module-level singleton (lazy)
_registry: ModelRegistry | None = None


def get_registry(path: Path | str | None = None) -> ModelRegistry:
    global _registry
    if _registry is None or path is not None:
        _registry = ModelRegistry(path)
    return _registry
