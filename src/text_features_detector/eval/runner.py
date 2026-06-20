"""Experiment runner: orchestrates async judge calls across models and strategies.

Supports:
  - Concurrent calls bounded by per-model max_concurrent limits
  - Self-consistency (n repeated calls, majority vote)
  - Checkpointing: partial results saved to JSONL; already-done sample+model+strategy
    triples are skipped on resume
  - Progress reporting via rich
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from text_features_detector.features import get_feature_spec
from text_features_detector.judges.registry import ModelConfig, get_registry
from text_features_detector.models import (
    EvalSample,
    JudgeResult,
    RunConfig,
    SelfConsistencyBundle,
)
from text_features_detector.strategies import STRATEGY_MAP

logger = logging.getLogger(__name__)


async def _run_single(
    sample: EvalSample,
    model_cfg: ModelConfig,
    strategy_name: str,
    run_cfg: RunConfig,
    llm_semaphore: asyncio.Semaphore,
) -> JudgeResult:
    """Run one (sample, model, strategy) call."""
    strategy_fn = STRATEGY_MAP.get(strategy_name)
    if strategy_fn is None:
        raise ValueError(f"Unknown strategy {strategy_name!r}. Available: {list(STRATEGY_MAP)}")

    feature_spec = get_feature_spec(sample.feature)
    async with llm_semaphore:
        result: JudgeResult = await strategy_fn(
            sample,
            feature_spec,
            model_cfg,
            temperature=run_cfg.temperature,
            timeout=run_cfg.timeout,
        )
    return result


async def _run_self_consistency(
    sample: EvalSample,
    model_cfg: ModelConfig,
    strategy_name: str,
    n: int,
    run_cfg: RunConfig,
    llm_semaphore: asyncio.Semaphore,
) -> SelfConsistencyBundle:
    """Run n independent judge calls and aggregate into a SelfConsistencyBundle."""
    tasks = [_run_single(sample, model_cfg, strategy_name, run_cfg, llm_semaphore) for _ in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    bundle = SelfConsistencyBundle(
        sample_id=sample.id,
        feature=sample.feature,
        model_id=model_cfg.model_id,
        strategy=strategy_name,
        n_runs=n,
        results=list(results),
    )
    bundle.compute_majority()
    return bundle


class ExperimentRunner:
    def __init__(self, run_cfg: RunConfig, samples: list[EvalSample]) -> None:
        self.run_cfg = run_cfg
        self.samples = samples
        self.registry = get_registry()
        self.output_dir = Path(run_cfg.output_dir) / run_cfg.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._results_path = self.output_dir / "results.jsonl"
        self._bundles_path = self.output_dir / "self_consistency.jsonl"

        # Load already-completed result keys to allow resuming
        self._done_keys: set[tuple[str, str, str]] = set()
        if self._results_path.exists():
            for line in self._results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    if not obj.get("failed", False) and obj.get("predicted_label") is not None:
                        self._done_keys.add((obj["sample_id"], obj["model_id"], obj["strategy"]))
        logger.info(
            "Output dir: %s | Already done: %d result(s)",
            self.output_dir,
            len(self._done_keys),
        )

    def _is_done(self, sample_id: str, model_id: str, strategy: str) -> bool:
        return (sample_id, model_id, strategy) in self._done_keys

    def _append_result(self, result: JudgeResult) -> None:
        with self._results_path.open("a", encoding="utf-8") as fh:
            fh.write(result.model_dump_json() + "\n")
        self._done_keys.add((result.sample_id, result.model_id, result.strategy))

    def _append_bundle(self, bundle: SelfConsistencyBundle) -> None:
        with self._bundles_path.open("a", encoding="utf-8") as fh:
            fh.write(bundle.model_dump_json() + "\n")

    async def _process_one(
        self,
        sample: EvalSample,
        model_cfg: ModelConfig,
        strategy: str,
        llm_semaphore: asyncio.Semaphore,
        progress: Progress,
        task_id: int,
    ) -> None:
        if self._is_done(sample.id, model_cfg.model_id, strategy):
            progress.advance(task_id)
            return

        result = await _run_single(sample, model_cfg, strategy, self.run_cfg, llm_semaphore)
        self._append_result(result)

        # Self-consistency
        sc_models = self.run_cfg.self_consistency_model_ids
        run_sc = self.run_cfg.self_consistency_n > 1 and (not sc_models or model_cfg.model_id in sc_models)
        if run_sc:
            bundle = await _run_self_consistency(
                sample,
                model_cfg,
                strategy,
                self.run_cfg.self_consistency_n,
                self.run_cfg,
                llm_semaphore,
            )
            self._append_bundle(bundle)

        progress.advance(task_id)

    async def run(self) -> list[JudgeResult]:
        """Run all (sample × model × strategy) combinations and return results."""
        model_cfgs = [self.registry.get(m) for m in self.run_cfg.model_ids]

        total = len(self.samples) * len(model_cfgs) * len(self.run_cfg.strategies)
        logger.info(
            "Experiment: %d samples × %d models × %d strategies = %d total calls",
            len(self.samples),
            len(model_cfgs),
            len(self.run_cfg.strategies),
            total,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Judge calls", total=total)

            max_concurrent = self.run_cfg.max_concurrent if self.run_cfg.max_concurrent > 0 else 1
            llm_semaphore = asyncio.Semaphore(max_concurrent)

            async def _limited(sample, model_cfg, strategy):
                await self._process_one(
                    sample,
                    model_cfg,
                    strategy,
                    llm_semaphore,
                    progress,
                    task_id,
                )

            coros = [
                _limited(sample, model_cfg, strategy)
                for sample in self.samples
                for model_cfg in model_cfgs
                for strategy in self.run_cfg.strategies
            ]
            await asyncio.gather(*coros)

        # Load all results from disk (includes any previously checkpointed)
        results: list[JudgeResult] = []
        if self._results_path.exists():
            for line in self._results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    results.append(JudgeResult.model_validate(json.loads(line)))
        logger.info("Run complete. Total results on disk: %d", len(results))
        return results
