"""Metrics aggregation: accuracy, precision, recall, F1, cost, latency, self-consistency."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from text_features_detector.models import Feature


@dataclass
class PerGroupMetrics:
    """Metrics for one (model_id, strategy, feature) group."""

    model_id: str
    strategy: str
    feature: Feature

    n_total: int = 0
    n_correct: int = 0
    n_abstained: int = 0
    n_failed: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    macro_f1: float = 0.0

    # Confusion matrix elements
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Cost
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_per_1k_samples_usd: float = 0.0
    cost_per_correct_label_usd: float = 0.0

    # Latency (ms)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0

    # API calls
    total_api_calls: int = 0
    total_retries: int = 0
    failure_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SelfConsistencyMetrics:
    """Self-consistency metrics for one (model_id, strategy, feature) group."""

    model_id: str
    strategy: str
    feature: Feature
    n_bundles: int = 0
    n_runs_per_bundle: int = 0
    mean_agreement_rate: float = 0.0
    mean_entropy: float = 0.0
    majority_vote_accuracy: float = 0.0
    majority_vote_f1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def compute_metrics(
    results: list[dict],
    golden_map: dict[str, bool],
) -> list[PerGroupMetrics]:
    """Compute per-group metrics from a list of result dicts + gold label lookup.

    Args:
        results: List of JudgeResult.model_dump() dicts.
        golden_map: {sample_id: gold_label}.
    """
    # Group results by (model_id, strategy, feature)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        key = (r["model_id"], r["strategy"], r["feature"])
        groups[key].append(r)

    out: list[PerGroupMetrics] = []
    for (model_id, strategy, feature), rows in sorted(groups.items()):
        g = PerGroupMetrics(model_id=model_id, strategy=strategy, feature=feature)
        g.n_total = len(rows)
        g.n_failed = sum(1 for r in rows if r.get("failed", False))

        # Only evaluate non-abstained rows (predicted_label present and not failed)
        valid = [
            r
            for r in rows
            if not r.get("failed", False) and r.get("predicted_label") is not None and r["sample_id"] in golden_map
        ]
        # Abstained = not failed, but no valid prediction (e.g. parse error with failed=False)
        non_failed = [r for r in rows if not r.get("failed", False)]
        g.n_abstained = len(non_failed) - len(valid)

        if valid:
            y_true = [golden_map[r["sample_id"]] for r in valid]
            y_pred = [r["predicted_label"] for r in valid]
            g.n_correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
            g.accuracy = accuracy_score(y_true, y_pred)
            g.precision = precision_score(y_true, y_pred, zero_division=0)
            g.recall = recall_score(y_true, y_pred, zero_division=0)
            g.f1 = f1_score(y_true, y_pred, zero_division=0)
            g.macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[False, True])
            g.tn, g.fp, g.fn, g.tp = cm.ravel().tolist()

        # Cost
        g.total_input_tokens = sum(r.get("input_tokens", 0) for r in rows)
        g.total_output_tokens = sum(r.get("output_tokens", 0) for r in rows)
        g.total_cost_usd = sum(r.get("estimated_cost_usd", 0.0) for r in rows)
        if g.n_total > 0:
            g.cost_per_1k_samples_usd = g.total_cost_usd / g.n_total * 1000
        if g.n_correct > 0:
            g.cost_per_correct_label_usd = g.total_cost_usd / g.n_correct

        # Latency
        latencies = [r.get("latency_ms", 0.0) for r in rows]
        if latencies:
            g.latency_mean_ms = sum(latencies) / len(latencies)
            g.latency_p50_ms = _percentile(latencies, 50)
            g.latency_p95_ms = _percentile(latencies, 95)

        # API calls
        g.total_api_calls = sum(r.get("api_calls", 1) for r in rows)
        g.total_retries = sum(r.get("retries", 0) for r in rows)
        g.failure_rate = g.n_failed / g.n_total if g.n_total else 0.0

        out.append(g)

    return out


def compute_self_consistency_metrics(
    bundles: list[dict],
    golden_map: dict[str, bool],
) -> list[SelfConsistencyMetrics]:
    """Compute self-consistency metrics from SelfConsistencyBundle dicts."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for b in bundles:
        key = (b["model_id"], b["strategy"], b["feature"])
        groups[key].append(b)

    out: list[SelfConsistencyMetrics] = []
    for (model_id, strategy, feature), rows in sorted(groups.items()):
        sc = SelfConsistencyMetrics(model_id=model_id, strategy=strategy, feature=feature)
        sc.n_bundles = len(rows)
        sc.n_runs_per_bundle = rows[0].get("n_runs", 1) if rows else 0
        agreement_rates = [r.get("agreement_rate", 0.0) for r in rows]
        entropies = [r.get("entropy", 0.0) for r in rows]
        sc.mean_agreement_rate = sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0.0
        sc.mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        # Majority-vote accuracy
        valid = [r for r in rows if r.get("majority_label") is not None and r["sample_id"] in golden_map]
        if valid:
            y_true = [golden_map[r["sample_id"]] for r in valid]
            y_pred = [r["majority_label"] for r in valid]
            sc.majority_vote_accuracy = accuracy_score(y_true, y_pred)
            sc.majority_vote_f1 = f1_score(y_true, y_pred, zero_division=0)

        out.append(sc)

    return out


def load_results_from_dir(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load results.jsonl and self_consistency.jsonl from a run directory."""
    results: list[dict] = []
    bundles: list[dict] = []
    rp = run_dir / "results.jsonl"
    bp = run_dir / "self_consistency.jsonl"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
    if bp.exists():
        for line in bp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                bundles.append(json.loads(line))
    return results, bundles
