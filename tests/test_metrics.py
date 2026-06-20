"""Tests for metrics computation."""

import pytest

from text_features_detector.eval.metrics import (
    compute_metrics,
)


def _make_result(
    sample_id: str,
    gold: bool,
    predicted: bool | None,
    model_id: str = "gpt-4.1-nano",
    strategy: str = "simple_binary",
    feature: str = "sentiment_positive",
    cost: float = 0.001,
    latency: float = 200.0,
) -> dict:
    return {
        "sample_id": sample_id,
        "feature": feature,
        "model_id": model_id,
        "strategy": strategy,
        "predicted_label": predicted,
        "verdict": "true" if predicted else ("false" if predicted is not None else "abstain"),
        "confidence": 0.9,
        "rationale": "test",
        "raw_response": "",
        "input_tokens": 150,
        "output_tokens": 50,
        "latency_ms": latency,
        "estimated_cost_usd": cost,
        "api_calls": 1,
        "retries": 0,
        "failed": predicted is None,
        "error_message": None,
        "timestamp": "2026-06-01T00:00:00+00:00",
        "prompt_used": None,
    }


GOLDEN = {
    "s0": True,
    "s1": True,
    "s2": False,
    "s3": False,
}

RESULTS_PERFECT = [
    _make_result("s0", True, True),
    _make_result("s1", True, True),
    _make_result("s2", False, False),
    _make_result("s3", False, False),
]

RESULTS_HALF = [
    _make_result("s0", True, True),
    _make_result("s1", True, False),  # wrong
    _make_result("s2", False, False),
    _make_result("s3", False, True),  # wrong
]


def test_perfect_accuracy():
    metrics = compute_metrics(RESULTS_PERFECT, GOLDEN)
    assert len(metrics) == 1
    m = metrics[0]
    assert m.accuracy == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.n_failed == 0
    assert m.n_abstained == 0


def test_half_accuracy():
    metrics = compute_metrics(RESULTS_HALF, GOLDEN)
    m = metrics[0]
    assert m.accuracy == pytest.approx(0.5)


def test_abstained_results_excluded_from_metrics():
    # failed=False but predicted_label=None → counted as abstained (not failed)
    abstained1 = _make_result("s2", False, None)
    abstained1["failed"] = False  # explicit: parse succeeded but returned abstain
    abstained2 = _make_result("s3", False, None)
    abstained2["failed"] = False
    results = RESULTS_PERFECT[:2] + [abstained1, abstained2]
    metrics = compute_metrics(results, GOLDEN)
    m = metrics[0]
    assert m.n_total == 4
    assert m.n_abstained == 2
    assert m.n_correct == 2


def test_cost_aggregation():
    metrics = compute_metrics(RESULTS_PERFECT, GOLDEN)
    m = metrics[0]
    assert m.total_cost_usd == pytest.approx(4 * 0.001)
    assert m.cost_per_1k_samples_usd == pytest.approx(0.001 * 1000)


def test_latency_aggregation():
    metrics = compute_metrics(RESULTS_PERFECT, GOLDEN)
    m = metrics[0]
    assert m.latency_mean_ms == pytest.approx(200.0)
    assert m.latency_p50_ms > 0
    assert m.latency_p95_ms > 0


def test_multiple_models():
    results = RESULTS_PERFECT + [
        _make_result("s0", True, False, model_id="claude-haiku-4-5"),
        _make_result("s1", True, False, model_id="claude-haiku-4-5"),
        _make_result("s2", False, True, model_id="claude-haiku-4-5"),
        _make_result("s3", False, True, model_id="claude-haiku-4-5"),
    ]
    metrics = compute_metrics(results, GOLDEN)
    assert len(metrics) == 2
    accs = {m.model_id: m.accuracy for m in metrics}
    assert accs["gpt-4.1-nano"] == pytest.approx(1.0)
    assert accs["claude-haiku-4-5"] == pytest.approx(0.0)


def test_empty_results():
    metrics = compute_metrics([], {})
    assert metrics == []
