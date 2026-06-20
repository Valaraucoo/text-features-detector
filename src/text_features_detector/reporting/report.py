"""Reporting: write evaluation results to CSV files.

Produces two files in <run_dir>/report/:
  metrics.csv           — per-(model, strategy, feature) classification + cost + latency
  self_consistency.csv  — per-(model, strategy, feature) self-consistency stats (if present)
"""

from __future__ import annotations

import logging
from pathlib import Path

from text_features_detector.eval.metrics import (
    compute_metrics,
    compute_self_consistency_metrics,
    load_results_from_dir,
)
from text_features_detector.reporting.helpers import load_golden_map, write_csv

logger = logging.getLogger(__name__)


def generate_report(run_dir: Path | str) -> Path:
    """Compute metrics from a completed run and write CSV summaries.

    Args:
        run_dir: Path to the run output directory (must contain results.jsonl
                 and optionally golden_set.jsonl and self_consistency.jsonl).

    Returns:
        Path to the report directory.
    """
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    results_raw, bundles_raw = load_results_from_dir(run_dir)
    golden_map = load_golden_map(run_dir)

    if not golden_map:
        logger.warning("golden_set.jsonl not found in %s; accuracy metrics will be empty.", run_dir)

    metrics = compute_metrics(results_raw, golden_map)
    sc_metrics = compute_self_consistency_metrics(bundles_raw, golden_map)

    write_csv(report_dir / "metrics.csv", metrics)
    if sc_metrics:
        write_csv(report_dir / "self_consistency.csv", sc_metrics)

    logger.info("Report written to %s (%d rows)", report_dir, len(metrics))
    return report_dir
