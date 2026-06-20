from __future__ import annotations

import csv
import json
from pathlib import Path

from text_features_detector.eval import PerGroupMetrics, SelfConsistencyMetrics


def load_golden_map(run_dir: Path) -> dict[str, bool]:
    gs_path = run_dir / "golden_set.jsonl"
    if not gs_path.exists():
        return {}
    golden_map: dict[str, bool] = {}
    for line in gs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            golden_map[obj["id"]] = bool(obj["gold_label"])
    return golden_map


def write_csv(path: Path, rows: list[PerGroupMetrics | SelfConsistencyMetrics]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
