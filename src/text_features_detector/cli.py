"""CLI entry point for the text-features-detector evaluation pipeline.

Commands:
  tfd run     --config configs/experiment_small.yaml
  tfd report  --run-dir results/small_benchmark_v1
  tfd list    (list available models, datasets, features)
  tfd prepare --config configs/experiment_small.yaml  (download & save golden set only)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(name="tfd", help="Text Feature Detector – LLM-Judge evaluation pipeline")
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command("list")
def cmd_list(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List available models, datasets, and features."""
    _setup_logging(verbose)

    from text_features_detector.data.loaders import DATASET_LOADERS
    from text_features_detector.features import list_features
    from text_features_detector.judges.registry import get_registry

    registry = get_registry()

    console.rule("[bold]Models[/bold]")
    t = Table("ID", "Provider", "Tier", "Cost in / 1M", "Cost out / 1M")
    for mid in registry.list_ids():
        m = registry.get(mid)
        t.add_row(
            mid,
            m.provider,
            m.tier,
            f"${m.cost_input_per_1m:.3f}",
            f"${m.cost_output_per_1m:.3f}",
        )
    console.print(t)

    console.rule("[bold]Datasets[/bold]")
    for name in sorted(DATASET_LOADERS):
        console.print(f"  • {name}")

    console.rule("[bold]Features[/bold]")
    for name in list_features():
        console.print(f"  • {name}")


@app.command("prepare")
def cmd_prepare(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download datasets and save the golden set to the run output directory."""
    _setup_logging(verbose)

    from text_features_detector.data import build_golden_set, save_golden_set
    from text_features_detector.models import RunConfig

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    run_cfg = RunConfig.model_validate(raw)

    out_dir = Path(run_cfg.output_dir) / run_cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_golden_set(run_cfg.datasets, seed=run_cfg.seed or 42)
    gs_path = out_dir / "golden_set.jsonl"
    save_golden_set(samples, gs_path)

    console.print(f"[green]Golden set saved:[/green] {gs_path} ({len(samples)} samples)")

    # Also save a copy of the config
    (out_dir / "run_config.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")


@app.command("run")
def cmd_run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    prepare_first: bool = typer.Option(
        True,
        "--prepare/--no-prepare",
        help="Build golden set before running (default: True)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full evaluation experiment."""
    _setup_logging(verbose)

    from text_features_detector.data import build_golden_set, load_golden_set, save_golden_set
    from text_features_detector.eval.runner import ExperimentRunner
    from text_features_detector.models import RunConfig

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    run_cfg = RunConfig.model_validate(raw)

    out_dir = Path(run_cfg.output_dir) / run_cfg.run_id
    gs_path = out_dir / "golden_set.jsonl"

    if prepare_first or not gs_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        samples = build_golden_set(run_cfg.datasets, seed=run_cfg.seed or 42)
        save_golden_set(samples, gs_path)
        (out_dir / "run_config.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        samples = load_golden_set(gs_path)

    console.print(
        f"[bold]Running experiment[/bold] [cyan]{run_cfg.run_id}[/cyan] "
        f"| {len(samples)} samples × {len(run_cfg.model_ids)} models × "
        f"{len(run_cfg.strategies)} strategies"
    )

    runner = ExperimentRunner(run_cfg, samples)
    results = asyncio.run(runner.run())

    console.print(f"[green]Done.[/green] {len(results)} results written to {out_dir}")
    console.print("Run [bold]tfd report[/bold] to generate the report.")


@app.command("report")
def cmd_report(
    run_dir: Path = typer.Option(..., "--run-dir", "-r", help="Path to run output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate CSV report from a completed run."""
    _setup_logging(verbose)

    from text_features_detector.reporting.report import generate_report

    report_dir = generate_report(run_dir)
    console.print(f"[green]Report:[/green] {report_dir}")


@app.command("cost-estimate")
def cmd_cost_estimate(
    config: Path = typer.Option(..., "--config", "-c"),
    avg_input_tokens: int = typer.Option(200, help="Estimated input tokens per call"),
    avg_output_tokens: int = typer.Option(80, help="Estimated output tokens per call"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Estimate total API cost for an experiment without running it."""
    _setup_logging(verbose)

    import yaml

    from text_features_detector.judges.registry import get_registry
    from text_features_detector.models import RunConfig

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    run_cfg = RunConfig.model_validate(raw)
    registry = get_registry()

    total_samples = sum(cfg.max_samples or 500 for cfg in run_cfg.datasets)

    t = Table("Model", "Strategy", "Calls", "Est. Input USD", "Est. Output USD", "Total USD")
    grand = 0.0
    for mid in run_cfg.model_ids:
        m = registry.get(mid)
        sc_enabled = run_cfg.self_consistency_n > 1 and (
            not run_cfg.self_consistency_model_ids or mid in run_cfg.self_consistency_model_ids
        )
        calls_per_sample = 1 + run_cfg.self_consistency_n if sc_enabled else 1
        for strategy in run_cfg.strategies:
            calls = total_samples * calls_per_sample
            in_usd = calls * avg_input_tokens / 1_000_000 * m.cost_input_per_1m
            out_usd = calls * avg_output_tokens / 1_000_000 * m.cost_output_per_1m
            total_usd = in_usd + out_usd
            grand += total_usd
            t.add_row(
                mid,
                strategy,
                str(calls),
                f"${in_usd:.4f}",
                f"${out_usd:.4f}",
                f"${total_usd:.4f}",
            )
    console.print(t)
    console.print(f"\n[bold]Grand total estimate:[/bold] ${grand:.4f} USD")


if __name__ == "__main__":
    app()
