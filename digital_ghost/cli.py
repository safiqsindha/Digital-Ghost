"""Digital Ghost CLI.

Every command takes `--config` (default: configs/study.yaml) and reads
whatever sub-configs it needs from there — nothing is hardcoded here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from digital_ghost.config import (
    REPO_ROOT,
    load_captioning_config,
    load_provider_config,
    load_rating_app_config,
    load_study_config,
    load_training_config,
)

app = typer.Typer(add_completion=False, help="Digital Ghost: identity-bleed dose-response study pipeline.")

DEFAULT_CONFIG = REPO_ROOT / "configs" / "study.yaml"
ConfigOpt = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to study.yaml")


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


def _parse_csv_set(value: Optional[str]) -> Optional[set]:
    if not value:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


@app.command()
def ingest(
    config: Path = ConfigOpt,
    dry_run: bool = typer.Option(False, "--dry-run", help="Only require dry_run.n_images per arm"),
) -> None:
    """Validate provenance + counts for all three arms, write manifests."""
    _setup_logging()
    from digital_ghost.ingest.manifest import ingest_all_arms

    study = load_study_config(config)
    min_count = study.dry_run.n_images if dry_run else None
    paths = ingest_all_arms(study, min_count=min_count)
    for arm, path in paths.items():
        typer.echo(f"  {arm}: {path}")


@app.command()
def caption(config: Path = ConfigOpt) -> None:
    """Apply the neutral captioning scheme to every ingested arm."""
    _setup_logging()
    from digital_ghost.caption.generate import caption_all_arms

    study = load_study_config(config)
    captioning = load_captioning_config(study)
    paths = caption_all_arms(study, captioning)
    for arm, path in paths.items():
        typer.echo(f"  {arm}: {path}")


@app.command("init-eval-prompts")
def init_eval_prompts_cmd(
    config: Path = ConfigOpt,
    force: bool = typer.Option(False, "--force", help="Overwrite an existing frozen prompt file"),
) -> None:
    """Write the frozen 30-prompt eval set. Refuses to overwrite without --force."""
    _setup_logging()
    from digital_ghost.generation.eval_prompts import init_eval_prompts

    study = load_study_config(config)
    path = init_eval_prompts(study, force=force)
    typer.echo(f"wrote {path}")


@app.command()
def train(
    config: Path = ConfigOpt,
    dry_run: bool = typer.Option(False, "--dry-run", help="Fast placeholder training, no GPU/model needed"),
    resume: bool = typer.Option(True, help="Skip cells already marked succeeded"),
    arms: Optional[str] = typer.Option(None, help="Comma-separated arm names to restrict to"),
    doses: Optional[str] = typer.Option(None, help="Comma-separated doses to restrict to"),
) -> None:
    """Run the training sweep (parallel across N GPUs, resumable, budget-capped)."""
    _setup_logging()
    from digital_ghost.training.orchestrator import run_sweep

    study = load_study_config(config)
    if dry_run:
        study = study.with_dry_run_paths()
    training = load_training_config(study)
    provider_cfg = load_provider_config(study)

    only_arms = _parse_csv_set(arms)
    only_doses = {int(d) for d in _parse_csv_set(doses)} if doses else None

    report = run_sweep(
        study, training, provider_cfg, dry_run=dry_run, resume=resume,
        only_arms=only_arms, only_doses=only_doses,
    )
    _print_report("training sweep", report.total_cells, report.completed, report.skipped_already_done,
                   report.skipped_budget, report.failed, report.total_cost_usd, report.aborted_on_budget)
    if report.failed:
        raise typer.Exit(1)


@app.command()
def generate(
    config: Path = ConfigOpt,
    dry_run: bool = typer.Option(False, "--dry-run", help="Fast placeholder generation, no GPU/model needed"),
    resume: bool = typer.Option(True, help="Skip checkpoints already fully generated"),
    include_baseline: bool = typer.Option(True, help="Also generate the stock-SDXL baseline"),
    require_trained: bool = typer.Option(True, help="Fail if any sweep cell hasn't finished training"),
) -> None:
    """Generate the full eval grid: baseline + every trained LoRA checkpoint."""
    _setup_logging()
    from digital_ghost.generation.generate_grid import run_generation_grid

    study = load_study_config(config)
    if dry_run:
        study = study.with_dry_run_paths()
    training = load_training_config(study)
    provider_cfg = load_provider_config(study)

    report = run_generation_grid(
        study, training, provider_cfg, dry_run=dry_run, resume=resume,
        include_baseline=include_baseline, require_checkpoints_trained=require_trained,
    )
    _print_report("generation grid", report.total_checkpoints, report.completed, report.skipped_already_done,
                   report.skipped_budget, report.failed, report.total_cost_usd, report.aborted_on_budget)
    if report.failed:
        raise typer.Exit(1)


@app.command("rate-app")
def rate_app_cmd(
    config: Path = ConfigOpt,
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the pairwise rating web app."""
    _setup_logging()
    import uvicorn

    from digital_ghost.rating_app.backend.app import create_app

    study = load_study_config(config)
    rating_cfg = load_rating_app_config(study)
    web_app = create_app(study, rating_cfg)
    uvicorn.run(web_app, host=host, port=port)


@app.command()
def analyze(config: Path = ConfigOpt) -> None:
    """Fit the Davidson model (weighted + unweighted, overall / by tier / by exposure), write CSVs + plots."""
    _setup_logging()
    from digital_ghost.analysis.run_analysis import run_full_analysis

    study = load_study_config(config)
    results = run_full_analysis(study)
    typer.echo(f"wrote {len(results)} breakdown(s) to {study.path('analysis_dir')}")
    for r in results:
        typer.echo(f"  {r.label}: {r.n_comparisons} comparisons")


@app.command("dry-run")
def dry_run_cmd(config: Path = ConfigOpt) -> None:
    """Exercise ingest -> caption -> eval-prompts -> train -> generate end to end,
    on dry_run.n_images per arm and dry_run.n_prompts prompts, at ~zero cost.
    Does not touch the rating app or analysis (there's nothing to rate yet).
    """
    _setup_logging()
    from digital_ghost.caption.generate import caption_all_arms
    from digital_ghost.generation.eval_prompts import EvalPromptsExistError, init_eval_prompts
    from digital_ghost.generation.generate_grid import run_generation_grid
    from digital_ghost.ingest.manifest import ingest_all_arms
    from digital_ghost.training.orchestrator import run_sweep

    study = load_study_config(config)
    # All training/generation output is redirected under outputs/_dryrun/ so a
    # dry run can never overwrite artifacts a real (paid) run produced.
    study = study.with_dry_run_paths()
    training = load_training_config(study)
    captioning = load_captioning_config(study)
    provider_cfg = load_provider_config(study)

    typer.echo(f"=== dry-run: {study.dry_run.n_images} image(s)/arm, {study.dry_run.n_prompts} prompt(s) ===")
    typer.echo(f"    outputs isolated under: {study.path('outputs_dir')}")

    typer.echo("\n--- ingest ---")
    ingest_all_arms(study, min_count=study.dry_run.n_images)

    typer.echo("\n--- caption ---")
    caption_all_arms(study, captioning)

    typer.echo("\n--- eval prompts ---")
    try:
        path = init_eval_prompts(study, force=False)
        typer.echo(f"wrote {path}")
    except EvalPromptsExistError:
        typer.echo(f"{study.eval_prompts_file()} already exists, reusing it")

    # One tiny cell per arm, at dose = dry_run.n_images, instead of the full
    # (arm x study.doses x seeds_per_cell) sweep — enough to exercise every
    # code path without requiring study.doses-sized data or GPU time.
    dry_grid = [(arm.name, study.dry_run.n_images, 1) for arm in study.arms]

    typer.echo("\n--- train (dry-run cells) ---")
    train_report = run_sweep(
        study, training, provider_cfg, dry_run=True, resume=True, grid_override=dry_grid
    )
    _print_report("dry-run training", train_report.total_cells, train_report.completed,
                   train_report.skipped_already_done, train_report.skipped_budget,
                   train_report.failed, train_report.total_cost_usd, train_report.aborted_on_budget)
    if train_report.failed:
        raise typer.Exit(1)

    typer.echo("\n--- generate (dry-run checkpoints) ---")
    gen_report = run_generation_grid(
        study, training, provider_cfg, dry_run=True, resume=True,
        include_baseline=True, require_checkpoints_trained=True, grid_override=dry_grid,
    )
    _print_report("dry-run generation", gen_report.total_checkpoints, gen_report.completed,
                   gen_report.skipped_already_done, gen_report.skipped_budget,
                   gen_report.failed, gen_report.total_cost_usd, gen_report.aborted_on_budget)
    if gen_report.failed:
        raise typer.Exit(1)

    typer.echo(
        "\n=== dry-run complete. Wiring, config, and cost tracking all exercised at "
        "~zero cost. Populate data/raw/ fully and drop in your GPU provider key "
        "before running the real sweep. ==="
    )


def _print_report(label, total, completed, skipped_done, skipped_budget, failed, cost, aborted) -> None:
    typer.echo(
        f"{label}: {len(completed)}/{total} completed, {len(skipped_done)} already done, "
        f"{len(skipped_budget)} skipped (budget), {len(failed)} failed. "
        f"Spend so far: ${cost:.4f}."
    )
    if aborted:
        typer.secho("BUDGET CAP REACHED — sweep stopped early.", fg=typer.colors.RED, bold=True)
    for cell_id, err in failed.items():
        typer.secho(f"  FAILED {cell_id}: {err}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
