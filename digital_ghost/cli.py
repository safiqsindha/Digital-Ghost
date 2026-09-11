"""Digital Ghost CLI.

Every command takes `--config` (default: configs/study.yaml) and reads
whatever sub-configs it needs from there — nothing is hardcoded here.

Three staged gates, cheapest first, each proving something before the next
costs anything:

    dry-run     CPU only, a handful of images and prompts. Proves config,
                paths, manifests and the whole orchestration path work.
                Nothing is rented.
    smoke-cell  One real cell end to end on the GPU: train, generate,
                validate, stop. You look at the images before going further.
    sweep       All 45 cells. Refuses to start until a smoke cell has
                succeeded on this machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from digital_ghost.config import (
    REPO_ROOT,
    load_captioning_config,
    load_rating_app_config,
    load_runtime_config,
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


def _print_sweep_report(label: str, report) -> None:
    typer.echo(
        f"\n{label}: {len(report.succeeded)} succeeded, {len(report.failed)} failed, "
        f"{len(report.skipped)} skipped of {report.total_cells}. "
        f"Spend ${report.total_cost_usd:.2f}, elapsed {report.elapsed_hours:.2f}h."
    )
    if report.aborted_on_budget:
        typer.secho("BUDGET CAP REACHED — sweep stopped early.", fg=typer.colors.RED, bold=True)
    for outcome in report.failed:
        typer.secho(f"  FAILED {outcome.cell_id}: {'; '.join(outcome.failures[:2])}", fg=typer.colors.RED)


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# the three gates
# --------------------------------------------------------------------------


@app.command("dry-run")
def dry_run_cmd(config: Path = ConfigOpt) -> None:
    """Gate 1. CPU only, no GPU, nothing rented.

    Exercises ingest, captioning, the frozen prompt set, and the whole
    train/generate/validate orchestration on dry_run.n_images per arm and
    dry_run.n_prompts prompts, using a placeholder trainer. Outputs are
    isolated under outputs/_dryrun/ so this can never touch real results.
    """
    _setup_logging()
    from digital_ghost.caption.generate import caption_all_arms
    from digital_ghost.generation.eval_prompts import EvalPromptsExistError, init_eval_prompts
    from digital_ghost.ingest.manifest import ingest_all_arms
    from digital_ghost.training.sweep import run_sweep

    study = load_study_config(config).with_dry_run_paths()
    training = load_training_config(study)
    captioning = load_captioning_config(study)
    runtime = load_runtime_config(study)

    typer.echo(f"=== gate 1: dry-run ({study.dry_run.n_images} images/arm, "
               f"{study.dry_run.n_prompts} prompts, CPU only) ===")
    typer.echo(f"    outputs isolated under: {study.path('outputs_dir')}\n")

    ingest_all_arms(study, min_count=study.dry_run.n_images)
    caption_all_arms(study, captioning)
    try:
        init_eval_prompts(study)
    except EvalPromptsExistError:
        typer.echo(f"{study.eval_prompts_file()} already exists, reusing it")

    # One tiny cell per arm at dose = dry_run.n_images, rather than the full
    # study.doses grid, so the pool can be as small as a handful of images.
    grid = [(arm.name, study.dry_run.n_images, 1) for arm in study.arms]
    report = run_sweep(study, training, runtime, dry_run=True, resume=True, grid_override=grid)
    _print_sweep_report("dry-run", report)

    if report.failed:
        raise typer.Exit(1)
    typer.secho(
        "\nGate 1 passed. Orchestration, config, sanity checks and cost tracking all work. "
        "Next: rent a GPU and run `digital-ghost smoke-cell`.",
        fg=typer.colors.GREEN,
    )


@app.command("smoke-cell")
def smoke_cell_cmd(
    config: Path = ConfigOpt,
    arm: str = typer.Option("standard", help="Which arm; standard is the positive control"),
    dose: Optional[int] = typer.Option(None, help="Defaults to the largest dose"),
    seed_index: int = typer.Option(1),
    allow_hardware_change: bool = typer.Option(False, "--allow-hardware-change"),
) -> None:
    """Gate 2. One real cell end to end on the GPU, then stop.

    Writes into the real run directory, so it counts as one of the sweep's
    cells rather than being paid for twice. Look at its images and validation
    samples before going further; if you don't like them, throw it away with
    `digital-ghost invalidate-cell`.

    Defaults to standard at the largest dose because that is the study's
    positive control: if ordinary photos at full dose don't produce a
    recognisable likeness, the hyperparameters are wrong and every other cell
    would be uninterpretable.
    """
    _setup_logging()
    from digital_ghost.training.sweep import run_sweep

    study = load_study_config(config)
    training = load_training_config(study)
    runtime = load_runtime_config(study)
    dose = dose if dose is not None else max(study.doses)

    typer.echo(f"=== gate 2: smoke cell ({arm}, dose {dose}, seed {seed_index}) ===\n")
    report = run_sweep(
        study, training, runtime, dry_run=False, resume=True,
        grid_override=[(arm, dose, seed_index)],
        allow_hardware_change=allow_hardware_change,
    )
    _print_sweep_report("smoke cell", report)

    if report.failed:
        raise typer.Exit(1)

    cell = report.succeeded[0] if report.succeeded else None
    if cell:
        runs = study.path("runs_dir") / cell.cell_id
        gens = study.path("generations_dir") / cell.cell_id
        typer.secho("\nGate 2 passed. Now LOOK AT THE IMAGES before spending on the full sweep:",
                    fg=typer.colors.GREEN)
        typer.echo(f"  mid-training samples: {runs / 'validation_samples'}")
        typer.echo(f"  eval generations:     {gens}")
        typer.echo(f"  log:                  {runs / 'cell.log'}")
        typer.echo(f"\nMeasured {cell.elapsed_s / 3600:.2f}h for this cell "
                   f"(${cell.cost_usd:.2f}). Put that in runtime.yaml as "
                   "execution.estimated_gpu_hours_per_cell, then run `digital-ghost sweep`.")


@app.command()
def sweep(
    config: Path = ConfigOpt,
    resume: bool = typer.Option(True, help="Skip cells whose outputs are already complete"),
    arms: Optional[str] = typer.Option(None, help="Comma-separated arms to restrict to"),
    doses: Optional[str] = typer.Option(None, help="Comma-separated doses to restrict to"),
    allow_hardware_change: bool = typer.Option(
        False, "--allow-hardware-change",
        help="Continue on different hardware, recording the change in every affected cell",
    ),
    force: bool = typer.Option(False, "--force", help="Skip the smoke-cell gate"),
) -> None:
    """Gate 3. The full sweep. Resume-by-default; safe to run under tmux.

    Refuses to start until a smoke cell has succeeded on this machine, unless
    --force. Nothing depends on the foreground terminal: logs and cell state
    are on disk, so an SSH drop costs you the ticker and nothing else.
    """
    _setup_logging()
    from digital_ghost.sampling.subsample import cell_grid
    from digital_ghost.training.status import read_sweep_status
    from digital_ghost.training.sweep import run_sweep

    study = load_study_config(config)
    training = load_training_config(study)
    runtime = load_runtime_config(study)

    if not force:
        status = read_sweep_status(study)
        if not status.has_successful_cell:
            typer.secho(
                "Refusing to start: no cell has completed successfully on this machine yet.\n"
                "Run `digital-ghost smoke-cell` first and inspect its output — it costs one "
                "cell and catches the mistakes that would otherwise cost forty-five.\n"
                "Override with --force if you know what you're doing.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(2)

    only_arms = _parse_csv_set(arms)
    only_doses = {int(d) for d in _parse_csv_set(doses)} if doses else None
    grid = [
        (a, d, s) for a, d, s in cell_grid(study)
        if (only_arms is None or a in only_arms) and (only_doses is None or d in only_doses)
    ]

    typer.echo(f"=== gate 3: sweep ({len(grid)} cells, "
               f"parallel={runtime.execution.parallel_cells}) ===\n")
    report = run_sweep(
        study, training, runtime, dry_run=False, resume=resume,
        grid_override=grid, allow_hardware_change=allow_hardware_change,
    )
    _print_sweep_report("sweep", report)
    if report.failed:
        raise typer.Exit(1)


@app.command()
def status(config: Path = ConfigOpt, tail: int = typer.Option(5, help="Sweep log lines to show")) -> None:
    """What actually happened: done / failed / remaining, read from disk.

    Works whether the sweep is running in a tmux session you've been
    disconnected from or died hours ago.
    """
    _setup_logging(verbose=False)
    from digital_ghost.training.status import format_status, read_sweep_status
    from digital_ghost.training.sweep import sweep_log_path

    study = load_study_config(config)
    typer.echo(format_status(read_sweep_status(study), sweep_log_path(study), tail=tail))


@app.command("invalidate-cell")
def invalidate_cell_cmd(
    cell: str = typer.Argument(..., help="Cell id, e.g. standard_dose0200_seed1"),
    config: Path = ConfigOpt,
) -> None:
    """Throw away a completed cell so the sweep redoes it.

    The smoke cell counts as a real cell, so this is how you discard one whose
    images you inspected and didn't like.
    """
    _setup_logging()
    from digital_ghost.sampling.subsample import cell_grid
    from digital_ghost.training.cell import build_cell_spec, invalidate
    from digital_ghost.training.cell import cell_id as make_id

    study = load_study_config(config)
    for arm, dose, seed_index in cell_grid(study):
        if make_id(arm, dose, seed_index) == cell:
            spec = build_cell_spec(study, arm, dose, seed_index)
            if invalidate(spec):
                typer.secho(f"invalidated {cell} — the next sweep will redo it", fg=typer.colors.YELLOW)
            else:
                typer.echo(f"{cell} has no recorded run to invalidate")
            return
    typer.secho(f"unknown cell id: {cell}", fg=typer.colors.RED)
    raise typer.Exit(1)


# --------------------------------------------------------------------------
# generation / rating / analysis
# --------------------------------------------------------------------------


@app.command()
def generate(
    config: Path = ConfigOpt,
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume: bool = typer.Option(True),
    include_baseline: bool = typer.Option(True),
    require_trained: bool = typer.Option(True),
) -> None:
    """Regenerate the eval grid for every checkpoint.

    The sweep already generates each cell's images as it goes; this exists for
    when you want to redo generation without retraining — a changed prompt
    set, or more seeds per prompt.
    """
    _setup_logging()
    from digital_ghost.generation.generate_grid import run_generation_grid

    study = load_study_config(config)
    if dry_run:
        study = study.with_dry_run_paths()
    training = load_training_config(study)
    runtime = load_runtime_config(study)

    report = run_generation_grid(
        study, training, runtime, dry_run=dry_run, resume=resume,
        include_baseline=include_baseline, require_checkpoints_trained=require_trained,
    )
    typer.echo(
        f"generation: {len(report.completed)}/{report.total_checkpoints} completed, "
        f"{len(report.skipped_already_done)} already done, {len(report.failed)} failed. "
        f"Spend ${report.total_cost_usd:.2f}."
    )
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
    uvicorn.run(create_app(study, rating_cfg), host=host, port=port)


@app.command()
def analyze(config: Path = ConfigOpt) -> None:
    """Fit the Davidson model (weighted + unweighted, overall / tier / exposure), write CSVs + plots."""
    _setup_logging()
    from digital_ghost.analysis.run_analysis import run_full_analysis

    study = load_study_config(config)
    results = run_full_analysis(study)
    typer.echo(f"wrote {len(results)} breakdown(s) to {study.path('analysis_dir')}")
    for r in results:
        typer.echo(f"  {r.label}: {r.n_comparisons} comparisons")


if __name__ == "__main__":
    app()
