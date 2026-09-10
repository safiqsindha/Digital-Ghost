from __future__ import annotations

from digital_ghost.caption.generate import caption_all_arms
from digital_ghost.config import load_captioning_config, load_provider_config, load_training_config
from digital_ghost.ingest.manifest import ingest_all_arms
from digital_ghost.training.orchestrator import run_sweep


def _prepare(study):
    captioning = load_captioning_config(study)
    ingest_all_arms(study, min_count=5)
    caption_all_arms(study, captioning)
    return load_training_config(study), load_provider_config(study)


def test_sweep_completes_all_cells_and_resumes(tiny_study):
    study = tiny_study
    training, provider_cfg = _prepare(study)
    provider_cfg.max_parallel_gpus = 2

    report = run_sweep(study, training, provider_cfg, dry_run=True, resume=True)
    assert report.total_cells == study.n_cells
    assert len(report.completed) == study.n_cells
    assert not report.failed

    report2 = run_sweep(study, training, provider_cfg, dry_run=True, resume=True)
    assert len(report2.completed) == 0
    assert len(report2.skipped_already_done) == study.n_cells


def test_sweep_aborts_on_tiny_budget(tiny_study):
    study = tiny_study
    training, provider_cfg = _prepare(study)
    provider_cfg.max_parallel_gpus = 1
    provider_cfg.pricing_usd_per_gpu_hour = 3600.0  # $1/sec wall time
    study.budget.cap_usd = 0.000001

    report = run_sweep(study, training, provider_cfg, dry_run=True, resume=True)
    assert report.aborted_on_budget
    assert len(report.skipped_budget) > 0
    assert len(report.completed) < report.total_cells


def test_sweep_respects_only_arms_and_only_doses(tiny_study):
    study = tiny_study
    training, provider_cfg = _prepare(study)

    report = run_sweep(
        study, training, provider_cfg, dry_run=True, resume=True,
        only_arms={"meme"}, only_doses={3},
    )
    assert report.total_cells == study.seeds_per_cell
    assert len(report.completed) == study.seeds_per_cell
