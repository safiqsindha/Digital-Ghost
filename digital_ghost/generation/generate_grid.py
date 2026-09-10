"""Orchestrates the full generation grid: the stock-SDXL baseline plus all
45 trained LoRA checkpoints, each rendered against the same frozen 30-prompt
eval set at `eval.seeds_per_prompt` seeds each.

Reuses the same GPU-slot / provider / cost-ledger machinery as the training
orchestrator (digital_ghost/training/orchestrator.py) — generation spend
counts against the same study-wide budget cap.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from digital_ghost.config import ProviderConfig, StudyConfig, TrainingConfig
from digital_ghost.generation.eval_prompts import eval_prompt_seeds, load_eval_prompts
from digital_ghost.sampling.subsample import cell_grid
from digital_ghost.training.cell import CellSpec, build_cell_spec, is_complete as cell_is_complete
from digital_ghost.training.cost import BudgetExceededError, CostLedger, study_cost_ledger_path
from digital_ghost.training.provider import GPUProvider, JobStatus, get_provider

logger = logging.getLogger(__name__)

BASELINE_LABEL = "baseline"


@dataclass
class CheckpointSpec:
    label: str
    lora_path: str | None  # None for the baseline
    arm: str | None
    dose: int | None
    seed_index: int | None


def all_checkpoints(
    study: StudyConfig,
    include_baseline: bool = True,
    grid_override: list[tuple[str, int, int]] | None = None,
) -> list[CheckpointSpec]:
    checkpoints: list[CheckpointSpec] = []
    if include_baseline:
        checkpoints.append(CheckpointSpec(BASELINE_LABEL, None, None, None, None))
    source_grid = grid_override if grid_override is not None else cell_grid(study)
    for arm, dose, seed_index in source_grid:
        cell = build_cell_spec(study, arm, dose, seed_index)
        checkpoints.append(
            CheckpointSpec(cell.id, str(cell.checkpoint_path), arm, dose, seed_index)
        )
    return checkpoints


def flat_prompts_path(study: StudyConfig) -> Path:
    return study.path("generations_dir") / "_prompts_flat.jsonl"


def write_flat_prompts(study: StudyConfig, dry_run: bool = False) -> Path:
    """Every (prompt, seed) pair to render, flattened to one row each.
    Same file is reused for every checkpoint — prompts and seeds are fixed.
    """
    data = load_eval_prompts(study)
    prompts = data["prompts"]
    if dry_run:
        prompts = prompts[: study.dry_run.n_prompts]

    out_path = flat_prompts_path(study)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for prompt in prompts:
            for gen_seed in eval_prompt_seeds(study, prompt["id"]):
                f.write(
                    json.dumps(
                        {
                            "prompt_id": prompt["id"],
                            "prompt_text": prompt["text"],
                            "tier": prompt["tier"],
                            "gen_seed": gen_seed,
                        }
                    )
                    + "\n"
                )
    return out_path


def checkpoint_output_dir(study: StudyConfig, label: str) -> Path:
    return study.path("generations_dir") / label


def checkpoint_manifest_path(study: StudyConfig, label: str) -> Path:
    return checkpoint_output_dir(study, label) / "manifest.jsonl"


def is_checkpoint_generated(study: StudyConfig, checkpoint: CheckpointSpec, expected_rows: int) -> bool:
    manifest_path = checkpoint_manifest_path(study, checkpoint.label)
    if not manifest_path.exists():
        return False
    with open(manifest_path) as f:
        n = sum(1 for line in f if line.strip())
    return n == expected_rows


def generation_command(
    checkpoint: CheckpointSpec,
    study: StudyConfig,
    training: TrainingConfig,
    prompts_path: Path,
    dry_run: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "digital_ghost.generation.generate_images",
        "--checkpoint-label", checkpoint.label,
        "--base-model", training.base_model,
        "--resolution", str(training.resolution),
        "--output-dir", str(checkpoint_output_dir(study, checkpoint.label)),
        "--eval-prompts", str(prompts_path),
    ]
    if checkpoint.lora_path:
        cmd += ["--lora-path", checkpoint.lora_path]
    if checkpoint.arm:
        cmd += ["--arm", checkpoint.arm]
    if checkpoint.dose is not None:
        cmd += ["--dose", str(checkpoint.dose)]
    if checkpoint.seed_index is not None:
        cmd += ["--seed-index", str(checkpoint.seed_index)]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


@dataclass
class GenerationReport:
    total_checkpoints: int
    completed: list[str] = field(default_factory=list)
    skipped_already_done: list[str] = field(default_factory=list)
    skipped_budget: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    aborted_on_budget: bool = False


def run_generation_grid(
    study: StudyConfig,
    training: TrainingConfig,
    provider_config: ProviderConfig,
    dry_run: bool = False,
    resume: bool = True,
    include_baseline: bool = True,
    require_checkpoints_trained: bool = True,
    provider: GPUProvider | None = None,
    grid_override: list[tuple[str, int, int]] | None = None,
) -> GenerationReport:
    """`grid_override` mirrors training/orchestrator.py's parameter of the
    same name: restrict which trained cells get generated (e.g. to the
    handful of cells `digital-ghost dry-run` actually trained), instead of
    the full sweep from `study.doses`.
    """
    provider = provider or get_provider(provider_config)
    ledger = CostLedger(
        study_cost_ledger_path(study), cap_usd=study.budget.cap_usd, hard_stop=study.budget.hard_stop
    )

    prompts_path = write_flat_prompts(study, dry_run=dry_run)
    expected_rows = sum(1 for _ in open(prompts_path))

    checkpoints = all_checkpoints(study, include_baseline=include_baseline, grid_override=grid_override)

    if require_checkpoints_trained:
        missing = []
        for arm, dose, seed_index in (grid_override if grid_override is not None else cell_grid(study)):
            cell = build_cell_spec(study, arm, dose, seed_index)
            if not cell_is_complete(cell):
                missing.append(cell.id)
        if missing:
            raise RuntimeError(
                f"{len(missing)} training cell(s) are not complete yet, cannot generate "
                f"their checkpoints: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
                "Run the training sweep first, or pass require_checkpoints_trained=False "
                "to generate only for checkpoints that do exist."
            )

    gpu_slots: queue.Queue[int] = queue.Queue()
    for i in range(provider_config.max_parallel_gpus):
        gpu_slots.put(i)

    aborted = threading.Event()
    budget_lock = threading.Lock()
    report = GenerationReport(total_checkpoints=len(checkpoints))

    def make_run_fn(checkpoint: CheckpointSpec):
        def run_fn() -> None:
            if aborted.is_set():
                raise BudgetExceededError(f"generation aborted before {checkpoint.label} could start")

            gpu_idx = gpu_slots.get()
            try:
                with budget_lock:
                    if aborted.is_set():
                        raise BudgetExceededError(f"generation aborted before {checkpoint.label} could start")
                    try:
                        ledger.check_budget(0.0)
                    except BudgetExceededError:
                        aborted.set()
                        raise

                cmd = generation_command(checkpoint, study, training, prompts_path, dry_run=dry_run)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

                logger.info("checkpoint %s starting on GPU slot %d", checkpoint.label, gpu_idx)
                start = time.monotonic()
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                elapsed_hours = (time.monotonic() - start) / 3600.0

                entry = ledger.record(checkpoint.label, elapsed_hours, provider_config.pricing_usd_per_gpu_hour)
                logger.info(
                    "checkpoint %s finished in %.3fh, cost $%.4f (cumulative $%.2f / cap $%.2f)",
                    checkpoint.label, elapsed_hours, entry.cost_usd, entry.cumulative_cost_usd, ledger.cap_usd,
                )
                if ledger.is_over_budget() and ledger.hard_stop:
                    aborted.set()
                    logger.warning(
                        "budget cap reached after checkpoint %s — no further checkpoints will be started",
                        checkpoint.label,
                    )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"generation subprocess for {checkpoint.label} exited {result.returncode}: "
                        f"{result.stderr[-2000:]}"
                    )
            finally:
                gpu_slots.put(gpu_idx)

        return run_fn

    handles = {}
    for checkpoint in checkpoints:
        if resume and is_checkpoint_generated(study, checkpoint, expected_rows):
            logger.info("checkpoint %s already generated, skipping", checkpoint.label)
            report.skipped_already_done.append(checkpoint.label)
            continue
        if aborted.is_set():
            report.skipped_budget.append(checkpoint.label)
            continue
        handles[checkpoint.label] = provider.submit_job(checkpoint.label, make_run_fn(checkpoint))

    for label, handle in handles.items():
        status = provider.wait(handle)
        if status == JobStatus.SUCCEEDED:
            report.completed.append(label)
        elif status == JobStatus.FAILED:
            err = provider.get_error(handle)
            if isinstance(err, BudgetExceededError):
                report.skipped_budget.append(label)
            else:
                report.failed[label] = str(err) if err else "unknown error"
        elif status == JobStatus.CANCELLED:
            report.skipped_budget.append(label)

    report.total_cost_usd = ledger.spent()
    report.aborted_on_budget = aborted.is_set()
    return report
