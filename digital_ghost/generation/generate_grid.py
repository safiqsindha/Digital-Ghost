"""Standalone generation of the eval grid: the stock-SDXL baseline plus every
trained LoRA checkpoint, each rendered against the same frozen 30-prompt eval
set at `eval.seeds_per_prompt` seeds.

The sweep generates each cell's images as it trains it, so this path exists
for regenerating without retraining. Generation spend counts against the same
study-wide budget cap.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from digital_ghost.config import RuntimeConfig, StudyConfig, TrainingConfig
from digital_ghost.generation.eval_prompts import eval_prompt_seeds, load_eval_prompts
from digital_ghost.sampling.subsample import cell_grid
from digital_ghost.training.cell import build_cell_spec, is_complete as cell_is_complete
from digital_ghost.proc import run_subprocess_streaming
from digital_ghost.training.cost import BudgetExceededError, CostLedger, study_cost_ledger_path

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
    runtime: RuntimeConfig,
    dry_run: bool = False,
    resume: bool = True,
    include_baseline: bool = True,
    require_checkpoints_trained: bool = True,
    grid_override: list[tuple[str, int, int]] | None = None,
) -> GenerationReport:
    """Regenerate images for checkpoints that already exist.

    The sweep generates each cell's images as it trains it, so this is the
    standalone path: redoing generation without retraining, after a change to
    the prompt set or the number of seeds per prompt.

    `grid_override` restricts which trained cells are covered, mirroring the
    sweep's parameter of the same name.
    """
    ledger = CostLedger(
        study_cost_ledger_path(study), cap_usd=study.budget.cap_usd, hard_stop=study.budget.hard_stop
    )

    prompts_path = write_flat_prompts(study, dry_run=dry_run)
    expected_rows = sum(1 for line in prompts_path.read_text().splitlines() if line.strip())

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
                "Run the sweep first, or pass require_checkpoints_trained=False "
                "to generate only for checkpoints that do exist."
            )

    gpu_slots: queue.Queue[int] = queue.Queue()
    for i in range(runtime.execution.parallel_cells):
        gpu_slots.put(i)

    aborted = threading.Event()
    budget_lock = threading.Lock()
    report = GenerationReport(total_checkpoints=len(checkpoints))

    def run_one(checkpoint: CheckpointSpec) -> None:
        gpu_idx = gpu_slots.get()
        reserved = 0.0
        try:
            with budget_lock:
                if aborted.is_set():
                    report.skipped_budget.append(checkpoint.label)
                    return
                estimate = ledger.estimate_job_cost(
                    runtime.pricing.usd_per_gpu_hour,
                    runtime.execution.estimated_gpu_hours_per_cell,
                )
                try:
                    reserved = ledger.reserve(estimate)
                except BudgetExceededError:
                    aborted.set()
                    report.skipped_budget.append(checkpoint.label)
                    return

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            log_path = checkpoint_output_dir(study, checkpoint.label) / "generation.log"
            cmd = generation_command(checkpoint, study, training, prompts_path, dry_run=dry_run)

            logger.info("checkpoint %s generating on GPU slot %d", checkpoint.label, gpu_idx)
            start = time.monotonic()
            returncode = run_subprocess_streaming(cmd, log_path, env, f"generate {checkpoint.label}")
            elapsed_hours = (time.monotonic() - start) / 3600.0

            ledger.release(reserved)
            reserved = 0.0
            entry = ledger.record(checkpoint.label, elapsed_hours, runtime.pricing.usd_per_gpu_hour)
            logger.info(
                "checkpoint %s finished in %.3fh, cost $%.4f (cumulative $%.2f / cap $%.2f)",
                checkpoint.label, elapsed_hours, entry.cost_usd, entry.cumulative_cost_usd, ledger.cap_usd,
            )
            if ledger.is_over_budget() and ledger.hard_stop:
                aborted.set()
                logger.warning(
                    "budget cap reached after checkpoint %s — no further checkpoints will start",
                    checkpoint.label,
                )

            if returncode != 0:
                report.failed[checkpoint.label] = f"generation exited {returncode}, see {log_path}"
            else:
                report.completed.append(checkpoint.label)
        finally:
            if reserved:
                ledger.release(reserved)
            gpu_slots.put(gpu_idx)

    pending = []
    for checkpoint in checkpoints:
        if resume and is_checkpoint_generated(study, checkpoint, expected_rows):
            logger.info("checkpoint %s already generated, skipping", checkpoint.label)
            report.skipped_already_done.append(checkpoint.label)
            continue
        pending.append(checkpoint)

    if runtime.execution.parallel_cells == 1:
        for checkpoint in pending:
            run_one(checkpoint)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=runtime.execution.parallel_cells) as pool:
            list(pool.map(run_one, pending))

    report.total_cost_usd = ledger.spent()
    report.aborted_on_budget = aborted.is_set()
    return report
