"""Runs the full 45-cell sweep: parallel across N GPUs, resumable, cost-tracked.

Concurrency model: every pending cell is submitted to the configured
`GPUProvider` immediately; each cell's work function blocks on a bounded
queue of GPU slot indices (size = provider.max_parallel_gpus) before it
actually launches its training subprocess with `CUDA_VISIBLE_DEVICES`
scoped to that slot. That keeps true concurrency bounded to N regardless of
how many cells are "submitted" at once, and works the same way whether the
provider runs jobs as local threads (the stub) or dispatches to a remote
service.

Resume: a cell already marked "succeeded" with its checkpoint file present
is skipped. A cell that failed or was interrupted is retried from scratch
(LoRA training is fast enough per-cell that step-level resume isn't worth
the complexity here — see run_metadata.json for a cell's last known state).

Budget: before every subprocess launch, the ledger's cumulative spend is
checked against `budget.cap_usd`. Once exceeded, no new subprocess starts;
already-running ones are left to finish rather than killed mid-write.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field

from digital_ghost.config import ProviderConfig, StudyConfig, TrainingConfig
from digital_ghost.sampling.subsample import cell_grid
from digital_ghost.training.cell import (
    CellSpec,
    build_cell_spec,
    cell_id,
    is_complete,
    training_command,
    write_status,
)
from digital_ghost.training.cost import BudgetExceededError, CostLedger, study_cost_ledger_path
from digital_ghost.training.provider import GPUProvider, JobStatus, get_provider

logger = logging.getLogger(__name__)


@dataclass
class SweepReport:
    total_cells: int
    completed: list[str] = field(default_factory=list)
    skipped_already_done: list[str] = field(default_factory=list)
    skipped_budget: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    aborted_on_budget: bool = False


def run_sweep(
    study: StudyConfig,
    training: TrainingConfig,
    provider_config: ProviderConfig,
    dry_run: bool = False,
    resume: bool = True,
    only_arms: set[str] | None = None,
    only_doses: set[int] | None = None,
    provider: GPUProvider | None = None,
    grid_override: list[tuple[str, int, int]] | None = None,
) -> SweepReport:
    """`grid_override` replaces the full (arm, dose, seed_index) sweep grid
    with an arbitrary list — used by `digital-ghost dry-run` to exercise
    one tiny cell per arm instead of the real 45-cell sweep, without
    touching `study.doses`.
    """
    provider = provider or get_provider(provider_config)
    ledger = CostLedger(
        study_cost_ledger_path(study), cap_usd=study.budget.cap_usd, hard_stop=study.budget.hard_stop
    )

    gpu_slots: queue.Queue[int] = queue.Queue()
    for i in range(provider_config.max_parallel_gpus):
        gpu_slots.put(i)

    aborted = threading.Event()
    budget_lock = threading.Lock()

    source_grid = grid_override if grid_override is not None else cell_grid(study)
    grid = [
        (arm, dose, seed_index)
        for arm, dose, seed_index in source_grid
        if (only_arms is None or arm in only_arms) and (only_doses is None or dose in only_doses)
    ]

    report = SweepReport(total_cells=len(grid))

    def make_run_fn(cell: CellSpec):
        def run_fn() -> None:
            if aborted.is_set():
                write_status(cell, "skipped_budget")
                raise BudgetExceededError(f"sweep aborted before {cell.id} could start")

            gpu_idx = gpu_slots.get()
            reserved = 0.0
            try:
                with budget_lock:
                    if aborted.is_set():
                        write_status(cell, "skipped_budget")
                        raise BudgetExceededError(f"sweep aborted before {cell.id} could start")
                    # Reserve this job's estimated cost against the cap BEFORE
                    # launching. Checking against recorded spend alone lets
                    # every parallel worker start while the ledger still reads
                    # zero, overshooting the cap ~max_parallel_gpus-fold.
                    estimate = ledger.estimate_job_cost(
                        provider_config.pricing_usd_per_gpu_hour,
                        provider_config.estimated_gpu_hours_per_job,
                    )
                    try:
                        reserved = ledger.reserve(estimate)
                    except BudgetExceededError:
                        aborted.set()
                        write_status(cell, "skipped_budget")
                        raise

                write_status(cell, "running")
                cmd = training_command(cell, study, training, dry_run=dry_run)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

                logger.info("cell %s starting on GPU slot %d", cell.id, gpu_idx)
                start = time.monotonic()
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                elapsed_hours = (time.monotonic() - start) / 3600.0

                # Cost is already incurred at this point regardless of outcome —
                # record it unconditionally, then decide whether to stop
                # scheduling further cells. Release the reservation first so
                # the estimate isn't double-counted alongside the actual.
                ledger.release(reserved)
                reserved = 0.0
                entry = ledger.record(cell.id, elapsed_hours, provider_config.pricing_usd_per_gpu_hour)
                logger.info(
                    "cell %s finished in %.3fh, cost $%.4f (cumulative $%.2f / cap $%.2f)",
                    cell.id, elapsed_hours, entry.cost_usd, entry.cumulative_cost_usd, ledger.cap_usd,
                )
                if ledger.is_over_budget() and ledger.hard_stop:
                    aborted.set()
                    logger.warning(
                        "budget cap reached after cell %s (spent $%.2f / cap $%.2f) — "
                        "no further cells will be started",
                        cell.id, entry.cumulative_cost_usd, ledger.cap_usd,
                    )

                if result.returncode != 0:
                    write_status(
                        cell, "failed",
                        cost_usd=entry.cost_usd,
                        gpu_hours=elapsed_hours,
                        stderr_tail=result.stderr[-4000:],
                    )
                    raise RuntimeError(
                        f"training subprocess for {cell.id} exited {result.returncode}: "
                        f"{result.stderr[-2000:]}"
                    )

                write_status(cell, "succeeded", cost_usd=entry.cost_usd, gpu_hours=elapsed_hours)
            finally:
                # Covers the paths where the job never reached `record` (crash,
                # exception); leaving the reservation standing would shrink the
                # remaining budget for every later cell.
                if reserved:
                    ledger.release(reserved)
                gpu_slots.put(gpu_idx)

        return run_fn

    handles = {}
    for arm, dose, seed_index in grid:
        try:
            cell = build_cell_spec(study, arm, dose, seed_index)
        except Exception as e:  # noqa: BLE001 - recorded per-cell below
            # Runs on the submission thread, so an uncaught failure here (a
            # cell missing a caption, say) would abandon the whole sweep
            # mid-flight, orphaning running jobs and losing the cost report
            # for money already spent. Fail just this cell instead.
            failed_id = cell_id(arm, dose, seed_index)
            logger.exception("could not build cell spec for %s", failed_id)
            report.failed[failed_id] = str(e)
            continue

        if resume and is_complete(cell):
            logger.info("cell %s already complete, skipping", cell.id)
            report.skipped_already_done.append(cell.id)
            continue

        if aborted.is_set():
            write_status(cell, "skipped_budget")
            report.skipped_budget.append(cell.id)
            continue

        handles[cell.id] = provider.submit_job(cell.id, make_run_fn(cell))

    for cell_id, handle in handles.items():
        status = provider.wait(handle)
        if status == JobStatus.SUCCEEDED:
            report.completed.append(cell_id)
        elif status == JobStatus.FAILED:
            err = provider.get_error(handle)
            if isinstance(err, BudgetExceededError):
                report.skipped_budget.append(cell_id)
            else:
                report.failed[cell_id] = str(err) if err else "unknown error"
        elif status == JobStatus.CANCELLED:
            report.skipped_budget.append(cell_id)

    report.total_cost_usd = ledger.spent()
    report.aborted_on_budget = aborted.is_set()
    return report
