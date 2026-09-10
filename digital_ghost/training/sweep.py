"""The sweep: train, generate and validate each cell, one cell at a time.

Design assumptions, all of them load-bearing:

* **One rented box, serial by default.** Cells are compared against each
  other, so identical hardware matters more than throughput. Parallelism is
  opt-in and only spreads across GPUs in the same machine.
* **Nothing depends on the foreground terminal.** Subprocess output streams
  to per-cell log files, progress goes to a sweep log, and cell state lives in
  each cell's run_metadata.json. An SSH drop loses the ticker and nothing
  else; `digital-ghost status` reconstructs everything from disk.
* **Train and generate are interleaved per cell.** A cell isn't done until it
  has produced eval images that pass sanity checks, which means a collapsed
  LoRA surfaces on cell 1 rather than after all 45 have been paid for.
* **A failing cell is logged loudly and skipped, not fatal.** Losing 44 good
  cells because the 12th died is worse than finishing with 44.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from digital_ghost.config import RuntimeConfig, StudyConfig, TrainingConfig
from digital_ghost.generation.generate_grid import (
    BASELINE_LABEL,
    CheckpointSpec,
    checkpoint_manifest_path,
    checkpoint_output_dir,
    generation_command,
    write_flat_prompts,
)
from digital_ghost.proc import run_subprocess_streaming
from digital_ghost.hardware import HardwareChangedError, capture, check_consistency, load_reference, save_reference
from digital_ghost.notify import get_notifier, notify_cell_failure, notify_sweep_complete
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
from digital_ghost.training.sanity import SanityThresholds, check_checkpoint, check_generated_images

logger = logging.getLogger(__name__)


def sweep_log_path(study: StudyConfig) -> Path:
    return study.path("outputs_dir") / "sweep.log"


def hardware_reference_path(study: StudyConfig) -> Path:
    return study.path("outputs_dir") / "hardware_fingerprint.json"


class SweepLog:
    """Append-only, timestamped, one line per event.

    This is the record that survives a dropped connection, so every line is
    flushed immediately rather than buffered.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp}  {event:<8} {message}"
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        logger.info("%s %s", event, message)


@dataclass
class CellOutcome:
    cell_id: str
    status: str  # succeeded | failed | skipped_complete | skipped_budget
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)


@dataclass
class SweepReport:
    total_cells: int
    outcomes: list[CellOutcome] = field(default_factory=list)
    total_cost_usd: float = 0.0
    aborted_on_budget: bool = False
    elapsed_hours: float = 0.0

    @property
    def succeeded(self) -> list[CellOutcome]:
        return [o for o in self.outcomes if o.status == "succeeded"]

    @property
    def failed(self) -> list[CellOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def skipped(self) -> list[CellOutcome]:
        return [o for o in self.outcomes if o.status.startswith("skipped")]


class Ticker:
    """Live progress on stdout: cell N of M, elapsed, spend, projections.

    Every write is guarded. After an SSH drop stdout may be a dead pipe, and a
    BrokenPipeError from a progress line must never take down a sweep that is
    otherwise running fine.
    """

    def __init__(self, total_cells: int, refresh_seconds: float, ledger: CostLedger, enabled: bool = True):
        self.total = total_cells
        self.refresh = refresh_seconds
        self.ledger = ledger
        self.enabled = enabled
        self.started = time.monotonic()
        self.done = 0
        self.current = "starting"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def update(self, done: int, current: str) -> None:
        self.done = done
        self.current = current

    def _loop(self) -> None:
        while not self._stop.wait(self.refresh):
            self._print()

    def _print(self) -> None:
        try:
            elapsed = time.monotonic() - self.started
            spent = self.ledger.spent()
            if self.done > 0:
                per_cell = elapsed / self.done
                remaining = (self.total - self.done) * per_cell
                eta = datetime.now() + timedelta(seconds=remaining)
                projected = spent / self.done * self.total
                eta_text = eta.strftime("%H:%M %d-%b")
                projected_text = f"${projected:.2f}"
            else:
                eta_text = "--:--"
                projected_text = "--"
            print(
                f"[sweep] cell {self.done}/{self.total} | {self.current} | "
                f"elapsed {_hms(elapsed)} | spent ${spent:.2f} | "
                f"projected {projected_text} | finish ~{eta_text}",
                flush=True,
            )
        except Exception:  # noqa: BLE001 - a dead terminal must not kill the sweep
            pass


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _generate_for_checkpoint(
    study: StudyConfig,
    training: TrainingConfig,
    checkpoint: CheckpointSpec,
    prompts_path: Path,
    log_path: Path,
    env: dict,
    dry_run: bool,
) -> int:
    cmd = generation_command(checkpoint, study, training, prompts_path, dry_run=dry_run)
    return run_subprocess_streaming(cmd, log_path, env, f"generate {checkpoint.label}")


def _generated_image_paths(study: StudyConfig, label: str) -> list[Path]:
    manifest = checkpoint_manifest_path(study, label)
    if not manifest.exists():
        return []
    import json

    paths = []
    for line in manifest.read_text().splitlines():
        if line.strip():
            paths.append(Path(json.loads(line)["image_path"]))
    return paths


def run_sweep(
    study: StudyConfig,
    training: TrainingConfig,
    runtime: RuntimeConfig,
    *,
    dry_run: bool = False,
    resume: bool = True,
    grid_override: list[tuple[str, int, int]] | None = None,
    allow_hardware_change: bool = False,
    include_baseline: bool = True,
) -> SweepReport:
    started = time.monotonic()
    sweep_log = SweepLog(sweep_log_path(study))
    notifier = get_notifier(runtime.notifications)

    ledger = CostLedger(
        study_cost_ledger_path(study), cap_usd=study.budget.cap_usd, hard_stop=study.budget.hard_stop
    )
    thresholds = SanityThresholds(
        min_image_bytes=runtime.sanity.min_image_bytes,
        min_pixel_std=runtime.sanity.min_pixel_std,
        max_identical_fraction=runtime.sanity.max_identical_fraction,
        min_mean_luminance=runtime.sanity.min_mean_luminance,
        max_mean_luminance=runtime.sanity.max_mean_luminance,
    )

    # --- hardware consistency ------------------------------------------------
    current_fp = capture()
    reference_fp = load_reference(hardware_reference_path(study))
    if reference_fp is None:
        save_reference(hardware_reference_path(study), current_fp)
        sweep_log.write("HARDWARE", f"reference recorded gpu={current_fp.gpu_name} digest={current_fp.digest[:12]}")
    else:
        diffs = check_consistency(current_fp, reference_fp, allow_change=True)
        if diffs:
            if runtime.hardware.enforce_consistency and not allow_hardware_change:
                sweep_log.write("ABORT", f"hardware changed: {'; '.join(diffs)}")
                raise HardwareChangedError(diffs)
            sweep_log.write("HARDWARE", f"CHANGED (allowed): {'; '.join(diffs)}")

    grid = grid_override if grid_override is not None else cell_grid(study)
    prompts_path = write_flat_prompts(study, dry_run=dry_run)
    expected_images = sum(1 for line in prompts_path.read_text().splitlines() if line.strip())

    report = SweepReport(total_cells=len(grid))
    ticker = Ticker(len(grid), runtime.ticker.refresh_seconds, ledger, enabled=runtime.ticker.enabled)

    sweep_log.write(
        "SWEEP",
        f"start cells={len(grid)} parallel={runtime.execution.parallel_cells} "
        f"dry_run={dry_run} gpu={current_fp.gpu_name}",
    )

    # --- baseline: no training, but the analysis is anchored on it -----------
    if include_baseline:
        baseline = CheckpointSpec(BASELINE_LABEL, None, None, None, None)
        if not (resume and len(_generated_image_paths(study, BASELINE_LABEL)) == expected_images):
            sweep_log.write("START", f"cell={BASELINE_LABEL} (baseline, generation only)")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            log_path = checkpoint_output_dir(study, BASELINE_LABEL) / "cell.log"
            t0 = time.monotonic()
            rc = _generate_for_checkpoint(study, training, baseline, prompts_path, log_path, env, dry_run)
            hours = (time.monotonic() - t0) / 3600
            entry = ledger.record(BASELINE_LABEL, hours, runtime.pricing.usd_per_gpu_hour)
            if rc != 0:
                sweep_log.write("FAIL", f"cell={BASELINE_LABEL} generation exited {rc} log={log_path}")
            else:
                sweep_log.write("DONE", f"cell={BASELINE_LABEL} elapsed={_hms(hours*3600)} cost=${entry.cost_usd:.4f}")
        else:
            sweep_log.write("SKIP", f"cell={BASELINE_LABEL} already generated")

    gpu_slots: queue.Queue[int] = queue.Queue()
    for i in range(runtime.execution.parallel_cells):
        gpu_slots.put(i)

    aborted = threading.Event()
    budget_lock = threading.Lock()
    progress_lock = threading.Lock()
    ticker.start()

    def run_one(index: int, arm: str, dose: int, seed_index: int) -> CellOutcome:
        cid = cell_id(arm, dose, seed_index)
        try:
            cell = build_cell_spec(study, arm, dose, seed_index)
        except Exception as e:  # noqa: BLE001 - one bad cell must not end the sweep
            sweep_log.write("FAIL", f"cell={cid} could not build spec: {e}")
            return CellOutcome(cid, "failed", failures=[f"spec: {e}"])

        if resume and is_complete(cell):
            sweep_log.write("SKIP", f"cell={cid} already complete")
            return CellOutcome(cid, "skipped_complete")

        if aborted.is_set():
            write_status(cell, "skipped_budget")
            return CellOutcome(cid, "skipped_budget")

        gpu_idx = gpu_slots.get()
        reserved = 0.0
        t0 = time.monotonic()
        try:
            with budget_lock:
                if aborted.is_set():
                    write_status(cell, "skipped_budget")
                    return CellOutcome(cid, "skipped_budget")
                estimate = ledger.estimate_job_cost(
                    runtime.pricing.usd_per_gpu_hour, runtime.execution.estimated_gpu_hours_per_cell
                )
                try:
                    reserved = ledger.reserve(estimate)
                except BudgetExceededError:
                    aborted.set()
                    write_status(cell, "skipped_budget")
                    sweep_log.write("ABORT", f"cell={cid} budget cap reached before start")
                    return CellOutcome(cid, "skipped_budget")

            sweep_log.write("START", f"cell={cid} ({index}/{len(grid)}) gpu_slot={gpu_idx}")
            write_status(cell, "running", hardware=current_fp.to_dict())

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

            failures: list[str] = []

            rc = run_subprocess_streaming(
                training_command(cell, study, training, dry_run=dry_run),
                cell.log_path, env, f"train {cid}",
            )
            if rc != 0:
                failures.append(f"training subprocess exited {rc}")

            if not failures:
                ckpt_report = check_checkpoint(
                    cid, cell.checkpoint_path, runtime.sanity.min_checkpoint_bytes
                )
                if not ckpt_report.passed:
                    failures.extend(ckpt_report.failures)

            if not failures:
                checkpoint = CheckpointSpec(cid, str(cell.checkpoint_path), arm, dose, seed_index)
                rc = _generate_for_checkpoint(
                    study, training, checkpoint, prompts_path, cell.log_path, env, dry_run
                )
                if rc != 0:
                    failures.append(f"generation subprocess exited {rc}")

            if not failures:
                images = _generated_image_paths(study, cid)
                img_report = check_generated_images(cid, images, expected_images, thresholds)
                if not img_report.passed:
                    failures.extend(img_report.failures)

            elapsed = time.monotonic() - t0
            ledger.release(reserved)
            reserved = 0.0
            entry = ledger.record(cid, elapsed / 3600, runtime.pricing.usd_per_gpu_hour)

            if ledger.is_over_budget() and ledger.hard_stop:
                aborted.set()
                sweep_log.write("ABORT", f"budget cap reached after cell={cid}")

            if failures:
                write_status(
                    cell, "failed", cost_usd=entry.cost_usd, gpu_hours=elapsed / 3600,
                    failures=failures, hardware=current_fp.to_dict(),
                )
                sweep_log.write(
                    "FAIL",
                    f"cell={cid} elapsed={_hms(elapsed)} cost=${entry.cost_usd:.4f} "
                    f"reasons={'; '.join(failures[:3])} log={cell.log_path}",
                )
                return CellOutcome(cid, "failed", elapsed, entry.cost_usd, failures)

            write_status(
                cell, "succeeded", cost_usd=entry.cost_usd, gpu_hours=elapsed / 3600,
                hardware=current_fp.to_dict(),
            )
            sweep_log.write(
                "DONE",
                f"cell={cid} elapsed={_hms(elapsed)} cost=${entry.cost_usd:.4f} "
                f"cumulative=${entry.cumulative_cost_usd:.2f}",
            )
            return CellOutcome(cid, "succeeded", elapsed, entry.cost_usd)
        finally:
            if reserved:
                ledger.release(reserved)
            gpu_slots.put(gpu_idx)

    try:
        if runtime.execution.parallel_cells == 1:
            for i, (arm, dose, seed_index) in enumerate(grid, start=1):
                ticker.update(len(report.outcomes), cell_id(arm, dose, seed_index))
                outcome = run_one(i, arm, dose, seed_index)
                with progress_lock:
                    report.outcomes.append(outcome)
                if outcome.status == "failed":
                    notify_cell_failure(
                        notifier, outcome.cell_id, "; ".join(outcome.failures),
                        len(report.outcomes), len(grid),
                    )
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=runtime.execution.parallel_cells) as pool:
                futures = {
                    pool.submit(run_one, i, arm, dose, seed_index): cell_id(arm, dose, seed_index)
                    for i, (arm, dose, seed_index) in enumerate(grid, start=1)
                }
                for fut in as_completed(futures):
                    outcome = fut.result()
                    with progress_lock:
                        report.outcomes.append(outcome)
                        ticker.update(len(report.outcomes), outcome.cell_id)
                    if outcome.status == "failed":
                        notify_cell_failure(
                            notifier, outcome.cell_id, "; ".join(outcome.failures),
                            len(report.outcomes), len(grid),
                        )
    finally:
        ticker.stop()

    report.total_cost_usd = ledger.spent()
    report.aborted_on_budget = aborted.is_set()
    report.elapsed_hours = (time.monotonic() - started) / 3600

    sweep_log.write(
        "SWEEP",
        f"end succeeded={len(report.succeeded)} failed={len(report.failed)} "
        f"skipped={len(report.skipped)} cost=${report.total_cost_usd:.2f} "
        f"elapsed={_hms(report.elapsed_hours * 3600)}",
    )
    notify_sweep_complete(
        notifier, len(report.succeeded), len(report.failed),
        report.total_cost_usd, report.elapsed_hours,
    )
    return report
