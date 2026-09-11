"""Behaviours the sweep has to get right while nobody is watching.

The sweep is a ~27 hour unattended job on a rented box, driven from tmux over
SSH. Everything here pins down a property that only matters *because* of that:
state has to survive on disk rather than in the foreground process, one bad
cell must not take the other 44 with it, a cell that quietly produced garbage
must be recorded as failed rather than succeeded, and the machine must not
change underneath a study whose whole claim rests on cells being comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import digital_ghost.training.sweep as sweep
from digital_ghost.config import (
    RuntimeConfig,
    StudyConfig,
    TrainingConfig,
    load_captioning_config,
    load_runtime_config,
    load_training_config,
)
from digital_ghost.hardware import HardwareChangedError
from digital_ghost.training.cell import CellSpec, build_cell_spec, cell_id
from digital_ghost.training.cost import CostLedger
from digital_ghost.training.sanity import SanityReport
from digital_ghost.training.sweep import Ticker, hardware_reference_path, run_sweep, sweep_log_path

# tiny_study is 3 arms x doses [3, 5] x 1 seed = 6 cells. Tests take the
# smallest slice that still proves the property, because every cell is two
# real subprocesses even in dry-run mode.
ONE_CELL: list[tuple[str, int, int]] = [("standard", 3, 1)]
TWO_CELLS: list[tuple[str, int, int]] = [("standard", 3, 1), ("meme", 3, 1)]
THREE_CELLS: list[tuple[str, int, int]] = [("standard", 3, 1), ("meme", 3, 1), ("control", 3, 1)]


@pytest.fixture
def prepared_study(tiny_study) -> StudyConfig:
    """tiny_study ingested, captioned and with eval prompts frozen — exactly
    the state a real sweep starts from, with no cell run yet.
    """
    from digital_ghost.caption.generate import caption_all_arms
    from digital_ghost.generation.eval_prompts import init_eval_prompts
    from digital_ghost.ingest.manifest import ingest_all_arms

    ingest_all_arms(tiny_study, min_count=5)
    caption_all_arms(tiny_study, load_captioning_config(tiny_study))
    init_eval_prompts(tiny_study)
    return tiny_study


@pytest.fixture
def training_cfg(prepared_study) -> TrainingConfig:
    return load_training_config(prepared_study)


@pytest.fixture
def runtime_cfg(prepared_study) -> RuntimeConfig:
    runtime = load_runtime_config(prepared_study)
    # The ticker is a daemon thread printing to stdout on a timer. Under pytest
    # it adds nothing and can outlive the test it belongs to.
    runtime.ticker.enabled = False
    return runtime


def _cell(study: StudyConfig, triple: tuple[str, int, int]) -> CellSpec:
    arm, dose, seed_index = triple
    return build_cell_spec(study, arm, dose, seed_index)


def _metadata(study: StudyConfig, triple: tuple[str, int, int]) -> dict:
    return json.loads(_cell(study, triple).metadata_path.read_text())


def _outcomes_by_id(report: sweep.SweepReport) -> dict[str, sweep.CellOutcome]:
    return {o.cell_id: o for o in report.outcomes}


class TestSweepWritesItsStateToDisk:
    def test_dry_run_sweep_leaves_a_full_paper_trail(self, prepared_study, training_cfg, runtime_cfg):
        """Nothing the sweep produces may live only in the foreground process.
        An SSH drop kills the terminal and the ticker; the sweep log, the
        per-cell cell.log and each run_metadata.json are what `digital-ghost
        status` reconstructs the run from afterwards, so all three have to be
        on disk by the time a cell reports success.
        """
        report = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=TWO_CELLS, include_baseline=False,
        )

        assert [o.status for o in report.outcomes] == ["succeeded", "succeeded"]

        log = sweep_log_path(prepared_study).read_text()
        for triple in TWO_CELLS:
            cid = cell_id(*triple)
            assert f"START    cell={cid}" in log
            assert f"DONE     cell={cid}" in log

            cell = _cell(prepared_study, triple)
            assert cell.log_path.exists(), "subprocess output must be captured per cell"
            assert cell.log_path.stat().st_size > 0
            assert _metadata(prepared_study, triple)["status"] == "succeeded"

        assert "SWEEP    start" in log and "SWEEP    end" in log

    def test_resume_skips_completed_cells_without_redoing_them(
        self, prepared_study, training_cfg, runtime_cfg
    ):
        """Resume is the recovery path for a sweep killed at hour 19. It has to
        be cheap and idempotent: a skipped cell must not be re-trained (that is
        real money) and must not have its metadata rewritten, or the resumed
        run would silently overwrite the results it was meant to preserve.
        """
        first = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=TWO_CELLS, include_baseline=False,
        )
        assert [o.status for o in first.outcomes] == ["succeeded", "succeeded"]
        before = {t: _metadata(prepared_study, t)["updated_at"] for t in TWO_CELLS}

        second = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=TWO_CELLS, include_baseline=False,
        )

        assert [o.status for o in second.outcomes] == ["skipped_complete", "skipped_complete"]
        assert {t: _metadata(prepared_study, t)["updated_at"] for t in TWO_CELLS} == before
        # A skipped cell costs nothing, so the second run must not add spend.
        assert second.total_cost_usd == pytest.approx(first.total_cost_usd)


class TestOneBadCellIsNotFatal:
    def test_failing_cell_does_not_abort_the_cells_after_it(
        self, prepared_study, training_cfg, runtime_cfg, monkeypatch
    ):
        """The single most important property in sweep.py. Losing 44 good cells
        because the 12th hit an OOM, a corrupt download or a transient CUDA
        error would mean paying for the whole 27 hours twice. The failure is
        recorded loudly against its own cell and the sweep walks on.
        """
        doomed = cell_id("meme", 3, 1)
        real_training_command = sweep.training_command

        def failing_for_one_cell(cell: CellSpec, study, training, dry_run: bool = False) -> list[str]:
            if cell.id == doomed:
                return [sys.executable, "-c", "raise SystemExit(1)"]
            return real_training_command(cell, study, training, dry_run=dry_run)

        monkeypatch.setattr(sweep, "training_command", failing_for_one_cell)

        report = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=THREE_CELLS, include_baseline=False,
        )

        outcomes = _outcomes_by_id(report)
        assert outcomes[doomed].status == "failed"
        assert any("exited 1" in f for f in outcomes[doomed].failures)

        # The doomed cell sits in the middle of the grid, so a surviving third
        # cell is proof the sweep did not stop at the failure.
        survivors = [cell_id(*t) for t in THREE_CELLS if cell_id(*t) != doomed]
        assert [outcomes[cid].status for cid in survivors] == ["succeeded", "succeeded"]
        assert len(report.succeeded) == 2 and len(report.failed) == 1

        assert _metadata(prepared_study, ("meme", 3, 1))["status"] == "failed"
        for triple in (("standard", 3, 1), ("control", 3, 1)):
            assert _metadata(prepared_study, triple)["status"] == "succeeded"

    def test_a_wedged_cell_hits_its_deadline_and_the_sweep_walks_on(
        self, prepared_study, training_cfg, runtime_cfg, monkeypatch
    ):
        """A hang is the one failure the sanity checks cannot see.

        They only run once the subprocess returns, so without a deadline a
        wedged trainer stalls the sweep for as long as the box stays up — no
        progress, no failure, no next cell, and the GPU billing throughout.
        The deadline has to turn that into an ordinary failed cell.
        """
        doomed = cell_id("meme", 3, 1)
        real_training_command = sweep.training_command

        def hangs_for_one_cell(cell: CellSpec, study, training, dry_run: bool = False) -> list[str]:
            if cell.id == doomed:
                return [sys.executable, "-c", "import time; time.sleep(300)"]
            return real_training_command(cell, study, training, dry_run=dry_run)

        monkeypatch.setattr(sweep, "training_command", hangs_for_one_cell)
        # Deadlines are configured in hours. 25s is comfortably longer than a
        # healthy dry-run cell (which still pays a torch import) and far
        # shorter than the doomed cell's 300s sleep, so only the hang is cut.
        runtime_cfg.execution.training_timeout_hours = 25 / 3600
        runtime_cfg.execution.kill_grace_seconds = 3.0

        report = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=THREE_CELLS, include_baseline=False,
        )

        outcomes = _outcomes_by_id(report)
        assert outcomes[doomed].status == "failed"
        assert any("timed out" in f for f in outcomes[doomed].failures), outcomes[doomed].failures

        survivors = [cell_id(*t) for t in THREE_CELLS if cell_id(*t) != doomed]
        assert [outcomes[cid].status for cid in survivors] == ["succeeded", "succeeded"]
        assert _metadata(prepared_study, ("meme", 3, 1))["status"] == "failed"

    def test_sanity_failure_marks_the_cell_failed_rather_than_succeeded(
        self, prepared_study, training_cfg, runtime_cfg, monkeypatch
    ):
        """A collapsed LoRA exits 0 and writes files: on disk it is
        indistinguishable from success. If the sanity report were merely
        logged, 150 black frames per cell would flow into the human rating
        study and quietly corrupt the result. The reasons have to reach
        run_metadata.json too, since that is all `status` can see later.
        """
        reason = "all 8 images are byte-identical"

        def collapsed_output(cid: str, image_paths, expected_count: int, thresholds=None) -> SanityReport:
            return SanityReport(
                cell_id=cid, passed=False, failures=[reason], checked_n_images=len(image_paths)
            )

        monkeypatch.setattr(sweep, "check_generated_images", collapsed_output)

        report = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=ONE_CELL, include_baseline=False,
        )

        (outcome,) = report.outcomes
        assert outcome.status == "failed"
        assert reason in outcome.failures

        meta = _metadata(prepared_study, ONE_CELL[0])
        assert meta["status"] == "failed"
        assert reason in meta["failures"]


class TestHardwareConsistency:
    def test_changed_gpu_aborts_the_sweep_unless_explicitly_allowed(
        self, prepared_study, training_cfg, runtime_cfg
    ):
        """Cells are only comparable to each other if they trained on the same
        box. Marketplace instances get reclaimed and replaced mid-rental, and a
        GPU swap at cell 20 puts a confound *inside* the comparison that no
        later analysis can undo — so the sweep stops rather than continuing
        onto different hardware, and only continues when told to, loudly.
        """
        run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=ONE_CELL, include_baseline=False,
        )

        reference_path: Path = hardware_reference_path(prepared_study)
        reference = json.loads(reference_path.read_text())
        assert reference["gpu_name"], "the first run must record what it trained on"

        swapped_gpu = "NVIDIA H100 80GB HBM3"
        reference["gpu_name"] = swapped_gpu
        reference_path.write_text(json.dumps(reference))

        with pytest.raises(HardwareChangedError, match="gpu_name"):
            run_sweep(
                prepared_study, training_cfg, runtime_cfg,
                dry_run=True, resume=True, grid_override=ONE_CELL, include_baseline=False,
            )
        assert "ABORT    hardware changed" in sweep_log_path(prepared_study).read_text()

        report = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=ONE_CELL,
            include_baseline=False, allow_hardware_change=True,
        )

        assert [o.status for o in report.outcomes] == ["skipped_complete"]
        log = sweep_log_path(prepared_study).read_text()
        # The escape hatch must leave evidence: an unexplained hardware change
        # in the results is worse than one that is written down.
        assert "CHANGED (allowed)" in log and swapped_gpu in log


class TestBudgetCap:
    def test_cap_stops_new_cells_instead_of_overspending(
        self, prepared_study, training_cfg, runtime_cfg
    ):
        """The cap exists because the sweep spends real money for a day while
        unattended. Budget is reserved *before* a cell launches, so hitting the
        cap has to stop the remaining cells starting rather than being noticed
        after they have already been paid for.
        """
        prepared_study.budget.cap_usd = 0.01
        runtime_cfg.pricing.usd_per_gpu_hour = 10_000.0

        report = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=TWO_CELLS, include_baseline=False,
        )

        assert [o.status for o in report.outcomes] == ["skipped_budget", "skipped_budget"]
        assert report.aborted_on_budget is True
        assert report.succeeded == []
        # Recorded on disk as well, so a resumed run can tell "we ran out of
        # money here" apart from "this cell was never reached".
        for triple in TWO_CELLS:
            assert _metadata(prepared_study, triple)["status"] == "skipped_budget"
        assert "ABORT" in sweep_log_path(prepared_study).read_text()

    def test_running_out_of_money_does_not_discard_a_paid_cell(
        self, prepared_study, training_cfg, runtime_cfg
    ):
        """Hitting the cap must not make the next resume re-buy work already done.

        `is_complete` decides whether to retrain by reading run_metadata.json,
        so stamping "skipped_budget" over a cell that already succeeded would
        send a resumed sweep back to the GPU for a checkpoint sitting on disk —
        the budget cap causing spend instead of preventing it. The succeeded
        record has to survive the abort.
        """
        run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=True, grid_override=ONE_CELL, include_baseline=False,
        )
        assert _metadata(prepared_study, ONE_CELL[0])["status"] == "succeeded"

        # resume=False so the finished cell is re-attempted rather than skipped,
        # and a cap that no cell can clear so it is aborted on the way in.
        prepared_study.budget.cap_usd = 0.01
        runtime_cfg.pricing.usd_per_gpu_hour = 10_000.0
        report = run_sweep(
            prepared_study, training_cfg, runtime_cfg,
            dry_run=True, resume=False, grid_override=ONE_CELL, include_baseline=False,
        )

        assert [o.status for o in report.outcomes] == ["skipped_budget"]
        assert _metadata(prepared_study, ONE_CELL[0])["status"] == "succeeded"


class TestTicker:
    def test_print_swallows_a_dead_stdout(self, tmp_path, monkeypatch):
        """After an SSH drop the sweep keeps running but its stdout is a dead
        pipe. A BrokenPipeError raised from a cosmetic progress line would
        propagate out of the ticker thread and, worse, out of any foreground
        write — killing a 27 hour job over a status update nobody is reading.
        """
        ledger = CostLedger(tmp_path / "cost.jsonl", cap_usd=100.0)
        ticker = Ticker(total_cells=45, refresh_seconds=0.01, ledger=ledger, enabled=False)

        def dead_terminal(*args, **kwargs):
            raise BrokenPipeError("stdout is gone")

        monkeypatch.setattr("builtins.print", dead_terminal)

        ticker._print()  # before any cell finished: the "--:--" projection branch
        ticker.update(12, "meme_dose0050_seed1")
        ticker._print()  # after cells finished: the ETA/projection branch
