"""What `digital-ghost status` can tell you about a run it never saw.

The sweep runs in a tmux session you will get disconnected from, so the status
reader deliberately holds no state of its own: it rebuilds the picture from the
run directory every time. These tests pin down the three answers people act on
— how far along is it, has anything actually worked yet, and what broke — for a
run this process did not launch.
"""

from __future__ import annotations

import json

from digital_ghost.config import StudyConfig
from digital_ghost.sampling.subsample import cell_grid
from digital_ghost.training.cell import build_cell_spec, cell_id
from digital_ghost.training.status import SweepStatus, format_status, read_sweep_status
from digital_ghost.training.sweep import sweep_log_path


def _mark_failed(study: StudyConfig, triple: tuple[str, int, int], reasons: list[str]) -> str:
    """Rewrite one finished cell's metadata as a failure, in place.

    Written as a file edit rather than through write_status on purpose: the
    file on disk is the only channel between the sweep process and the status
    reader, so that is what the reader has to be tested against.
    """
    spec = build_cell_spec(study, *triple)
    meta = json.loads(spec.metadata_path.read_text())
    meta["status"] = "failed"
    meta["failures"] = reasons
    spec.metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return spec.id


class TestReadSweepStatus:
    def test_untouched_study_is_all_not_started_and_gates_the_sweep(self, tiny_study):
        """A study directory with nothing in it must read as "nothing has run",
        not as an error. `digital-ghost sweep` refuses to start a 45-cell,
        27-hour, several-hundred-dollar run until has_successful_cell proves a
        single cell has completed end to end on this machine, so False here is
        the guard that stops an unvalidated setup burning a day of rental.
        """
        status = read_sweep_status(tiny_study)

        assert len(status.cells) == len(cell_grid(tiny_study))
        assert {c.status for c in status.cells} == {"not_started"}
        assert status.has_successful_cell is False
        assert status.done == [] and status.failed == []
        assert status.remaining == status.cells
        assert status.total_cost_usd == 0.0

    def test_after_a_dry_run_sweep_cells_are_counted_as_done(self, trained_and_generated_study):
        """The same gate seen from the other side: once a dry-run sweep has
        driven train -> generate -> sanity-check for real, every cell reads back
        as succeeded from disk alone and the sweep is allowed to proceed.
        """
        study = trained_and_generated_study
        status = read_sweep_status(study)

        assert len(status.done) == len(cell_grid(study))
        assert status.has_successful_cell is True
        assert status.failed == [] and status.running == [] and status.remaining == []
        assert {c.cell_id for c in status.done} == {cell_id(*t) for t in cell_grid(study)}

    def test_failed_cell_reports_its_reasons(self, trained_and_generated_study):
        """The reasons are the point. A cell that failed 14 hours ago is only
        actionable if the reason survived in its metadata — the terminal that
        printed it is long gone, and re-running a 45-cell sweep to find out why
        is not an option.
        """
        study = trained_and_generated_study
        reasons = ["generation subprocess exited 1", "0 of 8 expected images present"]
        failed_id = _mark_failed(study, ("meme", 3, 1), reasons)

        status = read_sweep_status(study)

        assert [c.cell_id for c in status.failed] == [failed_id]
        assert status.failed[0].failures == reasons
        assert failed_id not in {c.cell_id for c in status.done}
        # One failure does not invalidate the cells that did work.
        assert status.has_successful_cell is True


class TestFormatStatus:
    def test_summary_shows_counts_spend_and_what_broke(self, trained_and_generated_study):
        """This string is the whole interface for someone checking in from a
        phone over SSH, so it has to carry the three things worth waking up
        for: progress, money spent against the cap, and which cells died.
        """
        study = trained_and_generated_study
        failed_id = _mark_failed(study, ("control", 5, 1), ["checkpoint is 0 bytes"])
        status: SweepStatus = read_sweep_status(study)

        text = format_status(status, sweep_log=sweep_log_path(study))

        assert f"cells: {len(status.done)} done" in text
        assert f"{len(status.failed)} failed" in text
        assert f"{len(status.remaining)} remaining   (of {len(status.cells)})" in text
        assert f"spend: ${status.total_cost_usd:.2f} of ${status.cap_usd:.2f} cap" in text

        assert "failed cells:" in text
        assert failed_id in text
        assert "checkpoint is 0 bytes" in text

        # The log tail is what turns "3 failed" into something diagnosable.
        assert "sweep log entries:" in text
        assert "SWEEP" in text

    def test_summary_omits_the_log_tail_when_there_is_no_log(self, tiny_study):
        """Status is run before the first sweep as often as during one; a
        missing sweep.log is a normal state, not something to blow up on.
        """
        text = format_status(read_sweep_status(tiny_study), sweep_log=sweep_log_path(tiny_study))

        assert "sweep log entries:" not in text
        assert "failed cells:" not in text
        assert "spend: $0.00" in text
