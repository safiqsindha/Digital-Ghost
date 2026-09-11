"""Reconstruct sweep state from the run directory.

Everything here reads from disk and nothing from memory, which is what makes
it useful: the sweep may be running in a tmux session you've been
disconnected from, or may have died hours ago. `digital-ghost status` answers
"what actually happened" either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from digital_ghost.config import StudyConfig
from digital_ghost.sampling.subsample import cell_grid
from digital_ghost.training.cell import build_cell_spec, cell_id, read_status
from digital_ghost.training.cost import CostLedger, study_cost_ledger_path


@dataclass
class CellStatus:
    cell_id: str
    arm: str
    dose: int
    seed_index: int
    status: str  # succeeded | failed | running | invalidated | not_started
    cost_usd: float = 0.0
    gpu_hours: float = 0.0
    failures: list[str] = field(default_factory=list)
    updated_at: str = ""


@dataclass
class SweepStatus:
    cells: list[CellStatus]
    total_cost_usd: float
    cap_usd: float

    @property
    def done(self) -> list[CellStatus]:
        return [c for c in self.cells if c.status == "succeeded"]

    @property
    def failed(self) -> list[CellStatus]:
        return [c for c in self.cells if c.status == "failed"]

    @property
    def running(self) -> list[CellStatus]:
        return [c for c in self.cells if c.status == "running"]

    @property
    def remaining(self) -> list[CellStatus]:
        return [c for c in self.cells if c.status in ("not_started", "invalidated", "skipped_budget")]

    @property
    def has_successful_cell(self) -> bool:
        """Whether any cell has completed — the gate `--sweep` checks.

        The smoke cell writes into the real run directory, so a successful
        cell of any kind is proof that a full train-generate-validate pass has
        worked on this machine.
        """
        return bool(self.done)


def read_sweep_status(study: StudyConfig) -> SweepStatus:
    cells: list[CellStatus] = []
    for arm, dose, seed_index in cell_grid(study):
        cid = cell_id(arm, dose, seed_index)
        try:
            spec = build_cell_spec(study, arm, dose, seed_index)
            meta = read_status(spec)
        except Exception:  # noqa: BLE001 - a cell we can't even spec is "not started"
            meta = None

        if not meta:
            cells.append(CellStatus(cid, arm, dose, seed_index, "not_started"))
            continue

        cells.append(
            CellStatus(
                cell_id=cid,
                arm=arm,
                dose=dose,
                seed_index=seed_index,
                status=meta.get("status", "not_started"),
                cost_usd=float(meta.get("cost_usd") or 0.0),
                gpu_hours=float(meta.get("gpu_hours") or 0.0),
                failures=list(meta.get("failures") or []),
                updated_at=str(meta.get("updated_at") or ""),
            )
        )

    ledger = CostLedger(study_cost_ledger_path(study), cap_usd=study.budget.cap_usd)
    return SweepStatus(cells=cells, total_cost_usd=ledger.spent(), cap_usd=study.budget.cap_usd)


def format_status(status: SweepStatus, sweep_log: Path | None = None, tail: int = 5) -> str:
    total = len(status.cells)
    lines = [
        f"cells: {len(status.done)} done   {len(status.failed)} failed   "
        f"{len(status.running)} running   {len(status.remaining)} remaining   (of {total})",
        f"spend: ${status.total_cost_usd:.2f} of ${status.cap_usd:.2f} cap",
    ]

    if status.done:
        hours = [c.gpu_hours for c in status.done if c.gpu_hours > 0]
        if hours:
            mean_h = sum(hours) / len(hours)
            remaining_h = mean_h * len(status.remaining)
            lines.append(
                f"pace:  {mean_h * 60:.1f} min/cell observed -> ~{remaining_h:.1f}h left "
                f"for {len(status.remaining)} remaining"
            )

    if status.failed:
        lines.append("")
        lines.append("failed cells:")
        for c in status.failed:
            reason = c.failures[0] if c.failures else "see cell.log"
            lines.append(f"  {c.cell_id:<28} {reason[:80]}")

    if status.running:
        lines.append("")
        lines.append("currently running:")
        for c in status.running:
            lines.append(f"  {c.cell_id:<28} since {c.updated_at}")

    if sweep_log and sweep_log.exists():
        entries = [ln for ln in sweep_log.read_text().splitlines() if ln.strip()]
        if entries:
            lines.append("")
            lines.append(f"last {min(tail, len(entries))} sweep log entries:")
            for ln in entries[-tail:]:
                lines.append(f"  {ln}")

    return "\n".join(lines)
