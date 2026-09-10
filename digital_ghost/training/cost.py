"""Append-only cost ledger + budget enforcement.

Every GPU-consuming step (a training cell, a generation batch) records its
spend here before doing anything else with it. The sweep aborts the moment
projected cumulative spend would exceed `budget.cap_usd` from study.yaml —
never after the fact.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from digital_ghost.config import StudyConfig


def study_cost_ledger_path(study: StudyConfig) -> Path:
    """Single shared ledger for the whole study — training and generation
    spend both count against the same `budget.cap_usd`.
    """
    return study.path("outputs_dir") / "cost_ledger.jsonl"


class BudgetExceededError(Exception):
    pass


@dataclass
class CostEntry:
    timestamp: str
    cell_id: str
    gpu_hours: float
    cost_usd: float
    cumulative_cost_usd: float


class CostLedger:
    """Thread-safe, file-backed. Safe to share across parallel workers in
    the same process; each process appending to the same path should hold
    an external lock (the orchestrator serializes budget checks — see
    training/orchestrator.py) since plain file append isn't atomic across
    processes.
    """

    def __init__(self, path: Path, cap_usd: float, hard_stop: bool = True):
        self.path = path
        self.cap_usd = cap_usd
        self.hard_stop = hard_stop
        # Reentrant because record() calls spent() while already holding it.
        self._lock = threading.RLock()
        # Cost of jobs that have been launched but haven't reported actuals
        # yet. Without this, N parallel workers all see spent()==0, all pass
        # the cap check, and all launch — overshooting the cap by roughly a
        # factor of N. Reserved cost counts against the cap until replaced by
        # the real figure.
        self._reserved_usd = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def spent(self) -> float:
        with self._lock:
            total = 0.0
            if self.path.exists():
                with open(self.path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            total += json.loads(line)["cost_usd"]
            return total

    def committed(self) -> float:
        """Recorded spend plus cost reserved for in-flight jobs."""
        with self._lock:
            return self.spent() + self._reserved_usd

    def reserve(self, estimated_usd: float) -> float:
        """Claim budget for a job about to be launched.

        Raises BudgetExceededError (when hard_stop) if launching would put
        committed spend past the cap. Callers MUST call `release` with the
        same amount once the job's real cost is known.
        """
        with self._lock:
            projected = self.committed() + estimated_usd
            if projected > self.cap_usd and self.hard_stop:
                raise BudgetExceededError(
                    f"budget cap would be exceeded: committed ${self.committed():.2f} + "
                    f"estimated ${estimated_usd:.2f} = ${projected:.2f} > cap ${self.cap_usd:.2f}"
                )
            self._reserved_usd += estimated_usd
            return estimated_usd

    def release(self, estimated_usd: float) -> None:
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - estimated_usd)

    def estimate_job_cost(self, price_per_gpu_hour: float, fallback_hours: float) -> float:
        """Best available estimate of what the next job will cost.

        Uses the mean of observed actuals once there are any, but never less
        than the configured fallback — under-reserving is what lets the cap
        be breached, so bias high.
        """
        entries = self.all_entries()
        if entries:
            observed_mean_hours = sum(e.gpu_hours for e in entries) / len(entries)
            hours = max(observed_mean_hours, fallback_hours)
        else:
            hours = fallback_hours
        return hours * price_per_gpu_hour

    def check_budget(self, projected_additional_usd: float = 0.0) -> None:
        """Pre-flight guard: call BEFORE starting new work. Raises if spend
        already recorded (optionally plus a projected increment) is at or
        past the cap, so a new subprocess never launches once over budget.
        """
        projected = self.spent() + projected_additional_usd
        if projected > self.cap_usd:
            msg = (
                f"budget cap exceeded: spent so far ${self.spent():.2f} + "
                f"projected ${projected_additional_usd:.2f} = ${projected:.2f} "
                f"> cap ${self.cap_usd:.2f}"
            )
            if self.hard_stop:
                raise BudgetExceededError(msg)

    def is_over_budget(self) -> bool:
        return self.spent() > self.cap_usd

    def record(self, cell_id: str, gpu_hours: float, price_per_gpu_hour: float) -> CostEntry:
        """Record cost that has ALREADY been incurred (the job already ran).

        Never raises: work that already happened is sunk cost and must be
        recorded honestly even if it pushes spend over the cap. Callers
        should check `is_over_budget()` after recording and stop scheduling
        new cells if it's true — see training/orchestrator.py.
        """
        cost_usd = gpu_hours * price_per_gpu_hour
        with self._lock:
            entry = CostEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cell_id=cell_id,
                gpu_hours=gpu_hours,
                cost_usd=cost_usd,
                cumulative_cost_usd=self.spent() + cost_usd,
            )
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def all_entries(self) -> list[CostEntry]:
        with self._lock:
            entries = []
            if self.path.exists():
                with open(self.path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(CostEntry(**json.loads(line)))
            return entries
