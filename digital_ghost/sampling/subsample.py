"""Deterministic, seeded dose subsampling.

Given an arm's validated image pool, a dose, and a seed index, this always
returns the same subset of image ids — independent of process, machine, or
run order. This is what lets the sweep be reproduced from the config alone.
"""

from __future__ import annotations

import numpy as np

from digital_ghost.config import StudyConfig


def subsample_ids(
    study: StudyConfig,
    arm: str,
    dose: int,
    seed_index: int,
    pool_ids: list[str],
) -> list[str]:
    """Deterministically select `dose` ids from `pool_ids` for one sweep cell.

    `pool_ids` must be pre-sorted by the caller (manifest ids, sorted
    lexicographically) so that the same pool always presents in the same
    order regardless of filesystem iteration order.
    """
    if dose > len(pool_ids):
        raise ValueError(
            f"dose {dose} exceeds pool size {len(pool_ids)} for arm '{arm}'"
        )
    if list(pool_ids) != sorted(pool_ids):
        raise ValueError("pool_ids must be pre-sorted for deterministic sampling")

    rng = np.random.default_rng(study.cell_seed(arm, dose, seed_index))
    chosen_idx = rng.choice(len(pool_ids), size=dose, replace=False)
    return [pool_ids[i] for i in sorted(chosen_idx.tolist())]


def cell_grid(study: StudyConfig) -> list[tuple[str, int, int]]:
    """All (arm, dose, seed_index) triples in the sweep, in a stable order."""
    cells = []
    for arm in study.arms:
        for dose in study.doses:
            for seed_index in range(1, study.seeds_per_cell + 1):
                cells.append((arm.name, dose, seed_index))
    return cells
