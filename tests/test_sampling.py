from __future__ import annotations

import pytest

from digital_ghost.config import load_study_config
from digital_ghost.sampling.subsample import cell_grid, subsample_ids


def test_subsample_is_deterministic_across_calls():
    study = load_study_config()
    pool = sorted(f"img_{i:04d}" for i in range(200))
    a = subsample_ids(study, "meme", 50, 1, pool)
    b = subsample_ids(study, "meme", 50, 1, pool)
    assert a == b
    assert len(a) == 50


def test_different_seed_index_gives_different_subsample():
    study = load_study_config()
    pool = sorted(f"img_{i:04d}" for i in range(200))
    a = subsample_ids(study, "meme", 50, 1, pool)
    c = subsample_ids(study, "meme", 50, 2, pool)
    assert a != c


def test_dose_exceeding_pool_raises():
    study = load_study_config()
    pool = sorted(f"img_{i:04d}" for i in range(10))
    with pytest.raises(ValueError, match="exceeds pool size"):
        subsample_ids(study, "meme", 50, 1, pool)


def test_unsorted_pool_raises():
    study = load_study_config()
    pool = [f"img_{i:04d}" for i in range(10)][::-1]
    with pytest.raises(ValueError, match="pre-sorted"):
        subsample_ids(study, "meme", 5, 1, pool)


def test_cell_grid_size_matches_n_cells():
    study = load_study_config()
    assert len(cell_grid(study)) == study.n_cells
    assert len(set(cell_grid(study))) == study.n_cells  # all unique
