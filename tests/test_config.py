from __future__ import annotations

import pytest
from pydantic import ValidationError

from digital_ghost.config import (
    StudyConfig,
    load_captioning_config,
    load_eval_prompts_source,
    load_runtime_config,
    load_rating_app_config,
    load_study_config,
    load_training_config,
)
from tests.conftest import write_study_yaml


def test_real_configs_load_and_cross_reference():
    study = load_study_config()
    assert study.n_cells == len(study.arms) * len(study.doses) * study.seeds_per_cell
    assert study.n_cells == 45

    training = load_training_config(study)
    assert training.lora.rank > 0

    captioning = load_captioning_config(study)
    assert len(captioning.templates) >= 1

    runtime = load_runtime_config(study)
    assert runtime.pricing.usd_per_gpu_hour > 0
    assert runtime.execution.parallel_cells >= 1

    rating = load_rating_app_config(study)
    assert set(rating.rating_options) == {"A", "B", "BOTH", "NEITHER"}

    eval_source = load_eval_prompts_source(study)
    by_tier = eval_source.by_tier()
    assert {t: len(v) for t, v in by_tier.items()} == study.eval.tiers


def test_doses_must_fit_within_pool_size_min(study_root):
    path = write_study_yaml(study_root, doses=[10, 200], pool_size_min=50)
    with pytest.raises(ValidationError, match="exceeds pool_size_min"):
        load_study_config(path)


def test_cell_seed_is_deterministic_and_seed_index_sensitive():
    study = load_study_config()
    a = study.cell_seed("meme", 50, 1)
    b = study.cell_seed("meme", 50, 1)
    c = study.cell_seed("meme", 50, 2)
    assert a == b
    assert a != c


def test_raw_dir_and_paths_resolve_under_repo_root(study_root):
    from digital_ghost.config import REPO_ROOT

    path = write_study_yaml(study_root, doses=[3], pool_size_min=3)
    study = load_study_config(path)
    assert study.raw_dir("meme") == REPO_ROOT / "data" / "raw" / "meme"
    assert study.path("manifest_dir") == REPO_ROOT / "data" / "manifest"


def test_arms_must_be_exactly_the_three_named_arms():
    from digital_ghost.config import ArmConfig

    with pytest.raises(ValidationError):
        StudyConfig(
            study_name="x",
            seed_root=1,
            arms=[ArmConfig(name="standard", raw_dir="a"), ArmConfig(name="meme", raw_dir="b")],
            pool_size_min=10,
            doses=[5],
            seeds_per_cell=1,
            training_config="training.yaml",
            captioning_config="captioning.yaml",
            runtime_config="runtime.yaml",
            rating_app_config="rating_app.yaml",
            eval={"prompts_source": "a", "prompts_file": "b", "seeds_per_prompt": 1, "tiers": {"near": 1, "mid": 1, "far": 1}},
            budget={"cap_usd": 10},
            paths={
                "manifest_dir": "a", "captions_dir": "b", "outputs_dir": "c",
                "runs_dir": "d", "generations_dir": "e", "ratings_dir": "f", "analysis_dir": "g",
            },
            dry_run={"n_images": 1, "n_prompts": 1},
        )
