from __future__ import annotations

import pytest

from digital_ghost.config import load_study_config
from digital_ghost.generation.eval_prompts import (
    EvalPromptsExistError,
    eval_prompt_seeds,
    init_eval_prompts,
    load_eval_prompts,
)
from tests.conftest import write_study_yaml


def test_init_eval_prompts_writes_30_prompts(study_root):
    path = write_study_yaml(study_root, doses=[3], pool_size_min=3)
    study = load_study_config(path)

    out = init_eval_prompts(study)
    assert out.exists()
    data = load_eval_prompts(study)
    assert len(data["prompts"]) == 30
    tiers = {p["tier"] for p in data["prompts"]}
    assert tiers == {"near", "mid", "far"}


def test_init_eval_prompts_refuses_overwrite_without_force(study_root):
    path = write_study_yaml(study_root, doses=[3], pool_size_min=3)
    study = load_study_config(path)
    init_eval_prompts(study)
    with pytest.raises(EvalPromptsExistError):
        init_eval_prompts(study)
    init_eval_prompts(study, force=True)  # should not raise


def test_eval_prompt_seeds_deterministic_and_prompt_sensitive(study_root):
    path = write_study_yaml(study_root, doses=[3], pool_size_min=3, seeds_per_prompt=5)
    study = load_study_config(path)
    init_eval_prompts(study)

    a = eval_prompt_seeds(study, "near_01")
    b = eval_prompt_seeds(study, "near_01")
    c = eval_prompt_seeds(study, "near_02")
    assert a == b
    assert len(a) == 5
    assert a != c
