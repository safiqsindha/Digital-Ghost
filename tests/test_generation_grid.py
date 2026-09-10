from __future__ import annotations

import json

import pytest

from digital_ghost.generation.generate_grid import all_checkpoints, checkpoint_manifest_path, run_generation_grid


def test_generation_grid_completes_and_resumes(trained_and_generated_study):
    study = trained_and_generated_study
    checkpoints = all_checkpoints(study)
    assert len(checkpoints) == study.n_cells + 1  # + baseline

    # trained_and_generated_study runs generation with dry_run=True, which
    # limits prompts to study.dry_run.n_prompts (not the full frozen 30) —
    # see generate_grid.write_flat_prompts.
    expected_rows = study.eval.seeds_per_prompt * study.dry_run.n_prompts

    for cp in checkpoints:
        manifest = checkpoint_manifest_path(study, cp.label)
        assert manifest.exists()
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert len(rows) == expected_rows
        for row in rows:
            assert row["checkpoint_label"] == cp.label
            for f in ("prompt_id", "tier", "gen_seed", "image_path"):
                assert f in row


def test_generation_requires_training_complete_first(tiny_study):
    from digital_ghost.caption.generate import caption_all_arms
    from digital_ghost.config import load_captioning_config, load_runtime_config, load_training_config
    from digital_ghost.generation.eval_prompts import init_eval_prompts
    from digital_ghost.ingest.manifest import ingest_all_arms

    study = tiny_study
    ingest_all_arms(study, min_count=5)
    caption_all_arms(study, load_captioning_config(study))
    init_eval_prompts(study)

    training = load_training_config(study)
    runtime = load_runtime_config(study)

    with pytest.raises(RuntimeError, match="not complete yet"):
        run_generation_grid(study, training, runtime, dry_run=True, require_checkpoints_trained=True)
