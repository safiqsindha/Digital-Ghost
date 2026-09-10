from __future__ import annotations

import pytest

from digital_ghost.caption.generate import CaptionSchemeError, caption_all_arms, check_templates_clean
from digital_ghost.config import CaptioningConfig, load_captioning_config, load_study_config
from digital_ghost.ingest.manifest import ingest_all_arms
from tests.conftest import populate_arm, write_study_yaml


def test_real_captioning_templates_are_clean():
    study = load_study_config()
    captioning = load_captioning_config(study)
    check_templates_clean(captioning)  # must not raise


def test_check_templates_clean_rejects_banned_term():
    bad = CaptioningConfig(
        scheme="test", templates=["a photo of charlie kirk"], banned_terms=["charlie"]
    )
    with pytest.raises(CaptionSchemeError, match="banned term"):
        check_templates_clean(bad)


def test_caption_assignment_is_deterministic_and_arm_agnostic(study_root):
    for arm in ("standard", "meme", "control"):
        populate_arm(study_root, arm, 5)
    path = write_study_yaml(study_root, doses=[3], pool_size_min=5)
    study = load_study_config(path)
    captioning = load_captioning_config(study)

    ingest_all_arms(study, min_count=5)
    first = caption_all_arms(study, captioning)
    second = caption_all_arms(study, captioning)  # overwrite, should be identical
    assert first.keys() == second.keys()

    from digital_ghost.caption.generate import load_captions

    for arm in ("standard", "meme", "control"):
        pairs = load_captions(study, arm)
        assert len(pairs) == 5
        assert all(p.caption in captioning.templates for p in pairs)
