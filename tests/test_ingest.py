from __future__ import annotations

import pytest

from digital_ghost.config import load_study_config
from digital_ghost.ingest.manifest import ManifestError, ingest_all_arms, load_manifest
from digital_ghost.ingest.provenance import ProvenanceError
from tests.conftest import populate_arm, write_study_yaml


def test_ingest_succeeds_and_writes_manifests(study_root):
    for arm in ("standard", "meme", "control"):
        populate_arm(study_root, arm, 5)
    path = write_study_yaml(study_root, doses=[3], pool_size_min=5)
    study = load_study_config(path)

    paths = ingest_all_arms(study, min_count=5)
    assert set(paths) == {"standard", "meme", "control"}
    entries = load_manifest(study, "meme")
    assert len(entries) == 5
    assert all(e.id.startswith("meme/") for e in entries)


def test_ingest_fails_loudly_on_missing_provenance(study_root):
    for arm in ("standard", "meme", "control"):
        populate_arm(study_root, arm, 5)
    sidecar = study_root / "data" / "raw" / "standard" / "img0.jpg.provenance.json"
    sidecar.unlink()

    path = write_study_yaml(study_root, doses=[3], pool_size_min=5)
    study = load_study_config(path)

    with pytest.raises(ProvenanceError, match="missing provenance sidecar"):
        ingest_all_arms(study, min_count=5)


def test_ingest_fails_on_undersized_pool(study_root):
    for arm in ("standard", "meme", "control"):
        populate_arm(study_root, arm, 3)
    path = write_study_yaml(study_root, doses=[3], pool_size_min=3)
    study = load_study_config(path)

    with pytest.raises(ManifestError, match="at least 10"):
        ingest_all_arms(study, min_count=10)


def test_ingest_rejects_malformed_provenance_json(study_root):
    for arm in ("standard", "meme", "control"):
        populate_arm(study_root, arm, 5)
    sidecar = study_root / "data" / "raw" / "meme" / "img1.jpg.provenance.json"
    sidecar.write_text("{not valid json")

    path = write_study_yaml(study_root, doses=[3], pool_size_min=5)
    study = load_study_config(path)

    with pytest.raises(ProvenanceError, match="malformed JSON"):
        ingest_all_arms(study, min_count=5)
