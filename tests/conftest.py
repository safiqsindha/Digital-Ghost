from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

import digital_ghost.config as config_module
from digital_ghost.config import StudyConfig, load_study_config
from digital_ghost.rating_app.backend.db import reset_engine

REAL_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def study_root(tmp_path, monkeypatch) -> Path:
    """A self-contained repo-root-shaped directory with real config files
    copied in (so schema/content stays in sync with what ships), suitable
    for StudyConfig.path()/raw_dir() resolution via a patched REPO_ROOT.
    """
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    for fn in ("training.yaml", "captioning.yaml", "provider.yaml", "eval_prompts_source.yaml", "rating_app.yaml"):
        shutil.copy(REAL_CONFIGS_DIR / fn, tmp_path / fn)
    reset_engine()
    yield tmp_path
    reset_engine()


def write_study_yaml(
    study_root: Path,
    *,
    doses: list[int],
    pool_size_min: int,
    seeds_per_cell: int = 1,
    seeds_per_prompt: int = 2,
    budget_cap_usd: float = 500.0,
) -> Path:
    data = yaml.safe_load((REAL_CONFIGS_DIR / "study.yaml").read_text())
    data["doses"] = doses
    data["pool_size_min"] = pool_size_min
    data["seeds_per_cell"] = seeds_per_cell
    data["eval"]["seeds_per_prompt"] = seeds_per_prompt
    data["budget"]["cap_usd"] = budget_cap_usd
    path = study_root / "configs.yaml"
    path.write_text(yaml.dump(data))
    return path


def populate_arm(study_root: Path, arm: str, n_images: int, *, distinct: bool = True) -> None:
    d = study_root / "data" / "raw" / arm
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        color = (i * 7 % 256, i * 31 % 256, i * 53 % 256) if distinct else (10, 10, 10)
        Image.new("RGB", (8, 8), color=color).save(d / f"img{i}.jpg")
        prov = {
            "source_url": f"https://example.com/{arm}/{i}",
            "date": "2024-01-01",
            "platform": "example",
            "tool": "unknown" if arm == "meme" else None,
        }
        (d / f"img{i}.jpg.provenance.json").write_text(json.dumps(prov))


@pytest.fixture
def tiny_study(study_root) -> StudyConfig:
    """3 arms x 5 images each, doses=[3, 5], 1 seed per cell — small but
    exercises adjacent-dose logic (2+ doses) unlike a single-dose fixture.
    """
    for arm in ("standard", "meme", "control"):
        populate_arm(study_root, arm, 5)
    path = write_study_yaml(study_root, doses=[3, 5], pool_size_min=5)
    return load_study_config(path)


@pytest.fixture
def trained_and_generated_study(tiny_study):
    """tiny_study, fully ingested/captioned/trained/generated in dry-run mode."""
    from digital_ghost.caption.generate import caption_all_arms
    from digital_ghost.config import load_captioning_config, load_provider_config, load_training_config
    from digital_ghost.generation.eval_prompts import init_eval_prompts
    from digital_ghost.generation.generate_grid import run_generation_grid
    from digital_ghost.ingest.manifest import ingest_all_arms
    from digital_ghost.training.orchestrator import run_sweep

    study = tiny_study
    training = load_training_config(study)
    captioning = load_captioning_config(study)
    provider_cfg = load_provider_config(study)
    provider_cfg.max_parallel_gpus = 2

    ingest_all_arms(study, min_count=5)
    caption_all_arms(study, captioning)
    init_eval_prompts(study)

    train_report = run_sweep(study, training, provider_cfg, dry_run=True, resume=True)
    assert not train_report.failed, train_report.failed

    gen_report = run_generation_grid(study, training, provider_cfg, dry_run=True, resume=True)
    assert not gen_report.failed, gen_report.failed

    return study
