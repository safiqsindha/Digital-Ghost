"""End-to-end exercise of the REAL (non-dry-run) training and generation path.

These drive the vendored diffusers trainer and the real SDXL generation
pipeline against a tiny randomly-initialised stand-in model
(`hf-internal-testing/tiny-stable-diffusion-xl-pipe`, a few MB) instead of the
real 7GB SDXL. That makes the torch path testable on CPU in CI rather than
only discoverable on a rented GPU with real data.

What these prove:
  - the argv `cell.training_command()` builds actually drives the vendored
    trainer to completion
  - the imagefolder dataset it materializes is one the trainer accepts, with
    per-image captions rather than the DreamBooth instance prompt
  - gradients actually reach the LoRA weights (an all-zero adapter would mean
    the cell trained nothing while reporting success)
  - the patched validation hook writes mid-training samples to disk
  - the checkpoint round-trips: what training writes, generation can load
  - generation is reproducible at a fixed seed, and a loaded LoRA changes the
    output

What they cannot prove: that the frozen hyperparameters produce a usable
likeness at 1024px on a real GPU. That needs real data and real hardware,
which is what `digital-ghost smoke-cell` is for.

Marked `slow`: deselect with `-m "not slow"`. Skipped when the tiny model
can't be fetched (offline CI).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")
pytest.importorskip("datasets")

TINY_SDXL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
TINY_RESOLUTION = 64

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tiny_model() -> str:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(TINY_SDXL)
    except Exception as e:  # noqa: BLE001 - offline CI is a skip, not a failure
        pytest.skip(f"cannot fetch {TINY_SDXL}: {e}")
    return TINY_SDXL


@pytest.fixture(scope="module")
def trained_cell(tmp_path_factory, tiny_model):
    """Train one real cell through the same code path the sweep uses."""
    import yaml
    from PIL import Image

    import digital_ghost.config as config_module
    from digital_ghost.caption.generate import caption_all_arms
    from digital_ghost.config import load_captioning_config, load_study_config, load_training_config
    from digital_ghost.ingest.manifest import ingest_all_arms
    from digital_ghost.training.cell import build_cell_spec, training_command

    root = tmp_path_factory.mktemp("real_training")
    real_configs = config_module.REPO_ROOT / "configs"
    for fn in ("captioning.yaml", "eval_prompts_source.yaml", "rating_app.yaml", "runtime.yaml"):
        (root / fn).write_text((real_configs / fn).read_text())

    training_data = yaml.safe_load((real_configs / "training.yaml").read_text())
    training_data.update(
        base_model=tiny_model, resolution=TINY_RESOLUTION, max_train_steps=2,
        gradient_accumulation_steps=1, mixed_precision="no", gradient_checkpointing=False,
    )
    training_data["lora"]["rank"] = 4
    training_data["validation"]["num_images"] = 1
    (root / "training.yaml").write_text(yaml.dump(training_data))

    study_data = yaml.safe_load((real_configs / "study.yaml").read_text())
    study_data.update(doses=[3], pool_size_min=3, seeds_per_cell=1)
    study_data["eval"]["seeds_per_prompt"] = 1
    (root / "study.yaml").write_text(yaml.dump(study_data))

    for arm in ("standard", "meme", "control"):
        d = root / "data" / "raw" / arm
        d.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (128, 128), color=(i * 70, 40, 90)).save(d / f"img{i}.png")
            (d / f"img{i}.png.provenance.json").write_text(json.dumps({
                "source_url": f"https://example.invalid/{arm}/{i}",
                "date": "2024-01-01", "platform": "test",
            }))

    original_root = config_module.REPO_ROOT
    config_module.REPO_ROOT = root
    try:
        study = load_study_config(root / "study.yaml")
        training = load_training_config(study)
        ingest_all_arms(study)
        caption_all_arms(study, load_captioning_config(study))

        cell = build_cell_spec(study, "standard", 3, 1)
        cmd = training_command(cell, study, training, dry_run=False)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        assert result.returncode == 0, f"vendored trainer failed:\n{result.stderr[-4000:]}"
        yield cell
    finally:
        config_module.REPO_ROOT = original_root


def test_dataset_uses_per_image_captions_not_instance_prompt(trained_cell):
    """Captions are the variable this study holds constant across arms.

    The vendored trainer requires --instance_prompt even when captions come
    from the dataset, so this pins the behaviour that the dataset wins.
    """
    from digital_ghost.training.cell import UNUSED_INSTANCE_PROMPT

    metadata = trained_cell.dataset_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert all(row["text"] for row in rows)
    assert all(row["text"] != UNUSED_INSTANCE_PROMPT for row in rows)
    assert all((trained_cell.dataset_dir / row["file_name"]).exists() for row in rows)


def test_real_training_produces_a_checkpoint(trained_cell):
    assert trained_cell.checkpoint_path.exists()
    assert trained_cell.checkpoint_path.stat().st_size > 1000


def test_gradients_actually_reach_the_lora_weights(trained_cell):
    """LoRA up-matrices initialise to exactly zero.

    If the optimizer were updating the wrong parameter set, training would
    "succeed" and emit a checkpoint that is a mathematical no-op — every arm
    would look identical to baseline and the study would measure nothing.
    """
    from safetensors.torch import load_file

    state = load_file(str(trained_cell.checkpoint_path))
    assert state, "checkpoint has no tensors"

    up_weights = {k: v for k, v in state.items() if "lora.up" in k or "lora_B" in k}
    assert up_weights, f"no LoRA up/B matrices found in {sorted(state)[:5]}"
    for name, tensor in up_weights.items():
        assert tensor.abs().sum().item() > 0, f"{name} is still all zero — no gradient reached it"


def test_patched_trainer_writes_validation_samples_to_disk(trained_cell):
    """Upstream only sends these to tensorboard/wandb. The vendored patch
    writes them to disk, which is how a collapsing LoRA becomes visible during
    an unattended run with no tracker configured.
    """
    samples = sorted(trained_cell.validation_dir.glob("*.png"))
    assert samples, f"no validation samples in {trained_cell.validation_dir}"
    assert all(p.stat().st_size > 0 for p in samples)


def test_sanity_checks_pass_on_a_real_checkpoint(trained_cell):
    from digital_ghost.training.sanity import check_checkpoint

    report = check_checkpoint(trained_cell.id, trained_cell.checkpoint_path, min_bytes=1000)
    assert report.passed, report.failures


def _generate(out_dir, tiny_model, prompts_path, lora_path=None):
    cmd = [
        sys.executable, "-m", "digital_ghost.generation.generate_images",
        "--checkpoint-label", "test",
        "--base-model", tiny_model,
        "--resolution", str(TINY_RESOLUTION),
        "--output-dir", str(out_dir),
        "--eval-prompts", str(prompts_path),
        "--num-inference-steps", "2",
    ]
    if lora_path:
        cmd += ["--lora-path", str(lora_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    assert result.returncode == 0, f"real generation failed:\n{result.stderr[-4000:]}"
    return out_dir


@pytest.fixture(scope="module")
def prompts_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("prompts") / "prompts.jsonl"
    path.write_text(json.dumps({
        "prompt_id": "near_01", "prompt_text": "a man speaking at a podium",
        "tier": "near", "gen_seed": 12345,
    }) + "\n")
    return path


def test_checkpoint_round_trips_into_generation_and_is_reproducible(
    tmp_path, tiny_model, trained_cell, prompts_file
):
    """The format the trainer writes must be the format generation reads, and
    a fixed seed must reproduce exactly — the study's design assumes it.
    """
    import numpy as np
    from PIL import Image

    lora = trained_cell.checkpoint_path
    run1 = _generate(tmp_path / "run1", tiny_model, prompts_file, lora)
    run2 = _generate(tmp_path / "run2", tiny_model, prompts_file, lora)

    img1 = np.asarray(Image.open(run1 / "near_01" / "12345.png"))
    img2 = np.asarray(Image.open(run2 / "near_01" / "12345.png"))
    assert np.array_equal(img1, img2), "same seed + same checkpoint must reproduce exactly"


def test_loaded_lora_actually_changes_the_output(tmp_path, tiny_model, trained_cell, prompts_file):
    """If load_lora_weights silently no-opped, every arm would render
    identically to baseline and the curve would be flat for reasons that have
    nothing to do with the research question.
    """
    import numpy as np
    from PIL import Image

    with_lora = _generate(tmp_path / "with", tiny_model, prompts_file, trained_cell.checkpoint_path)
    baseline = _generate(tmp_path / "without", tiny_model, prompts_file, None)

    a = np.asarray(Image.open(with_lora / "near_01" / "12345.png")).astype(int)
    b = np.asarray(Image.open(baseline / "near_01" / "12345.png")).astype(int)
    assert not np.array_equal(a, b), "LoRA had no effect on generation — it was not applied"
