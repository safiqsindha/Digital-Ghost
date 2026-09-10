"""End-to-end exercise of the REAL (non-dry-run) training and generation code.

These run the actual diffusers/peft SDXL LoRA loop and the actual SDXL
generation pipeline, against a tiny randomly-initialised stand-in model
(`hf-internal-testing/tiny-stable-diffusion-xl-pipe`, a few MB) instead of
the real 7GB SDXL. That makes the torch code path testable on CPU in CI
rather than only discoverable on a rented GPU with real data.

What these prove:
  - the training loop runs, and gradients actually reach the LoRA weights
    (a silently no-op checkpoint would make the whole study measure nothing)
  - the checkpoint round-trips: what training writes, generation can load
  - generation is reproducible at a fixed seed
  - a loaded LoRA actually changes the output

What they cannot prove: that the frozen hyperparameters produce a usable
likeness at 1024px on a real GPU. That needs real data and real hardware.

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
pytest.importorskip("torchvision")

TINY_SDXL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
# Tiny stand-in model; 64px keeps the CPU run to seconds.
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
def trained_lora(tmp_path_factory, tiny_model):
    from PIL import Image

    work = tmp_path_factory.mktemp("real_training")
    dataset = work / "dataset.jsonl"
    with open(dataset, "w") as f:
        for i in range(3):
            img = work / f"img{i}.png"
            Image.new("RGB", (128, 128), color=(i * 60, 40, 90)).save(img)
            f.write(
                json.dumps(
                    {"id": f"arm/{i}", "image_path": str(img), "caption": "a man at a podium"}
                )
                + "\n"
            )

    out_dir = work / "out"
    cmd = [
        sys.executable, "-m", "digital_ghost.training.train_lora_sdxl",
        "--dataset", str(dataset),
        "--output-dir", str(out_dir),
        "--base-model", tiny_model,
        "--resolution", str(TINY_RESOLUTION),
        "--batch-size", "1",
        "--gradient-accumulation-steps", "2",
        "--max-train-steps", "2",
        "--lr", "1e-4",
        "--lr-scheduler", "constant",
        "--lr-warmup-steps", "0",
        "--weight-decay", "0.01",
        "--lora-rank", "4",
        "--lora-alpha", "4",
        "--lora-dropout", "0.0",
        "--lora-target-modules", "to_k,to_q,to_v,to_out.0",
        "--mixed-precision", "no",  # fp16 is a GPU concern; CPU runs fp32
        "--seed", "42",
        "--data-seed", "123",
        "--checkpointing-steps", "500",
        "--caption-dropout-rate", "0.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, f"real training failed:\n{result.stderr[-4000:]}"
    return out_dir


def test_real_training_produces_a_checkpoint_and_log(trained_lora):
    assert (trained_lora / "lora_weights.safetensors").exists()
    log = json.loads((trained_lora / "training_log.json").read_text())
    assert log["dry_run"] is False
    assert log["n_examples"] == 3
    assert len(log["losses"]) == 2
    assert all(loss > 0 for loss in log["losses"])


def test_gradients_actually_reach_the_lora_weights(trained_lora):
    """LoRA `up` matrices initialise to exactly zero. If the optimizer were
    updating the wrong parameter set, training would 'succeed' and emit a
    checkpoint that is a mathematical no-op — every arm would look identical
    to baseline and the study would measure nothing.
    """
    from safetensors.torch import load_file

    state = load_file(str(trained_lora / "lora_weights.safetensors"))
    assert state, "checkpoint has no tensors"

    up_weights = {k: v for k, v in state.items() if "lora.up" in k or "lora_B" in k}
    assert up_weights, f"no LoRA up/B matrices found in {sorted(state)[:5]}"
    for name, tensor in up_weights.items():
        assert tensor.abs().sum().item() > 0, f"{name} is still all zero — no gradient reached it"


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, f"real generation failed:\n{result.stderr[-4000:]}"
    return out_dir


@pytest.fixture(scope="module")
def prompts_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("prompts") / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt_id": "near_01",
                "prompt_text": "a man speaking at a podium",
                "tier": "near",
                "gen_seed": 12345,
            }
        )
        + "\n"
    )
    return path


def test_generation_loads_the_trained_checkpoint_and_is_reproducible(
    tmp_path, tiny_model, trained_lora, prompts_file
):
    """Covers the training->generation handoff: the format save_lora_weights
    writes must be the format load_lora_weights reads, and a fixed seed must
    give byte-identical output (the study's whole design assumes it).
    """
    import numpy as np
    from PIL import Image

    lora = trained_lora / "lora_weights.safetensors"
    run1 = _generate(tmp_path / "run1", tiny_model, prompts_file, lora)
    run2 = _generate(tmp_path / "run2", tiny_model, prompts_file, lora)

    img1 = np.asarray(Image.open(run1 / "near_01" / "12345.png"))
    img2 = np.asarray(Image.open(run2 / "near_01" / "12345.png"))
    assert np.array_equal(img1, img2), "same seed + same checkpoint must reproduce exactly"

    manifest = [json.loads(line) for line in (run1 / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 1
    assert manifest[0]["prompt_id"] == "near_01"
    assert manifest[0]["gen_seed"] == 12345
    assert manifest[0]["tier"] == "near"


def test_loaded_lora_actually_changes_the_output(tmp_path, tiny_model, trained_lora, prompts_file):
    """If load_lora_weights silently no-opped, every arm would render
    identically to baseline and the dose-response curve would be flat for
    reasons that have nothing to do with the research question.
    """
    import numpy as np
    from PIL import Image

    with_lora = _generate(
        tmp_path / "with", tiny_model, prompts_file, trained_lora / "lora_weights.safetensors"
    )
    baseline = _generate(tmp_path / "without", tiny_model, prompts_file, None)

    a = np.asarray(Image.open(with_lora / "near_01" / "12345.png")).astype(int)
    b = np.asarray(Image.open(baseline / "near_01" / "12345.png")).astype(int)
    assert not np.array_equal(a, b), "LoRA had no effect on generation — it was not applied"


def test_generation_manifest_is_written_atomically(tmp_path, tiny_model, prompts_file):
    """Resume decides a checkpoint is finished by reading this manifest, so a
    killed process must never leave a readable-but-partial one behind.
    """
    out = _generate(tmp_path / "atomic", tiny_model, prompts_file, None)
    assert (out / "manifest.jsonl").exists()
    assert not (out / "manifest.jsonl.tmp").exists()
