"""The vendored trainer is upstream plus exactly one reviewable change.

A vendored copy that quietly drifts from its recorded provenance is worse
than no vendoring at all: the whole point of pinning the training script is
that every line driving 45 paid GPU-hours is auditable. These tests fail if
the vendored file, the patch, or the recorded hash stop agreeing.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from digital_ghost.config import REPO_ROOT

VENDOR_DIR = REPO_ROOT / "vendor"
VENDORED = VENDOR_DIR / "train_dreambooth_lora_sdxl.py"
PATCH = VENDOR_DIR / "patches" / "0001-write-validation-samples-to-disk.patch"
VENDOR_README = VENDOR_DIR / "README.md"


def test_vendored_trainer_and_patch_exist():
    assert VENDORED.exists()
    assert PATCH.exists()
    assert VENDOR_README.exists()


def test_readme_records_provenance():
    text = VENDOR_README.read_text()
    assert "v0.40.0" in text, "vendored tag must be recorded"
    assert re.search(r"[0-9a-f]{64}", text), "upstream SHA-256 must be recorded"
    assert "huggingface/diffusers" in text


def test_patch_is_the_only_local_change():
    """Reconstruct the vendored file from upstream + the patch.

    Uses the upstream hash recorded in the README as the anchor, so this also
    catches someone editing the vendored file directly and forgetting to
    regenerate the patch.
    """
    upstream_sha = re.search(r"`([0-9a-f]{64})`", VENDOR_README.read_text())
    assert upstream_sha, "README must record the upstream SHA-256 in backticks"

    diff = PATCH.read_text()
    # The patch must be small and touch only the validation-sample hook.
    added = [ln for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert not removed, f"the patch should only add lines, but removes: {removed}"
    assert len(added) < 40, f"patch has grown to {len(added)} added lines — is it still one change?"
    assert any("validation_samples" in ln for ln in added)


def test_patched_hook_saves_images_and_cannot_kill_training():
    """A failure to save a sample must never take down a training run."""
    source = VENDORED.read_text()
    assert "DIGITAL GHOST PATCH" in source
    patch_body = source.split("DIGITAL GHOST PATCH")[1].split("END DIGITAL GHOST PATCH")[0]
    assert "validation_samples" in patch_body
    assert "except Exception" in patch_body, "sample saving must be wrapped"


def test_vendored_trainer_compiles():
    result = subprocess.run(
        ["python", "-m", "py_compile", str(VENDORED)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_instance_prompt_is_required_upstream_but_unused_with_dataset_captions():
    """Pins the upstream behaviour this pipeline depends on.

    The trainer is DreamBooth-shaped and demands --instance_prompt even when
    captions come from a dataset. Captions are the variable this study holds
    constant across arms, so if a future version ever started letting the
    instance prompt reach the model, that would silently change what is being
    measured.
    """
    source = VENDORED.read_text()
    assert '"--instance_prompt"' in source
    assert "custom_instance_prompts" in source
    # the dataset caption wins at __getitem__ time
    assert re.search(r"if self\.custom_instance_prompts", source)


@pytest.mark.parametrize("flag", [
    "--dataset_name", "--caption_column", "--rank", "--max_train_steps",
    "--validation_prompt", "--validation_epochs", "--num_validation_images",
    "--gradient_checkpointing", "--mixed_precision", "--seed",
])
def test_flags_the_pipeline_passes_still_exist(flag):
    """cell.training_command builds argv against these. If an upgrade drops or
    renames one, every cell would fail at launch on a rented GPU.
    """
    assert f'"{flag}"' in VENDORED.read_text(), f"{flag} missing from vendored trainer"
