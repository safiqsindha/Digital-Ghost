from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from digital_ghost.training.sanity import (
    SanityThresholds,
    check_checkpoint,
    check_generated_images,
)


def _noise_image(path: Path, seed: int, size: int = 128) -> Path:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def _good_set(tmp_path: Path, n: int = 6) -> list[Path]:
    return [_noise_image(tmp_path / f"good_{i}.png", seed=i) for i in range(n)]


def test_good_set_passes(tmp_path):
    paths = _good_set(tmp_path)
    report = check_generated_images("cell_a", paths, expected_count=len(paths))
    assert report.passed, report.summary()
    assert report.failures == []
    assert report.checked_n_images == len(paths)
    assert "PASS" in report.summary()
    assert "\n" not in report.summary()


def test_all_black_set_fails(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f"black_{i}.png"
        Image.new("RGB", (128, 128), (0, 0, 0)).save(path)
        paths.append(path)
    # Solid-colour PNGs compress to almost nothing, so drop the size floor to
    # isolate the luminance and flatness checks.
    thresholds = SanityThresholds(min_image_bytes=0, max_identical_fraction=1.0)
    report = check_generated_images("cell_black", paths, len(paths), thresholds)
    assert not report.passed
    assert any("mean luminance" in f for f in report.failures)
    assert any("pixel std" in f for f in report.failures)


def test_blown_out_set_fails(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f"white_{i}.png"
        Image.new("RGB", (128, 128), (255, 255, 255)).save(path)
        paths.append(path)
    thresholds = SanityThresholds(min_image_bytes=0, max_identical_fraction=1.0)
    report = check_generated_images("cell_white", paths, len(paths), thresholds)
    assert not report.passed
    assert any("above mean luminance" in f for f in report.failures)


def test_all_identical_set_fails(tmp_path):
    source = _noise_image(tmp_path / "source.png", seed=7)
    payload = source.read_bytes()
    paths = []
    for i in range(5):
        path = tmp_path / f"dup_{i}.png"
        path.write_bytes(payload)
        paths.append(path)
    report = check_generated_images("cell_dupes", paths, len(paths))
    assert not report.passed
    assert any("byte-identical" in f for f in report.failures)


def test_wrong_count_fails(tmp_path):
    paths = _good_set(tmp_path, n=3)
    report = check_generated_images("cell_count", paths, expected_count=6)
    assert not report.passed
    assert any("expected 6 images, got 3" in f for f in report.failures)


def test_missing_file_is_a_finding_not_a_crash(tmp_path):
    paths = _good_set(tmp_path, n=2) + [tmp_path / "never_written.png"]
    report = check_generated_images("cell_missing", paths, expected_count=3)
    assert not report.passed
    assert any("missing/unreadable" in f for f in report.failures)


def test_truncated_file_is_a_finding_not_a_crash(tmp_path):
    good = _good_set(tmp_path, n=2)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(_noise_image(tmp_path / "src.png", seed=99).read_bytes()[:6000])
    report = check_generated_images("cell_truncated", good + [truncated], expected_count=3)
    assert not report.passed
    assert any("corrupt image" in f for f in report.failures)


def test_undersized_file_fails(tmp_path):
    good = _good_set(tmp_path, n=2)
    stub = tmp_path / "stub.png"
    stub.write_bytes(b"\x89PNG\r\n\x1a\n")
    report = check_generated_images("cell_stub", good + [stub], expected_count=3)
    assert not report.passed
    assert any("under 1024 bytes" in f for f in report.failures)


def test_all_failures_accumulate(tmp_path):
    rng = np.random.default_rng(3)
    near_black = tmp_path / "near_black.png"
    Image.fromarray(rng.integers(0, 3, size=(256, 256, 3), dtype=np.uint8)).save(near_black)
    stub = tmp_path / "stub.png"
    stub.write_bytes(b"junk")
    report = check_generated_images(
        "cell_many", [near_black, stub, tmp_path / "gone.png"], expected_count=10
    )
    assert not report.passed
    assert len(report.failures) >= 4
    assert "FAIL" in report.summary()


def _lora_state(up_value: float) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "unet.attn.lora_A.weight": torch.randn(64, 256),
        "unet.attn.lora_B.weight": torch.full((256, 64), up_value),
    }


def test_healthy_checkpoint_passes(tmp_path):
    path = tmp_path / "lora_weights.safetensors"
    state = _lora_state(up_value=0.0)
    state["unet.attn.lora_B.weight"] = torch.randn(256, 64)
    save_file(state, str(path))
    report = check_checkpoint("cell_ok", path)
    assert report.passed, report.summary()


def test_all_zero_up_matrices_flagged_as_untrained(tmp_path):
    path = tmp_path / "lora_weights.safetensors"
    save_file(_lora_state(up_value=0.0), str(path))
    report = check_checkpoint("cell_untrained", path)
    assert not report.passed
    assert any("trained nothing" in f for f in report.failures)


def test_missing_checkpoint_fails(tmp_path):
    report = check_checkpoint("cell_gone", tmp_path / "nope.safetensors")
    assert not report.passed
    assert any("missing" in f for f in report.failures)
    assert report.checked_n_images == 0


def test_undersized_checkpoint_fails(tmp_path):
    path = tmp_path / "tiny.safetensors"
    save_file({"unet.attn.lora_B.weight": torch.randn(2, 2)}, str(path))
    report = check_checkpoint("cell_tiny", path, min_bytes=10_000_000)
    assert not report.passed
    assert any("byte floor" in f for f in report.failures)


def test_corrupt_checkpoint_is_a_finding_not_a_crash(tmp_path):
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(b"\x00" * 8192)
    report = check_checkpoint("cell_corrupt", path)
    assert not report.passed
    assert any("failed to load" in f for f in report.failures)
