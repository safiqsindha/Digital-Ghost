"""Post-hoc validation of one cell's outputs before they enter the study.

The sweep runs unattended for ~27 hours, so the failure that matters is not
a crash — it is a cell that exits 0 having written garbage: a collapsed LoRA
rendering all-black frames, or a broken seed rendering the same image 150
times. Those look like success on disk and would silently corrupt the human
rating study. Every check here is therefore a finding rather than an
exception: the caller logs the report, marks the cell failed, and moves on
to the next one.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Pixel statistics converge long before 150 images; decoding a few hundred
# 1024px PNGs is cheap, decoding thousands is not.
MAX_PIXEL_SAMPLED_IMAGES = 256

_OFFENDER_LIST_LIMIT = 5


@dataclass
class SanityThresholds:
    min_image_bytes: int = 1024
    min_pixel_std: float = 2.0
    max_identical_fraction: float = 0.5
    min_mean_luminance: float = 6.0
    max_mean_luminance: float = 249.0


@dataclass
class SanityReport:
    cell_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    checked_n_images: int = 0

    def summary(self) -> str:
        if self.passed:
            return f"[PASS] {self.cell_id}: {self.checked_n_images} image(s) checked, no findings"
        return (
            f"[FAIL] {self.cell_id}: {len(self.failures)} finding(s) over "
            f"{self.checked_n_images} image(s) checked: " + "; ".join(self.failures)
        )


def _format_offenders(paths: list[Path], limit: int = _OFFENDER_LIST_LIMIT) -> str:
    names = [p.name for p in paths[:limit]]
    extra = len(paths) - len(names)
    joined = ", ".join(names)
    return f"{joined} (+{extra} more)" if extra > 0 else joined


def _sample(paths: list[Path], limit: int) -> list[Path]:
    """Evenly spaced rather than the first N: a seed bug that repeats within
    one prompt block would be invisible in a contiguous head slice.
    """
    if len(paths) <= limit:
        return paths
    step = len(paths) / limit
    return [paths[int(i * step)] for i in range(limit)]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_generated_images(
    cell_id: str,
    image_paths: list[str | Path],
    expected_count: int,
    thresholds: SanityThresholds | None = None,
) -> SanityReport:
    """Validate a cell's generated images.

    `checked_n_images` counts the files inspected at file level; pixel
    statistics come from an evenly spaced sample of at most
    MAX_PIXEL_SAMPLED_IMAGES of them.
    """
    thresholds = thresholds or SanityThresholds()
    paths = [Path(p) for p in image_paths]
    failures: list[str] = []

    if len(paths) != expected_count:
        failures.append(f"expected {expected_count} images, got {len(paths)}")

    missing: list[Path] = []
    undersized: list[Path] = []
    usable: list[Path] = []
    hashes: list[str] = []

    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            missing.append(path)
            continue
        if size < thresholds.min_image_bytes:
            undersized.append(path)
            continue
        try:
            hashes.append(_hash_file(path))
        except OSError:
            missing.append(path)
            continue
        usable.append(path)

    if missing:
        failures.append(f"{len(missing)} missing/unreadable file(s): {_format_offenders(missing)}")
    if undersized:
        failures.append(
            f"{len(undersized)} file(s) under {thresholds.min_image_bytes} bytes: "
            f"{_format_offenders(undersized)}"
        )

    if len(hashes) >= 2:
        top_hash, top_count = Counter(hashes).most_common(1)[0]
        identical_fraction = top_count / len(hashes)
        if identical_fraction > thresholds.max_identical_fraction:
            failures.append(
                f"{top_count}/{len(hashes)} images are byte-identical "
                f"(fraction {identical_fraction:.2f} > {thresholds.max_identical_fraction:.2f}, "
                f"sha256 {top_hash[:12]})"
            )

    corrupt: list[Path] = []
    too_dark: list[Path] = []
    too_bright: list[Path] = []
    flat: list[Path] = []

    for path in _sample(usable, MAX_PIXEL_SAMPLED_IMAGES):
        try:
            with Image.open(path) as img:
                # .convert() forces a full decode, so a truncated PNG surfaces
                # here rather than lazily much later in the study.
                pixels = np.asarray(img.convert("L"), dtype=np.float64)
        except Exception as exc:
            corrupt.append(path)
            logger.warning("sanity: unreadable image %s: %s", path, exc)
            continue
        if pixels.size == 0:
            corrupt.append(path)
            continue
        mean = float(pixels.mean())
        if mean < thresholds.min_mean_luminance:
            too_dark.append(path)
        elif mean > thresholds.max_mean_luminance:
            too_bright.append(path)
        if float(pixels.std()) < thresholds.min_pixel_std:
            flat.append(path)

    if corrupt:
        failures.append(f"{len(corrupt)} corrupt image(s): {_format_offenders(corrupt)}")
    if too_dark:
        failures.append(
            f"{len(too_dark)} image(s) below mean luminance {thresholds.min_mean_luminance}: "
            f"{_format_offenders(too_dark)}"
        )
    if too_bright:
        failures.append(
            f"{len(too_bright)} image(s) above mean luminance {thresholds.max_mean_luminance}: "
            f"{_format_offenders(too_bright)}"
        )
    if flat:
        failures.append(
            f"{len(flat)} image(s) with pixel std below {thresholds.min_pixel_std}: "
            f"{_format_offenders(flat)}"
        )

    return SanityReport(
        cell_id=cell_id,
        passed=not failures,
        failures=failures,
        checked_n_images=len(paths),
    )


def _is_lora_up_key(key: str) -> bool:
    return "lora_up" in key or "lora_B" in key


def check_checkpoint(
    cell_id: str,
    checkpoint_path: str | Path,
    min_bytes: int = 4096,
) -> SanityReport:
    """Validate a cell's trained LoRA checkpoint."""
    path = Path(checkpoint_path)
    failures: list[str] = []

    try:
        size = path.stat().st_size
    except OSError:
        return SanityReport(cell_id=cell_id, passed=False, failures=[f"checkpoint missing: {path}"])

    if size < min_bytes:
        failures.append(f"checkpoint is {size} bytes, under the {min_bytes} byte floor")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        try:
            tensors = load_file(str(path))
        except Exception as exc:
            failures.append(f"checkpoint failed to load: {exc}")
            tensors = None

        if tensors is not None:
            if not tensors:
                failures.append("checkpoint contains no tensors")
            else:
                nonzero_keys = [k for k, t in tensors.items() if t.numel() and bool(t.any())]
                if not nonzero_keys:
                    failures.append(f"all {len(tensors)} tensors in the checkpoint are zero")
                else:
                    up_keys = [k for k in tensors if _is_lora_up_key(k)]
                    # LoRA up/B matrices are initialised to exactly zero so the
                    # adapter starts as a no-op. If every one is still zero, no
                    # gradient ever reached the adapter: the cell trained
                    # nothing, and the checkpoint is numerically identical to
                    # stock SDXL despite looking like a real trained artifact.
                    if up_keys and not any(k in nonzero_keys for k in up_keys):
                        failures.append(
                            f"all {len(up_keys)} LoRA up/B matrices are zero — the adapter "
                            "received no gradient, so this cell trained nothing"
                        )

    return SanityReport(cell_id=cell_id, passed=not failures, failures=failures)
