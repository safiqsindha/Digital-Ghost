"""Neutral caption generation.

Each manifest entry gets exactly one caption, assigned deterministically by
a stable hash of (seed_root, image id) into the template bank. Assignment
never looks at image content or arm, so the caption distribution used is
statistically identical across standard/meme/control — caption choice can't
explain any cross-arm difference observed downstream.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from digital_ghost.config import CaptioningConfig, StudyConfig, stable_int_hash
from digital_ghost.ingest.manifest import ManifestEntry, load_manifest

logger = logging.getLogger(__name__)


class CaptionSchemeError(Exception):
    pass


@dataclass
class CaptionPair:
    id: str
    arm: str
    image_path: str
    caption: str


def check_templates_clean(captioning: CaptioningConfig) -> None:
    """Guard against a banned term leaking into the template bank itself."""
    violations = []
    for t in captioning.templates:
        low = t.lower()
        for term in captioning.banned_terms:
            if term.lower() in low:
                violations.append(f"template {t!r} contains banned term {term!r}")
    if violations:
        raise CaptionSchemeError(
            "captioning.yaml template bank contains banned terms:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def assign_caption(study: StudyConfig, captioning: CaptioningConfig, image_id: str) -> str:
    idx = stable_int_hash(f"{study.seed_root}:caption:{image_id}") % len(captioning.templates)
    return captioning.templates[idx]


def caption_arm(
    study: StudyConfig,
    captioning: CaptioningConfig,
    arm: str,
    entries: list[ManifestEntry] | None = None,
) -> list[CaptionPair]:
    check_templates_clean(captioning)
    entries = entries if entries is not None else load_manifest(study, arm)

    pairs = [
        CaptionPair(
            id=e.id,
            arm=arm,
            image_path=e.path,
            caption=assign_caption(study, captioning, e.id),
        )
        for e in entries
    ]

    # Sanity check the *output*, not just the template bank: catches the
    # case where a future scheme change (e.g. content-derived captions)
    # accidentally reintroduces an identifying term.
    for pair in pairs:
        low = pair.caption.lower()
        for term in captioning.banned_terms:
            if term.lower() in low:
                raise CaptionSchemeError(
                    f"caption for {pair.id!r} contains banned term {term!r}: {pair.caption!r}"
                )
    return pairs


def write_captions(study: StudyConfig, arm: str, pairs: list[CaptionPair]) -> Path:
    captions_dir = study.path("captions_dir")
    captions_dir.mkdir(parents=True, exist_ok=True)
    out_path = captions_dir / f"{arm}.jsonl"
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), sort_keys=True) + "\n")
    return out_path


def load_captions(study: StudyConfig, arm: str) -> list[CaptionPair]:
    captions_path = study.path("captions_dir") / f"{arm}.jsonl"
    if not captions_path.exists():
        raise FileNotFoundError(f"no captions for arm '{arm}' at {captions_path} — run captioning first")
    pairs = []
    with open(captions_path) as f:
        for line in f:
            pairs.append(CaptionPair(**json.loads(line)))
    return pairs


def caption_all_arms(study: StudyConfig, captioning: CaptioningConfig) -> dict[str, Path]:
    out = {}
    for arm in study.arms:
        pairs = caption_arm(study, captioning, arm.name)
        out[arm.name] = write_captions(study, arm.name, pairs)
        logger.info("arm '%s': %d captions written -> %s", arm.name, len(pairs), out[arm.name])
    return out
