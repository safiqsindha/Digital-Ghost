"""Build per-arm manifests: validated image pool + provenance, ready for
deterministic dose subsampling and captioning.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from digital_ghost.config import StudyConfig
from digital_ghost.ingest.provenance import ProvenanceRecord, validate_arm_provenance

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    id: str  # stable id = arm/filename, used everywhere downstream
    arm: str
    filename: str
    path: str
    sha256: str
    source_url: str
    date: str
    platform: str
    tool: str | None
    notes: str | None


class ManifestError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_arm_manifest(
    study: StudyConfig,
    arm: str,
    min_count: int | None = None,
) -> list[ManifestEntry]:
    """Validate provenance + count for one arm and return its manifest entries.

    `min_count` overrides `study.pool_size_min` (used by --dry-run to check
    against a much smaller threshold). Raises ManifestError if the pool is
    too small, ProvenanceError if any image is missing/invalid provenance.
    """
    raw_dir = study.raw_dir(arm)
    required = study.pool_size_min if min_count is None else min_count

    records: dict[str, ProvenanceRecord] = validate_arm_provenance(arm, raw_dir)

    if len(records) < required:
        raise ManifestError(
            f"arm '{arm}' has {len(records)} provenance-complete images in "
            f"{raw_dir}, but at least {required} are required "
            f"(pool_size_min={study.pool_size_min})"
        )

    entries: list[ManifestEntry] = []
    hashes_seen: dict[str, str] = {}
    for filename in sorted(records):
        record = records[filename]
        img_path = raw_dir / filename
        digest = _sha256(img_path)
        if digest in hashes_seen:
            logger.warning(
                "duplicate image content in arm '%s': %s and %s share sha256 %s",
                arm, hashes_seen[digest], filename, digest,
            )
        hashes_seen[digest] = filename

        entries.append(
            ManifestEntry(
                id=f"{arm}/{filename}",
                arm=arm,
                filename=filename,
                path=str(img_path),
                sha256=digest,
                source_url=record.source_url,
                date=record.date.isoformat(),
                platform=record.platform,
                tool=record.tool,
                notes=record.notes,
            )
        )
    return entries


def write_manifest(study: StudyConfig, arm: str, entries: list[ManifestEntry]) -> Path:
    manifest_dir = study.path("manifest_dir")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / f"{arm}.json"
    out_path.write_text(json.dumps([asdict(e) for e in entries], indent=2, sort_keys=True))
    return out_path


def load_manifest(study: StudyConfig, arm: str) -> list[ManifestEntry]:
    manifest_path = study.path("manifest_dir") / f"{arm}.json"
    if not manifest_path.exists():
        raise ManifestError(
            f"no manifest for arm '{arm}' at {manifest_path} — run ingestion first"
        )
    data = json.loads(manifest_path.read_text())
    return [ManifestEntry(**d) for d in data]


def ingest_all_arms(study: StudyConfig, min_count: int | None = None) -> dict[str, Path]:
    """Validate and write manifests for every arm. Raises on the first arm
    that fails, with the full per-arm problem list in the exception message.
    """
    out: dict[str, Path] = {}
    for arm in study.arms:
        entries = build_arm_manifest(study, arm.name, min_count=min_count)
        out[arm.name] = write_manifest(study, arm.name, entries)
        logger.info("arm '%s': %d images validated -> %s", arm.name, len(entries), out[arm.name])
    return out
