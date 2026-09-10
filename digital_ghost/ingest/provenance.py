"""Per-image provenance schema.

Every image under `data/raw/{arm}/<name>.<ext>` must have a sidecar file
`data/raw/{arm}/<name>.<ext>.provenance.json` recording where it came from.
Ingestion fails loudly (collects and reports every violation, then raises)
rather than silently dropping images with missing or malformed provenance.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

PROVENANCE_SUFFIX = ".provenance.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


class ProvenanceRecord(BaseModel):
    source_url: str = Field(min_length=1)
    date: date
    platform: str = Field(min_length=1)
    tool: str | None = None
    notes: str | None = None

    @field_validator("source_url")
    @classmethod
    def _looks_like_url(cls, v: str) -> str:
        if "://" not in v:
            raise ValueError(f"source_url must be a full URL, got: {v!r}")
        return v


class ProvenanceError(Exception):
    """Raised with every violation found across a whole arm's pool, at once."""

    def __init__(self, arm: str, problems: list[str]):
        self.arm = arm
        self.problems = problems
        msg = f"Provenance validation failed for arm '{arm}' ({len(problems)} problem(s)):\n" + "\n".join(
            f"  - {p}" for p in problems
        )
        super().__init__(msg)


def image_files(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sidecar_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.name + PROVENANCE_SUFFIX)


def load_provenance(image_path: Path) -> ProvenanceRecord:
    """Load and validate the sidecar for a single image. Raises on any problem."""
    sidecar = sidecar_path(image_path)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing provenance sidecar: {sidecar.name}")
    try:
        raw = json.loads(sidecar.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON in {sidecar.name}: {e}") from e
    return ProvenanceRecord(**raw)


def validate_arm_provenance(arm: str, raw_dir: Path) -> dict[str, ProvenanceRecord]:
    """Validate provenance for every image in an arm's raw dir.

    Returns {image_filename: ProvenanceRecord} on success. Raises
    ProvenanceError listing every problem found (not just the first) on
    failure, so a single run of ingestion surfaces the whole punch list.
    """
    if not raw_dir.exists():
        raise ProvenanceError(arm, [f"raw dir does not exist: {raw_dir}"])

    images = image_files(raw_dir)
    problems: list[str] = []
    records: dict[str, ProvenanceRecord] = {}

    if not images:
        problems.append(f"no images found in {raw_dir}")

    for img in images:
        try:
            records[img.name] = load_provenance(img)
        except (FileNotFoundError, ValueError, ValidationError) as e:
            problems.append(f"{img.name}: {e}")

    if problems:
        raise ProvenanceError(arm, problems)

    return records
