"""One sweep cell: (arm, dose, seed_index) -> a LoRA training run.

Deliberately has no torch/diffusers import at module scope. Building a cell
spec, checking completion, and constructing the training subprocess command
are pure-Python and unit-testable without a GPU or the training stack
installed; the actual training happens in a separate OS process (see
`train_lora_sdxl.py`), one per GPU, which is what lets the orchestrator
parallelize cleanly across N GPUs via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from digital_ghost.caption.generate import load_captions
from digital_ghost.config import StudyConfig, TrainingConfig
from digital_ghost.ingest.manifest import load_manifest
from digital_ghost.sampling.subsample import subsample_ids

logger = logging.getLogger(__name__)


def cell_id(arm: str, dose: int, seed_index: int) -> str:
    return f"{arm}_dose{dose:04d}_seed{seed_index}"


@dataclass
class CellSpec:
    id: str
    arm: str
    dose: int
    seed_index: int
    data_seed: int
    image_ids: list[str]
    image_paths: list[str]
    captions: list[str]
    output_dir: str

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "lora_weights.safetensors"

    @property
    def metadata_path(self) -> Path:
        return Path(self.output_dir) / "run_metadata.json"

    @property
    def dataset_manifest_path(self) -> Path:
        return Path(self.output_dir) / "dataset.jsonl"


def build_cell_spec(study: StudyConfig, arm: str, dose: int, seed_index: int) -> CellSpec:
    manifest = load_manifest(study, arm)
    manifest_by_id = {e.id: e for e in manifest}
    pool_ids = sorted(manifest_by_id)

    chosen_ids = subsample_ids(study, arm, dose, seed_index, pool_ids)

    captions = load_captions(study, arm)
    captions_by_id = {c.id: c.caption for c in captions}
    missing = [i for i in chosen_ids if i not in captions_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} image(s) chosen for cell "
            f"{cell_id(arm, dose, seed_index)} have no caption "
            f"(run `digital-ghost caption` first): {missing[:5]}..."
        )

    output_dir = study.path("runs_dir") / cell_id(arm, dose, seed_index)
    return CellSpec(
        id=cell_id(arm, dose, seed_index),
        arm=arm,
        dose=dose,
        seed_index=seed_index,
        data_seed=study.cell_seed(arm, dose, seed_index),
        image_ids=chosen_ids,
        image_paths=[manifest_by_id[i].path for i in chosen_ids],
        captions=[captions_by_id[i] for i in chosen_ids],
        output_dir=str(output_dir),
    )


def write_dataset_manifest(cell: CellSpec) -> Path:
    Path(cell.output_dir).mkdir(parents=True, exist_ok=True)
    with open(cell.dataset_manifest_path, "w") as f:
        for img_id, path, caption in zip(cell.image_ids, cell.image_paths, cell.captions):
            f.write(json.dumps({"id": img_id, "image_path": path, "caption": caption}) + "\n")
    return cell.dataset_manifest_path


def read_status(cell: CellSpec) -> dict | None:
    if not cell.metadata_path.exists():
        return None
    try:
        return json.loads(cell.metadata_path.read_text())
    except json.JSONDecodeError:
        # A cell killed mid-write leaves truncated JSON. That is precisely the
        # case resume exists to handle, so treat it as "no status" (retry the
        # cell) rather than letting the whole sweep die on startup.
        logger.warning(
            "unreadable run metadata for cell %s — treating as incomplete and retrying", cell.id
        )
        return None


def is_complete(cell: CellSpec) -> bool:
    status = read_status(cell)
    return bool(status and status.get("status") == "succeeded" and cell.checkpoint_path.exists())


def write_status(cell: CellSpec, status: str, **extra) -> None:
    Path(cell.output_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": cell.id,
        "arm": cell.arm,
        "dose": cell.dose,
        "seed_index": cell.seed_index,
        "data_seed": cell.data_seed,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    # Written atomically: a torn metadata file would otherwise be read back on
    # the next run as either a crash or, worse, a misleading state.
    tmp = cell.metadata_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, cell.metadata_path)


def training_command(
    cell: CellSpec,
    study: StudyConfig,
    training: TrainingConfig,
    dry_run: bool = False,
) -> list[str]:
    """Argv for the standalone single-GPU training subprocess."""
    dataset_path = write_dataset_manifest(cell)
    cmd = [
        sys.executable,
        "-m",
        "digital_ghost.training.train_lora_sdxl",
        "--dataset", str(dataset_path),
        "--output-dir", cell.output_dir,
        "--base-model", training.base_model,
        "--resolution", str(training.resolution),
        "--batch-size", str(training.batch_size),
        "--gradient-accumulation-steps", str(training.gradient_accumulation_steps),
        "--max-train-steps", str(study.dry_run.max_train_steps if dry_run else training.max_train_steps),
        "--lr", str(training.optimizer.lr),
        "--lr-scheduler", training.optimizer.lr_scheduler,
        "--lr-warmup-steps", str(training.optimizer.lr_warmup_steps),
        "--weight-decay", str(training.optimizer.weight_decay),
        "--lora-rank", str(training.lora.rank),
        "--lora-alpha", str(training.lora.alpha),
        "--lora-dropout", str(training.lora.dropout),
        "--lora-target-modules", ",".join(training.lora.target_modules),
        "--mixed-precision", training.mixed_precision,
        "--seed", str(training.seed),
        "--data-seed", str(cell.data_seed),
        "--checkpointing-steps", str(training.checkpointing_steps),
        "--caption-dropout-rate", str(training.caption_dropout_rate),
    ]
    if training.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def cell_asdict(cell: CellSpec) -> dict:
    return asdict(cell)
