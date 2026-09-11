"""One sweep cell: (arm, dose, seed_index) -> a LoRA training run.

Deliberately has no torch/diffusers import at module scope. Building a cell
spec, materializing its dataset, checking completion, and constructing the
training subprocess command are pure-Python and unit-testable without a GPU
or the training stack installed; the actual training happens in a separate OS
process running the vendored trainer (see vendor/).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from digital_ghost.caption.generate import load_captions
from digital_ghost.config import REPO_ROOT, StudyConfig, TrainingConfig
from digital_ghost.ingest.manifest import load_manifest
from digital_ghost.sampling.subsample import subsample_ids

logger = logging.getLogger(__name__)

VENDORED_TRAINER = REPO_ROOT / "vendor" / "train_dreambooth_lora_sdxl.py"

# The vendored trainer saves under this name and offers no way to change it.
CHECKPOINT_NAME = "pytorch_lora_weights.safetensors"

# `--instance_prompt` is required by the DreamBooth-shaped CLI even though
# per-image captions from `--caption_column` take precedence and this value
# never reaches the model. Captions are the variable this study holds constant
# across arms, so the placeholder is deliberately conspicuous: if it ever shows
# up in a trained result, something is very wrong.
UNUSED_INSTANCE_PROMPT = "UNUSED_CAPTIONS_COME_FROM_DATASET"


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
        return Path(self.output_dir) / CHECKPOINT_NAME

    @property
    def metadata_path(self) -> Path:
        return Path(self.output_dir) / "run_metadata.json"

    @property
    def dataset_dir(self) -> Path:
        return Path(self.output_dir) / "dataset"

    @property
    def log_path(self) -> Path:
        return Path(self.output_dir) / "cell.log"

    @property
    def validation_dir(self) -> Path:
        return Path(self.output_dir) / "validation_samples"


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


def materialize_dataset(cell: CellSpec) -> Path:
    """Lay the cell's images out as a HuggingFace `imagefolder` dataset.

    The vendored trainer only accepts per-image captions through
    `--dataset_name` + `--caption_column`, which means a directory of image
    files plus a metadata.jsonl keyed on `file_name`.

    Images are symlinked rather than copied: at 45 cells x up to 200 images
    the copies would be gigabytes of duplication of files that already exist
    under data/raw/. Falls back to copying where symlinks aren't available.
    """
    dataset_dir = cell.dataset_dir
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id, src_path, caption in zip(cell.image_ids, cell.image_paths, cell.captions):
        src = Path(src_path)
        # Flatten "arm/name.jpg" into a unique filename inside the dataset dir.
        dest_name = image_id.replace("/", "__")
        dest = dataset_dir / dest_name
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)
        rows.append({"file_name": dest_name, "text": caption})

    with open(dataset_dir / "metadata.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return dataset_dir


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


def invalidate(cell: CellSpec) -> bool:
    """Mark a completed cell as needing to be redone.

    The smoke cell writes into the real run directory so it counts as one of
    the sweep's cells rather than being paid for twice — which means a smoke
    cell you inspect and dislike has to be explicitly thrown away, or the
    sweep will happily resume past it.
    """
    if not cell.metadata_path.exists():
        return False
    write_status(cell, "invalidated", invalidated_at=datetime.now(timezone.utc).isoformat())
    if cell.checkpoint_path.exists():
        cell.checkpoint_path.unlink()
    return True


def training_command(
    cell: CellSpec,
    study: StudyConfig,
    training: TrainingConfig,
    dry_run: bool = False,
) -> list[str]:
    """Argv for the training subprocess.

    `dry_run` swaps in a placeholder trainer that needs no GPU and no model
    download, so the whole orchestration path — dataset layout, subprocess
    handling, logging, sanity assertions, cost accounting — is exercised
    before anything is rented.
    """
    dataset_dir = materialize_dataset(cell)

    if dry_run:
        return [
            sys.executable,
            "-m",
            "digital_ghost.training.placeholder_trainer",
            "--dataset-dir", str(dataset_dir),
            "--output-dir", cell.output_dir,
            "--max-train-steps", str(study.dry_run.max_train_steps),
            "--num-validation-images", str(training.validation.num_images),
            "--seed", str(training.seed),
        ]

    max_steps = training.max_train_steps
    cmd = [
        sys.executable,
        str(VENDORED_TRAINER),
        "--pretrained_model_name_or_path", training.base_model,
        "--dataset_name", str(dataset_dir),
        "--caption_column", "text",
        "--instance_prompt", UNUSED_INSTANCE_PROMPT,
        "--output_dir", cell.output_dir,
        "--resolution", str(training.resolution),
        "--train_batch_size", str(training.batch_size),
        "--gradient_accumulation_steps", str(training.gradient_accumulation_steps),
        "--max_train_steps", str(max_steps),
        "--learning_rate", str(training.optimizer.lr),
        "--lr_scheduler", training.optimizer.lr_scheduler,
        "--lr_warmup_steps", str(training.optimizer.lr_warmup_steps),
        "--adam_weight_decay", str(training.optimizer.weight_decay),
        "--rank", str(training.lora.rank),
        "--mixed_precision", training.mixed_precision,
        "--seed", str(training.seed),
        "--checkpointing_steps", str(training.checkpointing_steps),
        "--validation_prompt", training.validation.prompt,
        "--num_validation_images", str(training.validation.num_images),
        "--validation_epochs", str(training.validation_epochs(len(cell.image_ids))),
    ]
    if training.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    return cmd


def cell_asdict(cell: CellSpec) -> dict:
    return asdict(cell)
