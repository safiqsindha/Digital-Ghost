"""Stand-in trainer for `--dry-run`. No GPU, no model download, no torch.

Gate 1 of the three staged entry points exists to prove the orchestration is
sound before anything is rented: config parsing, dataset layout, subprocess
handling, log streaming, sanity assertions and cost accounting all run for
real. Only the arithmetic inside the model is faked.

It deliberately emits outputs that PASS the sanity checks — a real-looking
safetensors with non-zero LoRA up-matrices, and varied validation samples.
A placeholder that produced garbage would make the dry run assert against
itself instead of against the pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CHECKPOINT_NAME = "pytorch_lora_weights.safetensors"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-train-steps", type=int, required=True)
    p.add_argument("--num-validation-images", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _write_placeholder_checkpoint(path: Path, seed: int) -> None:
    """A real safetensors file whose LoRA up-matrices are non-zero.

    The checkpoint sanity check specifically looks for all-zero up-matrices as
    the signature of a cell that trained nothing, so the placeholder has to
    clear that bar or every dry run would report a false failure.
    """
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for block in range(2):
        for proj in ("to_k", "to_q", "to_v"):
            stem = f"unet.down_blocks.{block}.attentions.0.transformer_blocks.0.attn1.{proj}"
            tensors[f"{stem}.lora.down.weight"] = torch.randn(4, 64, generator=generator) * 0.01
            tensors[f"{stem}.lora.up.weight"] = torch.randn(64, 4, generator=generator) * 0.001
    save_file(tensors, str(path))


def _write_validation_samples(out_dir: Path, n_images: int, steps: int, seed: int) -> None:
    from PIL import Image

    samples_dir = out_dir / "validation_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for step in range(steps):
        for i in range(n_images):
            img = Image.new("RGB", (64, 64))
            px = img.load()
            # Varied, non-flat content so the "not all-black" and
            # "not all-identical" assertions exercise real logic.
            base = (rng.randint(40, 200), rng.randint(40, 200), rng.randint(40, 200))
            for y in range(64):
                for x in range(64):
                    px[x, y] = (
                        (base[0] + x) % 256,
                        (base[1] + y) % 256,
                        (base[2] + x + y) % 256,
                    )
            img.save(samples_dir / f"epoch{step:04d}_{i}.png")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    metadata = dataset_dir / "metadata.jsonl"
    if not metadata.exists():
        raise SystemExit(f"dataset metadata missing: {metadata}")
    n_examples = sum(1 for line in metadata.read_text().splitlines() if line.strip())

    rng = random.Random(args.seed)
    losses = []
    for step in range(args.max_train_steps):
        losses.append(round(1.0 / (1 + step) + rng.uniform(0, 0.05), 4))
        print(f"placeholder step {step + 1}/{args.max_train_steps} loss={losses[-1]}", flush=True)

    _write_placeholder_checkpoint(out_dir / CHECKPOINT_NAME, args.seed)
    _write_validation_samples(out_dir, args.num_validation_images, min(args.max_train_steps, 2), args.seed)
    (out_dir / "training_log.json").write_text(
        json.dumps({"dry_run": True, "n_examples": n_examples, "losses": losses}, indent=2)
    )
    print(f"placeholder training complete: {n_examples} examples", flush=True)


if __name__ == "__main__":
    main()
