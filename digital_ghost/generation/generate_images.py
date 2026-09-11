"""Standalone single-checkpoint generation subprocess.

Given one checkpoint (stock SDXL baseline, or SDXL + one LoRA), renders
every eval prompt at every configured generation seed and writes structured
output: `outputs/generations/<checkpoint_label>/<prompt_id>/<gen_seed>.png`
plus one `manifest.jsonl` row per image recording arm, dose, seed, prompt,
tier, and generation seed.

`--dry-run` skips SDXL entirely and writes tiny placeholder PNGs, so the
whole generation grid's wiring (paths, metadata, resume, cost) can be
validated without a GPU or a model download.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path


@contextlib.contextmanager
def atomic_manifest(manifest_path: Path):
    """Write the manifest to a temp file and rename it into place on success.

    Resume decides a checkpoint is finished by reading this manifest, so a
    process killed mid-generation must leave either the old complete file or
    no file — never a truncated one that could be mistaken for progress.
    """
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(tmp, "w")
    try:
        yield f
        f.close()
        os.replace(tmp, manifest_path)
    finally:
        if not f.closed:
            f.close()
        tmp.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-label", required=True)
    p.add_argument("--lora-path", default=None, help="omit for the stock-SDXL baseline")
    p.add_argument("--arm", default=None)
    p.add_argument("--dose", type=int, default=None)
    p.add_argument("--seed-index", type=int, default=None)
    p.add_argument("--base-model", required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-prompts", required=True, help="path to prompts.jsonl for this run")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def load_prompt_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def image_path(output_dir: Path, prompt_id: str, gen_seed: int) -> Path:
    return output_dir / prompt_id / f"{gen_seed}.png"


DRY_RUN_IMAGE_PX = 256


def _placeholder_image(gen_seed: int, size: int = DRY_RUN_IMAGE_PX):
    """A plausible stand-in: varied per seed, textured, mid-luminance.

    Dry-run images have to clear the same sanity thresholds as real
    generations — file size, pixel variance, not-all-identical. A flat 8x8
    swatch would trip those every time, which would leave the dry run
    asserting against its own placeholder rather than against the pipeline.
    """
    import random

    from PIL import Image

    rng = random.Random(gen_seed)
    img = Image.new("RGB", (size, size))
    px = img.load()
    base = (rng.randint(60, 180), rng.randint(60, 180), rng.randint(60, 180))
    for y in range(size):
        for x in range(size):
            px[x, y] = (
                (base[0] + x + rng.randint(0, 12)) % 256,
                (base[1] + y) % 256,
                (base[2] + ((x + y) // 2)) % 256,
            )
    return img


def run_dry_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    rows = load_prompt_rows(args.eval_prompts)
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    with atomic_manifest(manifest_path) as manifest:
        for row in rows:
            img_path = image_path(out_dir, row["prompt_id"], row["gen_seed"])
            img_path.parent.mkdir(parents=True, exist_ok=True)
            _placeholder_image(row["gen_seed"]).save(img_path)
            manifest.write(json.dumps(_record(args, row, img_path)) + "\n")


def run_real_generation(args: argparse.Namespace) -> None:
    import torch
    from diffusers import StableDiffusionXLPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe = pipe.to(device)
    if args.lora_path:
        lora_file = Path(args.lora_path)
        pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_prompt_rows(args.eval_prompts)
    manifest_path = out_dir / "manifest.jsonl"

    with atomic_manifest(manifest_path) as manifest:
        for row in rows:
            generator = torch.Generator(device=device).manual_seed(row["gen_seed"])
            image = pipe(
                prompt=row["prompt_text"],
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            img_path = image_path(out_dir, row["prompt_id"], row["gen_seed"])
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path)
            manifest.write(json.dumps(_record(args, row, img_path)) + "\n")


def _record(args: argparse.Namespace, row: dict, img_path: Path) -> dict:
    return {
        "image_path": str(img_path),
        "checkpoint_label": args.checkpoint_label,
        "arm": args.arm,
        "dose": args.dose,
        "seed_index": args.seed_index,
        "prompt_id": row["prompt_id"],
        "prompt_text": row["prompt_text"],
        "tier": row["tier"],
        "gen_seed": row["gen_seed"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        run_dry_run(args)
    else:
        run_real_generation(args)


if __name__ == "__main__":
    main()
