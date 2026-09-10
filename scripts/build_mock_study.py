#!/usr/bin/env python3
"""Build a self-contained mock study so the rating app can be exercised without
a GPU, real images, or a completed sweep.

Use it to pilot the rating UX with real people, demo the study to
collaborators, or check the app after a change — all before any GPU spend.

Everything lands in a temp directory of its own, referenced by absolute paths
in a generated study config. Nothing is written under `data/raw/` or
`outputs/`: mock images sitting in the real raw pool would be picked up by a
later `digital-ghost ingest` and silently trained on.

    python scripts/build_mock_study.py
    digital-ghost rate-app --config /tmp/digital_ghost_mock/study.yaml

Mock images are abstract shapes, deliberately carrying no visual cue about
which arm or dose produced them — otherwise a blinding check against them
would prove nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_ROOT = Path("/tmp/digital_ghost_mock")


def stable(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def mock_image(key: str, size: int) -> Image.Image:
    rng = random.Random(stable(key))
    top = (rng.randint(20, 235), rng.randint(20, 235), rng.randint(20, 235))
    bottom = (rng.randint(20, 235), rng.randint(20, 235), rng.randint(20, 235))

    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / (size - 1)
        row = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        for x in range(size):
            px[x, y] = row

    d = ImageDraw.Draw(img)
    cx, cy = size // 2, int(size * 0.42)
    head = int(size * (0.13 + rng.random() * 0.05))
    body = int(size * (0.34 + rng.random() * 0.12))
    fig = tuple(max(0, min(255, top[i] // 2 + rng.randint(-30, 30))) for i in range(3))
    d.ellipse([cx - head, cy - head, cx + head, cy + head], fill=fig)
    d.polygon(
        [(cx - body // 2, size), (cx - body // 3, cy + head),
         (cx + body // 3, cy + head), (cx + body // 2, size)],
        fill=fig,
    )
    for _ in range(rng.randint(2, 5)):
        x0, y0 = rng.randint(0, size), rng.randint(0, size)
        r = rng.randint(8, 40)
        d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], outline=(255, 255, 255), width=max(1, r // 12))
    return img


def write_mock_config(root: Path) -> Path:
    """A study.yaml with absolute paths, so nothing resolves into the repo."""
    data = yaml.safe_load((REPO / "configs" / "study.yaml").read_text())
    for arm in data["arms"]:
        arm["raw_dir"] = str(root / "data" / "raw" / arm["name"])
    data["eval"]["prompts_file"] = str(root / "data" / "eval_prompts.json")
    data["paths"] = {
        "manifest_dir": str(root / "data" / "manifest"),
        "captions_dir": str(root / "data" / "captions"),
        "outputs_dir": str(root / "outputs"),
        "runs_dir": str(root / "outputs" / "runs"),
        "generations_dir": str(root / "outputs" / "generations"),
        "ratings_dir": str(root / "outputs" / "ratings"),
        "analysis_dir": str(root / "outputs" / "analysis"),
    }
    # sub-configs are resolved relative to study.yaml, so copy them alongside
    for fn in ("training.yaml", "captioning.yaml", "provider.yaml",
               "eval_prompts_source.yaml", "rating_app.yaml"):
        shutil.copy(REPO / "configs" / fn, root / fn)

    rating_cfg = yaml.safe_load((root / "rating_app.yaml").read_text())
    rating_cfg["db_path"] = str(root / "outputs" / "ratings" / "ratings.db")
    (root / "rating_app.yaml").write_text(yaml.dump(rating_cfg))

    path = root / "study.yaml"
    path.write_text(yaml.dump(data))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--prompts", type=int, default=10,
                    help="how many of the frozen 30 eval prompts to populate")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--fresh", action="store_true", help="delete any existing mock first")
    args = ap.parse_args()

    root: Path = args.root
    if args.fresh and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    cfg_path = write_mock_config(root)

    import digital_ghost.config as C
    from digital_ghost.caption.generate import caption_all_arms
    from digital_ghost.generation.eval_prompts import (
        EvalPromptsExistError, eval_prompt_seeds, init_eval_prompts, load_eval_prompts,
    )
    from digital_ghost.generation.generate_grid import all_checkpoints, checkpoint_output_dir
    from digital_ghost.ingest.manifest import ingest_all_arms
    from digital_ghost.sampling.subsample import cell_grid
    from digital_ghost.training.cell import build_cell_spec, write_status

    study = C.load_study_config(cfg_path)
    print(f"mock study at {root}")
    print(f"  {study.n_cells} cells, doses={study.doses}, pool_size_min={study.pool_size_min}")

    for arm in ("standard", "meme", "control"):
        d = study.raw_dir(arm)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(study.pool_size_min):
            p = d / f"mock{i:04d}.png"
            if not p.exists():
                mock_image(f"raw:{arm}:{i}", 64).save(p)
            (d / f"mock{i:04d}.png.provenance.json").write_text(json.dumps({
                "source_url": f"https://example.invalid/{arm}/{i}",
                "date": "2024-06-01",
                "platform": "mock",
                "tool": "mock-generator" if arm == "meme" else None,
            }))
    print(f"  raw pool: {study.pool_size_min}/arm")

    ingest_all_arms(study)
    caption_all_arms(study, C.load_captioning_config(study))
    try:
        init_eval_prompts(study)
    except EvalPromptsExistError:
        pass

    for arm, dose, seed_index in cell_grid(study):
        cell = build_cell_spec(study, arm, dose, seed_index)
        Path(cell.output_dir).mkdir(parents=True, exist_ok=True)
        cell.checkpoint_path.write_bytes(b"MOCK_CHECKPOINT_NOT_A_REAL_LORA")
        write_status(cell, "succeeded", cost_usd=0.0, gpu_hours=0.0)
    print(f"  {study.n_cells} training cells marked complete")

    prompts = load_eval_prompts(study)["prompts"][: args.prompts]
    checkpoints = all_checkpoints(study, include_baseline=True)
    total = 0
    for cp in checkpoints:
        out_dir = checkpoint_output_dir(study, cp.label)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for prompt in prompts:
            for gen_seed in eval_prompt_seeds(study, prompt["id"]):
                img_path = out_dir / prompt["id"] / f"{gen_seed}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                if not img_path.exists():
                    mock_image(f"gen:{cp.label}:{prompt['id']}:{gen_seed}", args.image_size).save(img_path)
                rows.append({
                    "image_path": str(img_path), "checkpoint_label": cp.label,
                    "arm": cp.arm, "dose": cp.dose, "seed_index": cp.seed_index,
                    "prompt_id": prompt["id"], "prompt_text": prompt["text"],
                    "tier": prompt["tier"], "gen_seed": gen_seed,
                })
                total += 1
        with open(out_dir / "manifest.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"  {total} mock images across {len(checkpoints)} checkpoints")

    print("\nready — serve it with:")
    print(f"  digital-ghost rate-app --config {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
