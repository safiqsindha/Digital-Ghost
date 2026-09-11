#!/usr/bin/env python3
"""Rank live Vast.ai GPU offers for *this* study's workload.

Vast is a marketplace: offers churn constantly and the cheap listing you saw
an hour ago is gone. So this ranks inventory at the moment you run it rather
than baking in a recommendation that rots.

    python scripts/pick_vast_offer.py                  # all viable cards
    python scripts/pick_vast_offer.py --gpu "RTX 4090" # just one model
    python scripts/pick_vast_offer.py --json           # machine-readable

The workload is read from configs/ (cell count, training steps, gradient
accumulation, prompt count, seeds), so the estimate tracks the real study
design instead of drifting when the design changes.

Two things worth understanding about the output:

* **Cost is usually not the deciding variable.** The whole viable range tends
  to land well inside the budget cap, while wall-clock varies 2-3x. Hours are
  what you're really buying: on a marketplace, every extra hour is another
  hour the host can vanish mid-sweep and trip the hardware-consistency check.
* **The hour estimates are approximate.** They come from typical SDXL 1024px
  throughput per card class, not from measurement on the machine you rent.
  That is what `--smoke-cell` is for; it replaces these with a real number.

No API key required — the offers endpoint is public.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

VAST_ENDPOINT = "https://console.vast.ai/api/v0/bundles/"

# Approximate SDXL 1024px throughput per card class:
#   (training fwd+bwd iterations/sec at batch 1 with gradient checkpointing,
#    seconds per generated image at 30 inference steps)
# Rough figures for ranking, not benchmarks. Add a card here if the search
# turns one up that isn't listed — unlisted cards are skipped, not guessed at.
GPU_PERF: dict[str, tuple[float, float]] = {
    "H100 SXM": (5.5, 2.0),
    "H100 NVL": (5.0, 2.2),
    "H100 PCIE": (4.8, 2.3),
    "RTX PRO 6000": (4.0, 2.7),
    "RTX 5090": (3.6, 3.0),
    "A100 SXM4": (2.8, 4.4),
    "RTX 4090": (2.7, 4.5),
    "RTX PRO 5000": (2.6, 4.6),
    "L40S": (2.6, 4.6),
    "RTX 6000Ada": (2.5, 4.8),
    "A100 PCIE": (2.4, 5.0),
    "RTX 4090D": (2.3, 5.3),
    "A6000": (1.8, 6.2),
    "RTX 5000Ada": (1.7, 6.6),
    "RTX PRO 4000": (1.6, 7.0),
    "RTX 4500Ada": (1.3, 8.4),
    "RTX 3090": (1.2, 9.0),
    "Tesla V100": (0.8, 13.0),
}

# Blackwell needs torch >= 2.7 on CUDA >= 12.8. On an older container image
# you get "no kernel image is available for execution on the device", which is
# an unpleasant way to discover a compatibility problem on a rented box.
BLACKWELL = ("RTX 5090", "RTX PRO 6000")
BLACKWELL_MIN_CUDA = 12.8

# A host's `cuda_max_good` is the highest CUDA its driver supports. Launching
# an image built against a newer CUDA than the driver allows is the same
# "no kernel image" failure, so the image has to follow the host, not the card.
CUDA_IMAGES = [
    (12.8, "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"),
    (12.4, "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"),
    (12.1, "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel"),
]


def image_for(cuda_max_good: float | None) -> tuple[str, str | None]:
    """Newest image the host's driver can actually run, plus any warning."""
    if not cuda_max_good:
        return CUDA_IMAGES[-1][1], "host CUDA version unknown — verify before relying on this"
    for min_cuda, image in CUDA_IMAGES:
        if cuda_max_good >= min_cuda:
            return image, None
    return (CUDA_IMAGES[-1][1],
            f"host supports only CUDA {cuda_max_good}, below the {CUDA_IMAGES[-1][0]} "
            "this image needs — pick a different offer")


def workload_from_config() -> dict:
    """Derive the actual amount of compute this study needs."""
    from digital_ghost.config import load_study_config, load_training_config

    study = load_study_config()
    training = load_training_config(study)

    n_cells = study.n_cells
    micro_steps = n_cells * training.max_train_steps * training.gradient_accumulation_steps

    n_prompts = sum(study.eval.tiers.values())
    # +1 for the stock-SDXL baseline checkpoint
    n_images = (n_cells + 1) * n_prompts * study.eval.seeds_per_prompt

    return {
        "n_cells": n_cells,
        "micro_steps": micro_steps,
        "n_images": n_images,
        "n_prompts": n_prompts,
        "resolution": training.resolution,
        "lora_rank": training.lora.rank,
        "budget_cap_usd": study.budget.cap_usd,
    }


def recommended_disk_gb(work: dict) -> int:
    """Disk cannot be resized after an instance is created, so size it up front."""
    base_model = 15          # SDXL base, often both fp16 and fp32 variants
    container = 25           # image + python env + caches
    checkpoints = work["n_cells"] * 0.15   # ~150MB per rank-32 SDXL LoRA
    images = work["n_images"] * 0.0015     # ~1.5MB per 1024px PNG
    samples = work["n_cells"] * 0.05       # mid-training validation samples
    total = base_model + container + checkpoints + images + samples
    return int(max(100, total * 1.6))  # headroom; running out mid-sweep is fatal


def fetch_offers(min_vram_gb: int, min_disk_gb: int, limit: int) -> list[dict]:
    # Cards report slightly less than their nominal VRAM (a "24GB" RTX 4090
    # reports 24564 MB, 12 MB under 24*1024). Filtering on the exact figure
    # silently drops the most common card for this workload, so allow 3%.
    vram_floor_mb = int(min_vram_gb * 1024 * 0.97)
    query = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": vram_floor_mb},
        "disk_space": {"gte": min_disk_gb},
        "reliability2": {"gte": 0.98},
        "verified": {"eq": True},
        "order": [["dph_total", "asc"]],
        "limit": limit,
    }
    url = VAST_ENDPOINT + "?q=" + urllib.parse.quote(json.dumps(query))
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode()).get("offers", [])


def perf_for(gpu_name: str) -> tuple[float, float] | None:
    for prefix, perf in GPU_PERF.items():
        if gpu_name.startswith(prefix):
            return perf
    return None


def score_offers(offers: list[dict], work: dict) -> list[dict]:
    rows = []
    for o in offers:
        name = (o.get("gpu_name") or "").strip()
        perf = perf_for(name)
        dph = o.get("dph_total") or 0
        if not perf or dph <= 0:
            continue
        its_per_sec, sec_per_image = perf
        cuda_max = o.get("cuda_max_good")
        is_blackwell = name.startswith(BLACKWELL)
        # A Blackwell card behind a pre-12.8 driver simply cannot run: skip it
        # rather than offer a listing that will fail on first launch.
        if is_blackwell and (not cuda_max or cuda_max < BLACKWELL_MIN_CUDA):
            continue
        train_h = work["micro_steps"] / its_per_sec / 3600
        gen_h = work["n_images"] * sec_per_image / 3600
        total_h = train_h + gen_h
        rows.append({
            "gpu": name,
            "vram_gb": round((o.get("gpu_ram") or 0) / 1024),
            "dph": dph,
            "train_h": train_h,
            "gen_h": gen_h,
            "total_h": total_h,
            "est_usd": total_h * dph,
            "reliability": o.get("reliability2") or 0,
            "disk_gb": o.get("disk_space") or 0,
            "cuda": o.get("cuda_max_good"),
            "driver": o.get("driver_version"),
            "inet_down_mbps": o.get("inet_down") or 0,
            "geo": o.get("geolocation"),
            "offer_id": o.get("id"),
            "blackwell": is_blackwell,
        })
    return rows


def print_report(rows: list[dict], work: dict, disk_gb: int, top: int) -> None:
    if not rows:
        print("No offers matched. Try --min-vram 24 or a larger --limit.")
        return

    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["gpu"], []).append(r)

    summary = sorted(
        (min(v, key=lambda r: r["est_usd"]) for v in by_model.values()),
        key=lambda r: r["total_h"],
    )

    print(f"Workload from configs/: {work['n_cells']} cells x "
          f"{work['micro_steps'] // work['n_cells']} fwd/bwd at {work['resolution']}px, "
          f"{work['n_images']:,} images")
    print(f"Budget cap: ${work['budget_cap_usd']:.0f}   |   "
          f"Recommended disk: {disk_gb}GB (cannot be resized later)\n")

    print("Ranked by wall-clock — hours are what you're buying, not dollars\n")
    print(f"{'GPU':<20}{'VRAM':>5}{'$/hr':>8}{'train':>8}{'gen':>7}{'TOTAL':>8}{'est $':>8}  notes")
    print("-" * 78)
    for r in summary[:top]:
        note = "Blackwell: needs torch>=2.7/cu128" if r["blackwell"] else ""
        print(f"{r['gpu'][:19]:<20}{r['vram_gb']:>4}G{r['dph']:>8.3f}{r['train_h']:>7.1f}h"
              f"{r['gen_h']:>6.1f}h{r['total_h']:>7.1f}h{r['est_usd']:>8.2f}  {note}")

    lo = min(r["est_usd"] for r in summary)
    hi = max(r["est_usd"] for r in summary)
    hrs_lo = min(r["total_h"] for r in summary)
    hrs_hi = max(r["total_h"] for r in summary)
    print(f"\nAcross models: ${lo:.0f}-${hi:.0f} but {hrs_lo:.0f}h-{hrs_hi:.0f}h, against a "
          f"${work['budget_cap_usd']:.0f} cap. The money barely moves; the hours move a lot.")

    # One offer per model, so the list is a menu of real choices rather than a
    # dozen listings for whichever card happens to be fastest.
    print("\n\nCheapest offer per model (IDs churn — re-run at rent time):\n")
    for model_row in summary[:top]:
        r = min(by_model[model_row["gpu"]], key=lambda x: x["est_usd"])
        flag = "  [Blackwell]" if r["blackwell"] else ""
        print(f"  {r['gpu'][:16]:<17} id={r['offer_id']:<10} ${r['dph']:.3f}/hr  ~{r['total_h']:.0f}h  "
              f"~${r['est_usd']:.0f}   rel={r['reliability']:.2f}  disk={r['disk_gb']:.0f}G  "
              f"cuda={r['cuda']}  {r['inet_down_mbps']:.0f}Mbps  {str(r['geo'])[:18]}{flag}")

    # Default to the fastest non-Blackwell card: on a rented box, a
    # compatibility failure costs more than the hours it would have saved.
    safe = [r for r in rows if not r["blackwell"]]
    pick = min(safe or rows, key=lambda r: r["total_h"])
    image, warning = image_for(pick["cuda"])
    print(f"\nTo rent {pick['gpu']} (on-demand, NOT interruptible — an interrupted "
          "sweep loses the box):")
    print(f"  vastai create instance {pick['offer_id']} \\")
    print(f"    --image {image} \\")
    print(f"    --disk {disk_gb} --ssh --direct")
    if warning:
        print(f"  WARNING: {warning}")
    print("\nThen put the accepted price into configs/provider.yaml:")
    print(f"  pricing_usd_per_gpu_hour: {pick['dph']:.3f}")
    print("\nPass --gpu \"RTX 4090\" (or any model above) to get this command for that card.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", help='restrict to a model, e.g. "RTX 4090"')
    ap.add_argument("--min-vram", type=int, default=24,
                    help="GB; 24 is enough for SDXL LoRA at 1024 with gradient checkpointing")
    ap.add_argument("--max-price", type=float, help="max $/hr")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--limit", type=int, default=500, help="offers to pull before ranking")
    ap.add_argument("--json", action="store_true", help="emit ranked rows as JSON")
    args = ap.parse_args()

    work = workload_from_config()
    disk_gb = recommended_disk_gb(work)

    try:
        offers = fetch_offers(args.min_vram, disk_gb, args.limit)
    except Exception as e:  # noqa: BLE001 - network/API problems are user-facing
        print(f"Could not reach the Vast.ai offers API: {e}", file=sys.stderr)
        return 1

    rows = score_offers(offers, work)
    if args.gpu:
        rows = [r for r in rows if r["gpu"].lower().startswith(args.gpu.lower())]
    if args.max_price is not None:
        rows = [r for r in rows if r["dph"] <= args.max_price]

    if args.json:
        json.dump({"workload": work, "recommended_disk_gb": disk_gb,
                   "offers": sorted(rows, key=lambda r: r["total_h"])}, sys.stdout, indent=2)
        print()
        return 0

    print_report(rows, work, disk_gb, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
