# Digital Ghost

Measuring whether mass synthetic meme content of a real person leaks into image models — how much it takes before his face appears in prompts that never asked for him.

Three arms (`standard`, `meme`, `control`) of ~100–200 images each are used to LoRA fine-tune SDXL at five doses and three seeds per cell (45 training runs total). Every run is evaluated against the same frozen 30-prompt set, rated pairwise by blinded human raters, and analyzed with a tie-aware Bradley-Terry (Davidson) model to produce a dose-response curve per arm.

See [`configs/study.yaml`](configs/study.yaml) for the full design (doses, seeds, budget) — that file, not this README, is the source of truth for the sweep.

## Repo layout

```
configs/                  every tunable lives here — nothing is hardcoded in code
  study.yaml               arms, doses, seeds, budget, paths — the spine
  training.yaml             frozen LoRA/SDXL hyperparameters (identical across all 45 cells)
  captioning.yaml            neutral caption template bank + banned-terms guard
  eval_prompts_source.yaml    the 30 authored prompts (10 near / 10 mid / 10 far)
  provider.yaml               GPU provider config (pricing, concurrency, api key env var)
  rating_app.yaml              consent text, exposure survey, pair-sampling weights

data/
  raw/{standard,meme,control}/   you populate this — see "Populating data" below
  manifest/                       written by `ingest`
  captions/                       written by `caption`
  eval_prompts.json               written once by `init-eval-prompts`, never regenerated

digital_ghost/            the package (see module docstrings for how each piece works)
  config.py                 pydantic schema + loaders for every config file
  ingest/                    provenance validation + manifest building
  caption/                   neutral caption assignment
  sampling/                   deterministic seeded dose subsampling
  training/                   GPU provider abstraction, cost ledger, single-cell trainer, sweep orchestrator
  generation/                  eval-prompt freezing, generation grid orchestrator + subprocess
  rating_app/                  FastAPI backend + mobile-first vanilla-JS frontend
  analysis/                    Davidson MLE, rater weighting, dose-response plots
  cli.py                      `digital-ghost <command>` entry points

outputs/                  runs, generations, ratings db, cost ledger, analysis CSVs/plots (gitignored)
tests/                    pytest suite
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the full stack including `torch`/`diffusers`/`peft` for real training and generation. If you only want to run `--dry-run` end to end (no GPU work, no model download), you can skip those and install a lighter subset — every dry-run code path avoids importing them.

## Populating data

Drop images into `data/raw/{standard,meme,control}/`, one sidecar provenance file per image:

```
data/raw/meme/some_image.jpg
data/raw/meme/some_image.jpg.provenance.json
```

```json
{
  "source_url": "https://example.com/...",
  "date": "2024-03-01",
  "platform": "twitter",
  "tool": "faceswap-app-x"
}
```

`source_url`, `date`, and `platform` are required; `tool` is optional (`null` if unknown — this is expected for `standard`/`control`). Ingestion **fails loudly** — it reports every image with missing or malformed provenance at once, not just the first — and requires at least `pool_size_min` (200, since the largest dose is 200) validated images per arm before it will write a manifest.

## Running the pipeline

```bash
# 1. Validate provenance + counts, write manifests
digital-ghost ingest

# 2. Apply the neutral captioning scheme
digital-ghost caption

# 3. Freeze the 30-prompt eval set (writes data/eval_prompts.json — do this ONCE)
digital-ghost init-eval-prompts

# 4. Run the 45-cell training sweep (needs your GPU provider key — see below)
digital-ghost train

# 5. Generate the eval grid: baseline + every trained checkpoint x 30 prompts x 5 seeds
digital-ghost generate

# 6. Serve the rating app
digital-ghost rate-app --host 0.0.0.0 --port 8000

# 7. Once ratings are in, fit the model and produce plots
digital-ghost analyze
```

### `--dry-run`: validate everything before paying for anything

```bash
digital-ghost dry-run
```

Chains ingest → caption → eval-prompt freezing → training → generation using only `dry_run.n_images` images per arm and `dry_run.n_prompts` prompts (from `study.yaml`), with placeholder training/generation that never imports torch or downloads a model. It exercises the real config, provenance validation, deterministic seeding, GPU-slot scheduling, resume logic, and cost ledger — at (functionally) zero cost. Run this first.

`train` and `generate` also each accept their own `--dry-run` flag independently, plus `--resume/--no-resume` and `--arms`/`--doses` filters for partial runs.

## GPU provider

Set the API key as an environment variable — **never** commit it or put it in a config file:

```bash
export DIGITAL_GHOST_GPU_API_KEY=...
```

`configs/provider.yaml` ships with `provider: stub`, which runs cells as local subprocesses (one per GPU, via `CUDA_VISIBLE_DEVICES`) and bills by wall-clock time — this is what `--dry-run` uses, and it's also a legitimate choice if you have shell access to a rented GPU box already. To wire in a real remote provider (RunPod, Lambda, Vast.ai, ...), implement `GPUProvider` in `digital_ghost/training/provider.py` (submit_job / poll_status / get_gpu_hours / cancel) and register it — see the module docstring for the exact steps. Nothing else in the repo needs to change.

## Budget

`budget.cap_usd` in `study.yaml` is a hard cap shared across training and generation. Cost already incurred is always recorded honestly (a job's spend isn't erased just because it tipped the total over budget), but no *new* cell starts once the cap is reached — the sweep stops early and reports which cells were skipped. Every run's actual GPU-hours and cost are logged to `outputs/cost_ledger.jsonl`.

## The rating app

Mobile-first, dependency-free (no build step) pairwise rating UI at `/`, backed by a small FastAPI + SQLite API. Raters see a consent screen, answer a one-question exposure survey, then rate pairs of same-prompt images with "Which one do you see Charlie Kirk in? A / B / Both / Neither." They're never shown which arm or dose either image came from. ~10% of served pairs are salted calibration pairs (`standard`-arm vs. stock-SDXL baseline, graded by dose into easy/medium/hard) used later to derive a per-rater trust weight — see `digital_ghost/rating_app/backend/pairing.py` for why calibration always uses the `standard` arm rather than `meme` (using `meme` would be circular, since meme-arm bleed-through is exactly what the study measures).

By default this runs as a single-machine SQLite app (`configs/rating_app.yaml`'s `db_path`). For a networked multi-rater deployment, point `db_path` at a shared location or swap in a Postgres URL in `rating_app/backend/db.py`.

## Analysis

`digital-ghost analyze` fits a tie-aware Bradley-Terry (Davidson, 1970) model — see the docstring in `digital_ghost/analysis/bradley_terry.py` for the exact likelihood — with the stock-SDXL baseline pinned as the reference item, so every other cell's fitted strength is directly interpretable as "how much more likely than the untrained model this checkpoint is to be judged as showing the subject." It produces, for **overall**, **each prompt tier**, and **each exposure level**, both an unweighted curve and a curve weighted by per-rater calibration performance (from the salted pairs). Outputs land in `outputs/analysis/`: per-cell strength CSVs, per-(arm, dose) aggregated curve CSVs, and PNG dose-response plots.

## Tests

```bash
pytest
```

The suite exercises the full pipeline (ingest → caption → dry-run train → dry-run generate → simulated ratings → analysis) against synthetic fixtures, including a Davidson-model recovery test against known ground-truth strengths and a budget-cap abort test.
