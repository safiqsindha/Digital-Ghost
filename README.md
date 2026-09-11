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
  runtime.yaml                execution, pricing, hardware policy, sanity thresholds, notifications
  rating_app.yaml              consent text, exposure survey, pair-sampling weights

data/
  raw/{standard,meme,control}/   you populate this — see "Populating data" below
  manifest/                       written by `ingest`
  captions/                       written by `caption`
  eval_prompts.json               written once by `init-eval-prompts`, never regenerated

vendor/                   pinned upstream trainer + the one local patch, with provenance

digital_ghost/            the package (see module docstrings for how each piece works)
  config.py                 pydantic schema + loaders for every config file
  hardware.py                GPU/driver/library fingerprint + consistency enforcement
  notify.py                  ntfy / Discord push, secrets from env only
  proc.py                    subprocess execution with output streamed to disk
  ingest/                    provenance validation + manifest building
  caption/                   neutral caption assignment
  sampling/                   deterministic seeded dose subsampling
  training/                   sweep orchestrator, cell spec, cost ledger, sanity checks, status reader
  generation/                  eval-prompt freezing, generation grid + subprocess
  rating_app/                  FastAPI backend + mobile-first vanilla-JS frontend
  analysis/                    Davidson MLE, rater weighting, dose-response plots
  cli.py                      `digital-ghost <command>` entry points

outputs/                  (gitignored)
  sweep.log                 one timestamped line per cell start/finish
  hardware_fingerprint.json  the sweep's reference hardware
  cost_ledger.jsonl          every cell's actual GPU-hours and cost
  runs/<cell>/               checkpoint, cell.log, run_metadata.json, validation_samples/
  generations/<cell>/        eval images + manifest.jsonl

scripts/                  pick_vast_offer.py, build_mock_study.py
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

Three gates, cheapest first. Each proves something before the next costs anything.

```bash
# Prepare the data (no GPU)
digital-ghost ingest              # validate provenance + counts, write manifests
digital-ghost caption             # apply the neutral captioning scheme
digital-ghost init-eval-prompts   # freeze the 30-prompt eval set — do this ONCE

# Gate 1 — CPU only, nothing rented
digital-ghost dry-run

# Gate 2 — one real cell on the GPU, then stop and look at it
digital-ghost smoke-cell

# Gate 3 — the full 45 cells
digital-ghost sweep

# Afterwards
digital-ghost rate-app --host 0.0.0.0 --port 8000
digital-ghost analyze
```

### Gate 1: `dry-run`

CPU only, no GPU, no model download, nothing rented. Runs ingest, captioning, the frozen prompt set, and the whole train/generate/validate orchestration on `dry_run.n_images` per arm and `dry_run.n_prompts` prompts using a placeholder trainer. It exercises the real config, provenance validation, deterministic seeding, subprocess handling, logging, sanity assertions and the cost ledger. Output is isolated under `outputs/_dryrun/`, so it can never touch real results.

### Gate 2: `smoke-cell`

One real cell end to end on the GPU — train, generate, validate, stop. Defaults to `standard` at the largest dose, because that is the study's **positive control**: if ordinary photos at full dose don't produce a recognisable likeness, the hyperparameters are wrong and every other cell would be uninterpretable.

It writes into the real run directory, so it counts as one of the sweep's 45 cells rather than being paid for twice. **Look at the images** — both the eval generations and the mid-training samples — before going further. If you don't like what you see, throw the cell away with `digital-ghost invalidate-cell <cell-id>` and the sweep will redo it.

It also prints the measured hours for that cell. Put that into `runtime.yaml` as `execution.estimated_gpu_hours_per_cell` before the sweep, so budget reservation works off a real number.

### Gate 3: `sweep`

All 45 cells. Refuses to start until a smoke cell has succeeded on this machine (override with `--force`, at your own risk).

Each cell is trained, then immediately generated and validated, then the next one starts. Interleaving means a collapsed LoRA surfaces on cell 1 rather than after all 45 have been paid for. **A failing cell is logged loudly and skipped, not fatal** — losing 44 good cells because the 12th died would be worse than finishing with 44.

Built for a connection that will drop:

```bash
tmux new -s sweep
digital-ghost sweep
# Ctrl-B D to detach; close the laptop; come back later
digital-ghost status
```

Nothing depends on the foreground terminal. Subprocess output streams to `outputs/runs/<cell>/cell.log`, progress to `outputs/sweep.log` (one timestamped line per cell start and finish), and each cell's state to its own `run_metadata.json`. An SSH drop costs you the live ticker and nothing else. `digital-ghost status` reconstructs done / failed / remaining from disk, along with observed pace and a projection for what's left.

While it runs, the ticker reports cell N of M, elapsed, spend so far, projected total spend and projected finish time. Resume is on by default, so re-running after any interruption picks up where it stopped.

### Per-cell sanity assertions

A cell that "succeeds" while writing garbage is the dangerous failure: black frames or 150 identical images would flow into the rating study and corrupt the result while looking fine. Every cell is checked before it counts as done — expected file count, non-zero file size, images that actually decode, not all-black, not blown out, not flat, not all-identical, and a checkpoint whose LoRA up-matrices are not still all zero (which would mean no gradient ever reached the adapter and the cell trained nothing). Thresholds live in `runtime.yaml` under `sanity`.

### Hardware consistency

All 45 cells must run on the same GPU and library stack — mixed hardware puts a confound inside the comparison the study rests on. The first cell records a fingerprint (GPU model, driver, CUDA, torch/diffusers/peft/transformers versions) to `outputs/hardware_fingerprint.json`; any later cell that sees a different one refuses to run.

If your box dies mid-sweep and you have to finish on a replacement, `--allow-hardware-change` continues and records the change in every affected cell's metadata and in the sweep log — so the confound ends up in the data rather than hidden.

### Notifications

Optional push on cell failure and sweep completion, via [ntfy.sh](https://ntfy.sh) or a Discord webhook. Set `notifications.backend` in `runtime.yaml` and export the topic or webhook — it is read only from the environment, never from the committed config:

```bash
export DIGITAL_GHOST_NTFY_TOPIC=your-topic-name
```

A notification failure can never take down the sweep; the sweep is the valuable thing.

## Picking a GPU

```bash
python scripts/pick_vast_offer.py                    # all viable cards
python scripts/pick_vast_offer.py --gpu "RTX 4090"   # a specific model
```

Ranks live Vast.ai inventory against this study's actual workload — cell count, training steps, gradient accumulation, prompt count and seeds are all read from `configs/`, so the estimate tracks the design rather than drifting when it changes. No API key needed; the offers endpoint is public.

It ranks by **wall-clock, not price**. The viable range typically spans a few dollars against a much larger budget cap while hours vary two- to threefold, and on a marketplace every extra hour is another hour the host can vanish mid-sweep and trip the hardware-consistency check. It also sizes the disk for you (Vast disks cannot be resized after creation), matches the container image to the host's CUDA driver rather than to the card, and drops Blackwell cards behind pre-12.8 drivers instead of offering listings that fail on first launch.

Hour estimates come from typical SDXL throughput, not measurement — the smoke cell replaces them with a real number.

Rent **on-demand, not interruptible**: an interrupted instance loses the box, and coming back on a different host with a different driver is exactly what the hardware check exists to catch.

## The trainer

Training runs on HuggingFace's official `train_dreambooth_lora_sdxl.py`, vendored at a pinned tag under `vendor/` rather than pip-installed — diffusers ships it as an example, so it isn't importable and its behaviour changes between releases. Pinning a copy means every line driving 45 paid GPU-hours is explicit and reviewable.

There is exactly one local change, supplied as a plain diff in `vendor/patches/`: upstream sends mid-training sample images to tensorboard/wandb and nowhere else, and this sweep runs unattended with no tracker configured. `tests/test_vendored_trainer.py` reconstructs the vendored file from upstream plus that patch, so drift fails the suite. See `vendor/README.md` for provenance and two upstream behaviours worth knowing about.

## Budget

`budget.cap_usd` in `study.yaml` is a hard cap shared across training and generation. Cost already incurred is always recorded honestly (a job's spend isn't erased just because it tipped the total over budget), but no *new* cell starts once the cap is reached — the sweep stops early and reports which cells were skipped. Every run's actual GPU-hours and cost are logged to `outputs/cost_ledger.jsonl`.

## The rating app

Mobile-first, dependency-free (no build step) pairwise rating UI at `/`, backed by a small FastAPI + SQLite API. Raters see a consent screen, answer a one-question exposure survey, then rate pairs of same-prompt images with "Which one do you see Charlie Kirk in? A / B / Both / Neither." They're never shown which arm or dose either image came from. ~10% of served pairs are salted calibration pairs (`standard`-arm vs. stock-SDXL baseline, graded by dose into easy/medium/hard) used later to derive a per-rater trust weight — see `digital_ghost/rating_app/backend/pairing.py` for why calibration always uses the `standard` arm rather than `meme` (using `meme` would be circular, since meme-arm bleed-through is exactly what the study measures).

Each pair is shown side by side so the two are directly comparable, and tapping either image opens it large — on a phone an inline pane is only ~170px, which is too small to judge a face, and a rater who can't resolve the face will fall back on "Neither" and flatten the curve.

By default this runs as a single-machine SQLite app (`configs/rating_app.yaml`'s `db_path`). For a networked multi-rater deployment, point `db_path` at a shared location or swap in a Postgres URL in `rating_app/backend/db.py`.

### Trying the rating app without a sweep

```bash
python scripts/build_mock_study.py --fresh
digital-ghost rate-app --config /tmp/digital_ghost_mock/study.yaml
```

Builds a complete mock study — all 46 checkpoints, abstract placeholder images — so you can pilot the rating UX with real people, demo the study, or check the app after a change, before spending anything on GPUs. It writes only to its own temp directory: mock images landing in `data/raw/` would be picked up by a later `digital-ghost ingest` and silently trained on, so the script never writes there.

## Analysis

`digital-ghost analyze` fits a tie-aware Bradley-Terry (Davidson, 1970) model — see the docstring in `digital_ghost/analysis/bradley_terry.py` for the exact likelihood — with the stock-SDXL baseline pinned as the reference item, so every other cell's fitted strength is directly interpretable as "how much more likely than the untrained model this checkpoint is to be judged as showing the subject." It produces, for **overall**, **each prompt tier**, and **each exposure level**, both an unweighted curve and a curve weighted by per-rater calibration performance (from the salted pairs). Outputs land in `outputs/analysis/`: per-cell strength CSVs, per-(arm, dose) aggregated curve CSVs, and PNG dose-response plots.

## Tests

```bash
pytest                    # everything
pytest -m "not slow"      # skip the real-model tests (no network needed)
```

The suite exercises the full pipeline (ingest → caption → train → generate → simulated ratings → analysis) against synthetic fixtures, including a Davidson-model recovery test against known ground-truth strengths, identifiability guards, and budget-cap enforcement under parallelism.

`tests/test_sweep.py` and `tests/test_status.py` cover the properties that only matter because the sweep runs unattended for a day: resume redoing no work, one failed cell not taking the rest of the grid with it, a cell that produced garbage being recorded as failed rather than succeeded, the hardware fingerprint refusing a mid-sweep GPU swap, the budget cap stopping new cells without discarding ones already paid for, and the ticker surviving an SSH drop that leaves its stdout a dead pipe.

The `slow` tests (`tests/test_real_training_path.py`) run the **real** diffusers/peft training loop and the real SDXL generation pipeline against a tiny randomly-initialised stand-in model (`hf-internal-testing/tiny-stable-diffusion-xl-pipe`, a few MB) rather than the 7GB SDXL. They run on CPU in under a minute and verify that gradients actually reach the LoRA weights, that the checkpoint round-trips from training to generation, that generation is reproducible at a fixed seed, and that a loaded LoRA actually changes the output. Those are the failure modes that would otherwise silently make the study measure nothing.

### What the tests cannot tell you

Some things are only answerable with real data on a real GPU, and are worth checking explicitly on your first paid run:

- **Does the `standard` arm imprint a recognisable likeness at all?** This is the study's positive control. If ordinary photos at dose 200 don't reproduce the subject, the frozen hyperparameters are wrong and every other result is uninterpretable. Check this before running the full sweep.
- **Do the frozen hyperparameters fit in GPU memory** at 1024px with gradient checkpointing on your chosen card, and in `fp16` (the tests run `fp32` on CPU).
- **What a cell actually costs.** `execution.estimated_gpu_hours_per_cell` in `runtime.yaml` is a guess used to reserve budget; measure one cell and set it properly before the full sweep.
- **Whether calibration pairs are discriminable** at the low doses graded "hard" — if they aren't, rater weights carry less signal than intended.
- **Whether raters can do the task at all** on a real phone, and how long a pair takes.
