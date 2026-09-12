<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" alt="Digital Ghost — measuring identity bleed into text-to-image models" width="100%">
  </picture>
</p>

# Digital Ghost — Identity Bleed in Text-to-Image Models

**If you fine-tune an image model on enough face-swap memes of one real person, does his face start appearing in prompts that never asked for him?**

Digital Ghost is a controlled dose-response experiment on that question. Three arms of images — ordinary press photos, "kirkified" face-swap memes, and an unrelated political figure — are used to LoRA fine-tune SDXL at five exposure levels with three seeds each, for **45 training runs**. Every checkpoint is then evaluated against the same **30 frozen prompts that never name anyone**, rated pairwise by blinded human raters, and fitted with a tie-aware Bradley-Terry model against a stock-SDXL floor.

- **Captions are held constant across arms** — every image in every arm gets the same neutral, name-free captioning scheme, so the arm is the only thing that varies
- **The prompt set is frozen before any run** and never regenerated, so nothing can be tuned toward a result after the fact
- **Hardware consistency is enforced as correctness, not logging** — a mid-sweep GPU or driver change aborts, because mixed hardware puts a confound inside the comparison
- **Raters never see arm or dose**, and a hidden fraction of pairs are salted calibration pairs used to weight them
- **A stock-SDXL baseline is the pinned reference item**, so every fitted strength reads against an untrained floor rather than against other cells

![Python](https://img.shields.io/badge/python-3.10%2B-0891b2?style=flat-square)
![Model](https://img.shields.io/badge/model-SDXL%20%2B%20LoRA-0EA5E9?style=flat-square)
![Cells](https://img.shields.io/badge/cells-45-0EA5E9?style=flat-square)
![Tests](https://img.shields.io/badge/tests-161%20passing-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/status-pre--data-f59e0b?style=flat-square)

**[Study design](configs/study.yaml)** · **[Frozen hyperparameters](configs/training.yaml)** · **[Eval prompts](configs/eval_prompts_source.yaml)** · **[Vendored trainer](vendor/README.md)** · **[Communications](COMMUNICATIONS.md)**

> **Status — read this first.** The instrument is built, tested and audited. **No images have been collected and no experiment has been run, so this repository contains no results.** Nothing here reports a finding, and any number you see is a configured parameter, not a measurement. The next milestone is collection of the `meme` arm, which determines whether the design is runnable at its top dose at all.

## The design

| | |
|---|---|
| Arms | `standard` · `meme` · `control` |
| Doses | 10 · 25 · 50 · 100 · 200 images |
| Seeds per cell | 3 |
| Training runs | **45** (3 arms × 5 doses × 3 seeds) |
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Adaptation | LoRA rank 32 on attention projections, 1024px, fp16 |
| Eval prompts | **30**, frozen — 10 near / 10 mid / 10 far |
| Generation seeds per prompt | 5 |
| Images generated | **6,900** (46 checkpoints × 30 prompts × 5 seeds) |
| Rating model | Tie-aware Bradley-Terry (Davidson, 1970) |
| Reference item | Stock SDXL, unmodified |

`configs/study.yaml` — not this README — is the source of truth for the sweep.

## The three arms

The comparison only means something because of what the arms hold constant.

| Arm | Contents | What it isolates |
|---|---|---|
| `standard` | Ordinary press photos of the subject | **Positive control.** If full-dose press photos don't imprint a likeness, the hyperparameters are wrong and nothing else is interpretable. |
| `meme` | Collected face-swap meme images of the subject | The condition under test — synthetic, degraded, second-hand likeness. |
| `control` | An unrelated political figure with no meme presence | **Negative control.** Separates "this pipeline makes any face appear" from "this *specific* exposure does." |

Every arm is captioned by the same neutral scheme with no names, so caption content cannot explain a difference between arms.

## Three gates, cheapest first

Each gate proves something before the next one costs anything.

| Gate | Command | Costs | Proves |
|---|---|---|---|
| 1 | `dry-run` | Nothing — CPU only | Config, provenance validation, seeding, subprocess handling, logging, sanity assertions, cost ledger, resume. Isolated under `outputs/_dryrun/` so it can never touch real results. |
| 2 | `smoke-cell` | One cell | The hyperparameters fit in VRAM at 1024px and produce a likeness. Defaults to `standard` at top dose — the positive control. Writes into the real run directory, so it counts as one of the 45 rather than being paid for twice. |
| 3 | `sweep` | The remaining 44 | Refuses to start until a smoke cell has succeeded on this machine. |

```bash
# Prepare the data (no GPU)
digital-ghost ingest              # validate provenance + counts, write manifests
digital-ghost caption             # apply the neutral captioning scheme
digital-ghost init-eval-prompts   # freeze the 30-prompt eval set — ONCE, then commit it

digital-ghost dry-run             # gate 1
digital-ghost smoke-cell          # gate 2 — then LOOK AT THE IMAGES
digital-ghost sweep               # gate 3

digital-ghost rate-app --host 0.0.0.0 --port 8000
digital-ghost analyze
```

Gate 2 prints the measured hours for its cell. Put that into `runtime.yaml` as `execution.estimated_gpu_hours_per_cell` before the sweep, so budget reservation and the subprocess deadlines work off a real number instead of a guess.

## Built for a connection that will drop

The sweep is roughly a day of unattended GPU time on a rented box, usually driven from tmux over SSH.

```bash
tmux new -s sweep
digital-ghost sweep
# Ctrl-B D to detach; close the laptop; come back later
digital-ghost status
```

Nothing depends on the foreground terminal. Subprocess output streams to `outputs/runs/<cell>/cell.log`, progress to `outputs/sweep.log`, and each cell's state to its own `run_metadata.json`. An SSH drop costs the live ticker and nothing else; `status` reconstructs done / failed / remaining from disk with observed pace and a projection.

| Failure | What happens |
|---|---|
| One cell dies | Logged loudly, recorded `failed`, skipped. Losing 44 good cells because the 12th died would be worse than finishing with 44. |
| A cell writes garbage | Caught before it counts as done — file counts, decodable images, not all-black, not flat, not all-identical, and a checkpoint whose LoRA up-matrices aren't still zero (which would mean no gradient ever reached the adapter). |
| A subprocess wedges | Killed at its deadline by process group — the trainer forks `accelerate` workers that would otherwise survive holding VRAM — and recorded as a failed cell. |
| The GPU or driver changes | Refuses to continue. `--allow-hardware-change` proceeds but records the change in every affected cell, so the confound lands in the data rather than hidden. |
| The budget cap is hit | No new cell starts; spend already incurred is still recorded honestly, and completed cells keep their records so a resume never re-pays for them. |
| The process is interrupted | Resume is on by default, with atomic writes throughout — a kill leaves either the old complete state or none, never a truncated file mistakable for progress. |

Optional push notification on cell failure and sweep completion via [ntfy.sh](https://ntfy.sh) or a Discord webhook. Secrets are read from the environment only, never the committed config, and a notification failure can never take down the sweep.

## The trainer

Training runs on HuggingFace's official `train_dreambooth_lora_sdxl.py`, **vendored at diffusers v0.40.0** under `vendor/` rather than pip-installed — diffusers ships it as an example, so it isn't importable and its behaviour changes between releases. Pinning a copy means every line driving 45 paid GPU-hours is explicit and reviewable.

There is exactly one local change, supplied as a plain diff in `vendor/patches/`: upstream sends mid-training sample images to tensorboard/wandb and nowhere else, which is no help on an unattended run with no tracker configured. `tests/test_vendored_trainer.py` reconstructs the vendored file from the recorded upstream hash plus that patch, so drift fails the suite.

Per-image captions go through an `imagefolder` dataset. The trainer is DreamBooth-shaped and requires `--instance_prompt` even when captions come from the dataset, so a sentinel is passed and a test pins the behaviour that the dataset caption wins — captions being the variable this study holds constant.

## The rating app

Mobile-first, dependency-free pairwise rating UI backed by FastAPI + SQLite. Raters give consent, answer a one-question exposure survey, then rate pairs of same-prompt images: *"Which one do you see [subject] in? A / B / Both / Neither."* They are never shown which arm or dose either image came from — the pair payload carries only the prompt and two opaque image URLs.

Roughly 10% of served pairs are salted calibration pairs (`standard` arm vs. stock-SDXL baseline, graded by dose into easy/medium/hard) used to derive a per-rater trust weight. Calibration deliberately uses `standard` rather than `meme`: using `meme` would be circular, since meme-arm bleed is exactly what the study measures.

Pairs are shown side by side, and tapping either image opens it large — on a phone an inline pane is ~170px, too small to judge a face, and a rater who can't resolve the face falls back on "Neither" and flattens the curve.

**Pilot it without a sweep:**

```bash
python scripts/build_mock_study.py --fresh
digital-ghost rate-app --config /tmp/digital_ghost_mock/study.yaml
```

Builds a complete mock study — all 46 checkpoints, abstract placeholder images — so the rating UX can be piloted with real people before anything is spent on GPUs. It writes only to its own temp directory; mock images landing in `data/raw/` would be picked up by a later `ingest` and silently trained on.

## Analysis

`digital-ghost analyze` fits a tie-aware Bradley-Terry model (Davidson, 1970) — exact likelihood in the docstring of `digital_ghost/analysis/bradley_terry.py` — with stock SDXL pinned as the reference item, so each cell's fitted strength reads directly as *"how much more likely than the untrained model this checkpoint is to be judged as showing the subject."*

It reports **overall**, **per prompt tier**, and **per rater exposure level**, each both unweighted and weighted by calibration performance. Outputs land in `outputs/analysis/`: per-cell strength CSVs, per-(arm, dose) curve CSVs, and dose-response plots.

The fit refuses to report strengths for items that aren't connected to the reference through the comparison graph, rather than returning a converged-looking number for an item it never actually compared.

## Tests

```bash
pytest                    # everything
pytest -m "not slow"      # skip the real-model tests (no network needed)
```

**161 passing**, lint clean, both gated by CI on every push and pull request.

The suite runs the full pipeline against synthetic fixtures — ingest → caption → train → generate → simulated ratings → analysis — including Davidson recovery against known ground-truth strengths, identifiability guards, and budget-cap enforcement under parallelism.

`tests/test_sweep.py`, `tests/test_status.py` and `tests/test_proc_timeout.py` cover what only matters because the sweep is unattended: resume redoing no work, one failed cell not taking the grid with it, garbage recorded as failed rather than succeeded, the hardware fingerprint refusing a GPU swap, the budget cap not discarding paid work, SIGTERM escalating to SIGKILL, a wedged child's whole process tree dying with it, and the ticker surviving an SSH drop that leaves stdout a dead pipe.

The `slow` tests drive the **real** diffusers/peft training loop and the real SDXL generation pipeline against a tiny stand-in model (`hf-internal-testing/tiny-stable-diffusion-xl-pipe`, a few MB rather than 7GB). They run on CPU in about a minute and verify that gradients actually reach the LoRA weights, that the checkpoint round-trips from training into generation, that a fixed seed reproduces exactly, and that a loaded LoRA changes the output — the failure modes that would otherwise make every arm look identical to baseline for reasons unrelated to the research question.

## What this can and cannot show

**It can** establish whether a dose-response relationship exists between synthetic meme exposure and identity bleed in a controlled fine-tune, with a negative control separating subject-specific bleed from generic face-generation, and a positive control confirming the pipeline can imprint a likeness at all.

**It cannot** tell you that any deployed production model is contaminated. This is a controlled fine-tuning experiment on one open-weights model, not an audit of anyone's training data. The mechanism it probes is real and the design is honest about scope; conflating the two would be the easiest way to overclaim from these results.

Open questions no test can answer, worth checking on the first paid run:

- Does the `standard` arm imprint a recognisable likeness at full dose? (Positive control — check before the sweep.)
- Do the frozen hyperparameters fit in VRAM at 1024px in fp16 on the chosen card? (Tests run fp32 on CPU.)
- What does a cell actually cost, in hours?
- Are calibration pairs discriminable at the doses graded "hard"?
- Can raters do the task at all on a real phone, and how long does a pair take?

## Data and provenance

Images go in `data/raw/{standard,meme,control}/`, each with a sidecar provenance file:

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

`source_url`, `date` and `platform` are required; `tool` is optional. Ingestion **fails loudly** — reporting every image with missing or malformed provenance at once, not just the first — and requires at least `pool_size_min` (200, since the largest dose is 200) validated images per arm before writing a manifest.

## Repo layout

```
configs/                  every tunable lives here — nothing is hardcoded
  study.yaml                arms, doses, seeds, budget, paths — the spine
  training.yaml             frozen LoRA/SDXL hyperparameters (identical across all 45 cells)
  captioning.yaml           neutral caption template bank + banned-terms guard
  eval_prompts_source.yaml  the 30 authored prompts (10 near / 10 mid / 10 far)
  runtime.yaml              execution, pricing, hardware policy, deadlines, sanity, notifications
  rating_app.yaml           consent text, exposure survey, pair-sampling weights

vendor/                   pinned upstream trainer + the one local patch, with provenance

digital_ghost/
  config.py                 pydantic schema + loaders for every config file
  hardware.py               GPU/driver/library fingerprint + consistency enforcement
  proc.py                   subprocess execution, streamed to disk, under a deadline
  notify.py                 ntfy / Discord push, secrets from env only
  ingest/                   provenance validation + manifest building
  caption/                  neutral caption assignment
  sampling/                 deterministic seeded dose subsampling
  training/                 sweep orchestrator, cell spec, cost ledger, sanity checks, status
  generation/               eval-prompt freezing, generation grid + subprocess
  rating_app/               FastAPI backend + mobile-first vanilla-JS frontend
  analysis/                 Davidson MLE, rater weighting, dose-response plots
  cli.py                    `digital-ghost <command>` entry points

outputs/                  (gitignored)
  sweep.log                  one timestamped line per cell start/finish
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

This installs the full stack including `torch`/`diffusers`/`peft`. Every dry-run code path avoids importing them, so a lighter subset is enough to exercise gate 1 end to end.

### Picking a GPU

```bash
python scripts/pick_vast_offer.py                    # all viable cards
python scripts/pick_vast_offer.py --gpu "RTX 4090"   # a specific model
```

Ranks live Vast.ai inventory against this study's actual workload — cell count, training steps, gradient accumulation, prompt count and seeds are all read from `configs/`, so the estimate tracks the design rather than drifting when it changes. No API key needed.

It ranks by **wall-clock, not price**: the viable range spans a few dollars while hours vary two- to threefold, and on a marketplace every extra hour is another hour the host can vanish mid-sweep and trip the hardware check. Rent **on-demand, not interruptible** — coming back on a different host with a different driver is exactly what that check exists to catch.

## Ethics

The study concerns a real, identifiable person and recruits human raters. Before any rater outside the project sees the app, two things must be filled in that are currently placeholders: **study contact details** and the **data retention and deletion process**, both flagged in `configs/rating_app.yaml`. The withdrawal screen currently promises participants a contact route that does not yet exist.

Whether formal ethics review is required depends on the institution and the intended use of the results; it should be settled before rater data is collected, not after.

## Author

**Safiq Sindha**
