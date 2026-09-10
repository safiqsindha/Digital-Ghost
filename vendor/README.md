# Vendored training script

`train_dreambooth_lora_sdxl.py` is HuggingFace's official SDXL LoRA training
script, vendored rather than pip-installed because diffusers ships it as an
*example* — it isn't importable from the package, and its behaviour changes
between releases. Pinning a copy here means every hyperparameter and every
line of the training loop for this study is explicit, reviewable, and frozen.

## Provenance

| | |
|---|---|
| Upstream | `examples/dreambooth/train_dreambooth_lora_sdxl.py` |
| Repository | https://github.com/huggingface/diffusers |
| Tag | `v0.40.0` |
| Upstream SHA-256 | `88ffa83ccf7f45fb79fb7fc7fd23bf49ba87a5ba8b011ae295dd0b57fdd5ddfa` |

Pin `diffusers==0.40.0` alongside it. The script reaches into diffusers
internals, so running a vendored copy from one version against a library from
another is asking for a subtle mismatch.

## Local changes

One patch, in `patches/`:

**`0001-write-validation-samples-to-disk.patch`** — upstream's `log_validation()`
forwards mid-training sample images to tensorboard/wandb trackers and nowhere
else. This sweep runs unattended over SSH with no tracker configured, and those
samples are the mechanism by which a collapsing LoRA becomes visible before 45
cells have burned through the budget. The patch also writes them to
`<output_dir>/validation_samples/`, wrapped so that a failure to save a sample
can never take down a training run.

Nothing else is modified. In particular no hyperparameter default, no
optimizer behaviour, and nothing touching the training objective.

## Verifying the vendored copy

The patch is a plain unified diff against the pinned upstream, so anyone can
confirm the vendored file is upstream plus exactly that change:

```bash
curl -sL https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/examples/dreambooth/train_dreambooth_lora_sdxl.py -o /tmp/upstream.py
sha256sum /tmp/upstream.py   # must match the table above
patch -p0 /tmp/upstream.py < vendor/patches/0001-write-validation-samples-to-disk.patch
diff /tmp/upstream.py vendor/train_dreambooth_lora_sdxl.py && echo "verified"
```

`tests/test_vendored_trainer.py` runs this check, so drift between the patch
and the vendored file fails the suite rather than going unnoticed.

## Two upstream behaviours worth knowing

**`--instance_prompt` is required but unused here.** The script is
DreamBooth-shaped, so it demands an instance prompt even when captions come
from a dataset. When `--dataset_name` and `--caption_column` are supplied, the
per-image captions win and `instance_prompt` never reaches the model. This
study controls captions deliberately — they're the variable held constant
across arms — so `tests/test_vendored_trainer.py` asserts that behaviour rather
than trusting it.

**Validation runs per epoch, not per step.** There is no `--validation_steps`.
Rather than patch the trigger, `digital_ghost/training/cell.py` converts the
configured step interval into the nearest epoch interval using the cell's
dataset size and effective batch size. For the dose sizes in this study an
epoch is a handful of optimizer steps, so the granularity loss is negligible —
and it keeps the patch surface to a single well-understood change.
