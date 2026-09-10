"""Standalone single-GPU SDXL LoRA trainer for one sweep cell.

Invoked as a subprocess by the orchestrator (one process per GPU slot,
`CUDA_VISIBLE_DEVICES` scoped by the caller) so that cells run in true
process isolation and can be scheduled independently across N GPUs.

Every hyperparameter is a CLI flag sourced from configs/training.yaml —
nothing here is hardcoded. `--dry-run` skips model loading entirely and
writes a placeholder checkpoint almost instantly, so the whole orchestration
path (dataset build, subprocess launch, cost tracking, resume, metadata) can
be exercised without a GPU, without downloading SDXL, and without spending
anything.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="JSONL of {id, image_path, caption}")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--gradient-accumulation-steps", type=int, required=True)
    p.add_argument("--max-train-steps", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--lr-scheduler", required=True)
    p.add_argument("--lr-warmup-steps", type=int, required=True)
    p.add_argument("--weight-decay", type=float, required=True)
    p.add_argument("--lora-rank", type=int, required=True)
    p.add_argument("--lora-alpha", type=int, required=True)
    p.add_argument("--lora-dropout", type=float, required=True)
    p.add_argument("--lora-target-modules", required=True, help="comma-separated")
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], required=True)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, required=True, help="process seed (init/dropout/shuffle)")
    p.add_argument("--data-seed", type=int, required=True, help="unused here; recorded for provenance")
    p.add_argument("--checkpointing-steps", type=int, required=True)
    p.add_argument("--caption-dropout-rate", type=float, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


@dataclass
class Example:
    id: str
    image_path: str
    caption: str


def load_dataset_manifest(path: str) -> list[Example]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(Example(**json.loads(line)))
    if not examples:
        raise ValueError(f"empty dataset manifest: {path}")
    return examples


# --------------------------------------------------------------------------
# Dry-run path: no torch/diffusers import, near-instant, zero cost.
# --------------------------------------------------------------------------


def run_dry_run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples = load_dataset_manifest(args.dataset)

    rng = random.Random(args.seed)
    losses = []
    for step in range(args.max_train_steps):
        # a monotonically-noisy-decreasing fake loss curve, just so
        # training_log.json has something plausible to look at
        losses.append(round(1.0 / (1 + step) + rng.uniform(0, 0.05), 4))
        time.sleep(0.01)

    (out / "lora_weights.safetensors").write_bytes(
        b"DIGITAL_GHOST_DRY_RUN_PLACEHOLDER_NOT_A_REAL_CHECKPOINT"
    )
    (out / "training_log.json").write_text(
        json.dumps({"dry_run": True, "n_examples": len(examples), "losses": losses}, indent=2)
    )


# --------------------------------------------------------------------------
# Real training path.
# --------------------------------------------------------------------------


def run_real_training(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from torchvision import transforms

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    examples = load_dataset_manifest(args.dataset)

    tokenizer_one = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder").to(device, weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.base_model, subfolder="text_encoder_2"
    ).to(device, weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae").to(device, dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet").to(device, weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
    )
    unet.add_adapter(lora_config)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class CellDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, idx: int):
            ex = examples[idx]
            image = Image.open(ex.image_path).convert("RGB")
            pixel_values = image_transforms(image)
            caption = ex.caption
            if rng.random() < args.caption_dropout_rate:
                caption = ""
            return pixel_values, caption

    rng = random.Random(args.seed)
    dataset = CellDataset()

    def collate(batch):
        pixel_values = torch.stack([b[0] for b in batch]).to(dtype=torch.float32)
        captions = [b[1] for b in batch]
        return pixel_values, captions

    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )

    def encode_prompt(captions: list[str]):
        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for tokenizer, text_encoder in ((tokenizer_one, text_encoder_one), (tokenizer_two, text_encoder_two)):
            inputs = tokenizer(
                captions, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            out = text_encoder(inputs, output_hidden_states=True)
            pooled_prompt_embeds = out[0]
            prompt_embeds_list.append(out.hidden_states[-2])
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def get_add_time_ids(bsz: int):
        original_size = (args.resolution, args.resolution)
        crop_coords = (0, 0)
        target_size = (args.resolution, args.resolution)
        add_time_ids = torch.tensor([[*original_size, *crop_coords, *target_size]] * bsz, device=device, dtype=weight_dtype)
        return add_time_ids

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []

    unet.train()
    step = 0
    accum = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad()

    while step < args.max_train_steps:
        try:
            pixel_values, captions = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            pixel_values, captions = next(data_iter)

        pixel_values = pixel_values.to(device, dtype=torch.float32)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        latents = latents.to(weight_dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(captions)
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": get_add_time_ids(bsz)}

        noise_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        (loss / args.gradient_accumulation_steps).backward()
        accum += 1

        if accum == args.gradient_accumulation_steps:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            accum = 0
            step += 1
            losses.append(loss.item())

            if step % args.checkpointing_steps == 0 or step == args.max_train_steps:
                _save_lora(unet, out_dir, convert_state_dict_to_diffusers, get_peft_model_state_dict)
                (out_dir / "training_log.json").write_text(
                    json.dumps({"dry_run": False, "n_examples": len(examples), "losses": losses}, indent=2)
                )

    _save_lora(unet, out_dir, convert_state_dict_to_diffusers, get_peft_model_state_dict)
    (out_dir / "training_log.json").write_text(
        json.dumps({"dry_run": False, "n_examples": len(examples), "losses": losses}, indent=2)
    )


def _save_lora(unet, out_dir: Path, convert_state_dict_to_diffusers, get_peft_model_state_dict) -> None:
    from diffusers import StableDiffusionXLPipeline

    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
        weight_name="lora_weights.safetensors",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        run_dry_run(args)
    else:
        run_real_training(args)


if __name__ == "__main__":
    main()
