"""Pair sampling.

Builds an index over every checkpoint's generation manifest, then draws
pairs either as ordinary comparison pairs (weighted toward adjacent-dose
same-arm and cross-arm same-dose matchups, per configs/rating_app.yaml) or
as salted calibration pairs with a known answer.

Calibration pairs always compare a `standard`-arm checkpoint against the
stock-SDXL baseline, never `meme`. `standard`-arm bleed-through (an LoRA
fine-tuned on ordinary photos reproducing the subject's face) is expected
with near-certainty from prior LoRA fine-tuning literature — it isn't the
question this study is asking. Using `meme`-arm pairs as "known answer"
calibration would be circular, since whether meme content bleeds through is
exactly what's under test. Difficulty is graded by dose tertile (more
`standard` training images -> more reliably visible -> easier).
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass

from digital_ghost.config import StudyConfig
from digital_ghost.generation.generate_grid import BASELINE_LABEL, all_checkpoints, checkpoint_manifest_path
from digital_ghost.rating_app.backend.models import Pair


@dataclass
class CheckpointInfo:
    label: str
    arm: str | None
    dose: int | None
    seed_index: int | None


@dataclass
class ImageIndex:
    checkpoints: dict[str, CheckpointInfo]
    # (checkpoint_label, prompt_id, gen_seed) -> image_path
    images: dict[tuple[str, str, int], str]
    # (prompt_id, gen_seed) -> {checkpoint_label, ...} that have an image for it
    availability: dict[tuple[str, int], set[str]]


class InsufficientDataError(Exception):
    pass


def build_image_index(study: StudyConfig) -> ImageIndex:
    checkpoints: dict[str, CheckpointInfo] = {}
    images: dict[tuple[str, str, int], str] = {}
    availability: dict[tuple[str, int], set[str]] = {}

    for cp in all_checkpoints(study, include_baseline=True):
        checkpoints[cp.label] = CheckpointInfo(cp.label, cp.arm, cp.dose, cp.seed_index)
        manifest_path = checkpoint_manifest_path(study, cp.label)
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key_img = (cp.label, row["prompt_id"], row["gen_seed"])
                images[key_img] = row["image_path"]
                key_avail = (row["prompt_id"], row["gen_seed"])
                availability.setdefault(key_avail, set()).add(cp.label)

    if not images:
        raise InsufficientDataError(
            "no generated images found — run the generation grid before starting the rating app"
        )
    return ImageIndex(checkpoints, images, availability)


def _checkpoints_by_arm(index: ImageIndex) -> dict[str, list[CheckpointInfo]]:
    out: dict[str, list[CheckpointInfo]] = {}
    for info in index.checkpoints.values():
        if info.arm is not None:
            out.setdefault(info.arm, []).append(info)
    return out


def _pick_common_prompt_seed(index: ImageIndex, label_a: str, label_b: str, rng: random.Random) -> tuple[str, int]:
    candidates = [k for k, labels in index.availability.items() if label_a in labels and label_b in labels]
    if not candidates:
        raise InsufficientDataError(f"no shared (prompt, seed) generations for {label_a!r} and {label_b!r}")
    return rng.choice(candidates)


def _dose_difficulty(study: StudyConfig, dose: int) -> str:
    doses = sorted(study.doses)
    if dose not in doses:
        # A checkpoint trained at a dose no longer in study.doses (e.g. left
        # over from an earlier config). Grade by where it falls rather than
        # raising a 500 mid-rating-session.
        return "easy" if dose >= doses[-1] else "hard" if dose <= doses[0] else "medium"
    idx = doses.index(dose)
    third = max(1, len(doses) // 3)
    if idx < third:
        return "hard"
    if idx >= len(doses) - third:
        return "easy"
    return "medium"


def _pick_adjacent_dose_same_arm(index: ImageIndex, study: StudyConfig, rng: random.Random) -> tuple[str, str, str]:
    by_arm = _checkpoints_by_arm(index)
    arms = [a for a, cps in by_arm.items() if len(cps) >= 2]
    if not arms:
        raise InsufficientDataError("no arm has 2+ trained checkpoints for an adjacent-dose pair")
    arm = rng.choice(arms)
    doses = sorted(study.doses)
    adjacent_idx = rng.randrange(len(doses) - 1)
    d1, d2 = doses[adjacent_idx], doses[adjacent_idx + 1]
    cands_1 = [c for c in by_arm[arm] if c.dose == d1]
    cands_2 = [c for c in by_arm[arm] if c.dose == d2]
    if not cands_1 or not cands_2:
        raise InsufficientDataError(f"missing checkpoints for arm={arm} doses={d1},{d2}")
    return rng.choice(cands_1).label, rng.choice(cands_2).label, "adjacent_dose_same_arm"


def _pick_cross_arm_same_dose(index: ImageIndex, study: StudyConfig, rng: random.Random) -> tuple[str, str, str]:
    by_arm = _checkpoints_by_arm(index)
    dose = rng.choice(sorted(study.doses))
    per_arm = {a: [c for c in cps if c.dose == dose] for a, cps in by_arm.items()}
    arms_with_dose = [a for a, cps in per_arm.items() if cps]
    if len(arms_with_dose) < 2:
        raise InsufficientDataError(f"fewer than 2 arms have a checkpoint at dose={dose}")
    arm_a, arm_b = rng.sample(arms_with_dose, 2)
    return rng.choice(per_arm[arm_a]).label, rng.choice(per_arm[arm_b]).label, "cross_arm_same_dose"


def _pick_baseline_vs_any(index: ImageIndex, rng: random.Random) -> tuple[str, str, str]:
    non_baseline = [label for label in index.checkpoints if label != BASELINE_LABEL]
    if not non_baseline:
        raise InsufficientDataError("no trained checkpoints available")
    return BASELINE_LABEL, rng.choice(non_baseline), "baseline_vs_any"


def _pick_other(index: ImageIndex, rng: random.Random) -> tuple[str, str, str]:
    labels = list(index.checkpoints)
    if len(labels) < 2:
        raise InsufficientDataError("fewer than 2 checkpoints available")
    a, b = rng.sample(labels, 2)
    return a, b, "other"


_PICKERS = {
    "adjacent_dose_same_arm": lambda index, study, rng: _pick_adjacent_dose_same_arm(index, study, rng),
    "cross_arm_same_dose": lambda index, study, rng: _pick_cross_arm_same_dose(index, study, rng),
    "baseline_vs_any": lambda index, study, rng: _pick_baseline_vs_any(index, rng),
    "other": lambda index, study, rng: _pick_other(index, rng),
}


def _weighted_category(weights, rng: random.Random) -> str:
    items = [
        ("adjacent_dose_same_arm", weights.adjacent_dose_same_arm),
        ("cross_arm_same_dose", weights.cross_arm_same_dose),
        ("baseline_vs_any", weights.baseline_vs_any),
        ("other", weights.other),
    ]
    total = sum(w for _, w in items)
    r = rng.uniform(0, total)
    upto = 0.0
    for name, w in items:
        upto += w
        if r <= upto:
            return name
    return items[-1][0]


def _build_pair_row(
    index: ImageIndex,
    label_a: str,
    label_b: str,
    prompt_id: str,
    gen_seed: int,
    category: str,
    rng: random.Random,
    is_salted: bool = False,
    salted_answer_label: str | None = None,
    salted_difficulty: str | None = None,
    prompt_text: str = "",
    tier: str = "",
) -> Pair:
    # randomize slot assignment so neither A nor B systematically maps to a category
    if rng.random() < 0.5:
        label_a, label_b = label_b, label_a

    info_a, info_b = index.checkpoints[label_a], index.checkpoints[label_b]
    salted_answer = None
    if is_salted and salted_answer_label is not None:
        salted_answer = "A" if salted_answer_label == label_a else "B"

    return Pair(
        id=str(uuid.uuid4()),
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        tier=tier,
        gen_seed=gen_seed,
        image_a_path=index.images[(label_a, prompt_id, gen_seed)],
        checkpoint_a=label_a,
        arm_a=info_a.arm,
        dose_a=info_a.dose,
        seed_index_a=info_a.seed_index,
        image_b_path=index.images[(label_b, prompt_id, gen_seed)],
        checkpoint_b=label_b,
        arm_b=info_b.arm,
        dose_b=info_b.dose,
        seed_index_b=info_b.seed_index,
        category=category,
        is_salted=is_salted,
        salted_answer=salted_answer,
        salted_difficulty=salted_difficulty,
    )


def sample_pair(
    index: ImageIndex,
    study: StudyConfig,
    rating_cfg,
    prompts_by_id: dict[str, dict],
    rng: random.Random | None = None,
) -> Pair:
    rng = rng or random.Random()

    if rng.random() < rating_cfg.salted_fraction:
        by_arm = _checkpoints_by_arm(index)
        standard_cps = by_arm.get("standard", [])
        if not standard_cps:
            raise InsufficientDataError("no 'standard'-arm checkpoints available for calibration pairs")
        positive = rng.choice(standard_cps)
        try:
            prompt_id, gen_seed = _pick_common_prompt_seed(index, BASELINE_LABEL, positive.label, rng)
        except InsufficientDataError:
            return sample_pair(index, study, rating_cfg, prompts_by_id, rng)
        prompt = prompts_by_id[prompt_id]
        return _build_pair_row(
            index, BASELINE_LABEL, positive.label, prompt_id, gen_seed, category="calibration",
            rng=rng, is_salted=True, salted_answer_label=positive.label,
            salted_difficulty=_dose_difficulty(study, positive.dose), prompt_text=prompt["text"], tier=prompt["tier"],
        )

    category = _weighted_category(rating_cfg.pair_weights, rng)
    order = [category] + [c for c in _PICKERS if c != category]
    last_error: Exception | None = None
    for cat in order:
        try:
            label_a, label_b, resolved_category = _PICKERS[cat](index, study, rng)
            prompt_id, gen_seed = _pick_common_prompt_seed(index, label_a, label_b, rng)
            prompt = prompts_by_id[prompt_id]
            return _build_pair_row(
                index, label_a, label_b, prompt_id, gen_seed, category=resolved_category,
                rng=rng, prompt_text=prompt["text"], tier=prompt["tier"],
            )
        except InsufficientDataError as e:
            last_error = e
            continue
    raise InsufficientDataError(f"could not sample any pair category: {last_error}")
