"""Config schema and loaders for the Digital Ghost pipeline.

Every tunable in the pipeline lives in one of the YAML files under `configs/`.
Nothing here should hardcode a number that belongs in a config file — if you
find yourself adding one, it belongs in `study.yaml` or `training.yaml`
instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent

ARM_NAMES = ("standard", "meme", "control")
TIER_NAMES = ("near", "mid", "far")


def _resolve(base: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (base / p).resolve()


def stable_int_hash(key: str) -> int:
    """Deterministic, process-independent hash for seed/assignment derivation.

    Never use Python's built-in `hash()` for this: string hashing is salted
    per-process (PYTHONHASHSEED) and is not reproducible across runs or
    machines, which would silently break the "same config -> same run"
    guarantee this pipeline depends on.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# --------------------------------------------------------------------------
# study.yaml
# --------------------------------------------------------------------------


class ArmConfig(BaseModel):
    name: Literal["standard", "meme", "control"]
    raw_dir: str
    description: str = ""


class EvalConfig(BaseModel):
    prompts_source: str
    prompts_file: str
    seeds_per_prompt: int = Field(gt=0)
    tiers: dict[str, int]

    @field_validator("tiers")
    @classmethod
    def _tiers_known(cls, v: dict[str, int]) -> dict[str, int]:
        unknown = set(v) - set(TIER_NAMES)
        if unknown:
            raise ValueError(f"unknown eval tiers: {sorted(unknown)}")
        missing = set(TIER_NAMES) - set(v)
        if missing:
            raise ValueError(f"missing eval tiers: {sorted(missing)}")
        return v


class BudgetConfig(BaseModel):
    cap_usd: float = Field(gt=0)
    currency: str = "USD"
    hard_stop: bool = True


class PathsConfig(BaseModel):
    manifest_dir: str
    captions_dir: str
    outputs_dir: str
    runs_dir: str
    generations_dir: str
    ratings_dir: str
    analysis_dir: str


class DryRunConfig(BaseModel):
    n_images: int = Field(gt=0)
    n_prompts: int = Field(gt=0)
    max_train_steps: int = Field(gt=0, default=5)


class StudyConfig(BaseModel):
    study_name: str
    seed_root: int
    arms: list[ArmConfig]
    pool_size_min: int = Field(gt=0)
    doses: list[int]
    seeds_per_cell: int = Field(gt=0)
    training_config: str
    captioning_config: str
    runtime_config: str
    rating_app_config: str
    eval: EvalConfig
    budget: BudgetConfig
    paths: PathsConfig
    dry_run: DryRunConfig

    # populated by the loader, not read from YAML directly
    config_dir: Path = Field(default=REPO_ROOT, exclude=True)

    @field_validator("arms")
    @classmethod
    def _exactly_three_arms(cls, v: list[ArmConfig]) -> list[ArmConfig]:
        names = [a.name for a in v]
        if sorted(names) != sorted(ARM_NAMES):
            raise ValueError(
                f"study.arms must contain exactly {ARM_NAMES}, got {names}"
            )
        return v

    @field_validator("doses")
    @classmethod
    def _doses_ascending_positive(cls, v: list[int]) -> list[int]:
        if not v or any(d <= 0 for d in v):
            raise ValueError("doses must be a non-empty list of positive integers")
        if list(v) != sorted(set(v)):
            raise ValueError("doses must be strictly ascending with no duplicates")
        return v

    @model_validator(mode="after")
    def _doses_within_pool(self) -> "StudyConfig":
        if max(self.doses) > self.pool_size_min:
            raise ValueError(
                f"max dose ({max(self.doses)}) exceeds pool_size_min "
                f"({self.pool_size_min}); every arm must supply at least "
                "pool_size_min images"
            )
        return self

    @property
    def n_cells(self) -> int:
        return len(self.arms) * len(self.doses) * self.seeds_per_cell

    def arm(self, name: str) -> ArmConfig:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(name)

    def path(self, key: str) -> Path:
        # paths.* and arms[].raw_dir are repo-root-relative (e.g. "data/manifest");
        # only the sub-config references (training_config, etc.) are relative to
        # config_dir, since those files live alongside study.yaml.
        return _resolve(REPO_ROOT, getattr(self.paths, key))

    def raw_dir(self, arm_name: str) -> Path:
        return _resolve(REPO_ROOT, self.arm(arm_name).raw_dir)

    def eval_prompts_file(self) -> Path:
        return _resolve(REPO_ROOT, self.eval.prompts_file)

    def resolve_repo_path(self, relative: str) -> Path:
        """Resolve any repo-root-relative path string from a sub-config
        (e.g. rating_app.yaml's db_path) the same way paths.* are resolved.
        """
        return _resolve(REPO_ROOT, relative)

    def with_dry_run_paths(self, subdir: str = "_dryrun") -> "StudyConfig":
        """A copy whose every output path is redirected under `outputs/<subdir>/`.

        Dry runs must not be able to touch real artifacts. Without this, a
        dry run after a real run sees a different expected image count (it
        uses fewer prompts), judges finished checkpoints incomplete, and
        rewrites their manifests with placeholder rows — destroying results
        that cost real GPU money. Isolating the paths makes that structurally
        impossible rather than relying on every call site to be careful.
        """
        outputs = Path(self.paths.outputs_dir) / subdir
        redirected = PathsConfig(
            manifest_dir=self.paths.manifest_dir,  # inputs, read-only here
            captions_dir=self.paths.captions_dir,
            outputs_dir=str(outputs),
            runs_dir=str(outputs / "runs"),
            generations_dir=str(outputs / "generations"),
            ratings_dir=str(outputs / "ratings"),
            analysis_dir=str(outputs / "analysis"),
        )
        clone = self.model_copy(update={"paths": redirected})
        clone.config_dir = self.config_dir
        return clone

    def cell_seed(self, arm: str, dose: int, seed_index: int) -> int:
        """Deterministic per-cell seed derived from the study's seed_root.

        Same (arm, dose, seed_index) always yields the same seed, independent
        of run order or machine — this is what makes the sweep reproducible.
        """
        key = f"{self.seed_root}:{arm}:{dose}:{seed_index}"
        return stable_int_hash(key) % (2**31 - 1)


# --------------------------------------------------------------------------
# training.yaml
# --------------------------------------------------------------------------


class LoraConfig(BaseModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1, default=0.0)
    target_modules: list[str]


class OptimizerConfig(BaseModel):
    lr: float = Field(gt=0)
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = Field(ge=0, default=0)
    weight_decay: float = Field(ge=0, default=0.0)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


class ValidationConfig(BaseModel):
    every_n_steps: int = Field(gt=0, default=250)
    num_images: int = Field(gt=0, default=2)
    prompt: str

    @field_validator("prompt")
    @classmethod
    def _prompt_is_neutral(cls, v: str) -> str:
        # The sample prompt is subject to the same rule as the eval prompts:
        # naming the subject would defeat the thing the study measures.
        if not v.strip():
            raise ValueError("validation prompt must not be blank")
        return v


class TrainingConfig(BaseModel):
    base_model: str
    lora: LoraConfig
    optimizer: OptimizerConfig
    resolution: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_train_steps: int = Field(gt=0)
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    gradient_checkpointing: bool = True
    seed: int
    checkpointing_steps: int = Field(gt=0)
    caption_dropout_rate: float = Field(ge=0, lt=1, default=0.0)
    validation: ValidationConfig

    def validation_epochs(self, n_images: int) -> int:
        """Convert the configured step interval into the epoch interval the
        vendored trainer actually accepts (it has no --validation_steps).
        """
        effective_batch = self.batch_size * self.gradient_accumulation_steps
        steps_per_epoch = max(1, n_images // effective_batch)
        return max(1, round(self.validation.every_n_steps / steps_per_epoch))


# --------------------------------------------------------------------------
# captioning.yaml
# --------------------------------------------------------------------------


class CaptioningConfig(BaseModel):
    scheme: str
    templates: list[str] = Field(min_length=1)
    banned_terms: list[str] = Field(default_factory=list)

    @field_validator("templates")
    @classmethod
    def _templates_clean(cls, v: list[str]) -> list[str]:
        for t in v:
            if not t.strip():
                raise ValueError("caption templates must not be blank")
        return v


# --------------------------------------------------------------------------
# rating_app.yaml
# --------------------------------------------------------------------------


class ExposureSurveyConfig(BaseModel):
    question: str
    options: list[str] = Field(min_length=2)


class ConsentConfig(BaseModel):
    title: str
    body: str
    can_withdraw_notice: str


class PairWeightsConfig(BaseModel):
    adjacent_dose_same_arm: float = Field(ge=0)
    cross_arm_same_dose: float = Field(ge=0)
    baseline_vs_any: float = Field(ge=0)
    other: float = Field(ge=0)

    @model_validator(mode="after")
    def _positive_total(self) -> "PairWeightsConfig":
        if self.adjacent_dose_same_arm + self.cross_arm_same_dose + self.baseline_vs_any + self.other <= 0:
            raise ValueError("pair_weights must sum to something positive")
        return self


class CalibrationConfig(BaseModel):
    # Probability of a "correct" answer by pure chance, used to rescale raw
    # salted-pair accuracy into a 0..1 rater weight (chance -> 0, perfect -> 1).
    chance_rate: float = Field(gt=0, lt=1)
    # Raters with fewer than this many answered calibration pairs get
    # weight 0 in the weighted analysis (not enough evidence to trust them),
    # but are still included in the unweighted analysis.
    min_pairs_for_weight: int = Field(gt=0)


class RatingAppConfig(BaseModel):
    db_path: str
    salted_fraction: float = Field(ge=0, lt=1)
    pair_weights: PairWeightsConfig
    exposure_survey: ExposureSurveyConfig
    consent: ConsentConfig
    rating_options: list[str] = Field(min_length=2)
    question_text: str
    calibration: CalibrationConfig

    @field_validator("rating_options")
    @classmethod
    def _known_options(cls, v: list[str]) -> list[str]:
        if sorted(v) != sorted({"A", "B", "BOTH", "NEITHER"}):
            raise ValueError('rating_options must be exactly ["A", "B", "BOTH", "NEITHER"]')
        return v


# --------------------------------------------------------------------------
# runtime.yaml
# --------------------------------------------------------------------------


class ExecutionConfig(BaseModel):
    # Serial by default. Cells are compared against each other, so running one
    # at a time on one GPU is both the simplest thing to reason about and the
    # easiest to keep hardware-identical. Raise this only with several GPUs
    # in the same box.
    parallel_cells: int = Field(gt=0, default=1)
    # Up-front estimate of one cell's GPU time, used to reserve budget before
    # launching. Reservation is what stops parallel workers from each seeing
    # "nothing spent yet" and collectively blowing past the cap. Overestimating
    # is safe; the ledger switches to observed actuals after the first cell.
    estimated_gpu_hours_per_cell: float = Field(gt=0, default=0.5)
    # Deadlines for the two subprocesses a cell runs. Without them a wedged
    # child stalls the whole sweep for as long as the box stays up, billing
    # the whole time, and the per-cell sanity checks cannot help because they
    # only run once the subprocess returns. Defaults are deliberately loose
    # placeholders — tighten both to roughly 3x what the smoke cell actually
    # takes, or a hang costs most of a day before it trips.
    training_timeout_hours: float = Field(gt=0, default=6.0)
    generation_timeout_hours: float = Field(gt=0, default=2.0)
    # Grace period between SIGTERM and SIGKILL when a deadline is hit.
    kill_grace_seconds: float = Field(gt=0, default=30.0)

    @model_validator(mode="after")
    def _deadlines_exceed_the_estimate(self) -> "ExecutionConfig":
        """A deadline below the expected runtime would kill every healthy cell.

        Cheap to typo (hours vs minutes) and expensive to discover: the sweep
        would run to completion, report 45 failures, and bill for all of them.
        """
        for name, hours in (
            ("training_timeout_hours", self.training_timeout_hours),
            ("generation_timeout_hours", self.generation_timeout_hours),
        ):
            if hours <= self.estimated_gpu_hours_per_cell:
                raise ValueError(
                    f"{name}={hours}h is below estimated_gpu_hours_per_cell="
                    f"{self.estimated_gpu_hours_per_cell}h — every cell would be killed mid-run"
                )
        return self


class PricingConfig(BaseModel):
    # Whatever you actually accepted on the marketplace. Spend tracking is only
    # as honest as this number.
    usd_per_gpu_hour: float = Field(gt=0)
    gpu_model: str = ""


class HardwareConfig(BaseModel):
    # Every cell must run on the same GPU and library stack: mixed hardware
    # puts a confound inside the comparison the study is built on.
    enforce_consistency: bool = True


class TickerConfig(BaseModel):
    refresh_seconds: float = Field(gt=0, default=10.0)
    enabled: bool = True


class SanityConfig(BaseModel):
    min_image_bytes: int = Field(gt=0, default=10_000)
    # A collapsed LoRA renders flat frames; near-zero pixel variance catches it.
    min_pixel_std: float = Field(ge=0, default=2.0)
    max_identical_fraction: float = Field(gt=0, le=1.0, default=0.9)
    min_mean_luminance: float = Field(ge=0, default=2.0)
    max_mean_luminance: float = Field(gt=0, default=253.0)
    min_checkpoint_bytes: int = Field(gt=0, default=1_000)


class NotificationConfig(BaseModel):
    backend: Literal["none", "ntfy", "discord"] = "none"
    # Read only from these environment variables. A webhook URL is a secret and
    # must never live in a committed config file.
    ntfy_topic_env: str = "DIGITAL_GHOST_NTFY_TOPIC"
    discord_webhook_env: str = "DIGITAL_GHOST_DISCORD_WEBHOOK"
    notify_on: list[str] = Field(default_factory=lambda: ["failure", "completion"])


class RuntimeConfig(BaseModel):
    """How the sweep executes on the box you rented.

    Replaces the old provider abstraction: with a single rented machine there
    is no remote job API to talk to, so cells are plain local subprocesses and
    what remains worth configuring is concurrency, price, and monitoring.
    """

    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    pricing: PricingConfig
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    ticker: TickerConfig = Field(default_factory=TickerConfig)
    sanity: SanityConfig = Field(default_factory=SanityConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)


# --------------------------------------------------------------------------
# eval_prompts_source.yaml
# --------------------------------------------------------------------------


class EvalPromptSpec(BaseModel):
    id: str
    tier: Literal["near", "mid", "far"]
    text: str

    @field_validator("text")
    @classmethod
    def _no_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt text must not be blank")
        return v


class EvalPromptsSource(BaseModel):
    prompts: list[EvalPromptSpec]

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalPromptsSource":
        ids = [p.id for p in self.prompts]
        if len(ids) != len(set(ids)):
            raise ValueError("eval prompt ids must be unique")
        return self

    def by_tier(self) -> dict[str, list[EvalPromptSpec]]:
        out: dict[str, list[EvalPromptSpec]] = {t: [] for t in TIER_NAMES}
        for p in self.prompts:
            out[p.tier].append(p)
        return out


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_study_config(path: str | Path = REPO_ROOT / "configs" / "study.yaml") -> StudyConfig:
    path = Path(path).resolve()
    data = _read_yaml(path)
    cfg = StudyConfig(**data, config_dir=path.parent)
    cfg.config_dir = path.parent
    return cfg


def load_training_config(study: StudyConfig) -> TrainingConfig:
    return TrainingConfig(**_read_yaml(_resolve(study.config_dir, study.training_config)))


def load_captioning_config(study: StudyConfig) -> CaptioningConfig:
    return CaptioningConfig(**_read_yaml(_resolve(study.config_dir, study.captioning_config)))


def load_runtime_config(study: StudyConfig) -> RuntimeConfig:
    return RuntimeConfig(**_read_yaml(_resolve(study.config_dir, study.runtime_config)))


def load_rating_app_config(study: StudyConfig) -> RatingAppConfig:
    return RatingAppConfig(**_read_yaml(_resolve(study.config_dir, study.rating_app_config)))


def load_eval_prompts_source(study: StudyConfig) -> EvalPromptsSource:
    data = _read_yaml(_resolve(study.config_dir, study.eval.prompts_source))
    return EvalPromptsSource(**data)
