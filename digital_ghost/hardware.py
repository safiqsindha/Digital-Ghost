"""Hardware and library fingerprinting for the sweep.

The study's claims come from comparing its 45 cells against each other, so
every cell has to be trained on the same GPU and the same library versions.
A GPU swap or a `pip install -U` halfway through the sweep puts a confound
*inside* the comparison rather than beside it, and no amount of downstream
analysis can separate it from the effect being measured. The sweep therefore
captures a fingerprint at the start, stores one alongside every cell's run
metadata, and refuses to keep going if the machine changes underneath it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

UNKNOWN = "unknown"

# The fields that define "the same machine". `captured_at` is deliberately
# absent: two captures of one unchanged box, minutes apart, must produce the
# same digest or every comparison against the reference would report a change.
IDENTITY_FIELDS = (
    "gpu_name",
    "gpu_count",
    "driver_version",
    "cuda_version",
    "torch_version",
    "diffusers_version",
    "peft_version",
    "transformers_version",
)

_NVIDIA_SMI_TIMEOUT_S = 10.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HardwareFingerprint:
    gpu_name: str
    gpu_count: int
    driver_version: str
    cuda_version: str
    torch_version: str
    diffusers_version: str
    peft_version: str
    transformers_version: str
    captured_at: str = field(default_factory=_utc_now)
    digest: str = ""

    def __post_init__(self) -> None:
        # Always recomputed rather than trusted from the input, so a digest
        # read back from disk can never disagree with the fields beside it.
        self.digest = self.compute_digest()

    def compute_digest(self) -> str:
        payload = json.dumps(
            {name: getattr(self, name) for name in IDENTITY_FIELDS}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareFingerprint":
        # Fingerprints are stored inside larger per-cell metadata blobs, so
        # accept a superset of keys instead of choking on the neighbours.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def describe(self) -> str:
        return (
            f"{self.gpu_count}x {self.gpu_name} (driver {self.driver_version}, "
            f"CUDA {self.cuda_version}), torch {self.torch_version}, "
            f"diffusers {self.diffusers_version}, peft {self.peft_version}, "
            f"transformers {self.transformers_version}"
        )


class HardwareChangedError(Exception):
    pass


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return UNKNOWN


def _query_nvidia_smi() -> list[tuple[str, str]]:
    """One (gpu name, driver version) pair per visible GPU, empty on any failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def capture() -> HardwareFingerprint:
    """Fingerprint the current machine. Never raises.

    A CPU-only box (dry runs, CI, a laptop) yields gpu_name "cpu" rather than
    an error, so the whole pipeline stays exercisable without a GPU.
    """
    # Imported here, not at module scope: this module is pulled in by the CLI
    # and by metadata readers that must stay importable without the training
    # stack installed.
    try:
        import torch
    except ImportError:
        torch = None

    gpu_name = "cpu"
    gpu_count = 0
    cuda_version = UNKNOWN
    smi_rows = _query_nvidia_smi()
    # torch never exposes the NVIDIA driver version, so it comes from
    # nvidia-smi even when torch supplies everything else.
    driver_version = smi_rows[0][1] if smi_rows else UNKNOWN

    if torch is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or UNKNOWN
    elif smi_rows:
        gpu_name = smi_rows[0][0]
        gpu_count = len(smi_rows)

    return HardwareFingerprint(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        driver_version=driver_version,
        cuda_version=cuda_version,
        torch_version=_package_version("torch"),
        diffusers_version=_package_version("diffusers"),
        peft_version=_package_version("peft"),
        transformers_version=_package_version("transformers"),
    )


def check_consistency(
    current: HardwareFingerprint,
    previous: HardwareFingerprint,
    allow_change: bool = False,
) -> list[str]:
    """Compare against the sweep's reference fingerprint.

    Returns one human-readable line per differing field. Raises
    HardwareChangedError when there are differences and `allow_change` is
    False; with `allow_change` True the differences come back so the caller
    can record them loudly instead.
    """
    differences = [
        f"{name}: {getattr(previous, name)} -> {getattr(current, name)}"
        for name in IDENTITY_FIELDS
        if getattr(previous, name) != getattr(current, name)
    ]
    if differences and not allow_change:
        raise HardwareChangedError(
            "hardware or library versions changed mid-sweep:\n"
            + "\n".join(f"  - {d}" for d in differences)
            + "\n\nEvery cell in this study is compared against every other "
            "cell, so all of them must be trained on the same GPU with the "
            "same library versions. Continuing here would bake the change "
            "above into the results as a confound that no later analysis can "
            "separate from the effect being measured.\n"
            "Fix: restore the original machine and library versions, or start "
            "a fresh study directory.\n"
            "Escape hatch: pass --allow-hardware-change to continue anyway "
            "and have the difference recorded in the run metadata."
        )
    return differences


def save_reference(path: str | Path, fp: HardwareFingerprint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written atomically: a sweep interrupted mid-write would otherwise leave
    # a torn reference that fails to parse on resume, stranding the run.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(fp.to_dict(), indent=2, sort_keys=True))
    os.replace(tmp, path)


def load_reference(path: str | Path) -> HardwareFingerprint | None:
    """The stored reference fingerprint, or None if the sweep has none yet."""
    path = Path(path)
    if not path.exists():
        return None
    return HardwareFingerprint.from_dict(json.loads(path.read_text()))
