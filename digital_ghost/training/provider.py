"""GPU provider integration — provider-agnostic interface + a local stub.

Real providers (RunPod, Lambda, Vast.ai, ...) submit remote jobs and poll a
remote API. This module defines the interface every provider must satisfy
and ships one concrete implementation, `LocalStubProvider`, which runs jobs
as local threads and bills by wall-clock time. It's what `--dry-run` uses,
and it's also a legitimate choice for single-machine execution on a rented
GPU box you already have shell access to.

To wire in a real remote provider:
  1. Subclass `GPUProvider` and implement submit_job / poll_status /
     get_gpu_hours / cancel against that provider's API.
  2. Read the API key ONLY from `os.environ[self.config.api_key_env]` —
     never from a config file, never hardcoded.
  3. Register it in `get_provider()` below.
  4. Set `provider: <your-provider-name>` in configs/provider.yaml.
No other code in this repo needs to change.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from digital_ghost.config import ProviderConfig

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobHandle:
    job_id: str
    cell_id: str


class ProviderAuthError(Exception):
    pass


class GPUProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = os.environ.get(config.api_key_env)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ProviderAuthError(
                f"{self.config.api_key_env} is not set in the environment. "
                f"Export it before running against provider '{self.config.provider}'."
            )
        return self.api_key

    @abstractmethod
    def submit_job(self, cell_id: str, run_fn: Callable[[], None]) -> JobHandle: ...

    @abstractmethod
    def poll_status(self, handle: JobHandle) -> JobStatus: ...

    @abstractmethod
    def get_gpu_hours(self, handle: JobHandle) -> float: ...

    @abstractmethod
    def get_error(self, handle: JobHandle) -> BaseException | None: ...

    @abstractmethod
    def cancel(self, handle: JobHandle) -> None: ...

    def wait(self, handle: JobHandle, poll_interval_s: float | None = None) -> JobStatus:
        interval = poll_interval_s if poll_interval_s is not None else self.config.poll_interval_s
        while True:
            status = self.poll_status(handle)
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                return status
            time.sleep(interval)


class LocalStubProvider(GPUProvider):
    """Runs each job as a local thread; "GPU-hours" = wall-clock hours.

    Does not require an API key. This is the default (`provider: stub` in
    provider.yaml) until a real backend is wired in.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit_job(self, cell_id: str, run_fn: Callable[[], None]) -> JobHandle:
        job_id = str(uuid.uuid4())
        state = {
            "cell_id": cell_id,
            "status": JobStatus.RUNNING,
            "start": time.monotonic(),
            "end": None,
            "error": None,
        }

        def _target():
            try:
                run_fn()
                state["status"] = JobStatus.SUCCEEDED
            # BaseException, not Exception: a SystemExit or KeyboardInterrupt
            # escaping here would leave status stuck at RUNNING, and wait()
            # would spin on it forever.
            except BaseException as e:  # noqa: BLE001 - surfaced via get_error
                logger.exception("cell %s failed", cell_id)
                state["status"] = JobStatus.FAILED
                state["error"] = e
            finally:
                state["end"] = time.monotonic()

        thread = threading.Thread(target=_target, daemon=True)
        with self._lock:
            self._jobs[job_id] = state
            state["thread"] = thread
        thread.start()
        return JobHandle(job_id=job_id, cell_id=cell_id)

    def poll_status(self, handle: JobHandle) -> JobStatus:
        return self._jobs[handle.job_id]["status"]

    def get_gpu_hours(self, handle: JobHandle) -> float:
        state = self._jobs[handle.job_id]
        end = state["end"] or time.monotonic()
        return (end - state["start"]) / 3600.0

    def get_error(self, handle: JobHandle) -> BaseException | None:
        return self._jobs[handle.job_id]["error"]

    def cancel(self, handle: JobHandle) -> None:
        # Cooperative-only: local threads running training loops are expected
        # to check a cancellation flag; this stub just marks the bookkeeping.
        state = self._jobs[handle.job_id]
        if state["status"] == JobStatus.RUNNING:
            state["status"] = JobStatus.CANCELLED
            state["end"] = time.monotonic()


_REGISTRY: dict[str, type[GPUProvider]] = {
    "stub": LocalStubProvider,
}


def get_provider(config: ProviderConfig) -> GPUProvider:
    cls = _REGISTRY.get(config.provider)
    if cls is None:
        raise NotImplementedError(
            f"no GPUProvider registered for provider={config.provider!r}. "
            f"Known providers: {sorted(_REGISTRY)}. See the module docstring "
            "in digital_ghost/training/provider.py for how to add one."
        )
    return cls(config)
