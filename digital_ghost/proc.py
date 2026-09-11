"""Subprocess execution with output streamed to disk, under a deadline.

Shared by the sweep and the standalone generation grid. Kept in its own module
so neither has to import the other.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


class SubprocessTimeout(RuntimeError):
    """A child outlived its deadline and was killed.

    Carries the elapsed time so the caller can say how long it waited rather
    than just that something went wrong.
    """

    def __init__(self, header: str, timeout_s: float, elapsed_s: float):
        self.header = header
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s
        super().__init__(f"{header} exceeded its {timeout_s / 3600:.2f}h deadline")


def _kill_process_group(proc: subprocess.Popen, grace_s: float, log) -> None:
    """Take down the child and everything it spawned.

    The trainer is not a single process: it runs under `accelerate`, which
    forks workers of its own. Signalling only the direct child would leave
    those workers alive holding VRAM, and the next cell would then fail to
    allocate on a GPU that looks free. The child is started in its own
    session (see below), so its pid doubles as a process-group id and one
    killpg reaches the whole tree.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:  # already reaped
        return

    for sig, label in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return  # gone between the check and the signal
        log.write(f"[digital-ghost] sent {label} to process group {pgid}\n")
        log.flush()
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue  # ignored SIGTERM; escalate


def run_subprocess_streaming(
    cmd: list[str],
    log_path: Path,
    env: dict,
    header: str,
    timeout_s: float | None = None,
    kill_grace_s: float = 30.0,
) -> int:
    """Run a subprocess, appending combined stdout+stderr to a log file.

    Buffering output in memory (subprocess' `capture_output=True`) both wastes
    memory on a thousand-step training run and throws the log away on success
    — which is exactly the log you want when a cell later looks wrong. Writing
    straight to a file also means an SSH drop costs nothing: the record is on
    disk either way.

    With `timeout_s` set, a child that outlives its deadline is killed and
    `SubprocessTimeout` is raised. Without one, a wedged trainer would stall
    the sweep for as long as the machine stays up: no progress, no failure,
    no next cell, and a rented GPU billing by the hour the whole time. The
    per-cell sanity checks cannot catch that, because they only run once the
    subprocess returns.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\n===== {header} =====\n")
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()

        started = time.monotonic()
        # start_new_session puts the child in its own process group so the
        # whole tree can be killed without signalling ourselves. The cost is
        # that Ctrl-C no longer reaches it through the terminal, so the
        # KeyboardInterrupt branch below forwards it by hand.
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True, start_new_session=True
        )
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            log.write(
                f"\n[digital-ghost] {header} exceeded its "
                f"{timeout_s / 3600:.2f}h deadline after {elapsed / 3600:.2f}h — killing\n"
            )
            log.flush()
            _kill_process_group(proc, kill_grace_s, log)
            raise SubprocessTimeout(header, timeout_s, elapsed) from None
        except BaseException:
            # Ctrl-C, or anything else unwinding through here. The child is in
            # its own session and would otherwise be orphaned onto the GPU.
            _kill_process_group(proc, kill_grace_s, log)
            raise
