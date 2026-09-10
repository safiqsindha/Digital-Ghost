"""Subprocess execution with output streamed to disk.

Shared by the sweep and the standalone generation grid. Kept in its own module
so neither has to import the other.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_subprocess_streaming(cmd: list[str], log_path: Path, env: dict, header: str) -> int:
    """Run a subprocess, appending combined stdout+stderr to a log file.

    Buffering output in memory (subprocess' `capture_output=True`) both wastes
    memory on a thousand-step training run and throws the log away on success
    — which is exactly the log you want when a cell later looks wrong. Writing
    straight to a file also means an SSH drop costs nothing: the record is on
    disk either way.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\n===== {header} =====\n")
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True)
        return proc.wait()
