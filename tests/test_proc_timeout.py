"""A wedged child must die on a deadline, and take its children with it.

This is the failure mode the sweep is least able to notice on its own: the
per-cell sanity checks only run once the subprocess returns, so a training
process that hangs produces no progress, no failure and no next cell, while a
rented GPU bills by the hour. These tests pin the deadline, the escalation
from SIGTERM to SIGKILL, and the process-group kill that stops orphaned
workers sitting on VRAM.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from digital_ghost.proc import SubprocessTimeout, run_subprocess_streaming


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_returns_normally_when_the_child_finishes_in_time(tmp_path):
    log = tmp_path / "cell.log"
    rc = run_subprocess_streaming(
        _python("print('done')"), log, os.environ.copy(), "quick", timeout_s=60
    )
    assert rc == 0
    assert "done" in log.read_text()


def test_nonzero_exit_is_returned_not_raised(tmp_path):
    """A failing cell is data, not an exception — the sweep records it and
    carries on to the other 44."""
    rc = run_subprocess_streaming(
        _python("raise SystemExit(3)"), tmp_path / "cell.log", os.environ.copy(), "boom",
        timeout_s=60,
    )
    assert rc == 3


def test_no_timeout_means_no_deadline(tmp_path):
    """The default stays backwards compatible: callers that pass nothing get
    the old unbounded wait."""
    rc = run_subprocess_streaming(
        _python("import time; time.sleep(0.2)"), tmp_path / "cell.log", os.environ.copy(), "sleepy"
    )
    assert rc == 0


def test_hung_child_is_killed_and_reported(tmp_path):
    log = tmp_path / "cell.log"
    started = time.monotonic()
    with pytest.raises(SubprocessTimeout) as excinfo:
        run_subprocess_streaming(
            _python("import time; time.sleep(300)"), log, os.environ.copy(), "train wedged",
            timeout_s=1.0, kill_grace_s=5,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 30, "the deadline did not actually cut the wait short"
    assert excinfo.value.timeout_s == 1.0
    assert "train wedged" in str(excinfo.value)
    # The log is the only record an unattended run leaves behind, so the
    # reason has to be in it rather than only in the exception.
    text = log.read_text()
    assert "exceeded its" in text
    assert "SIGTERM" in text


def test_a_child_ignoring_sigterm_is_escalated_to_sigkill(tmp_path):
    """The trainer runs under accelerate, and a shutdown path that only ever
    asks politely can be stonewalled forever."""
    stubborn = _python(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ignoring sigterm', flush=True)\n"
        "time.sleep(300)\n"
    )
    log = tmp_path / "cell.log"
    started = time.monotonic()
    with pytest.raises(SubprocessTimeout):
        run_subprocess_streaming(
            stubborn, log, os.environ.copy(), "stubborn", timeout_s=1.0, kill_grace_s=2.0
        )
    assert time.monotonic() - started < 30

    text = log.read_text()
    assert "SIGTERM" in text and "SIGKILL" in text, "never escalated past SIGTERM"


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_the_whole_process_tree_dies_not_just_the_direct_child(tmp_path):
    """Signalling only the direct child would leave `accelerate`'s workers
    alive holding VRAM, and the next cell would fail to allocate on a GPU that
    looks free. The grandchild here stands in for those workers.
    """
    pid_file = tmp_path / "grandchild.pid"
    parent = _python(
        "import subprocess, sys, time\n"
        f"kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(kid.pid))\n"
        "time.sleep(300)\n"
    )

    with pytest.raises(SubprocessTimeout):
        run_subprocess_streaming(
            parent, tmp_path / "cell.log", os.environ.copy(), "tree", timeout_s=2.0, kill_grace_s=3.0
        )

    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not _alive(grandchild):
            break
        time.sleep(0.2)
    assert not _alive(grandchild), f"grandchild {grandchild} survived — it would still hold VRAM"


def _alive(pid: int) -> bool:
    """True only for a live process. A killed child of ours may linger as a
    zombie until reaped, which is dead for our purposes."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - ps missing is not the thing under test
        return True
    return bool(state) and not state.startswith("Z")


def test_killing_the_group_does_not_signal_the_sweep_itself(tmp_path):
    """The kill is a killpg, so the child must be in its own process group.

    If it inherited ours, the same killpg that cleans up a wedged trainer
    would take down the sweep orchestrating it — turning one bad cell into a
    dead 27-hour run.
    """
    received: list[int] = []
    previous = signal.signal(signal.SIGTERM, lambda *_: received.append(1))
    try:
        with pytest.raises(SubprocessTimeout):
            run_subprocess_streaming(
                _python("import time; time.sleep(300)"), tmp_path / "cell.log",
                os.environ.copy(), "isolated", timeout_s=1.0, kill_grace_s=2.0,
            )
        time.sleep(0.5)
        assert not received, "the parent got its own SIGTERM — child shared our process group"
    finally:
        signal.signal(signal.SIGTERM, previous)
