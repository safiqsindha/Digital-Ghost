"""The mock study is the only way to exercise the rating app before any GPU spend.

It is a script rather than library code, so nothing else imports it and it
rots silently — it was left referencing `provider.yaml` for the whole of the
runtime-config refactor and only broke when someone actually ran it, which is
exactly the moment you want it working: piloting the rating UX with real
people before committing to a day of GPU time.

These build a deliberately tiny mock (2 prompts at 64px, ~2s) and drive the
real app against it.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from digital_ghost.config import REPO_ROOT

BUILDER = REPO_ROOT / "scripts" / "build_mock_study.py"


@pytest.fixture(scope="module")
def mock_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("mock_study")
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--root", str(root), "--fresh",
         "--prompts", "2", "--image-size", "64"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"mock builder failed:\n{result.stdout}\n{result.stderr}"
    return root


def test_builder_runs_and_writes_a_usable_config(mock_study):
    from digital_ghost.config import load_study_config

    cfg = mock_study / "study.yaml"
    assert cfg.exists()
    study = load_study_config(cfg)
    assert study.n_cells == 45


def test_builder_copies_every_sub_config_it_references(mock_study):
    """The failure that actually happened: a renamed config left the script
    copying a file that no longer existed."""
    for fn in ("training.yaml", "captioning.yaml", "runtime.yaml",
               "eval_prompts_source.yaml", "rating_app.yaml"):
        assert (mock_study / fn).exists(), f"{fn} was not copied into the mock study"


def test_mock_writes_nothing_into_the_real_repo(mock_study):
    """Mock images sitting in the real raw pool would be picked up by a later
    `digital-ghost ingest` and silently trained on — synthetic shapes landing
    in a study about real photographs."""
    for arm in ("standard", "meme", "control"):
        real = REPO_ROOT / "data" / "raw" / arm
        if real.exists():
            assert not list(real.glob("*.png")), f"mock images leaked into {real}"


def test_app_serves_a_blind_pair_against_the_mock(mock_study):
    """End-to-end through the real app: consent, signup, a pair, both images.

    The pair payload must not carry arm, dose or file path — a rater who can
    see which side came from the meme arm is not rating blind, and the whole
    comparison would be worthless.
    """
    from fastapi.testclient import TestClient

    from digital_ghost.config import load_study_config
    from digital_ghost.rating_app.backend.app import create_app

    client = TestClient(create_app(load_study_config(mock_study / "study.yaml")))

    assert client.get("/api/consent").status_code == 200
    survey = client.get("/api/exposure-survey").json()
    rater = client.post("/api/signup", json={"exposure_level": survey["options"][0]}).json()

    pair = client.get(f"/api/next-pair?rater_id={rater['rater_id']}").json()
    assert set(pair) == {
        "pair_id", "prompt_text", "question", "options", "image_a_url", "image_b_url"
    }, f"unexpected keys in the pair payload: {sorted(pair)}"

    blob = json.dumps(pair).lower()
    for leak in ("arm", "dose", "checkpoint", "baseline", "/outputs/", "standard", "meme"):
        assert leak not in blob, f"pair payload leaks {leak!r}: {pair}"

    for url in (pair["image_a_url"], pair["image_b_url"]):
        img = client.get(url)
        assert img.status_code == 200
        assert img.content[:4] == b"\x89PNG"

    rated = client.post("/api/rate", json={
        "rater_id": rater["rater_id"], "pair_id": pair["pair_id"], "choice": "A",
    })
    assert rated.status_code == 200
