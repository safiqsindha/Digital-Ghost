from __future__ import annotations

from fastapi.testclient import TestClient

from digital_ghost.config import load_rating_app_config
from digital_ghost.rating_app.backend.app import create_app


def _client(study):
    rating_cfg = load_rating_app_config(study)
    app = create_app(study, rating_cfg)
    return TestClient(app), rating_cfg


def test_consent_and_survey_are_config_driven(trained_and_generated_study):
    client, rating_cfg = _client(trained_and_generated_study)
    r = client.get("/api/consent")
    assert r.status_code == 200
    assert r.json()["title"] == rating_cfg.consent.title

    r = client.get("/api/exposure-survey")
    assert r.status_code == 200
    assert r.json()["options"] == rating_cfg.exposure_survey.options


def test_signup_rejects_unknown_exposure_level(trained_and_generated_study):
    client, _ = _client(trained_and_generated_study)
    r = client.post("/api/signup", json={"exposure_level": "not-a-real-level"})
    assert r.status_code == 422


def test_full_rating_flow_is_blind_to_arm_and_dose(trained_and_generated_study):
    client, rating_cfg = _client(trained_and_generated_study)

    r = client.post("/api/signup", json={"exposure_level": rating_cfg.exposure_survey.options[0]})
    rater_id = r.json()["rater_id"]

    for _ in range(30):
        r = client.get("/api/next-pair", params={"rater_id": rater_id})
        assert r.status_code == 200
        pair = r.json()

        # blind API surface: never arm/dose/salted info
        assert set(pair.keys()) == {"pair_id", "prompt_text", "question", "options", "image_a_url", "image_b_url"}
        assert pair["question"] == rating_cfg.question_text
        assert pair["options"] == rating_cfg.rating_options

        img_a = client.get(pair["image_a_url"])
        img_b = client.get(pair["image_b_url"])
        assert img_a.status_code == 200
        assert img_b.status_code == 200

        rr = client.post(
            "/api/rate",
            json={"rater_id": rater_id, "pair_id": pair["pair_id"], "choice": "BOTH", "response_ms": 500},
        )
        assert rr.status_code == 200


def test_withdrawal_blocks_further_pairs_and_ratings(trained_and_generated_study):
    client, rating_cfg = _client(trained_and_generated_study)
    r = client.post("/api/signup", json={"exposure_level": rating_cfg.exposure_survey.options[0]})
    rater_id = r.json()["rater_id"]

    r = client.get("/api/next-pair", params={"rater_id": rater_id})
    pair = r.json()

    client.post("/api/withdraw", json={"rater_id": rater_id})

    assert client.get("/api/next-pair", params={"rater_id": rater_id}).status_code == 410
    assert client.post(
        "/api/rate", json={"rater_id": rater_id, "pair_id": pair["pair_id"], "choice": "A"}
    ).status_code == 410


def test_unknown_rater_is_rejected(trained_and_generated_study):
    client, _ = _client(trained_and_generated_study)
    assert client.get("/api/next-pair", params={"rater_id": "ghost"}).status_code == 404


def test_salted_pairs_appear_and_are_standard_vs_baseline(trained_and_generated_study):
    from sqlmodel import Session, select

    from digital_ghost.config import load_rating_app_config
    from digital_ghost.rating_app.backend.db import get_engine
    from digital_ghost.rating_app.backend.models import Pair

    study = trained_and_generated_study
    client, rating_cfg = _client(study)
    r = client.post("/api/signup", json={"exposure_level": rating_cfg.exposure_survey.options[0]})
    rater_id = r.json()["rater_id"]

    for _ in range(60):
        client.get("/api/next-pair", params={"rater_id": rater_id})

    db_path = study.resolve_repo_path(rating_cfg.db_path)
    with Session(get_engine(db_path)) as session:
        pairs = session.exec(select(Pair)).all()
    salted = [p for p in pairs if p.is_salted]
    assert salted, "expected at least one salted pair out of 60 draws"
    for p in salted:
        arms = {p.arm_a, p.arm_b}
        assert None in arms  # baseline slot
        assert "standard" in arms  # never "meme" — see pairing.py docstring
        assert p.salted_answer in ("A", "B")
        assert p.salted_difficulty in ("easy", "medium", "hard")
