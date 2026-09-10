"""FastAPI backend for the pairwise rating app.

Routes are deliberately minimal and never return arm/dose/salted-status to
the client — that information exists only in the database, read later by
analysis/. See rating_app/frontend/ for the mobile-first UI that talks to
this API.
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

from digital_ghost.config import RatingAppConfig, StudyConfig, load_rating_app_config, load_study_config
from digital_ghost.generation.eval_prompts import load_eval_prompts
from digital_ghost.rating_app.backend.db import get_session
from digital_ghost.rating_app.backend.models import Pair, Rater, Rating
from digital_ghost.rating_app.backend.pairing import ImageIndex, build_image_index, sample_pair

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class SignupRequest(BaseModel):
    exposure_level: str


class SignupResponse(BaseModel):
    rater_id: str


class PairResponse(BaseModel):
    pair_id: str
    prompt_text: str
    question: str
    options: list[str]
    image_a_url: str
    image_b_url: str


class RateRequest(BaseModel):
    rater_id: str
    pair_id: str
    choice: str
    response_ms: Optional[int] = None


class WithdrawRequest(BaseModel):
    rater_id: str


def create_app(study: Optional[StudyConfig] = None, rating_cfg: Optional[RatingAppConfig] = None) -> FastAPI:
    study = study or load_study_config()
    rating_cfg = rating_cfg or load_rating_app_config(study)
    db_path = study.resolve_repo_path(rating_cfg.db_path)

    app = FastAPI(title="Digital Ghost Rating App")
    app.state.study = study
    app.state.rating_cfg = rating_cfg
    app.state.db_path = db_path
    app.state.image_index = None  # built lazily so the API starts even pre-generation
    app.state.prompts_by_id = None

    def get_db() -> Session:
        with get_session(db_path) as session:
            yield session

    def get_index() -> ImageIndex:
        if app.state.image_index is None:
            app.state.image_index = build_image_index(study)
            data = load_eval_prompts(study)
            app.state.prompts_by_id = {p["id"]: p for p in data["prompts"]}
        return app.state.image_index

    @app.get("/api/consent")
    def get_consent():
        return rating_cfg.consent.model_dump()

    @app.get("/api/exposure-survey")
    def get_exposure_survey():
        return rating_cfg.exposure_survey.model_dump()

    @app.post("/api/signup", response_model=SignupResponse)
    def signup(req: SignupRequest, session: Session = Depends(get_db)):
        if req.exposure_level not in rating_cfg.exposure_survey.options:
            raise HTTPException(422, f"exposure_level must be one of {rating_cfg.exposure_survey.options}")
        rater = Rater(id=str(uuid.uuid4()), exposure_level=req.exposure_level)
        session.add(rater)
        session.commit()
        return SignupResponse(rater_id=rater.id)

    @app.post("/api/withdraw")
    def withdraw(req: WithdrawRequest, session: Session = Depends(get_db)):
        rater = session.get(Rater, req.rater_id)
        if not rater:
            raise HTTPException(404, "unknown rater_id")
        from datetime import datetime, timezone

        rater.withdrawn_at = datetime.now(timezone.utc)
        session.add(rater)
        session.commit()
        return {"ok": True}

    @app.get("/api/next-pair", response_model=PairResponse)
    def next_pair(rater_id: str, session: Session = Depends(get_db)):
        rater = session.get(Rater, rater_id)
        if not rater:
            raise HTTPException(404, "unknown rater_id — sign up first")
        if rater.withdrawn_at is not None:
            raise HTTPException(410, "this rater has withdrawn from the study")

        index = get_index()
        pair = sample_pair(index, study, rating_cfg, app.state.prompts_by_id, rng=random.Random())
        session.add(pair)
        session.commit()
        return PairResponse(
            pair_id=pair.id,
            prompt_text=pair.prompt_text,
            question=rating_cfg.question_text,
            options=rating_cfg.rating_options,
            image_a_url=f"/api/image/{pair.id}/a",
            image_b_url=f"/api/image/{pair.id}/b",
        )

    @app.get("/api/image/{pair_id}/{slot}")
    def get_image(pair_id: str, slot: str, session: Session = Depends(get_db)):
        if slot not in ("a", "b"):
            raise HTTPException(404)
        pair = session.get(Pair, pair_id)
        if not pair:
            raise HTTPException(404, "unknown pair_id")
        path = pair.image_a_path if slot == "a" else pair.image_b_path
        if not Path(path).exists():
            raise HTTPException(404, "image file missing on disk")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/rate")
    def rate(req: RateRequest, session: Session = Depends(get_db)):
        rater = session.get(Rater, req.rater_id)
        if not rater:
            raise HTTPException(404, "unknown rater_id")
        if rater.withdrawn_at is not None:
            raise HTTPException(410, "this rater has withdrawn from the study")
        pair = session.get(Pair, req.pair_id)
        if not pair:
            raise HTTPException(404, "unknown pair_id")
        if req.choice not in rating_cfg.rating_options:
            raise HTTPException(422, f"choice must be one of {rating_cfg.rating_options}")

        rating = Rating(
            rater_id=req.rater_id, pair_id=req.pair_id, choice=req.choice, response_ms=req.response_ms
        )
        session.add(rating)
        session.commit()
        return {"ok": True}

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


# For `uvicorn digital_ghost.rating_app.backend.app:app`. Tests and the CLI
# should call create_app(study, rating_cfg) directly instead of importing
# this module-level instance, so they aren't tied to the real repo's config.
app = create_app()
