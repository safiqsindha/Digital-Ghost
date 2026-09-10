"""Loads ratings out of the rating app's SQLite database into a flat table
for analysis. Everything downstream (rater weights, Davidson fits, plots)
works off this one DataFrame.
"""

from __future__ import annotations

import pandas as pd
from sqlmodel import Session, select

from digital_ghost.config import StudyConfig
from digital_ghost.rating_app.backend.db import get_engine
from digital_ghost.rating_app.backend.models import Pair, Rater, Rating


def load_ratings(study: StudyConfig, db_path=None) -> pd.DataFrame:
    from digital_ghost.config import load_rating_app_config

    if db_path is None:
        rating_cfg = load_rating_app_config(study)
        db_path = study.resolve_repo_path(rating_cfg.db_path)

    with Session(get_engine(db_path)) as session:
        ratings = session.exec(select(Rating)).all()
        pairs = {p.id: p for p in session.exec(select(Pair)).all()}
        raters = {r.id: r for r in session.exec(select(Rater)).all()}

    rows = []
    for r in ratings:
        pair = pairs.get(r.pair_id)
        rater = raters.get(r.rater_id)
        if pair is None or rater is None:
            continue
        # Withdrawing consent must remove a participant from the analysis,
        # not merely stop serving them new pairs. Anything already submitted
        # is excluded here so it cannot reach the published curve.
        if rater.withdrawn_at is not None:
            continue
        rows.append(
            {
                "rating_id": r.id,
                "rater_id": r.rater_id,
                "exposure_level": rater.exposure_level,
                "pair_id": r.pair_id,
                "choice": r.choice,
                "response_ms": r.response_ms,
                "created_at": r.created_at,
                "prompt_id": pair.prompt_id,
                "tier": pair.tier,
                "gen_seed": pair.gen_seed,
                "item_a": pair.checkpoint_a,
                "arm_a": pair.arm_a,
                "dose_a": pair.dose_a,
                "seed_index_a": pair.seed_index_a,
                "item_b": pair.checkpoint_b,
                "arm_b": pair.arm_b,
                "dose_b": pair.dose_b,
                "seed_index_b": pair.seed_index_b,
                "category": pair.category,
                "is_salted": pair.is_salted,
                "salted_answer": pair.salted_answer,
                "salted_difficulty": pair.salted_difficulty,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["outcome"] = df["choice"].map({"A": "i", "B": "j", "BOTH": "tie", "NEITHER": "tie"})
    # An unrecognised choice maps to NaN, which the Davidson likelihood would
    # otherwise silently route down the "tie" branch and quietly bias the tie
    # parameter. Fail instead — this can only happen via a direct DB write or
    # a schema change, both of which warrant a look.
    if df["outcome"].isna().any():
        bad = sorted(df.loc[df["outcome"].isna(), "choice"].unique())
        raise ValueError(
            f"unrecognised rating choice(s) in the database: {bad}. "
            "Expected one of A / B / BOTH / NEITHER."
        )
    return df


def to_comparisons(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    return list(zip(df["item_a"], df["item_b"], df["outcome"]))


def salted_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_salted"]].copy()


def substantive_only(df: pd.DataFrame) -> pd.DataFrame:
    """Non-calibration rows — what the dose-response fit is estimated on."""
    return df[~df["is_salted"]].copy()
