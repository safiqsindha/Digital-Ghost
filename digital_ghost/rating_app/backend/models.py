"""SQLModel tables for the rating app.

Raters are never shown arm/dose — those columns exist only for analysis to
read after the fact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Rater(SQLModel, table=True):
    id: str = Field(primary_key=True)  # random token, not linked to any real identity
    exposure_level: str
    consented_at: datetime = Field(default_factory=_now)
    created_at: datetime = Field(default_factory=_now)
    withdrawn_at: Optional[datetime] = None


class Pair(SQLModel, table=True):
    id: str = Field(primary_key=True)
    # The rater this pair was drawn for. Pairs are sampled per-request, so a
    # submission naming a pair served to somebody else is not a legitimate
    # rating and is rejected.
    served_to_rater_id: Optional[str] = Field(default=None, foreign_key="rater.id", index=True)
    prompt_id: str
    prompt_text: str
    tier: str
    gen_seed: int

    # slot A / slot B — position is randomized at creation time so neither
    # slot systematically corresponds to a particular arm
    image_a_path: str
    checkpoint_a: str
    arm_a: Optional[str] = None
    dose_a: Optional[int] = None
    seed_index_a: Optional[int] = None

    image_b_path: str
    checkpoint_b: str
    arm_b: Optional[str] = None
    dose_b: Optional[int] = None
    seed_index_b: Optional[int] = None

    category: str  # adjacent_dose_same_arm | cross_arm_same_dose | baseline_vs_any | other

    is_salted: bool = False
    salted_answer: Optional[str] = None  # "A" | "B" (never BOTH/NEITHER — see pairing.py)
    salted_difficulty: Optional[str] = None  # easy | medium | hard

    created_at: datetime = Field(default_factory=_now)


class Rating(SQLModel, table=True):
    # One rating per rater per pair: a retry or double-tap must not turn into
    # two observations of the same comparison in the Davidson fit.
    __table_args__ = (UniqueConstraint("rater_id", "pair_id", name="uq_rating_rater_pair"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    rater_id: str = Field(foreign_key="rater.id", index=True)
    pair_id: str = Field(foreign_key="pair.id", index=True)
    choice: str  # A | B | BOTH | NEITHER
    response_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=_now)
