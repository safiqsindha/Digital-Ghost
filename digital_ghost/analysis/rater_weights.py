"""Per-rater weight derived from performance on salted calibration pairs.

A rater's weight is their salted-pair accuracy rescaled so that
chance-level performance maps to 0 and perfect performance maps to 1:

    weight = clip((accuracy - chance_rate) / (1 - chance_rate), 0, 1)

Raters with fewer than `calibration.min_pairs_for_weight` answered
calibration pairs get weight 0 (not "unknown" — there's no positive
evidence to weight them by, so the conservative choice is to exclude them
from the weighted analysis while still keeping them in the unweighted one).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from digital_ghost.config import CalibrationConfig


@dataclass
class RaterPerformance:
    rater_id: str
    n_calibration: int
    n_correct: int
    accuracy: float
    weight: float
    accuracy_by_difficulty: dict[str, float]


def compute_rater_weights(
    salted_df: pd.DataFrame, calibration: CalibrationConfig
) -> dict[str, RaterPerformance]:
    out: dict[str, RaterPerformance] = {}
    if salted_df.empty:
        return out

    correct = salted_df["choice"] == salted_df["salted_answer"]
    salted_df = salted_df.assign(correct=correct)

    for rater_id, group in salted_df.groupby("rater_id"):
        n = len(group)
        n_correct = int(group["correct"].sum())
        accuracy = n_correct / n if n else 0.0

        by_diff: dict[str, float] = {}
        for diff, dgroup in group.groupby("salted_difficulty"):
            by_diff[str(diff)] = float(dgroup["correct"].mean())

        if n < calibration.min_pairs_for_weight:
            weight = 0.0
        else:
            weight = (accuracy - calibration.chance_rate) / (1 - calibration.chance_rate)
            weight = max(0.0, min(1.0, weight))

        out[rater_id] = RaterPerformance(
            rater_id=rater_id,
            n_calibration=n,
            n_correct=n_correct,
            accuracy=accuracy,
            weight=weight,
            accuracy_by_difficulty=by_diff,
        )
    return out


def weight_column(df: pd.DataFrame, performances: dict[str, RaterPerformance]) -> pd.Series:
    """Per-row weight for `df`, defaulting to 0 for raters with no calibration data at all."""
    return df["rater_id"].map(lambda r: performances[r].weight if r in performances else 0.0)


def performances_to_frame(performances: dict[str, RaterPerformance]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rater_id": p.rater_id,
                "n_calibration": p.n_calibration,
                "n_correct": p.n_correct,
                "accuracy": p.accuracy,
                "weight": p.weight,
                **{f"accuracy_{k}": v for k, v in p.accuracy_by_difficulty.items()},
            }
            for p in performances.values()
        ]
    ).sort_values("rater_id")
