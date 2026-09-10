"""Full analysis driver: loads ratings, derives rater weights from salted
pairs, fits the Davidson model per breakdown (overall, by prompt tier, by
exposure level — each both weighted and unweighted), aggregates to
per-(arm, dose) dose-response curves, and writes CSVs + plots to
`outputs/analysis/`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from digital_ghost.analysis.bradley_terry import DavidsonFit, fit_davidson
from digital_ghost.analysis.data import load_ratings, salted_only, substantive_only, to_comparisons
from digital_ghost.analysis.plots import plot_dose_response, plot_rater_calibration
from digital_ghost.analysis.rater_weights import compute_rater_weights, performances_to_frame, weight_column
from digital_ghost.config import StudyConfig, load_rating_app_config
from digital_ghost.generation.generate_grid import BASELINE_LABEL, all_checkpoints

logger = logging.getLogger(__name__)


@dataclass
class BreakdownResult:
    label: str
    n_comparisons: int
    unweighted_cells: pd.DataFrame
    weighted_cells: pd.DataFrame
    unweighted_curve: pd.DataFrame
    weighted_curve: pd.DataFrame


def _item_metadata(study: StudyConfig) -> dict[str, dict]:
    return {
        cp.label: {"arm": cp.arm, "dose": cp.dose, "seed_index": cp.seed_index}
        for cp in all_checkpoints(study)
    }


def _fit_to_cell_frame(fit: DavidsonFit, item_meta: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for item in fit.items:
        meta = item_meta.get(item, {"arm": None, "dose": None, "seed_index": None})
        rows.append(
            {
                "item": item,
                "arm": meta["arm"],
                "dose": meta["dose"],
                "seed_index": meta["seed_index"],
                "log_strength": fit.log_strength[item],
                "strength": fit.strength[item],
            }
        )
    return pd.DataFrame(rows)


def _aggregate_dose_response(cell_df: pd.DataFrame) -> pd.DataFrame:
    trained = cell_df[cell_df["arm"].notna()]
    if trained.empty:
        return pd.DataFrame(columns=["arm", "dose", "log_strength_mean", "log_strength_std", "n_seeds"])
    agg = (
        trained.groupby(["arm", "dose"])["log_strength"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "log_strength_mean", "std": "log_strength_std", "count": "n_seeds"})
    )
    return agg


def _fit_one(
    df: pd.DataFrame, item_meta: dict[str, dict], weight_col: str | None
) -> DavidsonFit | None:
    comparisons = to_comparisons(df)
    if len(comparisons) < 2:
        return None
    items_present = {i for i, _, _ in comparisons} | {j for i, j, _ in comparisons}
    reference = BASELINE_LABEL if BASELINE_LABEL in items_present else None
    if reference is None:
        logger.warning(
            "baseline checkpoint not present in this subset (%d comparisons) — "
            "strengths will be relative to an arbitrary reference item instead",
            len(comparisons),
        )
    weights = df[weight_col].tolist() if weight_col else None
    try:
        return fit_davidson(comparisons, weights=weights, reference_item=reference)
    except Exception:
        logger.exception("Davidson fit failed for a subset of %d comparisons", len(comparisons))
        return None


def run_breakdown(label: str, df: pd.DataFrame, item_meta: dict[str, dict]) -> BreakdownResult | None:
    unweighted_fit = _fit_one(df, item_meta, weight_col=None)
    weighted_fit = _fit_one(df, item_meta, weight_col="rater_weight")
    if unweighted_fit is None or weighted_fit is None:
        logger.warning("skipping breakdown %r: not enough data to fit", label)
        return None

    unweighted_cells = _fit_to_cell_frame(unweighted_fit, item_meta)
    weighted_cells = _fit_to_cell_frame(weighted_fit, item_meta)
    return BreakdownResult(
        label=label,
        n_comparisons=len(df),
        unweighted_cells=unweighted_cells,
        weighted_cells=weighted_cells,
        unweighted_curve=_aggregate_dose_response(unweighted_cells),
        weighted_curve=_aggregate_dose_response(weighted_cells),
    )


def write_breakdown(result: BreakdownResult, out_dir: Path, doses: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.unweighted_cells.to_csv(out_dir / f"{result.label}_unweighted_cells.csv", index=False)
    result.weighted_cells.to_csv(out_dir / f"{result.label}_weighted_cells.csv", index=False)
    result.unweighted_curve.to_csv(out_dir / f"{result.label}_unweighted_curve.csv", index=False)
    result.weighted_curve.to_csv(out_dir / f"{result.label}_weighted_curve.csv", index=False)
    plot_dose_response(
        result.unweighted_curve, result.weighted_curve, doses,
        out_dir / f"{result.label}_dose_response.png", title=result.label,
    )


def run_full_analysis(study: StudyConfig) -> list[BreakdownResult]:
    df = load_ratings(study)
    if df.empty:
        raise RuntimeError("no ratings found in the rating app database — has anyone rated pairs yet?")

    rating_cfg = load_rating_app_config(study)
    out_dir = study.path("analysis_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    performances = compute_rater_weights(salted_only(df), rating_cfg.calibration)
    perf_df = performances_to_frame(performances)
    perf_df.to_csv(out_dir / "rater_performance.csv", index=False)
    plot_rater_calibration(perf_df, out_dir / "rater_calibration.png")

    substantive = substantive_only(df)
    if substantive.empty:
        raise RuntimeError("no non-calibration ratings found — everything collected so far is salted pairs")
    substantive = substantive.assign(rater_weight=weight_column(substantive, performances))

    item_meta = _item_metadata(study)
    results: list[BreakdownResult] = []

    overall = run_breakdown("overall", substantive, item_meta)
    if overall:
        results.append(overall)

    for tier, group in substantive.groupby("tier"):
        r = run_breakdown(f"tier_{tier}", group, item_meta)
        if r:
            results.append(r)

    for exposure, group in substantive.groupby("exposure_level"):
        r = run_breakdown(f"exposure_{exposure}", group, item_meta)
        if r:
            results.append(r)

    for r in results:
        write_breakdown(r, out_dir, doses=study.doses)
        logger.info("breakdown %s: %d comparisons -> %s", r.label, r.n_comparisons, out_dir)

    return results
