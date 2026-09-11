from __future__ import annotations

import random

import numpy as np
import pytest

from digital_ghost.analysis.rater_weights import compute_rater_weights
from digital_ghost.analysis.run_analysis import run_full_analysis
from digital_ghost.config import load_rating_app_config
from digital_ghost.generation.eval_prompts import load_eval_prompts
from digital_ghost.generation.generate_grid import all_checkpoints
from digital_ghost.rating_app.backend.db import get_session
from digital_ghost.rating_app.backend.models import Rater, Rating
from digital_ghost.rating_app.backend.pairing import build_image_index, sample_pair


def _simulate_ratings(study, rating_cfg, n_raters=4, n_pairs=400, seed=0):
    db_path = study.resolve_repo_path(rating_cfg.db_path)
    index = build_image_index(study)
    prompts_by_id = {p["id"]: p for p in load_eval_prompts(study)["prompts"]}
    checkpoints = {cp.label: cp for cp in all_checkpoints(study)}

    def true_log_strength(label):
        cp = checkpoints[label]
        if cp.arm is None:
            return 0.0
        coeff = {"standard": 0.5, "meme": 0.1, "control": 0.0}[cp.arm]
        return coeff * cp.dose

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    with get_session(db_path) as session:
        for i in range(n_raters):
            session.add(Rater(id=f"r{i}", exposure_level="none" if i % 2 == 0 else "a_lot"))
        session.commit()

        for k in range(n_pairs):
            pair = sample_pair(index, study, rating_cfg, prompts_by_id, rng=random.Random(rng.random()))
            session.add(pair)
            session.commit()

            li, lj = true_log_strength(pair.checkpoint_a), true_log_strength(pair.checkpoint_b)
            m = max(li, lj)
            pi, pj = np.exp(li - m), np.exp(lj - m)
            tie = np.sqrt(pi * pj)
            denom = pi + pj + tie
            r = np_rng.random()
            choice = "A" if r < pi / denom else ("B" if r < (pi + pj) / denom else "BOTH")

            session.add(Rating(rater_id=f"r{k % n_raters}", pair_id=pair.id, choice=choice))
        session.commit()
    return db_path


def test_run_full_analysis_produces_sane_dose_response(trained_and_generated_study):
    study = trained_and_generated_study
    rating_cfg = load_rating_app_config(study)
    _simulate_ratings(study, rating_cfg, n_raters=4, n_pairs=500)

    results = run_full_analysis(study)
    labels = {r.label for r in results}
    assert "overall" in labels
    assert any(label.startswith("tier_") for label in labels)
    assert any(label.startswith("exposure_") for label in labels)

    overall = next(r for r in results if r.label == "overall")
    curve = overall.unweighted_curve
    assert set(curve["arm"]) <= {"standard", "meme", "control"}

    # standard arm (strong injected signal) should end up with higher
    # log-strength at the top dose than the control arm (no signal)
    standard_top = curve[(curve["arm"] == "standard")].sort_values("dose").iloc[-1]["log_strength_mean"]
    control_top = curve[(curve["arm"] == "control")].sort_values("dose").iloc[-1]["log_strength_mean"]
    assert standard_top > control_top

    out_dir = study.path("analysis_dir")
    assert (out_dir / "rater_performance.csv").exists()
    assert (out_dir / "overall_dose_response.png").exists()


def test_run_full_analysis_raises_with_no_ratings(trained_and_generated_study):
    with pytest.raises(RuntimeError, match="no ratings found"):
        run_full_analysis(trained_and_generated_study)


def test_rater_weights_zero_for_raters_below_min_pairs(trained_and_generated_study):
    study = trained_and_generated_study
    rating_cfg = load_rating_app_config(study)
    from digital_ghost.analysis.data import load_ratings, salted_only

    _simulate_ratings(study, rating_cfg, n_raters=4, n_pairs=500)
    df = load_ratings(study)
    performances = compute_rater_weights(salted_only(df), rating_cfg.calibration)
    for perf in performances.values():
        if perf.n_calibration < rating_cfg.calibration.min_pairs_for_weight:
            assert perf.weight == 0.0
        assert 0.0 <= perf.weight <= 1.0
