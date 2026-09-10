from __future__ import annotations

import numpy as np
import pytest

from digital_ghost.analysis.bradley_terry import fit_davidson


def _sample_outcome(pi, pj, nu, rng):
    denom = pi + pj + nu * np.sqrt(pi * pj)
    r = rng.random()
    if r < pi / denom:
        return "i"
    if r < (pi + pj) / denom:
        return "j"
    return "tie"


def test_fit_davidson_recovers_known_strengths_and_ties():
    rng = np.random.default_rng(0)
    true_strength = {"baseline": 1.0, "weak": 1.5, "strong": 8.0, "medium": 3.0}
    true_nu = 0.8
    items = list(true_strength)

    comparisons = []
    for _ in range(15000):
        i, j = rng.choice(items, size=2, replace=False)
        o = _sample_outcome(true_strength[i], true_strength[j], true_nu, rng)
        comparisons.append((i, j, o))

    fit = fit_davidson(comparisons, reference_item="baseline", ridge=1e-4)
    assert fit.converged
    assert fit.nu == pytest.approx(true_nu, rel=0.15)
    for item in items:
        assert fit.strength[item] == pytest.approx(
            true_strength[item] / true_strength["baseline"], rel=0.15
        )


def test_reference_item_pinned_at_zero_log_strength():
    comparisons = [("a", "b", "i"), ("a", "b", "tie"), ("b", "a", "j")]
    fit = fit_davidson(comparisons, reference_item="a")
    assert fit.log_strength["a"] == 0.0
    assert fit.strength["a"] == 1.0


def test_weighted_fit_downweights_a_noisy_rater_signal():
    rng = np.random.default_rng(1)
    # "good" comparisons all favor i; "noisy" comparisons are pure coin flips
    good = [("strong", "baseline", "i") for _ in range(200)]
    noisy = [("strong", "baseline", rng.choice(["i", "j", "tie"])) for _ in range(200)]
    comparisons = good + noisy
    weights = [1.0] * len(good) + [0.0] * len(noisy)

    fit_weighted = fit_davidson(comparisons, weights=weights, reference_item="baseline")
    fit_unweighted = fit_davidson(comparisons, reference_item="baseline")

    # downweighting the noisy half to 0 should push the strength estimate
    # further from 1 (more confidently "strong beats baseline") than
    # treating all the noise as equally informative
    assert fit_weighted.strength["strong"] > fit_unweighted.strength["strong"]


def test_fit_davidson_rejects_empty_input():
    with pytest.raises(ValueError):
        fit_davidson([])
