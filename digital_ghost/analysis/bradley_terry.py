"""Tie-aware Bradley-Terry (Davidson, 1970) model.

Each item i has a positive strength pi_i (we optimize its log, lambda_i,
for an unconstrained problem). For a comparison between items i and j:

    P(i beats j) = pi_i / (pi_i + pi_j + nu * sqrt(pi_i * pi_j))
    P(j beats i) = pi_j / (pi_i + pi_j + nu * sqrt(pi_i * pi_j))
    P(tie)       = nu * sqrt(pi_i * pi_j) / (pi_i + pi_j + nu * sqrt(pi_i * pi_j))

`nu >= 0` is a single shared tie-propensity parameter, estimated jointly
with the strengths by maximum likelihood.

In this study "beats" means "the rater said they saw Charlie Kirk in this
one and not the other" and "tie" collapses BOTH and NEITHER — from the
pairwise-strength-estimation point of view, both outcomes say the same
thing: the rater detected no *difference* between i and j, whether because
both clearly showed the subject or because neither did. That distinction
matters for other diagnostics (see analysis/data.py) but not for what
Davidson's tie parameter is estimating.

One item must be pinned as the reference (log-strength fixed at 0) since
only strength *ratios* are identifiable, exactly as in ordinary
Bradley-Terry. We use the stock-SDXL baseline checkpoint, so every other
item's fitted strength is directly interpretable as "how much more likely
than the untrained model this checkpoint is to be judged as showing the
subject."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

Outcome = str  # "i" | "j" | "tie"


@dataclass
class DavidsonFit:
    items: list[str]
    log_strength: dict[str, float]
    strength: dict[str, float]
    nu: float
    n_obs: int
    converged: bool
    neg_log_likelihood: float
    reference_item: str


def fit_davidson(
    comparisons: list[tuple[str, str, Outcome]],
    weights: list[float] | None = None,
    reference_item: str | None = None,
    ridge: float = 1e-3,
) -> DavidsonFit:
    """Fit strengths + tie parameter by (optionally weighted) MLE.

    `weights` (one per comparison, default all 1.0) is what makes this
    reusable for both the unweighted curve (weights=None) and the
    rater-weighted curve (weights=[rater_weight[r] for r in ...]) — a
    weighted comparison just scales its log-likelihood contribution.

    `ridge` is a small L2 penalty on log-strengths, purely for numerical
    stability when an item has a near-unanimous record (classic
    Bradley-Terry separability, where the unpenalized MLE strength runs
    off to infinity). It is not intended as an informative prior.
    """
    if not comparisons:
        raise ValueError("fit_davidson called with zero comparisons")

    items = sorted({i for i, j, _ in comparisons} | {j for i, j, _ in comparisons})
    if reference_item is None:
        reference_item = items[0]
    if reference_item not in items:
        raise ValueError(f"reference_item {reference_item!r} not among items {items}")

    other_items = [it for it in items if it != reference_item]
    idx = {it: k for k, it in enumerate(other_items)}

    i_idx = np.array([idx.get(i, -1) for i, _, _ in comparisons])
    j_idx = np.array([idx.get(j, -1) for _, j, _ in comparisons])
    outcomes = np.array([o for _, _, o in comparisons])
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(comparisons))
    if len(w) != len(comparisons):
        raise ValueError("weights must have the same length as comparisons")

    n_free = len(other_items)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, float]:
        lam = np.zeros(n_free + 1)  # index n_free reserved for reference (always 0)
        lam[:n_free] = params[:n_free]
        log_nu = params[n_free]
        return lam, log_nu

    def lam_at(lam: np.ndarray, pos_idx: np.ndarray) -> np.ndarray:
        # reference item's slot (-1) maps to lam[n_free] == 0.0 by construction
        return lam[np.where(pos_idx == -1, n_free, pos_idx)]

    def neg_log_lik(params: np.ndarray) -> float:
        lam, log_nu = unpack(params)
        nu = np.exp(log_nu)
        li = lam_at(lam, i_idx)
        lj = lam_at(lam, j_idx)

        m = np.maximum(li, lj)
        pi = np.exp(li - m)
        pj = np.exp(lj - m)
        tie_term = nu * np.sqrt(pi * pj)
        denom = pi + pj + tie_term

        p_i = pi / denom
        p_j = pj / denom
        p_tie = tie_term / denom

        p = np.where(outcomes == "i", p_i, np.where(outcomes == "j", p_j, p_tie))
        p = np.clip(p, 1e-12, 1.0)
        nll = -np.sum(w * np.log(p))
        nll += ridge * np.sum(lam[:n_free] ** 2)
        return float(nll)

    x0 = np.zeros(n_free + 1)
    result = minimize(neg_log_lik, x0, method="L-BFGS-B")

    lam, log_nu = unpack(result.x)
    log_strength = {it: float(lam[k]) for it, k in idx.items()}
    log_strength[reference_item] = 0.0
    strength = {k: float(np.exp(v)) for k, v in log_strength.items()}

    return DavidsonFit(
        items=items,
        log_strength=log_strength,
        strength=strength,
        nu=float(np.exp(log_nu)),
        n_obs=len(comparisons),
        converged=bool(result.success),
        neg_log_likelihood=float(result.fun),
        reference_item=reference_item,
    )
