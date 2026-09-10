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


class DisconnectedComparisonGraphError(Exception):
    """Some items are not connected to the reference item by any chain of comparisons.

    Their strengths are not identifiable relative to the reference: no amount
    of data in a separate component says anything about how it compares to
    the reference. The ridge penalty will still happily return a finite
    number for them, which is worse than an error — it looks like a result.
    """

    def __init__(self, reference_item: str, unreachable: list[str]):
        self.reference_item = reference_item
        self.unreachable = unreachable
        super().__init__(
            f"{len(unreachable)} item(s) are never compared to reference "
            f"{reference_item!r}, directly or transitively, so their strength "
            f"relative to it is not identifiable: {sorted(unreachable)[:8]}"
            f"{'...' if len(unreachable) > 8 else ''}. Either collect comparisons "
            "linking them to the reference, or fit this subset separately."
        )


def _unreachable_items(
    comparisons: list[tuple[str, str, Outcome]], items: list[str], reference_item: str
) -> list[str]:
    """Items in no connected component containing `reference_item`."""
    adjacency: dict[str, set[str]] = {it: set() for it in items}
    for i, j, _ in comparisons:
        adjacency[i].add(j)
        adjacency[j].add(i)

    seen = {reference_item}
    stack = [reference_item]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return [it for it in items if it not in seen]


@dataclass
class DavidsonFit:
    items: list[str]
    log_strength: dict[str, float]
    strength: dict[str, float]
    nu: float
    n_obs: int
    effective_n_obs: float
    converged: bool
    neg_log_likelihood: float
    reference_item: str
    # Items present in the input but carried only by zero-weight comparisons,
    # so they contribute no information and are absent from `log_strength`.
    # Callers must surface these rather than let a cell quietly disappear
    # from a dose-response curve with no explanation.
    dropped_zero_weight_items: list[str]


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

    raw_weights = (
        np.asarray(weights, dtype=float) if weights is not None else np.ones(len(comparisons))
    )
    if len(raw_weights) != len(comparisons):
        raise ValueError("weights must have the same length as comparisons")
    if np.any(raw_weights < 0):
        raise ValueError("weights must be non-negative")

    # A zero-weight comparison contributes nothing to the likelihood, so it
    # must not contribute to identifiability either. Dropping these up front
    # is what keeps an item whose only ratings come from zero-weight raters
    # from being silently pinned at the reference's strength (i.e. reported
    # as "indistinguishable from the untrained model") by the ridge alone.
    all_items = sorted({i for i, j, _ in comparisons} | {j for i, j, _ in comparisons})
    kept = [(c, wt) for c, wt in zip(comparisons, raw_weights) if wt > 0]
    if not kept:
        raise ValueError("every comparison has zero weight; nothing to fit")
    comparisons = [c for c, _ in kept]
    w = np.array([wt for _, wt in kept], dtype=float)

    items = sorted({i for i, j, _ in comparisons} | {j for i, j, _ in comparisons})
    dropped_zero_weight_items = [it for it in all_items if it not in set(items)]
    if reference_item is None:
        reference_item = items[0]
    if reference_item not in items:
        raise ValueError(f"reference_item {reference_item!r} not among items {items}")

    # Identifiability, not just numerics: an item in a separate component of
    # the comparison graph has no estimable strength relative to the
    # reference, but `ridge` would still pull it to a plausible-looking
    # finite value and the fit would report converged=True. Refuse instead.
    unreachable = _unreachable_items(comparisons, items, reference_item)
    if unreachable:
        raise DisconnectedComparisonGraphError(reference_item, unreachable)

    other_items = [it for it in items if it != reference_item]
    idx = {it: k for k, it in enumerate(other_items)}

    i_idx = np.array([idx.get(i, -1) for i, _, _ in comparisons])
    j_idx = np.array([idx.get(j, -1) for _, j, _ in comparisons])
    outcomes = np.array([o for _, _, o in comparisons])

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
        # n_obs counts comparisons that actually entered the likelihood
        # (zero-weight ones were dropped above); effective_n_obs is the
        # weight mass behind them, which is the honest N for a weighted fit.
        n_obs=len(comparisons),
        effective_n_obs=float(w.sum()),
        converged=bool(result.success),
        neg_log_likelihood=float(result.fun),
        reference_item=reference_item,
        dropped_zero_weight_items=dropped_zero_weight_items,
    )
