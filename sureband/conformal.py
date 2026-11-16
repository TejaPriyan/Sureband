"""Split conformal prediction over typed decisions.

This is the statistical core of sureband. Given a calibration set of
(Decision, true_label) pairs held out from training, it computes a single
threshold that turns any future Decision's probabilities into a prediction
set with a guaranteed marginal coverage rate -- e.g. "the true answer is in
this set at least 90% of the time" -- under the standard conformal
assumption that calibration and future examples are exchangeable.

Two nonconformity scores are implemented:

  * `aps_score` / `aps_set`     -- unordered labels ("choice", "noul").
    Adaptive Prediction Sets (Romano, Sesia & Candes, 2020): rank labels by
    probability, accumulate until the true label is included.

  * `ordinal_score` / `ordinal_set` -- ordered labels ("score"). Grows a
    *contiguous* interval outward from the most likely level, so a rubric
    like low/medium/high never returns a set like {low, high} skipping
    medium.

Both reduce to the same idea: define a deterministic way to grow a set from
a probability vector, one label at a time, and use where the true label
first appears as the nonconformity score.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

_EPS = 1e-12


def _neumaier_step(total: float, comp: float, value: float) -> Tuple[float, float]:
    """Accumulate `value` into `(total, comp)` using Neumaier compensated summation."""
    next_total = total + value
    if abs(total) >= abs(value):
        comp += (total - next_total) + value
    else:
        comp += (value - next_total) + total
    return next_total, comp


def _ordered_score(probs: Sequence[float], order: Sequence[int], true_index: int) -> float:
    """Cumulative probability mass along `order` through `true_index`."""
    total, comp = 0.0, 0.0
    for idx in order:
        total, comp = _neumaier_step(total, comp, probs[idx])
        if idx == true_index:
            return min(total + comp, 1.0)
    return 1.0  # pragma: no cover - unreachable if probs cover all indices


def _ordered_set(probs: Sequence[float], order: Sequence[int], q_hat: float) -> List[int]:
    """Prefix of `order` covering mass `q_hat` with floating-point tolerance."""
    total, comp = 0.0, 0.0
    included: List[int] = []
    for idx in order:
        cumulative = total + comp
        if (
            included
            and cumulative + _EPS >= q_hat
            and cumulative + probs[idx] > q_hat + _EPS
        ):
            break
        total, comp = _neumaier_step(total, comp, probs[idx])
        included.append(idx)
    return included


def quantile_level(n_calibration: int, alpha: float) -> float:
    """The finite-sample-corrected quantile level for split conformal.

    Using the ceil((n+1)(1-alpha))/n formula (Vovk, Gammerman & Shafer)
    rather than the naive (1-alpha) quantile is what makes the coverage
    guarantee hold exactly for finite calibration sets, not just
    asymptotically. When ceil((n+1)(1-alpha)) > n, returns float('inf') so
    prediction sets cover all labels and preserve the finite-sample guarantee.
    """
    if n_calibration < 1:
        raise ValueError("need at least one calibration example")
    q_rank = math.ceil((n_calibration + 1) * (1 - alpha))
    if q_rank > n_calibration:
        return float("inf")
    return q_rank / n_calibration


# ---------------------------------------------------------------------------
# Unordered labels: choice, noul
# ---------------------------------------------------------------------------

def aps_score(probs: Sequence[float], true_index: int) -> float:
    """Nonconformity score for one calibration example (unordered labels).

    Sorts labels by probability, descending, and returns the cumulative
    probability mass through (and including) the true label's rank. A
    confidently-correct example scores low; a confidently-wrong example
    scores high (close to 1.0, since most of the mass sits on other labels
    before the true one is reached).
    """
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return _ordered_score(probs, order, true_index)


def aps_set(probs: Sequence[float], q_hat: float) -> List[int]:
    """Prediction set (as label indices) for unordered labels at threshold q_hat."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return _ordered_set(probs, order, q_hat)


# ---------------------------------------------------------------------------
# Ordered labels: score (ordinal rubrics)
# ---------------------------------------------------------------------------

def _greedy_ordinal_order(probs: Sequence[float]) -> List[int]:
    """Deterministic growth order for a contiguous interval, starting at the
    mode and, at each step, extending to whichever open boundary (left or
    right neighbor) carries more probability mass. Used identically at
    calibration time and prediction time so scores are comparable.
    """
    n = len(probs)
    mode = max(range(n), key=lambda i: probs[i])
    order = [mode]
    lo, hi = mode, mode
    while len(order) < n:
        left_candidate = lo - 1
        right_candidate = hi + 1
        left_val = probs[left_candidate] if left_candidate >= 0 else -1.0
        right_val = probs[right_candidate] if right_candidate < n else -1.0
        if left_val >= right_val:
            lo = left_candidate
            order.append(lo)
        else:
            hi = right_candidate
            order.append(hi)
    return order


def ordinal_score(probs: Sequence[float], true_index: int) -> float:
    """Nonconformity score for one calibration example (ordered labels)."""
    order = _greedy_ordinal_order(probs)
    return _ordered_score(probs, order, true_index)


def ordinal_set(probs: Sequence[float], q_hat: float) -> List[int]:
    """Prediction set (as label indices) for ordered labels at threshold q_hat.

    Always contiguous: it is built by the same left/right expansion used
    to score calibration examples, so a "score" decision never returns a
    set that skips a level in the middle of the rubric.
    """
    order = _greedy_ordinal_order(probs)
    return _ordered_set(probs, order, q_hat)


NONCONFORMITY = {
    "choice": aps_score,
    "noul": aps_score,
    "score": ordinal_score,
}

PREDICTION_SET = {
    "choice": aps_set,
    "noul": aps_set,
    "score": ordinal_set,
}
