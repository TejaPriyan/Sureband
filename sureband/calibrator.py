"""The Calibrator: fit once on a held-out calibration set, wrap many times."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, Union

from .conformal import NONCONFORMITY, PREDICTION_SET, quantile_level
from .detect import detect_type
from .types import ConformalResult, Decision


class Calibrator:
    """Fits a conformal threshold per decision kind, then wraps raw outputs.

    Example:
        cal = Calibrator(alpha=0.1)  # target 90% coverage
        cal.fit([
            (model_output_1, "billing"),
            (model_output_2, "technical"),
            ...  # a few hundred labeled examples the model did NOT train on
        ])
        result = cal.wrap(new_model_output)

    A separate threshold is kept per decision kind ("choice"/"noul"/"score")
    because their nonconformity scores are not on a comparable scale.
    Calibration examples of a kind sureband hasn't seen at fit time simply
    won't have a threshold yet -- `wrap()` will raise a clear error rather
    than silently guessing one.
    """

    def __init__(self, alpha: float = 0.1):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1 (e.g. 0.1 for 90% coverage)")
        self.alpha = alpha
        self._q_hat: Dict[str, float] = {}
        self._n_calibration: Dict[str, int] = {}

    def fit(self, examples: Iterable[Tuple[Union[dict, Decision], str]]) -> "Calibrator":
        """Fit conformal thresholds from (raw_output_or_Decision, true_label) pairs.

        `examples` must be data the underlying model was *not* trained or
        fine-tuned on -- reusing training data here invalidates the coverage
        guarantee, the same way it would for any held-out evaluation.
        """
        scores_by_kind: Dict[str, List[float]] = {}

        for raw, true_label in examples:
            decision = raw if isinstance(raw, Decision) else detect_type(raw)
            try:
                true_index = list(decision.labels).index(true_label)
            except ValueError:
                raise ValueError(
                    f"true label {true_label!r} not among this decision's "
                    f"labels {list(decision.labels)!r}"
                )
            score_fn = NONCONFORMITY[decision.kind]
            score = score_fn(decision.probabilities, true_index)
            scores_by_kind.setdefault(decision.kind, []).append(score)

        if not scores_by_kind:
            raise ValueError("fit() received no calibration examples")

        for kind, scores in scores_by_kind.items():
            n = len(scores)
            level = quantile_level(n, self.alpha)
            scores_sorted = sorted(scores)
            # level is a fraction in (0, 1]; convert to a rank into the sorted scores
            rank = min(int(round(level * n)) - 1, n - 1)
            rank = max(rank, 0)
            self._q_hat[kind] = scores_sorted[rank]
            self._n_calibration[kind] = n

        return self

    def wrap(self, raw: Union[dict, Decision]) -> ConformalResult:
        """Calibrate one raw model output into a ConformalResult."""
        decision = raw if isinstance(raw, Decision) else detect_type(raw)
        if decision.kind not in self._q_hat:
            raise RuntimeError(
                f"no calibration threshold fitted for decision kind "
                f"{decision.kind!r}; call fit() with at least one "
                f"{decision.kind!r} example first. Fitted kinds: "
                f"{list(self._q_hat)}"
            )

        q_hat = self._q_hat[decision.kind]
        set_fn = PREDICTION_SET[decision.kind]
        included_indices = set_fn(decision.probabilities, q_hat)
        prediction_set = frozenset(decision.labels[i] for i in included_indices)
        raw_probs = dict(zip(decision.labels, decision.probabilities))

        confidence = max(raw_probs[l] for l in prediction_set) if prediction_set else 0.0

        return ConformalResult(
            prediction_set=prediction_set,
            confidence=confidence,
            covered_at=1 - self.alpha,
            kind=decision.kind,
            raw_probabilities=raw_probs,
        )

    def n_calibration(self, kind: str) -> int:
        """How many calibration examples of this kind were used to fit it."""
        return self._n_calibration.get(kind, 0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        fitted = {k: f"n={self._n_calibration[k]}" for k in self._q_hat}
        return f"Calibrator(alpha={self.alpha}, fitted={fitted})"
