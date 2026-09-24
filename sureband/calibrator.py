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

    def __init__(self, alpha: float = 0.1, mondrian: bool = False):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1 (e.g. 0.1 for 90% coverage)")
        self.alpha = alpha
        self.mondrian = mondrian
        self._q_hat: Dict[str, float] = {}
        self._q_hat_mondrian: Dict[Tuple[str, str], float] = {}
        self._n_calibration: Dict[str, int] = {}
        self._n_calibration_mondrian: Dict[Tuple[str, str], int] = {}

    def fit(self, examples: Iterable[Tuple[Union[dict, Decision], str]]) -> "Calibrator":
        """Fit conformal thresholds from (raw_output_or_Decision, true_label) pairs.

        `examples` must be data the underlying model was *not* trained or
        fine-tuned on -- reusing training data here invalidates the coverage
        guarantee, the same way it would for any held-out evaluation.
        """
        scores_by_kind: Dict[str, List[float]] = {}
        scores_by_class: Dict[Tuple[str, str], List[float]] = {}

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
            if self.mondrian:
                scores_by_class.setdefault((decision.kind, true_label), []).append(score)

        if not scores_by_kind:
            raise ValueError("fit() received no calibration examples")

        for kind, scores in scores_by_kind.items():
            n = len(scores)
            level = quantile_level(n, self.alpha)
            scores_sorted = sorted(scores)
            rank = min(int(round(level * n)) - 1, n - 1)
            rank = max(rank, 0)
            self._q_hat[kind] = scores_sorted[rank]
            self._n_calibration[kind] = n

        if self.mondrian:
            for (kind, label), scores in scores_by_class.items():
                n_c = len(scores)
                level_c = quantile_level(n_c, self.alpha)
                scores_sorted = sorted(scores)
                rank_c = min(int(round(level_c * n_c)) - 1, n_c - 1)
                rank_c = max(rank_c, 0)
                self._q_hat_mondrian[(kind, label)] = scores_sorted[rank_c]
                self._n_calibration_mondrian[(kind, label)] = n_c

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

        if self.mondrian:
            score_fn = NONCONFORMITY[decision.kind]
            included_labels = []
            for i, l in enumerate(decision.labels):
                cand_score = score_fn(decision.probabilities, i)
                q_hat_val = self._q_hat_mondrian.get((decision.kind, l), self._q_hat[decision.kind])
                if cand_score <= q_hat_val:
                    included_labels.append(l)
            if not included_labels:
                order = sorted(range(len(decision.probabilities)), key=lambda i: decision.probabilities[i], reverse=True)
                included_labels = [decision.labels[order[0]]]
            prediction_set = frozenset(included_labels)
        else:
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

    def n_calibration(self, kind: str, label: str | None = None) -> int:
        """How many calibration examples of this kind (or class) were used to fit it."""
        if label is not None and self.mondrian:
            return self._n_calibration_mondrian.get((kind, label), 0)
        return self._n_calibration.get(kind, 0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        fitted = {k: f"n={self._n_calibration[k]}" for k in self._q_hat}
        mondrian_tag = ", mondrian=True" if self.mondrian else ""
        return f"Calibrator(alpha={self.alpha}{mondrian_tag}, fitted={fitted})"
