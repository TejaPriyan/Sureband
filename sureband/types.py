"""Shared data types.

sureband normalizes every System One model's raw output into a single
Decision shape before calibrating it, so the same conformal machinery works
whether the output came from Jev, Laya, Von, typical, decider, or a model
you trained yourself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Union


@dataclass(frozen=True)
class Decision:
    """A normalized typed-decision output, ready for conformal calibration.

    kind:
        "choice" -- categorical pick among named options
        "noul"   -- yes/no, expressed as P(true)
        "score"  -- ordinal position on a rubric with K ordered levels

    labels:
        Ordered label names. For "noul" this is always ("false", "true").
        For "score" these are the rubric levels in ascending order.

    probabilities:
        A probability per label, aligned with `labels`, summing to ~1.0.
    """

    kind: str
    labels: Sequence[str]
    probabilities: Sequence[float]
    raw: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        if self.kind not in ("choice", "noul", "score"):
            raise ValueError(f"unknown decision kind: {self.kind!r}")
        if len(self.labels) != len(self.probabilities):
            raise ValueError("labels and probabilities must be the same length")
        total = sum(self.probabilities)
        if not (0.95 <= total <= 1.05):
            raise ValueError(
                f"probabilities must sum to ~1.0, got {total:.4f} "
                f"({dict(zip(self.labels, self.probabilities))})"
            )

    @property
    def top_label(self) -> str:
        best_i = max(range(len(self.labels)), key=lambda i: self.probabilities[i])
        return self.labels[best_i]

    @property
    def top_probability(self) -> float:
        return max(self.probabilities)


@dataclass(frozen=True)
class ConformalResult:
    """The calibrated output for one Decision."""

    prediction_set: FrozenSet[str]
    confidence: float
    covered_at: float
    kind: str
    raw_probabilities: Dict[str, float]

    @property
    def is_certain(self) -> bool:
        """True when the prediction set contains exactly one label."""
        return len(self.prediction_set) == 1

    @property
    def top_label(self) -> Optional[str]:
        if not self.prediction_set:
            return None
        return max(self.prediction_set, key=lambda lbl: self.raw_probabilities[lbl])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        certain = "certain" if self.is_certain else "uncertain"
        return (
            f"ConformalResult({sorted(self.prediction_set)}, "
            f"{certain}, covered_at={self.covered_at:.0%})"
        )
