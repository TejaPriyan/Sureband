"""Tests for Mondrian (class-conditional) conformal calibration."""

import random
import pytest
import sureband


def _make_imbalanced_examples(n, seed=42):
    """Synthetic dataset with 3 classes of varying difficulty and frequencies."""
    rng = random.Random(seed)
    labels = ["rare_hard", "common_easy", "medium"]
    # Class weights: 10% rare_hard, 60% common_easy, 30% medium
    weights = [0.1, 0.6, 0.3]
    examples = []
    for _ in range(n):
        true_label = rng.choices(labels, weights=weights, k=1)[0]
        if true_label == "common_easy":
            # Model is accurate on common_easy
            probs = {"common_easy": 0.85, "medium": 0.10, "rare_hard": 0.05}
        elif true_label == "medium":
            # Model is moderately confident
            probs = {"common_easy": 0.25, "medium": 0.65, "rare_hard": 0.10}
        else:
            # Model struggles on rare_hard
            probs = {"common_easy": 0.45, "medium": 0.30, "rare_hard": 0.25}

        # Add slight random perturbation to avoid identical ties
        noise = [rng.gauss(0, 0.02) for _ in range(3)]
        raw_vals = [max(0.01, p + n) for p, n in zip(probs.values(), noise)]
        total = sum(raw_vals)
        norm_probs = {k: v / total for k, v in zip(probs.keys(), raw_vals)}

        raw = {"type": "choice", "probabilities": norm_probs}
        examples.append((raw, true_label))
    return examples


def test_mondrian_per_class_coverage_guarantee():
    alpha = 0.10
    target = 1 - alpha

    cal_data = _make_imbalanced_examples(1500, seed=10)
    test_data = _make_imbalanced_examples(3000, seed=20)

    cal = sureband.Calibrator(alpha=alpha, mondrian=True).fit(cal_data)

    covered_by_class = {"rare_hard": 0, "common_easy": 0, "medium": 0}
    total_by_class = {"rare_hard": 0, "common_easy": 0, "medium": 0}

    for raw, true_label in test_data:
        res = cal.wrap(raw)
        total_by_class[true_label] += 1
        if true_label in res.prediction_set:
            covered_by_class[true_label] += 1

    for label in ["rare_hard", "common_easy", "medium"]:
        coverage = covered_by_class[label] / total_by_class[label]
        # Mondrian guarantees at least (1 - alpha) per individual class!
        assert coverage >= target - 0.04, (
            f"Mondrian failed on {label}: expected >= {target - 0.04:.2f}, got {coverage:.2f}"
        )


def test_mondrian_n_calibration_queries():
    cal_data = _make_imbalanced_examples(500, seed=30)
    cal = sureband.Calibrator(alpha=0.1, mondrian=True).fit(cal_data)

    assert cal.n_calibration("choice") == 500
    n_rare = cal.n_calibration("choice", label="rare_hard")
    n_easy = cal.n_calibration("choice", label="common_easy")
    n_med = cal.n_calibration("choice", label="medium")

    assert n_rare + n_easy + n_med == 500
    assert n_easy > n_rare
