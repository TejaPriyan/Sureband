"""Tests for Mondrian (class-conditional) conformal calibration."""

import random

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


def test_mondrian_score_preserves_ordinal_contiguity():
    labels = ["low", "med", "high"]
    cal_data = []
    # Extreme classes need high thresholds; middle class is sharply calibrated
    for _ in range(12):
        cal_data.append(
            ({"type": "score", "labels": labels, "distribution": [0.10, 0.20, 0.70]}, "low")
        )
        cal_data.append(
            ({"type": "score", "labels": labels, "distribution": [0.70, 0.20, 0.10]}, "high")
        )
        cal_data.append(
            ({"type": "score", "labels": labels, "distribution": [0.20, 0.60, 0.20]}, "med")
        )

    cal = sureband.Calibrator(alpha=0.1, mondrian=True).fit(cal_data)
    for dist in ([0.20, 0.30, 0.50], [0.50, 0.30, 0.20], [0.45, 0.10, 0.45]):
        res = cal.wrap({"type": "score", "labels": labels, "distribution": dist})
        indices = sorted(labels.index(lbl) for lbl in res.prediction_set)
        assert indices == list(range(indices[0], indices[-1] + 1)), (
            f"Mondrian score prediction set skipped intermediate levels: {res.prediction_set}"
        )
    assert cal.wrap(
        {"type": "score", "labels": labels, "distribution": [0.20, 0.30, 0.50]}
    ).prediction_set == frozenset({"low", "med", "high"})

    # Also test randomized 5-level ordinal distributions
    levels5 = ["1", "2", "3", "4", "5"]
    rng = random.Random(7)
    cal5 = []
    for _ in range(100):
        raw_vals = [rng.random() + 0.05 for _ in range(5)]
        total = sum(raw_vals)
        dist = [v / total for v in raw_vals]
        true_lbl = rng.choices(levels5, weights=dist, k=1)[0]
        cal5.append(({"type": "score", "labels": levels5, "distribution": dist}, true_lbl))
    cal5_fit = sureband.Calibrator(alpha=0.1, mondrian=True).fit(cal5)
    for _ in range(50):
        raw_vals = [rng.random() + 0.05 for _ in range(5)]
        total = sum(raw_vals)
        dist = [v / total for v in raw_vals]
        res = cal5_fit.wrap({"type": "score", "labels": levels5, "distribution": dist})
        indices = sorted(levels5.index(lbl) for lbl in res.prediction_set)
        assert indices == list(range(indices[0], indices[-1] + 1))


def test_mondrian_float_drift_full_mass_does_not_drop_label():
    cal_data = []
    for _ in range(12):
        cal_data.append(
            ({"type": "choice", "probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}}, "c")
        )
        cal_data.append(
            ({"type": "choice", "probabilities": {"a": 0.9, "b": 0.05, "c": 0.05}}, "a")
        )
        cal_data.append(
            ({"type": "choice", "probabilities": {"a": 0.05, "b": 0.9, "c": 0.05}}, "b")
        )

    cal = sureband.Calibrator(alpha=0.1, mondrian=True).fit(cal_data)
    res = cal.wrap({"type": "choice", "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}})
    assert "c" in res.prediction_set


def test_mondrian_rare_class_uses_infinite_threshold():
    cal_data = [
        ({"type": "choice", "probabilities": {"common": 0.95, "rare": 0.05}}, "common")
        for _ in range(20)
    ] + [
        ({"type": "choice", "probabilities": {"common": 0.9, "rare": 0.1}}, "rare")
        for _ in range(3)
    ]
    cal = sureband.Calibrator(alpha=0.1, mondrian=True).fit(cal_data)
    assert cal._q_hat_mondrian[("choice", "rare")] == float("inf")
    res = cal.wrap({"type": "choice", "probabilities": {"common": 0.99, "rare": 0.01}})
    assert "rare" in res.prediction_set


def test_mondrian_refit_clears_stale_class_thresholds():
    cal = sureband.Calibrator(alpha=0.2, mondrian=True)
    cal.fit([
        ({"type": "choice", "probabilities": {"a": 0.8, "b": 0.1, "c": 0.1}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.1, "b": 0.8, "c": 0.1}}, "b"),
        ({"type": "choice", "probabilities": {"a": 0.1, "b": 0.1, "c": 0.8}}, "c"),
    ])
    assert cal.n_calibration("choice", label="c") == 1

    cal.fit([
        ({"type": "choice", "probabilities": {"a": 0.8, "b": 0.2}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.2, "b": 0.8}}, "b"),
    ])
    assert cal.n_calibration("choice", label="c") == 0
    assert ("choice", "c") not in cal._q_hat_mondrian
