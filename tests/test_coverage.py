"""Does the coverage guarantee actually hold?

This is the test that matters most: it simulates a System One model that is
*deliberately overconfident* (its reported probabilities are sharper than
reality, the exact failure mode reported for Laya, typical, and friends
out of the box) and checks that sureband's prediction sets still hit their
target coverage rate on held-out data, even though the model's own raw
confidence does not.
"""

import random

import pytest

import sureband


def _softmax(logits, temperature=1.0):
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [pow(2.718281828, x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _make_overconfident_examples(n, n_classes=4, overconfidence=1.5, logit_std=2.0, seed=0):
    """Each example has a 'true' distribution and a reported distribution
    that is the true one sharpened by `overconfidence` -- i.e. exactly the
    kind of miscalibration flagged in real System One model benchmarks.
    Returns list[(raw_output_dict, true_label_str)].
    """
    rng = random.Random(seed)
    labels = [f"class_{i}" for i in range(n_classes)]
    examples = []
    for _ in range(n):
        logits = [rng.gauss(0, logit_std) for _ in range(n_classes)]
        true_dist = _softmax(logits, temperature=1.0)
        reported_dist = _softmax(logits, temperature=1.0 / overconfidence)
        true_label = rng.choices(labels, weights=true_dist, k=1)[0]
        raw = {"type": "choice", "probabilities": dict(zip(labels, reported_dist))}
        examples.append((raw, true_label))
    return examples


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_conformal_coverage_meets_or_exceeds_target(alpha):
    """The guarantee split conformal actually makes is one-sided: marginal
    coverage of *at least* (1-alpha), not exactly (1-alpha). The
    non-randomized APS score used here (chosen over the randomized variant
    so that wrap() is deterministic, which matters more than shaving off a
    few points of over-coverage for a library meant to sit in a decision
    pipeline) is conservative, especially with few labels -- see
    test_sets_tighten_with_more_labels below for why. So this test checks
    the lower bound only, which is what the docs promise.
    """
    calibration_examples = _make_overconfident_examples(1200, n_classes=4, seed=1)
    test_examples = _make_overconfident_examples(6000, n_classes=4, seed=2)

    cal = sureband.Calibrator(alpha=alpha)
    cal.fit(calibration_examples)

    covered = 0
    for raw, true_label in test_examples:
        result = cal.wrap(raw)
        if true_label in result.prediction_set:
            covered += 1

    empirical_coverage = covered / len(test_examples)
    target = 1 - alpha
    # Small slack below target only, for finite-calibration-set noise.
    assert empirical_coverage >= target - 0.03, (
        f"target={target:.2f} got={empirical_coverage:.3f} -- coverage "
        "guarantee violated, this would be a real bug"
    )


def test_sets_tighten_as_label_space_grows():
    """Documents the honest trade-off: with only a couple of labels (e.g. a
    'noul' yes/no, or a 4-way choice), non-randomized conformal sets
    over-cover more and are less informative, because there are only a
    handful of achievable set sizes to round up to. With more labels the
    achievable coverage levels are closer together, so sets shrink toward
    the target and become genuinely informative rather than "include
    everything to be safe."
    """
    alpha = 0.1
    target = 1 - alpha
    relative_sizes = {}
    for n_classes in (4, 10, 25):
        cal = _make_overconfident_examples(2000, n_classes=n_classes, seed=1)
        test = _make_overconfident_examples(8000, n_classes=n_classes, seed=2)
        calibrator = sureband.Calibrator(alpha=alpha).fit(cal)

        covered = 0
        total_size = 0
        for raw, true_label in test:
            result = calibrator.wrap(raw)
            total_size += len(result.prediction_set)
            if true_label in result.prediction_set:
                covered += 1

        coverage = covered / len(test)
        assert coverage >= target - 0.03
        relative_sizes[n_classes] = (total_size / len(test)) / n_classes

    # As the label space grows, prediction sets should take up a shrinking
    # fraction of it -- proof the calibration is doing real work, not just
    # defaulting to "everything," as it does in the degenerate 2-class case.
    assert relative_sizes[25] < relative_sizes[10] < relative_sizes[4]


def test_raw_top1_accuracy_understates_true_error_rate():
    """Sanity check that the synthetic model is actually miscalibrated --
    otherwise the coverage test above would be trivially true even for a
    Calibrator that does nothing. Confirms raw top-1 'confidence' overstates
    how often the model is actually right.
    """
    examples = _make_overconfident_examples(4000, seed=3)
    correct = 0
    mean_reported_confidence = 0.0
    for raw, true_label in examples:
        probs = raw["probabilities"]
        top_label = max(probs, key=probs.get)
        mean_reported_confidence += probs[top_label]
        if top_label == true_label:
            correct += 1
    top1_accuracy = correct / len(examples)
    mean_reported_confidence /= len(examples)

    # The whole premise of the library: reported confidence runs hotter
    # than reality for an overconfident model.
    assert mean_reported_confidence > top1_accuracy + 0.05
