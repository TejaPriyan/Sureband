import math
import typing

import pytest

import sureband
from sureband.conformal import (
    aps_score,
    aps_set,
    ordinal_score,
    ordinal_set,
    quantile_level,
)
from sureband.types import ConformalResult, Decision


def test_decision_validates_probabilities_sum():
    with pytest.raises(ValueError):
        Decision(kind="choice", labels=("a", "b"), probabilities=(0.9, 0.9))


def test_decision_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Decision(kind="mystery", labels=("a",), probabilities=(1.0,))


def test_decision_rejects_invalid_probabilities_and_labels():
    with pytest.raises(ValueError):
        Decision(kind="choice", labels=("a", "b"), probabilities=(-0.2, 1.2))
    with pytest.raises(ValueError):
        Decision(kind="choice", labels=("a", "b"), probabilities=(float("nan"), 1.0))
    with pytest.raises(ValueError):
        Decision(
            kind="choice",
            labels=("a", "b"),
            probabilities=(float("inf"), float("-inf")),
        )
    with pytest.raises(ValueError):
        Decision(kind="choice", labels=("a", "b"), probabilities=(True, False))
    with pytest.raises(ValueError):
        Decision(kind="choice", labels=(), probabilities=())
    with pytest.raises(ValueError):
        Decision(kind="choice", labels=("a", "a"), probabilities=(0.5, 0.5))


def test_detect_type_rejects_out_of_range_and_duplicate_labels():
    with pytest.raises(ValueError):
        sureband.detect_type({"type": "noul", "probability": 83})
    with pytest.raises(ValueError):
        sureband.detect_type({"type": "noul", "probability": -0.1})
    with pytest.raises(ValueError):
        sureband.detect_type({"type": "noul", "probability": float("nan")})
    with pytest.raises(ValueError):
        sureband.detect_type(
            {"type": "score", "labels": ["low", "low"], "distribution": [0.5, 0.5]}
        )


def test_decision_renormalizes_near_one_probabilities():
    d = Decision(kind="choice", labels=("a", "b"), probabilities=(0.48, 0.50))
    assert math.fsum(d.probabilities) == pytest.approx(1.0)
    assert d.probabilities == pytest.approx((0.48 / 0.98, 0.50 / 0.98))
    assert aps_score(d.probabilities, 1) <= 1.0


def test_detect_type_noul_scalar():
    d = sureband.detect_type({"type": "noul", "probability": 0.83})
    assert d.kind == "noul"
    assert d.labels == ("false", "true")
    assert d.probabilities == pytest.approx((0.17, 0.83))


def test_detect_type_choice_from_probabilities_dict():
    d = sureband.detect_type(
        {"type": "choice", "probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}}
    )
    assert d.kind == "choice"
    assert d.top_label == "billing"


def test_detect_type_infers_score_from_ordinal_words():
    d = sureband.detect_type({"probabilities": {"low": 0.1, "medium": 0.3, "high": 0.6}})
    assert d.kind == "score"


def test_detect_type_raises_on_unrecognized_shape():
    with pytest.raises(ValueError):
        sureband.detect_type({"nonsense": 123})


def test_quantile_level_finite_sample_boundary():
    # (5 + 1) * 0.9 = 5.4 > 5 -> cannot guarantee 90% coverage without +inf
    assert quantile_level(5, 0.1) == float("inf")
    assert quantile_level(8, 0.1) == float("inf")
    # Exact boundary: (9 + 1) * 0.9 = 9 <= 9 -> 1.0; (1 + 1) * 0.5 = 1 <= 1 -> 1.0
    assert quantile_level(9, 0.1) == 1.0
    assert quantile_level(1, 0.5) == 1.0


def test_quantile_level_matches_known_formula():
    # ceil((n+1)(1-alpha)) / n, from Vovk/Gammerman/Shafer split conformal
    n, alpha = 99, 0.1
    expected = math.ceil((n + 1) * (1 - alpha)) / n
    assert quantile_level(n, alpha) == pytest.approx(expected)


def test_aps_score_is_between_zero_and_one():
    probs = [0.5, 0.3, 0.2]
    for true_index in range(3):
        s = aps_score(probs, true_index)
        assert 0.0 < s <= 1.0


def test_float_drift_and_zero_probability_membership():
    assert aps_score([0.7, 0.2, 0.1], 2) == 1.0
    assert ordinal_score([0.7, 0.2, 0.1], 2) == 1.0
    assert aps_set([0.7, 0.2, 0.1], q_hat=0.9999999999999999) == [0, 1, 2]
    assert aps_set([0.5, 0.3, 0.2], q_hat=0.9999999999999999) == [0, 1, 2]

    for probs in ([0.6, 0.0, 0.4], [0.7, 0.3, 0.0]):
        for true_index in range(len(probs)):
            assert true_index in aps_set(probs, q_hat=aps_score(probs, true_index))
            assert true_index in ordinal_set(probs, q_hat=ordinal_score(probs, true_index))


def test_aps_set_always_contains_top_label_at_high_threshold():
    probs = [0.6, 0.3, 0.1]
    included = aps_set(probs, q_hat=0.99)
    assert 0 in included  # index of the top-probability label


def test_aps_set_is_singleton_at_low_threshold():
    probs = [0.6, 0.3, 0.1]
    included = aps_set(probs, q_hat=0.01)
    assert included == [0]


def test_ordinal_set_is_always_contiguous():
    probs = [0.05, 0.10, 0.55, 0.20, 0.10]  # mode at index 2
    for q in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        included = sorted(ordinal_set(probs, q_hat=q))
        assert included == list(range(included[0], included[-1] + 1)), (
            f"ordinal set not contiguous at q_hat={q}: {included}"
        )


def test_ordinal_score_matches_set_membership():
    probs = [0.05, 0.10, 0.55, 0.20, 0.10]
    for true_index in range(5):
        s = ordinal_score(probs, true_index)
        included = ordinal_set(probs, q_hat=s)
        assert true_index in included


def test_calibrator_rejects_bad_alpha():
    with pytest.raises(ValueError):
        sureband.Calibrator(alpha=0.0)
    with pytest.raises(ValueError):
        sureband.Calibrator(alpha=1.5)


def test_calibrator_end_to_end_small_example():
    cal = sureband.Calibrator(alpha=0.2)
    examples = [
        ({"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.8, "b": 0.2}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}}, "b"),
        ({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}, "b"),
    ]
    cal.fit(examples)
    result = cal.wrap({"type": "choice", "probabilities": {"a": 0.95, "b": 0.05}})
    assert result.kind == "choice"
    assert result.covered_at == pytest.approx(0.8)
    assert "a" in result.prediction_set


def test_calibrator_raises_on_unfitted_kind():
    cal = sureband.Calibrator(alpha=0.1)
    cal.fit([({"type": "noul", "probability": 0.7}, "true")])
    with pytest.raises(RuntimeError):
        cal.wrap({"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}})


def test_calibrator_refit_clears_prior_state():
    cal = sureband.Calibrator(alpha=0.2)
    cal.fit([
        ({"type": "noul", "probability": 0.9}, "true"),
        ({"type": "noul", "probability": 0.8}, "true"),
        ({"type": "noul", "probability": 0.2}, "false"),
        ({"type": "noul", "probability": 0.1}, "false"),
    ])
    assert cal.n_calibration("noul") == 4

    cal.fit([
        ({"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.8, "b": 0.2}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}}, "b"),
        ({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}, "b"),
    ])
    assert cal.n_calibration("noul") == 0
    assert cal.n_calibration("choice") == 4
    with pytest.raises(RuntimeError):
        cal.wrap({"type": "noul", "probability": 0.9})

    with pytest.raises(ValueError):
        cal.fit([])
    assert cal.n_calibration("choice") == 0


def test_convenience_wrap_end_to_end():
    examples = [
        ({"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.8, "b": 0.2}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}}, "b"),
        ({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}, "b"),
    ]
    result = sureband.wrap(
        {"type": "choice", "probabilities": {"a": 0.95, "b": 0.05}},
        examples,
        alpha=0.2,
    )
    assert isinstance(result, ConformalResult)
    assert result.covered_at == pytest.approx(0.8)
    assert "a" in result.prediction_set


def test_typing_get_type_hints_compatible():
    for target in (
        sureband.Calibrator.n_calibration,
        sureband.Calibrator.fit,
        sureband.Calibrator.wrap,
        sureband.wrap,
        sureband.detect_type,
        quantile_level,
        Decision,
        ConformalResult,
    ):
        hints = typing.get_type_hints(target)
        assert isinstance(hints, dict)


def test_version_matches_package():
    assert sureband.__version__ == "0.1.2"
