import math

import pytest

import sureband
from sureband.conformal import aps_score, aps_set, ordinal_score, ordinal_set, quantile_level
from sureband.types import Decision


def test_decision_validates_probabilities_sum():
    with pytest.raises(ValueError):
        Decision(kind="choice", labels=("a", "b"), probabilities=(0.9, 0.9))


def test_decision_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Decision(kind="mystery", labels=("a",), probabilities=(1.0,))


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


def test_quantile_level_never_exceeds_one():
    assert quantile_level(5, 0.1) <= 1.0
    assert quantile_level(1, 0.5) <= 1.0


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
