import asyncio

import pytest

import sureband


def _fitted_calibrator():
    examples = [
        ({"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.8, "b": 0.2}}, "a"),
        ({"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}}, "b"),
        ({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}, "b"),
    ]
    return sureband.Calibrator(alpha=0.2).fit(examples)


def test_calibrated_wraps_sync_function_output():
    cal = _fitted_calibrator()

    @sureband.calibrated(cal)
    def classify(text: str) -> dict:
        return {"type": "choice", "probabilities": {"a": 0.95, "b": 0.05}}

    result = classify("some ticket")
    assert isinstance(result, sureband.ConformalResult)
    assert "a" in result.prediction_set


def test_calibrated_preserves_function_metadata():
    cal = _fitted_calibrator()

    @sureband.calibrated(cal)
    def classify(text: str) -> dict:
        """docstring"""
        return {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}

    assert classify.__name__ == "classify"
    assert classify.__doc__ == "docstring"


def test_calibrated_wraps_async_function_output():
    cal = _fitted_calibrator()

    @sureband.calibrated(cal)
    async def classify(text: str) -> dict:
        return {"type": "choice", "probabilities": {"a": 0.95, "b": 0.05}}

    result = asyncio.run(classify("some ticket"))
    assert isinstance(result, sureband.ConformalResult)
    assert "a" in result.prediction_set


def test_calibrated_passes_through_args_and_kwargs():
    cal = _fitted_calibrator()
    seen = {}

    @sureband.calibrated(cal)
    def classify(text: str, *, priority: str = "normal") -> dict:
        seen["text"] = text
        seen["priority"] = priority
        return {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}

    classify("hello", priority="high")
    assert seen == {"text": "hello", "priority": "high"}
