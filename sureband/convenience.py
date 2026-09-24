"""One-line convenience API for quick scripts and demos.

For any real usage, prefer fitting a Calibrator once and reusing it --
refitting on every call as this function does is wasteful and, if your
calibration set is small, wastes statistical power too.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

from .calibrator import Calibrator
from .types import ConformalResult, Decision


def wrap(
    raw: Union[dict, Decision],
    calibration_set: Iterable[Tuple[Union[dict, Decision], str]],
    alpha: float = 0.1,
) -> ConformalResult:
    """Fit a Calibrator on `calibration_set` and immediately wrap `raw`.

    Handy for a quick script or a notebook cell. In a service, build one
    Calibrator with `sureband.Calibrator(...).fit(...)` at startup and call
    `.wrap()` on it per request instead.
    """
    cal = Calibrator(alpha=alpha)
    cal.fit(calibration_set)
    return cal.wrap(raw)

