"""
sureband — distribution-free coverage guarantees for System One decision models.

Wraps the raw output of any typed-decision model (Jev, Laya, Von, typical,
decider, or your own) and turns its confidence score into a prediction set
with a statistically guaranteed coverage rate, via split conformal prediction.

    import sureband

    cal = sureband.Calibrator(alpha=0.1)          # target 90% coverage
    cal.fit(calibration_examples)                  # list[(raw_output, true_label)]
    result = cal.wrap(new_raw_output)

    result.prediction_set   # e.g. {"billing"} or {"billing", "technical"}
    result.confidence       # calibrated confidence for the top choice
    result.covered_at       # 0.90 -- the guarantee this set was built to satisfy
"""

from .types import Decision, ConformalResult
from .detect import detect_type
from .calibrator import Calibrator
from .convenience import wrap
from .decorators import calibrated

__all__ = [
    "Decision",
    "ConformalResult",
    "Calibrator",
    "detect_type",
    "wrap",
    "calibrated",
]

__version__ = "0.1.0"
