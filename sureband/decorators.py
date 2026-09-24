"""A decorator so calibrating a function's output doesn't require restructuring
your code around sureband's object model.

    calibrator = sureband.Calibrator(alpha=0.05).fit(calibration_examples)

    @sureband.calibrated(calibrator)
    def classify_ticket(ticket_text: str) -> dict:
        return my_model.predict(ticket_text)   # returns a raw typed-decision dict

    result = classify_ticket("my invoice is wrong")
    # -> ConformalResult, not the raw dict -- callers get the calibrated
    #    prediction set instead of the model's raw (possibly overconfident) output

Works with any function that returns a dict in one of the shapes
`detect_type` recognizes -- a FastAPI route, a LangChain/Instructor call, a
raw OpenAI/Claude structured-output call, or your own model's `.predict()`.
No framework-specific integration code, and no new dependency: this is
plain Python, so it composes with whatever web framework's own decorators
you're already using (put `@app.post(...)` above this one).
"""

from __future__ import annotations

import functools
from typing import Callable

from .calibrator import Calibrator
from .types import ConformalResult


def calibrated(calibrator: Calibrator) -> Callable:
    """Wrap a function so its raw dict output is returned as a ConformalResult.

    The wrapped function must return something `sureband.detect_type` can
    parse (or a `Decision` directly). Async functions are supported.
    """
    def decorator(fn: Callable) -> Callable:
        if _is_coroutine_function(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs) -> ConformalResult:
                raw = await fn(*args, **kwargs)
                return calibrator.wrap(raw)
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> ConformalResult:
            raw = fn(*args, **kwargs)
            return calibrator.wrap(raw)
        return wrapper

    return decorator


def _is_coroutine_function(fn: Callable) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)
