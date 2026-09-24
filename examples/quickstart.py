"""Quickstart: calibrating a 'choice' decision from a support-ticket router.

Run with: python examples/quickstart.py

This uses hand-written example outputs shaped like what Laya or Jev return,
so it runs with no model, no API key, and no network access. Swap
`model_output` for whatever your model actually returns.
"""

import sureband

# A held-out calibration set: (raw model output, true label) pairs the
# model was NOT trained or fine-tuned on. A few hundred is a reasonable
# starting point; more gives a tighter, less noisy threshold.
calibration_set = [
    ({"type": "choice", "probabilities": {"billing": 0.91, "technical": 0.06, "sales": 0.03}}, "billing"),
    ({"type": "choice", "probabilities": {"billing": 0.55, "technical": 0.40, "sales": 0.05}}, "technical"),
    ({"type": "choice", "probabilities": {"billing": 0.30, "technical": 0.65, "sales": 0.05}}, "technical"),
    ({"type": "choice", "probabilities": {"billing": 0.10, "technical": 0.15, "sales": 0.75}}, "sales"),
    ({"type": "choice", "probabilities": {"billing": 0.48, "technical": 0.47, "sales": 0.05}}, "billing"),
    ({"type": "choice", "probabilities": {"billing": 0.20, "technical": 0.70, "sales": 0.10}}, "billing"),
    ({"type": "choice", "probabilities": {"billing": 0.85, "technical": 0.10, "sales": 0.05}}, "billing"),
    ({"type": "choice", "probabilities": {"billing": 0.15, "technical": 0.20, "sales": 0.65}}, "sales"),
    # ... a real calibration set should have a few hundred of these, not 8.
]

# alpha=0.1 targets at least 90% coverage: the true department is in the
# returned set at least 90% of the time, guaranteed, not just "usually."
calibrator = sureband.Calibrator(alpha=0.1)
calibrator.fit(calibration_set)

# A new, ambiguous ticket the router is unsure about
model_output = {
    "type": "choice",
    "probabilities": {"billing": 0.52, "technical": 0.45, "sales": 0.03},
}

result = calibrator.wrap(model_output)

print("Prediction set:", sorted(result.prediction_set))
print("Guaranteed coverage:", f"{result.covered_at:.0%}")
print("Certain (single answer)?", result.is_certain)

if result.is_certain:
    print(f"-> auto-route to {result.top_label}")
else:
    print(f"-> ambiguous between {sorted(result.prediction_set)}, send to a human")
