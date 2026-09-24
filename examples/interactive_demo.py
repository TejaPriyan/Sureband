"""Interactive demo of sureband conformal prediction.

Run with:
    python examples/interactive_demo.py
"""

import sys
import sureband

# Pre-packaged calibration data for support ticket routing
CALIBRATION_DATA = [
    # Billing
    ({"type": "choice", "probabilities": {"billing": 0.85, "technical": 0.10, "sales": 0.05}}, "billing"),
    ({"type": "choice", "probabilities": {"billing": 0.70, "technical": 0.20, "sales": 0.10}}, "billing"),
    ({"type": "choice", "probabilities": {"billing": 0.60, "technical": 0.30, "sales": 0.10}}, "billing"),
    ({"type": "choice", "probabilities": {"billing": 0.40, "technical": 0.50, "sales": 0.10}}, "billing"),
    # Technical
    ({"type": "choice", "probabilities": {"billing": 0.05, "technical": 0.88, "sales": 0.07}}, "technical"),
    ({"type": "choice", "probabilities": {"billing": 0.15, "technical": 0.75, "sales": 0.10}}, "technical"),
    ({"type": "choice", "probabilities": {"billing": 0.30, "technical": 0.55, "sales": 0.15}}, "technical"),
    ({"type": "choice", "probabilities": {"billing": 0.45, "technical": 0.40, "sales": 0.15}}, "technical"),
    # Sales
    ({"type": "choice", "probabilities": {"billing": 0.10, "technical": 0.10, "sales": 0.80}}, "sales"),
    ({"type": "choice", "probabilities": {"billing": 0.25, "technical": 0.15, "sales": 0.60}}, "sales"),
    ({"type": "choice", "probabilities": {"billing": 0.35, "technical": 0.20, "sales": 0.45}}, "sales"),
] * 20  # Replicate to simulate ~220 calibration examples


def simulate_model_inference(query: str) -> dict:
    """Mock model predicting probability distribution based on query keywords."""
    q = query.lower()
    if any(w in q for w in ["refund", "charge", "credit", "invoice", "payment", "card"]):
        return {"type": "choice", "probabilities": {"billing": 0.78, "technical": 0.14, "sales": 0.08}}
    elif any(w in q for w in ["bug", "crash", "error", "api", "500", "broken", "down"]):
        return {"type": "choice", "probabilities": {"billing": 0.06, "technical": 0.86, "sales": 0.08}}
    elif any(w in q for w in ["pricing", "enterprise", "plan", "demo", "buy", "discount"]):
        return {"type": "choice", "probabilities": {"billing": 0.35, "technical": 0.10, "sales": 0.55}}
    else:
        # Ambiguous query
        return {"type": "choice", "probabilities": {"billing": 0.42, "technical": 0.38, "sales": 0.20}}


def display_result(query: str, raw: dict, cal_90: sureband.Calibrator, cal_95: sureband.Calibrator):
    probs = raw["probabilities"]
    top_label = max(probs, key=probs.get)
    top_conf = probs[top_label]

    res_90 = cal_90.wrap(raw)
    res_95 = cal_95.wrap(raw)

    print("\n" + "=" * 65)
    print(f"Query: \"{query}\"")
    print("-" * 65)
    print("Raw Model Output:")
    for label, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(p * 30)
        print(f"  {label:<12} {p:>5.1%}  [{bar:<30}]")
    print(f"  Top Decision:   {top_label} (reported confidence: {top_conf:.1%})")

    print("\nsureband Calibrated Guarantees:")
    print(f"  Coverage >= 90% (alpha=0.10):")
    print(f"    Prediction Set: {sorted(list(res_90.prediction_set))}")
    print(f"    Certain?        {'YES (Singleton)' if res_90.is_certain else 'NO (Ambiguous, review required)'}")

    print(f"  Coverage >= 95% (alpha=0.05):")
    print(f"    Prediction Set: {sorted(list(res_95.prediction_set))}")
    print(f"    Certain?        {'YES (Singleton)' if res_95.is_certain else 'NO (Ambiguous, review required)'}")
    print("=" * 65)


def main():
    print("Initializing sureband Calibrators (alpha=0.10 and alpha=0.05)...")
    cal_90 = sureband.Calibrator(alpha=0.10).fit(CALIBRATION_DATA)
    cal_95 = sureband.Calibrator(alpha=0.05).fit(CALIBRATION_DATA)

    sample_queries = [
        "I was charged twice on my credit card last week.",
        "The Python API returns a 500 error when sending batch payloads.",
        "We are an enterprise team looking for volume pricing options.",
        "Can you help me with my account subscription settings?",
    ]

    print("\n--- Running Sample Inquiries ---")
    for q in sample_queries:
        raw = simulate_model_inference(q)
        display_result(q, raw, cal_90, cal_95)

    print("\nInteractive mode: Type an inquiry (or 'q' to quit):")
    while True:
        try:
            line = input("\nEnter query > ").strip()
            if not line or line.lower() == "q":
                break
            raw = simulate_model_inference(line)
            display_result(line, raw, cal_90, cal_95)
        except (EOFError, KeyboardInterrupt):
            break
    print("\nExiting interactive demo.")


if __name__ == "__main__":
    main()
