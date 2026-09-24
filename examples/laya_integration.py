"""Validate sureband against a REAL model: Laya (convaiinnovations/laya).

Run this on your own machine -- it needs `pip install laya` and network
access to Hugging Face to download weights (~421M params), neither of
which is available in the sandbox this was written in.

    pip install laya sureband
    python examples/laya_integration.py

What this does:
  1. Loads real Laya, asks it a batch of yes/no questions on a public
     dataset (AG News, reframed as noul-style binary questions so it's easy
     to get ground truth for free -- swap in your own labeled data for a
     real evaluation).
  2. Splits the data: half to calibrate sureband, half to test it.
  3. Reports Laya's raw confidence vs. its actual accuracy (the gap IS the
     miscalibration problem), then reports sureband's calibrated coverage
     on the same held-out half.

If Laya's raw confidence already matches its accuracy on your data, this
will correctly show sureband adding little value -- that's the honest
outcome, and worth reporting either way.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import sureband

try:
    import laya
except ImportError:
    raise SystemExit("Run: pip install laya   (see module docstring)")

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Run: pip install datasets   (see module docstring)")


def build_noul_examples(n_per_class=150, seed=0):
    """AG News, reframed as a batch of independent yes/no questions:
    'is this World news?', 'is this Sports?', etc. Gives free ground truth
    without needing a labeled typed-decisions set of your own to start.
    """
    ds = load_dataset("Chetah/ag_news", split="test")
    label_names = ["World", "Sports", "Business", "Sci/Tech"]  # Standard AG News labels
    rng = random.Random(seed)

    by_label = {i: [] for i in range(len(label_names))}
    for row in ds:
        # Combine title and description into a single text field
        text = f"{row['title']}. {row['description']}"
        # Labels are 1-indexed in this dataset, convert to 0-indexed
        label_idx = row["label"] - 1
        by_label[label_idx].append(text)

    examples = []  # (state, question_topic, is_actually_that_topic: bool)
    for label_idx, topic in enumerate(label_names):
        texts = by_label[label_idx][:n_per_class]
        for t in texts:
            examples.append((t, topic, True))
        # negatives: articles from other topics, asked "is this <topic>?"
        other_texts = [
            txt for i, txts in by_label.items() if i != label_idx for txt in txts
        ]
        for t in rng.sample(other_texts, n_per_class):
            examples.append((t, topic, False))

    rng.shuffle(examples)
    return examples


def run():
    print("Loading Laya (convaiinnovations/laya)...")
    agent = laya.load("convaiinnovations/laya")

    print("Building evaluation set from AG News...")
    examples = build_noul_examples(n_per_class=150)
    split = len(examples) // 2
    calibration_rows, test_rows = examples[:split], examples[split:]
    print(f"{len(calibration_rows)} calibration examples, {len(test_rows)} test examples")

    def ask(state_text, topic):
        result = agent.predict(
            {"article": state_text},
            {"is_topic": {
                "type": "noul",
                "instructions": f"Is this article about {topic}?",
            }},
        )
        # Laya's actual output structure: result["answers"]["is_topic"]
        # which contains "noul" and "confidence" keys
        return result["answers"]["is_topic"]

    print("\nQuerying Laya for the calibration set...")
    calibration_examples = []
    for i, (text, topic, is_true) in enumerate(calibration_rows):
        raw = ask(text, topic)
        calibration_examples.append((raw, "true" if is_true else "false"))
        if (i + 1) % 100 == 0 or (i + 1) == len(calibration_rows):
            print(f"  Calibration: {i + 1}/{len(calibration_rows)} done", flush=True)

    print("\nQuerying Laya for the test set...")
    test_examples = []
    for i, (text, topic, is_true) in enumerate(test_rows):
        raw = ask(text, topic)
        test_examples.append((raw, "true" if is_true else "false"))
        if (i + 1) % 100 == 0 or (i + 1) == len(test_rows):
            print(f"  Test: {i + 1}/{len(test_rows)} done", flush=True)

    # --- Raw Laya, uncalibrated ---
    correct, mean_conf = 0, 0.0
    for raw, true_label in test_examples:
        # Laya's output has "noul" and "confidence" keys
        p_true = raw.get("noul", raw.get("confidence", raw.get("probability", raw.get("p", 0.5))))
        predicted = "true" if p_true >= 0.5 else "false"
        confidence = p_true if predicted == "true" else 1 - p_true
        mean_conf += confidence
        if predicted == true_label:
            correct += 1
    n = len(test_examples)
    print("\n=== Raw Laya (uncalibrated) ===")
    print(f"Accuracy:            {correct / n:.1%}")
    print(f"Mean stated confidence: {mean_conf / n:.1%}")
    print(f"Gap (overconfidence):   {mean_conf / n - correct / n:+.1%}")

    # --- sureband, calibrated ---
    calibrator = sureband.Calibrator(alpha=0.1).fit(calibration_examples)
    covered, total_size = 0, 0
    for raw, true_label in test_examples:
        result = calibrator.wrap(raw)
        total_size += len(result.prediction_set)
        if true_label in result.prediction_set:
            covered += 1
    print("\n=== sureband (alpha=0.1, target >= 90% coverage) ===")
    print(f"Empirical coverage: {covered / n:.1%}")
    print(f"Mean set size:      {total_size / n:.2f} (of 2 possible)")


if __name__ == "__main__":
    run()
