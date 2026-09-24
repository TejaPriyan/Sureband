"""Validate sureband against Laya using its natural 4-way choice output.

Instead of four separate yes/no (noul) questions, we ask one question
with all four AG News categories as options -- the way Laya is actually
designed to be used for classification.

Run with:
    python examples/laya_choice_integration.py

Needs `pip install laya datasets sureband` and the weights already cached
(or network access to Hugging Face on first run).
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

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

CHOICE_QUESTION = {
    "type": "choice",
    "instructions": "What is the main topic of this news article?",
    "criteria": {
        "World":    "International news, politics, diplomacy, global affairs",
        "Sports":   "Sports, athletes, games, tournaments, competitions",
        "Business": "Economics, companies, finance, markets, trade",
        "Sci/Tech": "Science, technology, innovation, engineering, research",
    },
}


def build_examples(n_per_class: int = 150, seed: int = 0) -> list:
    """Sample n_per_class articles per AG News label (1200 total).

    Each example is (article_text, true_label_name).
    Labels in the dataset are 1-indexed; we convert to 0-indexed here.
    """
    ds = load_dataset("Chetah/ag_news", split="test")
    rng = random.Random(seed)

    by_label: dict[int, list[str]] = {i: [] for i in range(len(LABEL_NAMES))}
    for row in ds:
        text = f"{row['title']}. {row['description']}"
        # dataset labels are 1-indexed: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
        label_idx = row["label"] - 1
        by_label[label_idx].append(text)

    examples: list[tuple[str, str]] = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        sampled = rng.sample(by_label[label_idx], n_per_class)
        for text in sampled:
            examples.append((text, label_name))

    rng.shuffle(examples)
    return examples


def ask(agent, text: str) -> dict:
    """Single forward pass: one article, one 4-way choice question."""
    result = agent.predict(
        {"article": text},
        {"topic": CHOICE_QUESTION},
    )
    return result["answers"]["topic"]


def run():
    print("Loading Laya (convaiinnovations/laya)...")
    agent = laya.load("convaiinnovations/laya")

    print("Building evaluation set from AG News (4-way choice)...")
    examples = build_examples(n_per_class=150)
    split = len(examples) // 2
    calibration_rows, test_rows = examples[:split], examples[split:]
    print(f"{len(calibration_rows)} calibration examples, {len(test_rows)} test examples")

    # ------------------------------------------------------------------ #
    # Query Laya for both splits                                           #
    # ------------------------------------------------------------------ #
    print("\nQuerying Laya for the calibration set...")
    calibration_examples: list[tuple[dict, str]] = []
    for i, (text, true_label) in enumerate(calibration_rows):
        raw = ask(agent, text)
        calibration_examples.append((raw, true_label))
        if (i + 1) % 100 == 0 or (i + 1) == len(calibration_rows):
            print(f"  Calibration: {i + 1}/{len(calibration_rows)} done", flush=True)

    print("\nQuerying Laya for the test set...")
    test_examples: list[tuple[dict, str]] = []
    for i, (text, true_label) in enumerate(test_rows):
        raw = ask(agent, text)
        test_examples.append((raw, true_label))
        if (i + 1) % 100 == 0 or (i + 1) == len(test_rows):
            print(f"  Test: {i + 1}/{len(test_rows)} done", flush=True)

    # ------------------------------------------------------------------ #
    # Raw Laya: accuracy vs stated confidence                              #
    # ------------------------------------------------------------------ #
    n = len(test_examples)
    correct = 0
    mean_conf = 0.0
    for raw, true_label in test_examples:
        if raw["choice"] == true_label:
            correct += 1
        mean_conf += raw["confidence"]

    accuracy = correct / n
    mean_confidence = mean_conf / n

    print("\n=== Raw Laya (uncalibrated, 4-way choice) ===")
    print(f"Accuracy:               {accuracy:.1%}")
    print(f"Mean stated confidence: {mean_confidence:.1%}")
    print(f"Gap (conf - acc):       {mean_confidence - accuracy:+.1%}  "
          f"({'overconfident' if mean_confidence > accuracy else 'underconfident'})")

    # ------------------------------------------------------------------ #
    # sureband calibrated coverage                                         #
    # ------------------------------------------------------------------ #
    calibrator = sureband.Calibrator(alpha=0.1).fit(calibration_examples)
    covered = 0
    total_set_size = 0
    for raw, true_label in test_examples:
        result = calibrator.wrap(raw)
        total_set_size += len(result.prediction_set)
        if true_label in result.prediction_set:
            covered += 1

    print("\n=== sureband (alpha=0.1, target >= 90% coverage) ===")
    print(f"Empirical coverage: {covered / n:.1%}")
    print(f"Mean set size:      {total_set_size / n:.2f}  (of {len(LABEL_NAMES)} possible)")


if __name__ == "__main__":
    run()
