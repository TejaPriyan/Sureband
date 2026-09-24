"""Validate sureband on 20 Newsgroups — a 10-way choice task.

Uses sklearn's 20 Newsgroups dataset (no Hugging Face needed, already
cached locally). We pick 10 semantically well-separated categories to
give Laya a fair, unambiguous choice task, then test whether sureband's
mean set size is tighter than on the 4-way AG News run.

Hypothesis from test_sets_tighten_as_label_space_grows (synthetic):
  set_size / n_labels should *decrease* as n_labels grows, because
  more probability mass is spread across wrong options so the threshold
  separates them more cleanly.

Run with:
    python examples/newsgroups_choice_integration.py
"""

from __future__ import annotations

import os
import random
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sureband

try:
    import laya
except ImportError:
    raise SystemExit("Run: pip install laya")

try:
    from sklearn.datasets import fetch_20newsgroups
except ImportError:
    raise SystemExit("Run: pip install scikit-learn")

# 10 semantically distinct categories — avoids near-duplicate pairs like
# comp.sys.ibm.pc.hardware / comp.sys.mac.hardware or the two hockey/baseball
# groups that share massive boilerplate.
SELECTED_CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "misc.forsale",
    "rec.autos",
    "rec.sport.baseball",
    "sci.crypt",
    "sci.med",
    "sci.space",
    "talk.politics.guns",
]

# Human-readable label → Laya criteria description
CRITERIA = {
    "alt.atheism":            "Posts about atheism, secularism, and debate about religion from a non-religious perspective",
    "comp.graphics":          "Computer graphics, image rendering, 3D modelling, and visualization software",
    "comp.os.ms-windows.misc":"Microsoft Windows operating system: tips, problems, drivers, setup questions",
    "misc.forsale":           "Items for sale or wanted: classified ads, prices, buying/selling offers",
    "rec.autos":              "Cars, automobiles, driving, car maintenance, reviews, and road trips",
    "rec.sport.baseball":     "Baseball: games, teams, players, statistics, and league news",
    "sci.crypt":              "Cryptography, encryption, privacy, security, and related mathematics",
    "sci.med":                "Medicine, health, diseases, pharmaceuticals, and medical research",
    "sci.space":              "Space exploration, astronomy, NASA, rockets, satellites, and the universe",
    "talk.politics.guns":     "Gun control, firearms policy, the Second Amendment, and related politics",
}

N_LABELS = len(SELECTED_CATEGORIES)
N_PER_CLASS = 75   # 75 per class × 10 = 750 total → 375 calibration + 375 test


def build_examples(seed: int = 42) -> list[tuple[str, str]]:
    """Return (text, true_label) pairs, balanced across selected categories."""
    ds = fetch_20newsgroups(
        subset="all",
        categories=SELECTED_CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )
    rng = random.Random(seed)

    by_cat: dict[str, list[str]] = defaultdict(list)
    for text, label_idx in zip(ds.data, ds.target):
        cat = ds.target_names[label_idx]
        if len(text.strip()) > 50:          # skip near-empty posts
            by_cat[cat].append(text.strip())

    examples: list[tuple[str, str]] = []
    for cat in SELECTED_CATEGORIES:
        sampled = rng.sample(by_cat[cat], N_PER_CLASS)
        for text in sampled:
            # Trim to ~500 chars to keep Laya's context short on CPU
            examples.append((text[:500], cat))

    rng.shuffle(examples)
    return examples


def ask(agent, text: str) -> dict:
    """One forward pass: 10-way choice question."""
    result = agent.predict(
        {"post": text},
        {"category": {
            "type": "choice",
            "instructions": "Which online discussion category does this post belong to?",
            "criteria": CRITERIA,
        }},
    )
    return result["answers"]["category"]


def run():
    print("Loading Laya...")
    agent = laya.load("convaiinnovations/laya")

    print(f"\nBuilding evaluation set ({N_LABELS} categories × {N_PER_CLASS} per class)...")
    examples = build_examples()
    split = len(examples) // 2
    calibration_rows, test_rows = examples[:split], examples[split:]
    print(f"{len(calibration_rows)} calibration examples, {len(test_rows)} test examples")
    print(f"Categories ({N_LABELS}): {', '.join(SELECTED_CATEGORIES)}")

    # ------------------------------------------------------------------ #
    # Query Laya                                                           #
    # ------------------------------------------------------------------ #
    print("\nQuerying Laya for the calibration set...")
    calibration_examples: list[tuple[dict, str]] = []
    for i, (text, true_label) in enumerate(calibration_rows):
        raw = ask(agent, text)
        calibration_examples.append((raw, true_label))
        if (i + 1) % 50 == 0 or (i + 1) == len(calibration_rows):
            print(f"  Calibration: {i + 1}/{len(calibration_rows)} done", flush=True)

    print("\nQuerying Laya for the test set...")
    test_examples: list[tuple[dict, str]] = []
    for i, (text, true_label) in enumerate(test_rows):
        raw = ask(agent, text)
        test_examples.append((raw, true_label))
        if (i + 1) % 50 == 0 or (i + 1) == len(test_rows):
            print(f"  Test: {i + 1}/{len(test_rows)} done", flush=True)

    # ------------------------------------------------------------------ #
    # Raw Laya metrics                                                     #
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

    print(f"\n=== Raw Laya (uncalibrated, {N_LABELS}-way choice) ===")
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

    mean_set_size = total_set_size / n
    print(f"\n=== sureband (alpha=0.1, target >= 90% coverage) ===")
    print(f"Empirical coverage: {covered / n:.1%}")
    print(f"Mean set size:      {mean_set_size:.2f}  (of {N_LABELS} possible)")
    print(f"Set size ratio:     {mean_set_size / N_LABELS:.1%}  (mean_set / n_labels)")

    # ------------------------------------------------------------------ #
    # Comparison table                                                     #
    # ------------------------------------------------------------------ #
    print("\n=== Comparison: set size ratio across label-space sizes ===")
    print(f"{'Task':<30} {'Labels':>6} {'MeanSet':>8} {'Ratio':>8}")
    print("-" * 56)
    print(f"{'AG News noul (binary)':<30} {'2':>6} {'1.71':>8} {'85.5%':>8}")
    print(f"{'AG News 4-way choice':<30} {'4':>6} {'3.05':>8} {'76.3%':>8}")
    print(f"{'20 Newsgroups 10-way':<30} {N_LABELS:>6} {mean_set_size:>8.2f} {mean_set_size/N_LABELS:>8.1%}")


if __name__ == "__main__":
    run()
