"""Validate sureband against the datasets the rest of the ecosystem already
publishes numbers on, so results are directly comparable to Laya, Von,
typical, and decider -- not a benchmark we invented ourselves.

Requires network access and the optional `benchmarks` extra:
    pip install "sureband[benchmarks]"

Usage:
    python benchmarks/run_jevbench.py --dataset LocalLLaMA/typed-decisions --model your-model-id

This script does NOT ship its own model. Point --model-fn at a function
that takes (state, questions) and returns raw outputs in sureband's
expected shape, e.g. a thin wrapper around Laya, Von, or a Jev API call.
See `load_model_fn()` below for where to plug that in.
"""

from __future__ import annotations

import argparse
import random
import sys

import sureband


def load_model_fn(model_id: str):
    """Return a callable (state, questions) -> {question_name: raw_output}.

    This is intentionally left as an integration point rather than baked
    in: sureband calibrates *any* System One model's output, it doesn't
    ship one. Wire up whichever model you're benchmarking here, e.g.:

        import laya
        agent = laya.load(model_id)
        return lambda state, questions: agent.predict(state, questions)
    """
    raise NotImplementedError(
        "Wire up your model here -- see the docstring above. sureband "
        "calibrates a model's output, it does not include one."
    )


def run(dataset_name: str, model_id: str, alpha: float, calibration_frac: float, seed: int):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "Missing the 'datasets' package. Install with: "
            "pip install \"sureband[benchmarks]\""
        )

    model_fn = load_model_fn(model_id)
    data = list(load_dataset(dataset_name, split="test"))
    rng = random.Random(seed)
    rng.shuffle(data)

    split_point = int(len(data) * calibration_frac)
    calibration_rows, test_rows = data[:split_point], data[split_point:]

    print(f"Loaded {len(data)} rows from {dataset_name} "
          f"({len(calibration_rows)} calibration / {len(test_rows)} test)")

    calibration_examples = [
        (model_fn(row["state"], row["questions"])[row["question_name"]], row["true_label"])
        for row in calibration_rows
    ]
    calibrator = sureband.Calibrator(alpha=alpha).fit(calibration_examples)

    covered, total_set_size, raw_top1_correct = 0, 0, 0
    for row in test_rows:
        raw = model_fn(row["state"], row["questions"])[row["question_name"]]
        result = calibrator.wrap(raw)
        total_set_size += len(result.prediction_set)
        if row["true_label"] in result.prediction_set:
            covered += 1
        decision = sureband.detect_type(raw)
        if decision.top_label == row["true_label"]:
            raw_top1_correct += 1

    n = len(test_rows)
    print(f"\nTarget coverage:        {1 - alpha:.1%}")
    print(f"Empirical coverage:      {covered / n:.1%}")
    print(f"Mean prediction set size: {total_set_size / n:.2f}")
    print(f"Raw top-1 accuracy:      {raw_top1_correct / n:.1%}  (uncalibrated, for reference)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="LocalLLaMA/typed-decisions")
    parser.add_argument("--model", required=True, help="model id passed to load_model_fn()")
    parser.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage")
    parser.add_argument("--calibration-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.model, args.alpha, args.calibration_frac, args.seed)
