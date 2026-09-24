# sureband

**Distribution-free coverage guarantees for System One decision models.**

Jev, Laya, Von, `typical`, `decider` — every model in this new category
returns a confidence score with its typed decision. None of them guarantee
that score means anything. Out of the box, most are measurably
overconfident (Laya's raw ECE is 0.44; every project in the ecosystem ships
its own ad hoc post-hoc fix).

sureband wraps any of their outputs and, using [split conformal
prediction](https://en.wikipedia.org/wiki/Conformal_prediction), returns a
**prediction set with a statistically guaranteed coverage rate** — "the true
answer is in this set at least 90% of the time," provably, not just
calibrated-looking — from a few hundred labeled examples and no retraining.

```python
import sureband

calibrator = sureband.Calibrator(alpha=0.1)   # target: at least 90% coverage
calibrator.fit(calibration_examples)           # list[(raw_model_output, true_label)]

result = calibrator.wrap(new_model_output)
result.prediction_set   # e.g. {"billing"} or {"billing", "technical"} if ambiguous
result.is_certain       # False when the set has more than one label
result.covered_at        # 0.9 -- the guarantee this set was built to satisfy
```

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable,
no-dependencies example.

## Install

```bash
pip install sureband
```

## Why not just use the model's own confidence score?

Because "confidence" from these models is whatever the training objective
produced, not a calibrated probability. sureband doesn't change the model —
it sits after it, using a held-out calibration set to learn exactly how much
to trust (or distrust) the model's own probabilities, then builds sets that
hit your target coverage regardless of how mis-calibrated the raw scores
are. The [`test_coverage.py`](tests/test_coverage.py) suite demonstrates
this directly: it deliberately builds an overconfident synthetic model and
checks the guarantee still holds.

## Works with any typed-decision output

sureband auto-detects the three primitives this whole model category
shares:

| kind     | shape                                              | example |
|----------|-----------------------------------------------------|---------|
| `choice` | probability per named option                       | `{"type": "choice", "probabilities": {"billing": 0.7, "sales": 0.3}}` |
| `noul`   | a single P(true) scalar                             | `{"type": "noul", "probability": 0.83}` |
| `score`  | probability per level of an ordered rubric          | `{"type": "score", "labels": ["low","med","high"], "distribution": [.1,.3,.6]}` |

If your model's output doesn't match one of the common shapes `detect_type`
recognizes, build a `sureband.Decision` directly — see
[`sureband/types.py`](sureband/types.py).

## Honest limitations

This section exists because the rest of this ecosystem's benchmark claims
have not always held up under independent testing, and we'd rather you find
out the limits from us than from a critique.

- **The guarantee is a lower bound, not an exact rate.** Split conformal
  promises coverage *of at least* your target, not exactly your target. The
  non-randomized method used here (chosen deliberately, so `wrap()` is
  deterministic — important if it's sitting in an automated pipeline) tends
  to **over-cover**, sometimes substantially, when the label space is small.
  A binary `noul` decision, for instance, often just gets you "certain" or
  "could be either," with little room in between. Sets get meaningfully
  tighter and closer to your target as the number of options grows — see
  `test_sets_tighten_as_label_space_grows` in the test suite for measured
  numbers at 4, 10, and 25 classes.
- **The guarantee needs exchangeable calibration data.** If your production
  traffic drifts from what you calibrated on (new ticket categories, a
  shifted user base), the coverage guarantee degrades along with it. Recalibrate
  periodically on fresh held-out data, the same way you'd re-validate any
  monitored model.
- **Calibration data must not be training data.** Reusing examples the
  underlying model was trained or fine-tuned on will silently invalidate
  the guarantee — there is no way for sureband to detect this for you.
- **This does not make the underlying model more accurate.** A model that's
  frequently wrong will get frequently *large* prediction sets (including
  "everything, I'm not sure") at your target coverage — which is the
  correct, honest behavior, but it will not feel impressive in a demo.
  sureband tells you when to trust a decision, it doesn't improve the
  decision itself.

## Benchmarks

`benchmarks/run_jevbench.py` runs against the same datasets (JevBench,
`LocalLLaMA/typed-decisions`) that Laya, Von, and `typical` already publish
numbers against, rather than a benchmark invented for this project, so
results are directly comparable. It requires network access to Hugging
Face and a model to wrap (sureband calibrates a model's output; it doesn't
ship a model), so it isn't run automatically here — wire up your model in
`load_model_fn()` and run it yourself.

## Status

Early — v0.1.0. The core conformal math is tested (`tests/`, including an
end-to-end coverage check against a deliberately miscalibrated synthetic
model), but this hasn't yet been run against a real Jev/Laya/Von output at
scale. Issues and PRs welcome once this is public.

## License

Apache 2.0
