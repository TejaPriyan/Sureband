# sureband

[![PyPI version](https://img.shields.io/pypi/v/sureband.svg?color=blue)](https://pypi.org/project/sureband/)
[![Python versions](https://img.shields.io/pypi/pyversions/sureband.svg)](https://pypi.org/project/sureband/)
[![CI](https://github.com/TejaPriyan/Sureband/actions/workflows/ci.yml/badge.svg)](https://github.com/TejaPriyan/Sureband/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

For an existing function (a FastAPI route, a LangChain/Instructor call, a
raw model `.predict()`), skip the object model entirely with the decorator:

```python
@sureband.calibrated(calibrator)
def classify_ticket(text: str) -> dict:
    return my_model.predict(text)   # unchanged -- still returns a raw dict

classify_ticket("my invoice is wrong")   # now returns a ConformalResult
```

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
- **Coverage is marginal by default, not per-class.** The standard guarantee
  holds *on average* across all your data. If some categories are rare or
  especially critical, use class-conditional (Mondrian) calibration via
  `sureband.Calibrator(alpha=0.1, mondrian=True)` to guarantee coverage for
  each category independently.
- **Calibration is a snapshot, not a subscription.** `Calibrator` doesn't
  update itself after `fit()`. If your traffic distribution drifts, the
  guarantee quietly drifts with it until you refit on fresh data — there's
  no online/streaming recalibration yet.
- **Calibration data must not be training data.** Reusing examples the
  underlying model was trained or fine-tuned on will silently invalidate
  the guarantee — there is no way for sureband to detect this for you.
- **This does not make the underlying model more accurate.** A model that's
  frequently wrong will get frequently *large* prediction sets (including
  "everything, I'm not sure") at your target coverage — which is the
  correct, honest behavior, but it will not feel impressive in a demo.
  sureband tells you when to trust a decision, it doesn't improve the
  decision itself.

## Validated on a real model

Everything above is demonstrated with synthetic data. Here's what happened
running sureband against real Laya (`convaiinnovations/laya`) on real
datasets — no synthetic miscalibration, no cherry-picking:

| Dataset | Decision | Classes | Split | Top-1 acc. | Mean confidence | Gap | Coverage | Mean set size |
|---|---|---|---|---|---|---|---|---|
| AG News | noul (binary) | 2 | 600/600 | 91.5% | 90.5% | −1.0% | 100.0% | 1.71 / 2 |
| AG News | choice | 4 | 300/300 | 92.0% | 75.2% | −16.8% | 100.0% | 3.05 / 4 |
| 20 Newsgroups | choice | 10 | 375/375 | 59.7% | 30.6% | −29.1% | 94.7% | 5.97 / 10 |

Full setup in [`examples/laya_integration.py`](examples/laya_integration.py)
and [`examples/newsgroups_choice_integration.py`](examples/newsgroups_choice_integration.py).

Two findings worth calling out honestly:

- **The coverage guarantee held in all three runs** (≥90% every time), and
  the relative set size shrank monotonically as the label space grew
  (85.5% → 76.3% → 59.7% of all classes retained) — exactly what
  `test_sets_tighten_as_label_space_grows` predicts on synthetic data,
  now confirmed on a real model.
- **Laya was underconfident here, not overconfident** — the opposite of
  what's been reported for it on other tasks (e.g. phishing detection,
  where independent testing found it badly overconfident). This is the
  actual argument for sureband: calibration direction and severity aren't
  a fixed property of a model, they depend on the deployment. A confidence
  fix baked into a checkpoint can't account for that; a calibration step
  run on your own held-out data can.

## Benchmarks

`benchmarks/run_jevbench.py` runs against the same datasets (JevBench,
`LocalLLaMA/typed-decisions`) that Laya, Von, and `typical` already publish
numbers against, rather than a benchmark invented for this project, so
results are directly comparable. It requires network access to Hugging
Face and a model to wrap (sureband calibrates a model's output; it doesn't
ship a model), so it isn't run automatically here — wire up your model in
`load_model_fn()` and run it yourself.

## Status

v0.1.0. The core conformal math is tested (`tests/`, including an
end-to-end coverage check against a deliberately miscalibrated synthetic
model) and validated against real Laya output on AG News and 20
Newsgroups (see "Validated on a real model" above). Not yet tested against
Jev, Von, or `typical`, or on a task with a naturally larger label space
than 10. Issues and PRs welcome.

## License

Apache 2.0
