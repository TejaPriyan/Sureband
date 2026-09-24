# Sureband Community Launch Kit

This guide contains pre-formatted, ready-to-post announcements tailored for Hacker News, Reddit, Twitter/X, and LinkedIn.

---

## 1. Hacker News — "Show HN"

**Title**: 
`Show HN: Sureband – Distribution-free coverage guarantees for fast decision models`

**Body**:
```text
Hey HN,

Fast "System One" decision models (like Laya, Jev, and small fine-tunes) are increasingly used for low-latency classification (triage, routing, intent detection, content moderation). While they output a confidence score, that number is often poorly calibrated out of the box.

We built Sureband (https://github.com/TejaPriyan/Sureband), an open-source Python library that wraps any typed-decision model output and provides distribution-free coverage guarantees via split conformal prediction.

Instead of guessing whether 75% confidence means a 1-in-4 error rate, you specify your risk tolerance (e.g., alpha=0.10 for 90% target coverage). Sureband calibrates on a few hundred held-out labeled examples and returns a prediction set guaranteed to contain the ground truth >= (1 - alpha) of the time:

```python
import sureband

calibrator = sureband.Calibrator(alpha=0.10).fit(calibration_data)
result = calibrator.wrap(model_output)

result.prediction_set  # e.g., {'billing'} or {'billing', 'technical'} if ambiguous
result.is_certain      # False if human review or fallback routing is required
```

### What we found testing against real models:
We validated Sureband against real Laya (convaiinnovations/laya) across AG News (binary & 4-way) and 20 Newsgroups (10-way choice):

1. Guaranteed Coverage Held: Across all tasks, empirical coverage met or exceeded the 90% bound (100.0%, 100.0%, and 94.7%).
2. Sets Tighten as Label Space Scales: Because non-randomized APS is conservative on small label spaces, set size ratios shrank monotonically as categories grew:
   - 2-class: 85.5% of label space
   - 4-class: 76.3% of label space
   - 10-class: 59.7% of label space (pruning ~4 out of 10 labels on average)
3. Model Miscalibration Varies by Task: On 10-way choice, Laya was severely underconfident (59.7% accuracy vs 30.6% mean confidence). Conformal prediction adapts to overconfidence and underconfidence alike without retraining.

Available on PyPI: `pip install sureband`
Repo: https://github.com/TejaPriyan/Sureband

Would love to hear feedback or thoughts on class-conditional / Mondrian extensions!
```

---

## 2. Reddit — r/MachineLearning & r/LocalLLaMA

**Post Title**:
`[P] Sureband: Distribution-free conformal prediction sets for LLM classification & small decision models`

**Post Body**:
```text
Hi everyone,

Small classification models and LLM structured outputs (like Laya, Jev, Instructor, and small BERT/LLaMA classifiers) are great for real-time routing, but their reported confidence scores often don't correlate reliably with actual accuracy.

We created **Sureband** (https://github.com/TejaPriyan/Sureband), a lightweight library that brings split conformal prediction to typed decision outputs.

### Key Features:
- **Provable Guarantees**: Target 90% or 95% coverage, and prediction sets are mathematically guaranteed to contain the true label with that frequency without distributional assumptions.
- **Ambiguity Detection**: If `result.is_certain` is False, the set contains multiple plausible labels, indicating that automated decisions should be flagged for human review or secondary routing.
- **Mondrian (Class-Conditional) Support**: Independent coverage guarantees per category so rare classes aren't under-covered by the average.
- **Zero-Friction Decorator**:
  ```python
  @sureband.calibrated(calibrator)
  def classify_ticket(text: str) -> dict:
      return model.predict(text)
  ```

PyPI: `pip install sureband`
GitHub: https://github.com/TejaPriyan/Sureband

Check out the full empirical benchmarks and tests in the repository!
```

---

## 3. Twitter / X Thread

**Tweet 1**:
> Excited to release Sureband 🎯 — a Python library adding distribution-free coverage guarantees (via split conformal prediction) to fast decision models & LLM classifiers.
> 
> "The true answer is in this prediction set ≥ 90% of the time," mathematically guaranteed without retraining.
> 
> 📦 `pip install sureband`
> 🔗 https://github.com/TejaPriyan/Sureband

**Tweet 2**:
> Why does this matter?
> Most small decision models are miscalibrated. In our benchmarks against Laya on 10-way topic classification, the model achieved 59.7% accuracy with only 30.6% mean confidence (-29.1% underconfident).
> 
> Fixed temperature heuristics fail; held-out conformal calibration works.

**Tweet 3**:
> Sureband turns raw outputs into guaranteed prediction sets:
> - If `result.is_certain`: safe to automate with confidence.
> - If ambiguous: returns multiple candidates for human escalation.
> 
> Includes Mondrian class-conditional support & simple decorators.
> ⭐ Star it on GitHub: https://github.com/TejaPriyan/Sureband
