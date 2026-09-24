"""Turn a raw System One model output into a normalized Decision.

Different models spell their output slightly differently (Laya nests a
"probabilities" dict, a noul-style yes/no often comes back as a single
scalar, etc). This module accepts the common shapes so the rest of the
library only has to deal with one representation. If your model's output
doesn't match, build a Decision directly instead of relying on detection.
"""

from __future__ import annotations

from typing import Any, Dict

from .types import Decision

_PROB_DICT_KEYS = ("probabilities", "probs", "scores", "distribution_by_label")
_SCALAR_PROB_KEYS = ("noul", "p_true", "probability", "prob", "p", "confidence")
_LABELS_KEYS = ("labels", "criteria", "levels", "options")
_DIST_KEYS = ("distribution", "dist")


def detect_type(raw: Dict[str, Any]) -> Decision:
    """Normalize a raw model output dict into a Decision.

    Accepts, among other shapes:
        {"type": "choice", "probabilities": {"billing": 0.7, "sales": 0.3}}
        {"type": "noul", "probability": 0.83}
        {"type": "score", "labels": ["low","med","high"], "distribution": [.1,.3,.6]}
        {"probability": 0.2}                      # noul inferred, no "type"
        {"probabilities": {"yes": 0.2, "no": 0.8}} # choice inferred
    """
    if not isinstance(raw, dict):
        raise TypeError(f"expected a dict-like model output, got {type(raw)}")

    kind = raw.get("kind") or raw.get("type") or raw.get("question_type")
    prob_dict = _first_present(raw, _PROB_DICT_KEYS)
    dist = _first_present(raw, _DIST_KEYS)

    # --- noul: a single scalar P(true) ---
    scalar = _first_present(raw, _SCALAR_PROB_KEYS)
    if scalar is not None and (kind in ("noul", "bool", "yes_no") or (kind is None and prob_dict is None and dist is None)):
        p_true = float(scalar)
        return Decision(
            kind="noul",
            labels=("false", "true"),
            probabilities=(1.0 - p_true, p_true),
            raw=raw,
        )

    # --- choice / score: a probability per label ---
    if isinstance(prob_dict, dict):
        labels = tuple(prob_dict.keys())
        probs = tuple(float(v) for v in prob_dict.values())
        inferred_kind = kind or ("score" if _looks_ordinal(labels) else "choice")
        return Decision(kind=inferred_kind, labels=labels, probabilities=probs, raw=raw)

    labels_list = _first_present(raw, _LABELS_KEYS)
    if dist is not None:
        probs = tuple(float(v) for v in dist)
        labels = tuple(labels_list) if labels_list else tuple(
            str(i) for i in range(len(probs))
        )
        return Decision(kind=kind or "score", labels=labels, probabilities=probs, raw=raw)

    raise ValueError(
        "could not detect a decision shape in this output; expected one of "
        f"{_PROB_DICT_KEYS + _SCALAR_PROB_KEYS + _DIST_KEYS} as a key, "
        "or build a sureband.Decision directly."
    )


def _first_present(d: Dict[str, Any], keys) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _looks_ordinal(labels) -> bool:
    """Heuristic: short label sets that read like a rubric (low/med/high,
    1..5, etc) are treated as ordinal ("score") rather than unordered
    ("choice") when no explicit type is given. Best-effort only -- pass
    kind explicitly if this guesses wrong for your rubric.
    """
    ordinal_words = {
        "low", "medium", "med", "high", "none", "mild", "moderate", "severe",
        "not urgent", "soon", "blocking", "never", "sometimes", "always",
    }
    lowered = {str(l).lower() for l in labels}
    if lowered & ordinal_words:
        return True
    return all(str(l).lstrip("-").isdigit() for l in labels)
