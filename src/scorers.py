"""Custom scoring functions used in evaluation and optimization."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.metrics import f1_score

from mlflow.genai import scorer


_VALID_CATEGORIES = ["incident", "request", "problem", "change"]


def _extract_category(text: str) -> str:
    """Extract one of the known categories from the model output.

    This is intentionally tolerant: we allow extra text but prioritize the first known
    category token found.
    """

    normalized = str(text or "").lower().strip()
    for category in _VALID_CATEGORIES:
        if category in normalized:
            return category
    return "unknown"


@scorer(name="exact_category_match")
def exact_category_match(outputs: Any, expectations: Dict[str, Any]) -> bool:
    """Return True if predicted category matches ground truth (case-insensitive)."""

    expected = str(expectations.get("type", "")).strip().lower()

    if isinstance(outputs, dict):
        predicted = str(outputs.get("type", "")).strip().lower()
    else:
        predicted = _extract_category(outputs)

    return predicted == expected


@scorer(name="macro_f1")
def macro_f1(outputs: Any, expectations: Dict[str, Any], **kwargs) -> float:
    """Compute macro F1 across the evaluation set (tracks state across calls)."""

    state = kwargs.setdefault("state", {})
    y_true = state.setdefault("y_true", [])
    y_pred = state.setdefault("y_pred", [])

    true_label = str(expectations.get("type", "Unknown")).strip().capitalize()
    pred_label = _extract_category(outputs).capitalize()

    y_true.append(true_label)
    y_pred.append(pred_label)

    # Avoid raising until we have at least 2 examples
    if len(y_true) < 2:
        return 0.0

    current_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(current_macro_f1)
