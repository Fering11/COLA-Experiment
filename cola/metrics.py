from __future__ import annotations

from collections.abc import Sequence


LABELS = ("favor", "against", "neutral")


def classification_metrics(
    truth: Sequence[str], predictions: Sequence[str]
) -> dict[str, object]:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length")
    if not truth:
        raise ValueError("at least one prediction is required")

    per_class = {label: _f1(truth, predictions, label) for label in LABELS}
    accuracy = sum(a == b for a, b in zip(truth, predictions, strict=True)) / len(truth)
    return {
        "count": len(truth),
        "accuracy": accuracy,
        "macro_f1": sum(per_class.values()) / len(LABELS),
        "f_avg": (per_class["favor"] + per_class["against"]) / 2,
        "per_class_f1": per_class,
    }


def _f1(truth: Sequence[str], predictions: Sequence[str], label: str) -> float:
    true_positive = sum(
        actual == label and predicted == label
        for actual, predicted in zip(truth, predictions, strict=True)
    )
    false_positive = sum(
        actual != label and predicted == label
        for actual, predicted in zip(truth, predictions, strict=True)
    )
    false_negative = sum(
        actual == label and predicted != label
        for actual, predicted in zip(truth, predictions, strict=True)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator

