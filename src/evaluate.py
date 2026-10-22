"""Probabilistic evaluation metrics for 1X2 (Home/Draw/Away) predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score across Home/Draw/Away outcomes (0 = perfect, 2 = worst)."""
    y_onehot = np.eye(3)[y_true]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Ranked Probability Score: the standard football metric, since H > D > A is ordinal.

    Unlike Brier, RPS penalises putting mass on an away win more than on a draw when the home side wins.
    """
    y_onehot = np.eye(3)[y_true]
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.sum(cum_diff[:, :-1] ** 2, axis=1) / 2.0))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) of the top-class confidence."""
    confidences = np.max(probs, axis=1)
    accuracies = (np.argmax(probs, axis=1) == y_true).astype(float)
    bins = np.clip(np.digitize(confidences, np.linspace(0.0, 1.0, n_bins + 1)[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * in_bin.mean()
    return float(ece)


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Accuracy, log loss, Brier, RPS and calibration error for a set of 1X2 predictions."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, np.argmax(probs, axis=1))), 4),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1, 2])), 4),
        "brier_score": round(multiclass_brier_score(y_true, probs), 4),
        "rps": round(ranked_probability_score(y_true, probs), 4),
        "ece": round(expected_calibration_error(y_true, probs), 4),
    }
