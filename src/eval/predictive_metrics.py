from __future__ import annotations

import numpy as np

from src.utils.metrics import classification_metrics


def compute_predictive_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return classification_metrics(y_true, y_prob)
