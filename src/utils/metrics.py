from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = float("nan")
    auprc = float("nan")
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm,
    }


def classification_metrics_from_logits(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    probs = softmax(logits)[:, 1]
    metrics = classification_metrics(labels, probs)
    return {
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "accuracy": metrics["accuracy"],
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
