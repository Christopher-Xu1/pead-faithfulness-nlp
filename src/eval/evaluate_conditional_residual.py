from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, average_precision_score, mean_squared_error, roc_auc_score


def evaluate_call_level_predictions(
    call_predictions_df: pd.DataFrame,
    target_column: str = "pead_target",
    prediction_column: str = "final_pred",
) -> dict[str, float]:
    y_true = pd.to_numeric(call_predictions_df[target_column], errors="coerce").to_numpy(dtype=float)
    y_score = pd.to_numeric(call_predictions_df[prediction_column], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]
    if len(y_true) == 0:
        return {
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
            "accuracy": float("nan"),
            "MSE": float("nan"),
            "RMSE": float("nan"),
            "correlation_with_pead_target": float("nan"),
        }

    y_sign = (y_true > 0.0).astype(int)
    y_pred_sign = (y_score >= 0.0).astype(int)

    auroc = float("nan")
    auprc = float("nan")
    if len(np.unique(y_sign)) > 1:
        auroc = float(roc_auc_score(y_sign, y_score))
        auprc = float(average_precision_score(y_sign, y_score))

    correlation = float("nan")
    if len(np.unique(y_true)) > 1 and len(np.unique(y_score)) > 1:
        correlation = float(pearsonr(y_true, y_score)[0])

    mse = float(mean_squared_error(y_true, y_score))
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "accuracy": float(accuracy_score(y_sign, y_pred_sign)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "correlation_with_pead_target": correlation,
    }


def summarize_overall_metrics(
    fold_metrics: list[dict[str, Any]],
    overall_test_predictions: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "fold_count": int(len(fold_metrics)),
        "overall_test_metrics": evaluate_call_level_predictions(overall_test_predictions),
    }
    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        metric_columns = [column for column in metrics_df.columns if column != "fold"]
        summary["mean_fold_metrics"] = {
            column: float(pd.to_numeric(metrics_df[column], errors="coerce").mean())
            for column in metric_columns
            if pd.api.types.is_numeric_dtype(pd.to_numeric(metrics_df[column], errors="coerce"))
        }
    else:
        summary["mean_fold_metrics"] = {}
    return summary
