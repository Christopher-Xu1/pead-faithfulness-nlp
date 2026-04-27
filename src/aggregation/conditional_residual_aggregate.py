from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_AGGREGATIONS = {"mean", "max", "std", "topk_mean"}


def _aggregate_scores(scores: np.ndarray, method: str, top_k: int) -> float:
    if len(scores) == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(scores))
    if method == "max":
        return float(np.max(scores))
    if method == "std":
        return float(np.std(scores, ddof=0))
    if method == "topk_mean":
        k = max(1, min(int(top_k), len(scores)))
        return float(np.sort(scores)[-k:].mean())
    raise ValueError(f"Unsupported aggregation method: {method}")


def aggregate_pair_residuals(
    pair_predictions_df: pd.DataFrame,
    methods: list[str] | None = None,
    top_k: int = 3,
    prediction_column: str = "pair_residual_pred",
) -> pd.DataFrame:
    methods = methods or ["mean", "max"]
    unknown = [method for method in methods if method not in SUPPORTED_AGGREGATIONS]
    if unknown:
        raise ValueError(f"Unsupported aggregation methods requested: {unknown}")

    group_columns = [
        "call_id",
        "ticker",
        "call_date",
        "pead_target",
        "baseline_pred",
        "residual_target",
    ]
    missing = [column for column in group_columns + [prediction_column] if column not in pair_predictions_df.columns]
    if missing:
        raise ValueError(f"Pair prediction dataframe is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for keys, group in pair_predictions_df.groupby(group_columns, sort=False):
        call_id, ticker, call_date, pead_target, baseline_pred, residual_target = keys
        ordered = group.sort_values("pair_index")
        scores = ordered[prediction_column].to_numpy(dtype=float)
        row = {
            "call_id": call_id,
            "ticker": ticker,
            "call_date": call_date,
            "pead_target": float(pead_target),
            "baseline_pred": float(baseline_pred),
            "residual_target": float(residual_target),
            "pair_count": int(len(ordered)),
        }
        for method in methods:
            row[f"{method}_pair_residual"] = _aggregate_scores(scores, method=method, top_k=top_k)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["call_date", "call_id"]).reset_index(drop=True)


def attach_aggregated_residuals(
    call_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
) -> pd.DataFrame:
    aggregation_columns = [column for column in aggregated_df.columns if column.endswith("_pair_residual")]
    out = call_df.merge(
        aggregated_df[["call_id", "pair_count", *aggregation_columns]],
        on="call_id",
        how="left",
    )
    for column in aggregation_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    if "pair_count" in out.columns:
        out["pair_count"] = pd.to_numeric(out["pair_count"], errors="coerce").fillna(0).astype(int)
    return out


def add_final_predictions(
    train_call_df: pd.DataFrame,
    val_call_df: pd.DataFrame,
    test_call_df: pd.DataFrame,
    method: str = "simple",
    simple_residual_column: str = "mean_pair_residual",
    meta_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if method == "simple":
        augmented = []
        for frame in [train_call_df, val_call_df, test_call_df]:
            out = frame.copy()
            if simple_residual_column not in out.columns:
                raise ValueError(f"Simple final prediction requires column {simple_residual_column!r}")
            out["final_pred"] = out["baseline_pred"] + out[simple_residual_column].fillna(0.0)
            augmented.append(out)
        metadata = {
            "method": "simple",
            "simple_residual_column": simple_residual_column,
        }
        return augmented[0], augmented[1], augmented[2], metadata

    if method != "meta_ridge":
        raise ValueError(f"Unsupported final prediction method: {method!r}")

    meta_feature_columns = meta_feature_columns or ["baseline_pred", "mean_pair_residual", "max_pair_residual"]
    missing = [column for column in meta_feature_columns if column not in train_call_df.columns]
    if missing:
        raise ValueError(f"Meta ridge requires missing feature columns: {missing}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )
    model.fit(train_call_df[meta_feature_columns], train_call_df["pead_target"])

    augmented = []
    for frame in [train_call_df, val_call_df, test_call_df]:
        out = frame.copy()
        out["final_pred"] = model.predict(out[meta_feature_columns])
        augmented.append(out)
    metadata = {
        "method": "meta_ridge",
        "feature_columns": meta_feature_columns,
        "ridge_alpha": float(model.named_steps["ridge"].alpha_),
    }
    return augmented[0], augmented[1], augmented[2], metadata
