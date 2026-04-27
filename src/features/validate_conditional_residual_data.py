from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_coverage(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").notna().mean())


def validate_conditional_residual_training_data(
    call_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    splits: list[dict[str, Any]],
    validation_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation_cfg = validation_cfg or {}
    feature_thresholds = validation_cfg.get(
        "min_feature_coverage",
        {
            "SUE_EPS": 0.65,
            "SUE_REV": 0.85,
            "pre_event_return": 0.90,
            "volatility": 0.90,
            "market_cap": 0.99,
            "qa_count": 0.99,
        },
    )
    min_train_pairs = int(validation_cfg.get("min_train_pairs_per_fold", 200))
    min_eval_pairs = int(validation_cfg.get("min_eval_pairs_per_fold", 50))
    min_total_pairs = int(validation_cfg.get("min_total_pairs", 1000))
    min_total_calls = int(validation_cfg.get("min_total_calls", 500))
    min_folds = int(validation_cfg.get("min_folds", 1))

    errors: list[str] = []
    warnings: list[str] = []

    if len(call_df) < min_total_calls:
        errors.append(f"Only {len(call_df)} calls remain after pair filtering; require at least {min_total_calls}.")
    if len(pair_df) < min_total_pairs:
        errors.append(f"Only {len(pair_df)} pairs remain after filtering; require at least {min_total_pairs}.")
    if len(splits) < min_folds:
        errors.append(f"Only {len(splits)} rolling folds are available; require at least {min_folds}.")

    if pair_df["pair_id"].duplicated().any():
        errors.append("Pair IDs are not unique after normalization.")
    if not set(pair_df["call_id"]).issubset(set(call_df["call_id"])):
        errors.append("Pair dataframe contains call_ids that are absent from the filtered call dataframe.")

    overall_coverage = {column: _safe_coverage(call_df, column) for column in feature_thresholds}
    for column, threshold in feature_thresholds.items():
        coverage = overall_coverage[column]
        if not np.isfinite(coverage) or coverage < float(threshold):
            errors.append(
                f"Overall coverage for {column} is {coverage:.4f}, below the required threshold of {float(threshold):.4f}."
            )

    split_summaries: list[dict[str, Any]] = []
    for split in splits:
        train_ids = set(split["train_ids"])
        val_ids = set(split["val_ids"])
        test_ids = set(split["test_ids"])
        train_calls = call_df[call_df["call_id"].isin(train_ids)]
        val_calls = call_df[call_df["call_id"].isin(val_ids)]
        test_calls = call_df[call_df["call_id"].isin(test_ids)]
        train_pairs = pair_df[pair_df["call_id"].isin(train_ids)]
        val_pairs = pair_df[pair_df["call_id"].isin(val_ids)]
        test_pairs = pair_df[pair_df["call_id"].isin(test_ids)]

        if train_pairs.empty or len(train_pairs) < min_train_pairs:
            errors.append(
                f"Fold {split['fold']} has only {len(train_pairs)} training pairs; require at least {min_train_pairs}."
            )
        if val_pairs.empty or len(val_pairs) < min_eval_pairs:
            errors.append(
                f"Fold {split['fold']} has only {len(val_pairs)} validation pairs; require at least {min_eval_pairs}."
            )
        if test_pairs.empty or len(test_pairs) < min_eval_pairs:
            errors.append(
                f"Fold {split['fold']} has only {len(test_pairs)} test pairs; require at least {min_eval_pairs}."
            )

        overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
        if overlap:
            errors.append(f"Fold {split['fold']} has overlapping call_ids across train/val/test splits.")

        train_cov = {column: _safe_coverage(train_calls, column) for column in feature_thresholds}
        for column, threshold in feature_thresholds.items():
            coverage = train_cov[column]
            if not np.isfinite(coverage) or coverage < float(threshold):
                warnings.append(
                    f"Fold {split['fold']} train coverage for {column} is {coverage:.4f}, below {float(threshold):.4f}."
                )

        split_summaries.append(
            {
                "fold": int(split["fold"]),
                "train_calls": int(len(train_calls)),
                "val_calls": int(len(val_calls)),
                "test_calls": int(len(test_calls)),
                "train_pairs": int(len(train_pairs)),
                "val_pairs": int(len(val_pairs)),
                "test_pairs": int(len(test_pairs)),
                "train_feature_coverage": train_cov,
            }
        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "call_rows": int(len(call_df)),
            "pair_rows": int(len(pair_df)),
            "calls_with_pairs": int(call_df["call_id"].nunique()),
            "fold_count": int(len(splits)),
            "overall_feature_coverage": overall_coverage,
            "split_summaries": split_summaries,
        },
    }
