from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.build_qa_pair_dataset import build_qa_pair_dataset

REQUIRED_CALL_LEVEL_COLUMNS = [
    "call_id",
    "ticker",
    "call_date",
    "pead_target",
    "SUE_EPS",
    "SUE_REV",
    "pre_event_return",
    "volatility",
    "market_cap",
    "qa_count",
]

REQUIRED_PAIR_LEVEL_COLUMNS = [
    "call_id",
    "pair_id",
    "pair_index",
    "question_text",
    "answer_text",
]

DEFAULT_BASELINE_FEATURE_COLUMNS = [
    "SUE_EPS",
    "SUE_REV",
    "pre_event_return",
    "volatility",
    "market_cap",
    "qa_count",
]

DEFAULT_PAIR_CONDITIONING_COLUMNS = [
    "SUE_EPS",
    "SUE_REV",
    "baseline_pred",
]


def _choose_market_cap_column(call_features_df: pd.DataFrame) -> str:
    for column in ["snapshot_market_cap_usd", "hist_market_cap", "snapshot_log_market_cap", "hist_log_market_cap"]:
        if column in call_features_df.columns:
            return column
    raise ValueError("Could not derive market_cap because no market-cap column was found in call features")


def _choose_qa_count_column(call_features_df: pd.DataFrame) -> str:
    for column in ["num_pairs", "num_questions", "num_qa_turns"]:
        if column in call_features_df.columns:
            return column
    raise ValueError("Could not derive qa_count because no QA count column was found in call features")


def normalize_call_level_dataframe(call_features_df: pd.DataFrame) -> pd.DataFrame:
    market_cap_column = _choose_market_cap_column(call_features_df)
    qa_count_column = _choose_qa_count_column(call_features_df)
    required_mapping = {
        "call_id": "call_id",
        "ticker": "ticker",
        "event_date": "call_date",
        "car_horizon": "pead_target",
        "sue_eps": "SUE_EPS",
        "sue_rev": "SUE_REV",
        "pre_event_return_5d": "pre_event_return",
        "volatility_20d": "volatility",
        market_cap_column: "market_cap",
        qa_count_column: "qa_count",
    }
    missing = [source for source in required_mapping if source not in call_features_df.columns]
    if missing:
        raise ValueError(f"Call-level features are missing required source columns: {missing}")

    call_df = call_features_df[list(required_mapping)].rename(columns=required_mapping).copy()
    call_df["call_date"] = pd.to_datetime(call_df["call_date"]).dt.strftime("%Y-%m-%d")
    numeric_cols = [column for column in REQUIRED_CALL_LEVEL_COLUMNS if column not in {"call_id", "ticker", "call_date"}]
    for column in numeric_cols:
        call_df[column] = pd.to_numeric(call_df[column], errors="coerce")
    call_df = call_df.drop_duplicates(subset=["call_id"]).sort_values(["call_date", "call_id"]).reset_index(drop=True)
    return call_df


def normalize_pair_level_dataframe(pair_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ["call_id", "pair_index", "question_text", "answer_text"] if column not in pair_df.columns]
    if missing:
        raise ValueError(f"Pair-level features are missing required columns: {missing}")
    out = pair_df[["call_id", "pair_index", "question_text", "answer_text"]].copy()
    out["pair_id"] = out["call_id"].astype(str) + "::" + out["pair_index"].astype(int).astype(str)
    out["pair_index"] = pd.to_numeric(out["pair_index"], errors="coerce").fillna(0).astype(int)
    out["question_text"] = out["question_text"].fillna("").astype(str)
    out["answer_text"] = out["answer_text"].fillna("").astype(str)
    out = out[REQUIRED_PAIR_LEVEL_COLUMNS].sort_values(["call_id", "pair_index"]).reset_index(drop=True)
    return out


def build_conditional_residual_inputs(
    parsed_df: pd.DataFrame,
    qa_summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    label_config_path: str | Path,
    earnings_fundamentals_df: pd.DataFrame | None = None,
    pair_filter_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_pair_df, raw_call_df, dataset_summary = build_qa_pair_dataset(
        parsed_df=parsed_df,
        qa_summary_df=qa_summary_df,
        metadata_df=metadata_df,
        labels_df=labels_df,
        prices_df=prices_df,
        market_df=market_df,
        label_config_path=label_config_path,
        earnings_fundamentals_df=earnings_fundamentals_df,
        pair_filter_config=pair_filter_config,
    )

    call_df = normalize_call_level_dataframe(raw_call_df)
    pair_df = normalize_pair_level_dataframe(raw_pair_df)
    calls_with_pairs = pair_df["call_id"].nunique()
    summary = {
        **dataset_summary,
        "normalized_call_rows": int(len(call_df)),
        "normalized_pair_rows": int(len(pair_df)),
        "normalized_calls_with_pairs": int(calls_with_pairs),
    }
    return call_df, pair_df, summary


def build_rolling_call_splits(
    call_df: pd.DataFrame,
    min_train_calls: int,
    val_calls: int,
    test_calls: int,
    step_calls: int,
    date_column: str = "call_date",
) -> list[dict[str, Any]]:
    ordered = (
        call_df[["call_id", date_column]]
        .drop_duplicates()
        .assign(**{date_column: lambda df: pd.to_datetime(df[date_column])})
        .sort_values([date_column, "call_id"])
        .reset_index(drop=True)
    )
    call_ids = ordered["call_id"].tolist()
    splits: list[dict[str, Any]] = []
    train_end = int(min_train_calls)
    fold = 0
    while train_end + int(val_calls) + int(test_calls) <= len(call_ids):
        val_end = train_end + int(val_calls)
        test_end = val_end + int(test_calls)
        splits.append(
            {
                "fold": fold,
                "train_ids": call_ids[:train_end],
                "val_ids": call_ids[train_end:val_end],
                "test_ids": call_ids[val_end:test_end],
            }
        )
        train_end += int(step_calls)
        fold += 1
    return splits


def fit_fundamentals_ridge(
    train_call_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[Pipeline, list[str]]:
    feature_columns = feature_columns or list(DEFAULT_BASELINE_FEATURE_COLUMNS)
    available_features = [column for column in feature_columns if column in train_call_df.columns]
    if not available_features:
        train_call_df = train_call_df.copy()
        train_call_df["constant_feature"] = 0.0
        available_features = ["constant_feature"]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )
    model.fit(train_call_df[available_features], train_call_df["pead_target"])
    return model, available_features


def add_baseline_and_residual_targets(
    train_call_df: pd.DataFrame,
    val_call_df: pd.DataFrame,
    test_call_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model, used_features = fit_fundamentals_ridge(train_call_df=train_call_df, feature_columns=feature_columns)
    augmented_frames: list[pd.DataFrame] = []
    for frame in [train_call_df, val_call_df, test_call_df]:
        out = frame.copy()
        for column in used_features:
            if column not in out.columns:
                out[column] = 0.0
        out["baseline_pred"] = model.predict(out[used_features])
        out["residual_target"] = out["pead_target"] - out["baseline_pred"]
        augmented_frames.append(out)
    metadata = {
        "feature_columns": used_features,
        "ridge_alpha": float(model.named_steps["ridge"].alpha_),
    }
    return augmented_frames[0], augmented_frames[1], augmented_frames[2], metadata


def merge_call_fields_into_pairs(
    pair_df: pd.DataFrame,
    call_df: pd.DataFrame,
    conditioning_columns: list[str] | None = None,
) -> pd.DataFrame:
    conditioning_columns = conditioning_columns or list(DEFAULT_PAIR_CONDITIONING_COLUMNS)
    required_call_columns = ["call_id", "ticker", "call_date", "pead_target", "residual_target", *conditioning_columns]
    missing = [column for column in required_call_columns if column not in call_df.columns]
    if missing:
        raise ValueError(f"Call dataframe is missing columns required for pair merge: {missing}")
    merged = pair_df.merge(call_df[required_call_columns], on="call_id", how="inner")
    merged = merged.sort_values(["call_date", "call_id", "pair_index"]).reset_index(drop=True)
    return merged
