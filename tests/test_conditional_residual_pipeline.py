from __future__ import annotations

import numpy as np
import pandas as pd

from src.aggregation.conditional_residual_aggregate import (
    add_final_predictions,
    aggregate_pair_residuals,
    attach_aggregated_residuals,
)
from src.eval.evaluate_conditional_residual import evaluate_call_level_predictions
from src.features.build_residual_targets import (
    add_baseline_and_residual_targets,
    build_rolling_call_splits,
    merge_call_fields_into_pairs,
    normalize_call_level_dataframe,
    normalize_pair_level_dataframe,
)
from src.features.validate_conditional_residual_data import validate_conditional_residual_training_data
from src.experiments.conditional_residual_qa_pead import _is_fold_complete


def test_normalize_call_and_pair_level_dataframes() -> None:
    raw_call_df = pd.DataFrame(
        {
            "call_id": ["c1", "c2"],
            "ticker": ["AAA", "BBB"],
            "event_date": ["2024-01-01", "2024-01-02"],
            "car_horizon": [0.1, -0.2],
            "sue_eps": [1.0, -1.0],
            "sue_rev": [0.5, -0.5],
            "pre_event_return_5d": [0.02, -0.01],
            "volatility_20d": [0.3, 0.4],
            "snapshot_market_cap_usd": [100.0, 200.0],
            "num_pairs": [3, 2],
        }
    )
    raw_pair_df = pd.DataFrame(
        {
            "call_id": ["c1", "c1", "c2"],
            "pair_index": [0, 1, 0],
            "question_text": ["q1", "q2", "q3"],
            "answer_text": ["a1", "a2", "a3"],
        }
    )

    call_df = normalize_call_level_dataframe(raw_call_df)
    pair_df = normalize_pair_level_dataframe(raw_pair_df)

    assert list(call_df.columns) == [
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
    assert list(pair_df.columns) == ["call_id", "pair_id", "pair_index", "question_text", "answer_text"]
    assert pair_df["pair_id"].tolist() == ["c1::0", "c1::1", "c2::0"]


def test_build_rolling_call_splits_keeps_calls_grouped_and_ordered() -> None:
    call_df = pd.DataFrame(
        {
            "call_id": [f"c{i}" for i in range(8)],
            "call_date": pd.date_range("2024-01-01", periods=8, freq="D").strftime("%Y-%m-%d"),
        }
    )
    splits = build_rolling_call_splits(call_df, min_train_calls=4, val_calls=2, test_calls=1, step_calls=1)
    assert len(splits) == 2
    assert splits[0]["train_ids"] == ["c0", "c1", "c2", "c3"]
    assert splits[0]["val_ids"] == ["c4", "c5"]
    assert splits[0]["test_ids"] == ["c6"]
    assert set(splits[0]["train_ids"]).isdisjoint(splits[0]["val_ids"])
    assert set(splits[0]["train_ids"]).isdisjoint(splits[0]["test_ids"])


def test_merge_call_fields_into_pairs_copies_fold_specific_targets() -> None:
    pair_df = pd.DataFrame(
        {
            "call_id": ["c1", "c1", "c2"],
            "pair_id": ["c1::0", "c1::1", "c2::0"],
            "pair_index": [0, 1, 0],
            "question_text": ["q1", "q2", "q3"],
            "answer_text": ["a1", "a2", "a3"],
        }
    )
    call_df = pd.DataFrame(
        {
            "call_id": ["c1", "c2"],
            "ticker": ["AAA", "BBB"],
            "call_date": ["2024-01-01", "2024-01-02"],
            "pead_target": [0.2, -0.1],
            "residual_target": [0.05, -0.02],
            "SUE_EPS": [1.0, -1.0],
            "SUE_REV": [0.5, -0.5],
            "baseline_pred": [0.15, -0.08],
        }
    )
    merged = merge_call_fields_into_pairs(pair_df, call_df)
    assert merged.loc[merged["call_id"] == "c1", "residual_target"].tolist() == [0.05, 0.05]
    assert merged.loc[merged["call_id"] == "c2", "baseline_pred"].tolist() == [-0.08]


def test_baseline_residual_targets_and_aggregation_flow() -> None:
    call_df = pd.DataFrame(
        {
            "call_id": ["c1", "c2", "c3", "c4"],
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "call_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "pead_target": [0.2, -0.1, 0.3, -0.2],
            "SUE_EPS": [1.0, -1.0, 0.5, -0.5],
            "SUE_REV": [0.2, -0.2, 0.1, -0.1],
            "pre_event_return": [0.01, -0.02, 0.03, -0.01],
            "volatility": [0.3, 0.4, 0.25, 0.35],
            "market_cap": [100.0, 200.0, 150.0, 180.0],
            "qa_count": [2, 1, 2, 1],
        }
    )
    train_calls, val_calls, test_calls, metadata = add_baseline_and_residual_targets(
        train_call_df=call_df.iloc[:2].copy(),
        val_call_df=call_df.iloc[2:3].copy(),
        test_call_df=call_df.iloc[3:].copy(),
    )
    assert "ridge_alpha" in metadata
    assert "baseline_pred" in train_calls.columns
    assert np.allclose(train_calls["pead_target"] - train_calls["baseline_pred"], train_calls["residual_target"])

    pair_predictions = pd.DataFrame(
        {
            "call_id": ["c1", "c1", "c2"],
            "pair_id": ["c1::0", "c1::1", "c2::0"],
            "pair_index": [0, 1, 0],
            "question_text": ["q1", "q2", "q3"],
            "answer_text": ["a1", "a2", "a3"],
            "ticker": ["AAA", "AAA", "BBB"],
            "call_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "pead_target": [0.2, 0.2, -0.1],
            "baseline_pred": [0.1, 0.1, -0.05],
            "residual_target": [0.1, 0.1, -0.05],
            "pair_residual_pred": [0.15, 0.05, -0.02],
        }
    )
    aggregated = aggregate_pair_residuals(pair_predictions, methods=["mean", "max", "std", "topk_mean"], top_k=1)
    assert aggregated.loc[aggregated["call_id"] == "c1", "mean_pair_residual"].iloc[0] == 0.1
    assert aggregated.loc[aggregated["call_id"] == "c1", "max_pair_residual"].iloc[0] == 0.15
    assert aggregated.loc[aggregated["call_id"] == "c1", "topk_mean_pair_residual"].iloc[0] == 0.15

    attached = attach_aggregated_residuals(train_calls.iloc[:2].copy(), aggregated)
    assert "mean_pair_residual" in attached.columns


def test_final_prediction_and_call_level_metrics() -> None:
    train_df = pd.DataFrame(
        {
            "call_id": ["c1", "c2"],
            "baseline_pred": [0.10, -0.20],
            "mean_pair_residual": [0.05, 0.03],
            "max_pair_residual": [0.08, 0.06],
            "pead_target": [0.18, -0.12],
        }
    )
    val_df = pd.DataFrame(
        {
            "call_id": ["c3", "c4"],
            "baseline_pred": [0.01, -0.03],
            "mean_pair_residual": [0.02, -0.01],
            "max_pair_residual": [0.03, 0.00],
            "pead_target": [0.05, -0.04],
        }
    )
    test_df = val_df.copy()

    _, val_preds, test_preds, metadata = add_final_predictions(
        train_call_df=train_df,
        val_call_df=val_df,
        test_call_df=test_df,
        method="simple",
    )
    assert metadata["method"] == "simple"
    assert np.allclose(val_preds["final_pred"], [0.03, -0.04])

    metrics = evaluate_call_level_predictions(test_preds)
    assert metrics["accuracy"] == 1.0
    assert metrics["MSE"] >= 0.0
    assert np.isfinite(metrics["RMSE"])


def test_validation_report_passes_for_sufficient_dataset() -> None:
    call_df = pd.DataFrame(
        {
            "call_id": [f"c{i}" for i in range(12)],
            "ticker": ["AAA"] * 12,
            "call_date": pd.date_range("2024-01-01", periods=12, freq="D").strftime("%Y-%m-%d"),
            "pead_target": np.linspace(-0.2, 0.2, 12),
            "SUE_EPS": np.linspace(-1.0, 1.0, 12),
            "SUE_REV": np.linspace(-0.5, 0.5, 12),
            "pre_event_return": np.linspace(-0.1, 0.1, 12),
            "volatility": np.linspace(0.2, 0.4, 12),
            "market_cap": np.linspace(100, 200, 12),
            "qa_count": [2] * 12,
        }
    )
    pair_rows = []
    for idx in range(12):
        for pair_idx in range(10):
            pair_rows.append(
                {
                    "call_id": f"c{idx}",
                    "pair_id": f"c{idx}::{pair_idx}",
                    "pair_index": pair_idx,
                    "question_text": "q",
                    "answer_text": "a",
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    splits = build_rolling_call_splits(call_df, min_train_calls=8, val_calls=2, test_calls=1, step_calls=1)
    report = validate_conditional_residual_training_data(
        call_df=call_df,
        pair_df=pair_df,
        splits=splits,
        validation_cfg={
            "min_total_calls": 10,
            "min_total_pairs": 100,
            "min_folds": 1,
            "min_train_pairs_per_fold": 50,
            "min_eval_pairs_per_fold": 10,
            "min_feature_coverage": {
                "SUE_EPS": 0.9,
                "SUE_REV": 0.9,
                "pre_event_return": 0.9,
                "volatility": 0.9,
                "market_cap": 0.9,
                "qa_count": 0.9,
            },
        },
    )
    assert report["passed"] is True
    assert report["errors"] == []


def test_is_fold_complete_requires_full_artifact_set(tmp_path) -> None:
    fold_dir = tmp_path / "fold_00"
    fold_dir.mkdir()
    required = [
        "metrics.json",
        "train_pair_predictions.csv",
        "val_pair_predictions.csv",
        "test_pair_predictions.csv",
        "train_aggregated_residuals.csv",
        "val_aggregated_residuals.csv",
        "test_aggregated_residuals.csv",
        "train_call_predictions.csv",
        "val_call_predictions.csv",
        "test_call_predictions.csv",
    ]
    for name in required[:-1]:
        (fold_dir / name).write_text("x", encoding="utf-8")
    assert _is_fold_complete(fold_dir) is False
    (fold_dir / required[-1]).write_text("x", encoding="utf-8")
    assert _is_fold_complete(fold_dir) is True
