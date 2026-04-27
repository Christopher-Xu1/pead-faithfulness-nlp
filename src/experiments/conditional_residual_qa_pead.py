from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.aggregation.conditional_residual_aggregate import (
    add_final_predictions,
    aggregate_pair_residuals,
    attach_aggregated_residuals,
)
from src.eval.evaluate_conditional_residual import evaluate_call_level_predictions, summarize_overall_metrics
from src.features.build_residual_targets import (
    DEFAULT_BASELINE_FEATURE_COLUMNS,
    DEFAULT_PAIR_CONDITIONING_COLUMNS,
    REQUIRED_CALL_LEVEL_COLUMNS,
    REQUIRED_PAIR_LEVEL_COLUMNS,
    add_baseline_and_residual_targets,
    build_conditional_residual_inputs,
    build_rolling_call_splits,
    merge_call_fields_into_pairs,
)
from src.features.validate_conditional_residual_data import validate_conditional_residual_training_data
from src.models.conditional_residual_model import predict_pair_residuals, train_conditional_residual_model
from src.utils.io import ensure_dir, load_yaml, save_json, write_csv
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_inputs(exp_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prepared_cfg = exp_cfg.get("prepared_inputs", {})
    prepared_call_path = prepared_cfg.get("call_level_path")
    prepared_pair_path = prepared_cfg.get("pair_level_path")
    if prepared_call_path and prepared_pair_path:
        call_df = pd.read_csv(prepared_call_path)
        pair_df = pd.read_csv(prepared_pair_path)
        _validate_required_columns(call_df, REQUIRED_CALL_LEVEL_COLUMNS, "Prepared call-level dataframe")
        _validate_required_columns(pair_df, REQUIRED_PAIR_LEVEL_COLUMNS, "Prepared pair-level dataframe")
        summary = {
            "input_source": "prepared",
            "normalized_call_rows": int(len(call_df)),
            "normalized_pair_rows": int(len(pair_df)),
            "normalized_calls_with_pairs": int(pair_df["call_id"].nunique()),
        }
        return call_df, pair_df, summary

    parsed_df = pd.read_csv(exp_cfg["parsed_calls_path"])
    qa_summary_df = pd.read_csv(exp_cfg["qa_summary_path"])
    metadata_df = pd.read_csv(exp_cfg["metadata_path"])
    labels_df = pd.read_csv(exp_cfg["labels_path"])
    prices_df = pd.read_csv(exp_cfg["prices_path"])
    market_df = pd.read_csv(exp_cfg["market_path"])
    earnings_path = exp_cfg.get("earnings_fundamentals_path")
    earnings_df = pd.read_csv(earnings_path) if earnings_path and Path(earnings_path).exists() else None
    return build_conditional_residual_inputs(
        parsed_df=parsed_df,
        qa_summary_df=qa_summary_df,
        metadata_df=metadata_df,
        labels_df=labels_df,
        prices_df=prices_df,
        market_df=market_df,
        label_config_path=exp_cfg["label_config"],
        earnings_fundamentals_df=earnings_df,
        pair_filter_config=exp_cfg.get("pair_filters"),
    )


def _build_run_report(
    report_path: str | Path,
    run_name: str,
    input_summary: dict[str, Any],
    split_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    overall_summary: dict[str, Any],
) -> None:
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    lines = [
        "# Conditional Residual QA PEAD Report",
        "",
        f"Run name: `{run_name}`",
        "",
        "## Inputs",
        "",
        f"- Source: `{input_summary.get('input_source', 'builder')}`",
        f"- Normalized calls: `{input_summary.get('normalized_call_rows', 0)}`",
        f"- Normalized pairs: `{input_summary.get('normalized_pair_rows', 0)}`",
        f"- Calls with pairs: `{input_summary.get('normalized_calls_with_pairs', 0)}`",
        "",
        "## Splits",
        "",
        "| Fold | Train Calls | Val Calls | Test Calls | Train Start | Train End | Test Start | Test End |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in split_rows:
        lines.append(
            f"| {row['fold']} | {row['train_calls']} | {row['val_calls']} | {row['test_calls']} | "
            f"{row['train_start']} | {row['train_end']} | {row['test_start']} | {row['test_end']} |"
        )

    lines.extend(
        [
            "",
            "## Fold Metrics",
            "",
            "| Fold | Val AUROC | Val AUPRC | Val Accuracy | Val RMSE | Test AUROC | Test AUPRC | Test Accuracy | Test RMSE | Test Corr |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['val_AUROC']:.4f} | {row['val_AUPRC']:.4f} | {row['val_accuracy']:.4f} | "
            f"{row['val_RMSE']:.4f} | {row['test_AUROC']:.4f} | {row['test_AUPRC']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_RMSE']:.4f} | {row['test_correlation_with_pead_target']:.4f} |"
        )

    overall_metrics = overall_summary.get("overall_test_metrics", {})
    mean_fold_metrics = overall_summary.get("mean_fold_metrics", {})
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- OOF test AUROC: `{overall_metrics.get('AUROC', float('nan')):.4f}`",
            f"- OOF test AUPRC: `{overall_metrics.get('AUPRC', float('nan')):.4f}`",
            f"- OOF test accuracy: `{overall_metrics.get('accuracy', float('nan')):.4f}`",
            f"- OOF test MSE: `{overall_metrics.get('MSE', float('nan')):.6f}`",
            f"- OOF test RMSE: `{overall_metrics.get('RMSE', float('nan')):.4f}`",
            f"- OOF test correlation: `{overall_metrics.get('correlation_with_pead_target', float('nan')):.4f}`",
            "",
            "## Mean Fold Metrics",
            "",
        ]
    )
    for key, value in mean_fold_metrics.items():
        lines.append(f"- {key}: `{value:.4f}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_summary_row(
    fold: int,
    train_calls: pd.DataFrame,
    val_calls: pd.DataFrame,
    test_calls: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "fold": fold,
        "train_calls": int(len(train_calls)),
        "val_calls": int(len(val_calls)),
        "test_calls": int(len(test_calls)),
        "train_start": str(train_calls["call_date"].min()),
        "train_end": str(train_calls["call_date"].max()),
        "test_start": str(test_calls["call_date"].min()),
        "test_end": str(test_calls["call_date"].max()),
    }


def _is_fold_complete(fold_dir: Path) -> bool:
    required_files = [
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
    return all((fold_dir / name).exists() for name in required_files)


def _load_completed_fold_outputs(
    fold_dir: Path,
    fold: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    test_call_predictions = pd.read_csv(fold_dir / "test_call_predictions.csv")
    test_call_predictions["fold"] = fold
    test_call_predictions["split"] = "test"
    return metrics, test_call_predictions


def run_experiment(config_path: str, validate_only: bool = False, resume: bool = False) -> dict[str, Any]:
    exp_cfg = load_yaml(config_path)
    model_cfg = load_yaml(exp_cfg["model_config"])
    rolling_cfg = exp_cfg.get("rolling", {})
    aggregation_cfg = exp_cfg.get("aggregation", {})
    final_cfg = exp_cfg.get("final_prediction", {})

    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    run_name = exp_cfg.get("run_name", "conditional_residual_qa_pead")
    dataset_output_dir = ensure_dir(exp_cfg.get("dataset_output_dir", "outputs/datasets/conditional_residual_qa_pead"))
    model_output_dir = ensure_dir(exp_cfg.get("output_dir", "outputs/models/conditional_residual_qa_pead"))

    call_df, pair_df, input_summary = _load_inputs(exp_cfg)
    call_df["call_date"] = pd.to_datetime(call_df["call_date"]).dt.strftime("%Y-%m-%d")
    pair_call_ids = set(pair_df["call_id"])
    zero_pair_calls = int((~call_df["call_id"].isin(pair_call_ids)).sum())
    call_df = call_df[call_df["call_id"].isin(pair_call_ids)].copy()
    call_df = call_df.sort_values(["call_date", "call_id"]).reset_index(drop=True)

    input_summary["zero_pair_calls_excluded_in_rolling_eval"] = zero_pair_calls
    input_summary["rolling_eval_calls"] = int(len(call_df))
    write_csv(call_df, Path(dataset_output_dir) / "call_level_inputs.csv")
    write_csv(pair_df, Path(dataset_output_dir) / "pair_level_inputs.csv")
    save_json(input_summary, Path(dataset_output_dir) / "input_summary.json")

    splits = build_rolling_call_splits(
        call_df=call_df,
        min_train_calls=int(rolling_cfg.get("min_train_calls", 400)),
        val_calls=int(rolling_cfg.get("val_calls", 64)),
        test_calls=int(rolling_cfg.get("test_calls", 64)),
        step_calls=int(rolling_cfg.get("step_calls", 64)),
    )
    max_folds = rolling_cfg.get("max_folds")
    if max_folds is not None:
        splits = splits[: int(max_folds)]
    if not splits:
        raise ValueError("No rolling splits were produced for conditional residual QA PEAD")

    validation_report = validate_conditional_residual_training_data(
        call_df=call_df,
        pair_df=pair_df,
        splits=splits,
        validation_cfg=exp_cfg.get("validation"),
    )
    save_json(validation_report, Path(model_output_dir) / "validation_report.json")
    for warning in validation_report["warnings"]:
        LOGGER.warning("Data validation warning: %s", warning)
    if not validation_report["passed"]:
        raise ValueError(
            "Conditional residual training data validation failed. See "
            f"{Path(model_output_dir) / 'validation_report.json'} for details."
        )
    if validate_only:
        LOGGER.info("Conditional residual data validation passed")
        return {
            "run_name": run_name,
            "dataset_output_dir": str(dataset_output_dir),
            "model_output_dir": str(model_output_dir),
            "validation_report": validation_report,
        }

    baseline_feature_columns = exp_cfg.get("baseline", {}).get("feature_columns", DEFAULT_BASELINE_FEATURE_COLUMNS)
    conditioning_columns = exp_cfg.get("conditioning", {}).get("feature_columns", DEFAULT_PAIR_CONDITIONING_COLUMNS)
    aggregation_methods = aggregation_cfg.get("methods", ["mean", "max"])
    top_k = int(aggregation_cfg.get("top_k", 3))

    split_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    overall_test_frames: list[pd.DataFrame] = []

    for split in splits:
        fold = int(split["fold"])
        fold_dir = ensure_dir(Path(model_output_dir) / f"fold_{fold:02d}")

        train_ids = set(split["train_ids"])
        val_ids = set(split["val_ids"])
        test_ids = set(split["test_ids"])

        train_calls = call_df[call_df["call_id"].isin(train_ids)].copy()
        val_calls = call_df[call_df["call_id"].isin(val_ids)].copy()
        test_calls = call_df[call_df["call_id"].isin(test_ids)].copy()

        split_rows.append(
            _split_summary_row(
                fold=fold,
                train_calls=train_calls,
                val_calls=val_calls,
                test_calls=test_calls,
            )
        )

        if resume and _is_fold_complete(fold_dir):
            LOGGER.info("Skipping completed fold %02d", fold)
            completed_metrics, completed_test_predictions = _load_completed_fold_outputs(fold_dir=fold_dir, fold=fold)
            val_metrics = completed_metrics["val_metrics"]
            test_metrics = completed_metrics["test_metrics"]
            fold_metric_rows.append(
                {
                    "fold": fold,
                    "val_AUROC": float(val_metrics["AUROC"]),
                    "val_AUPRC": float(val_metrics["AUPRC"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_MSE": float(val_metrics["MSE"]),
                    "val_RMSE": float(val_metrics["RMSE"]),
                    "val_correlation_with_pead_target": float(val_metrics["correlation_with_pead_target"]),
                    "test_AUROC": float(test_metrics["AUROC"]),
                    "test_AUPRC": float(test_metrics["AUPRC"]),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_MSE": float(test_metrics["MSE"]),
                    "test_RMSE": float(test_metrics["RMSE"]),
                    "test_correlation_with_pead_target": float(test_metrics["correlation_with_pead_target"]),
                }
            )
            overall_test_frames.append(completed_test_predictions.copy())
            continue

        train_calls, val_calls, test_calls, baseline_metadata = add_baseline_and_residual_targets(
            train_call_df=train_calls,
            val_call_df=val_calls,
            test_call_df=test_calls,
            feature_columns=baseline_feature_columns,
        )

        train_pairs = merge_call_fields_into_pairs(
            pair_df[pair_df["call_id"].isin(train_ids)].copy(),
            train_calls,
            conditioning_columns=conditioning_columns,
        )
        val_pairs = merge_call_fields_into_pairs(
            pair_df[pair_df["call_id"].isin(val_ids)].copy(),
            val_calls,
            conditioning_columns=conditioning_columns,
        )
        test_pairs = merge_call_fields_into_pairs(
            pair_df[pair_df["call_id"].isin(test_ids)].copy(),
            test_calls,
            conditioning_columns=conditioning_columns,
        )

        pair_model_bundle, pair_model_metadata = train_conditional_residual_model(
            train_df=train_pairs,
            val_df=val_pairs,
            model_name=model_cfg.get("model_name", "ProsusAI/finbert"),
            conditioning_columns=conditioning_columns,
            max_length=int(model_cfg.get("max_length", 192)),
            output_dir=fold_dir / "pair_model",
            learning_rate=float(model_cfg.get("learning_rate", 2e-5)),
            weight_decay=float(model_cfg.get("weight_decay", 0.01)),
            batch_size=int(model_cfg.get("batch_size", 16)),
            eval_batch_size=int(model_cfg.get("eval_batch_size", 32)),
            num_train_epochs=int(model_cfg.get("num_train_epochs", 2)),
            warmup_ratio=float(model_cfg.get("warmup_ratio", 0.1)),
            gradient_accumulation_steps=int(model_cfg.get("gradient_accumulation_steps", 1)),
            head_hidden_dim=int(model_cfg.get("head_hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            patience=int(model_cfg.get("patience", 2)),
            seed=seed + fold,
            freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        )

        pair_split_frames = {
            "train": train_pairs.copy(),
            "val": val_pairs.copy(),
            "test": test_pairs.copy(),
        }
        aggregated_split_frames: dict[str, pd.DataFrame] = {}
        call_split_frames: dict[str, pd.DataFrame] = {
            "train": train_calls.copy(),
            "val": val_calls.copy(),
            "test": test_calls.copy(),
        }

        for split_name, pair_frame in pair_split_frames.items():
            pair_frame["pair_residual_pred"] = predict_pair_residuals(pair_frame, model_bundle=pair_model_bundle)
            pair_frame["fold"] = fold
            pair_frame["split"] = split_name
            write_csv(pair_frame, fold_dir / f"{split_name}_pair_predictions.csv")

            aggregated = aggregate_pair_residuals(
                pair_predictions_df=pair_frame,
                methods=list(aggregation_methods),
                top_k=top_k,
            )
            aggregated["fold"] = fold
            aggregated["split"] = split_name
            aggregated_split_frames[split_name] = aggregated
            write_csv(aggregated, fold_dir / f"{split_name}_aggregated_residuals.csv")

            call_split_frames[split_name] = attach_aggregated_residuals(call_split_frames[split_name], aggregated)

        train_call_preds, val_call_preds, test_call_preds, final_metadata = add_final_predictions(
            train_call_df=call_split_frames["train"],
            val_call_df=call_split_frames["val"],
            test_call_df=call_split_frames["test"],
            method=str(final_cfg.get("method", "simple")),
            simple_residual_column=str(final_cfg.get("simple_residual_column", "mean_pair_residual")),
            meta_feature_columns=final_cfg.get("meta_feature_columns"),
        )

        for split_name, frame in {
            "train": train_call_preds,
            "val": val_call_preds,
            "test": test_call_preds,
        }.items():
            frame["fold"] = fold
            frame["split"] = split_name
            write_csv(frame, fold_dir / f"{split_name}_call_predictions.csv")

        val_metrics = evaluate_call_level_predictions(val_call_preds)
        test_metrics = evaluate_call_level_predictions(test_call_preds)
        fold_metric_rows.append(
            {
                "fold": fold,
                "val_AUROC": float(val_metrics["AUROC"]),
                "val_AUPRC": float(val_metrics["AUPRC"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_MSE": float(val_metrics["MSE"]),
                "val_RMSE": float(val_metrics["RMSE"]),
                "val_correlation_with_pead_target": float(val_metrics["correlation_with_pead_target"]),
                "test_AUROC": float(test_metrics["AUROC"]),
                "test_AUPRC": float(test_metrics["AUPRC"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_MSE": float(test_metrics["MSE"]),
                "test_RMSE": float(test_metrics["RMSE"]),
                "test_correlation_with_pead_target": float(test_metrics["correlation_with_pead_target"]),
            }
        )

        fold_metadata = {
            "fold": fold,
            "baseline_model": baseline_metadata,
            "pair_model": pair_model_metadata,
            "final_prediction": final_metadata,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        save_json(fold_metadata, fold_dir / "metrics.json")
        overall_test_frames.append(test_call_preds.copy())

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    overall_test_df = pd.concat(overall_test_frames, ignore_index=True) if overall_test_frames else pd.DataFrame()
    overall_summary = summarize_overall_metrics(fold_metric_rows, overall_test_df)

    write_csv(fold_metrics_df, Path(model_output_dir) / "fold_metrics.csv")
    write_csv(overall_test_df, Path(model_output_dir) / "overall_test_call_predictions.csv")
    save_json(overall_summary, Path(model_output_dir) / "overall_metrics.json")
    save_json(split_rows, Path(model_output_dir) / "split_summary.json")

    report_path = exp_cfg.get("report_path")
    if report_path:
        _build_run_report(
            report_path=report_path,
            run_name=run_name,
            input_summary=input_summary,
            split_rows=split_rows,
            fold_rows=fold_metric_rows,
            overall_summary=overall_summary,
        )

    LOGGER.info("Saved conditional residual artifacts to %s", model_output_dir)
    return {
        "run_name": run_name,
        "dataset_output_dir": str(dataset_output_dir),
        "model_output_dir": str(model_output_dir),
        "report_path": report_path,
        "overall_metrics": overall_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/conditional_residual_qa_pead.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_experiment(args.config, validate_only=bool(args.validate_only), resume=bool(args.resume))


if __name__ == "__main__":
    main()
