from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.build_qa_pair_dataset import build_qa_pair_dataset
from src.utils.io import ensure_dir, load_yaml, save_json, write_csv
from src.utils.logging_utils import get_logger
from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)


@dataclass
class TargetTransform:
    lower: float
    upper: float
    mean: float
    std: float

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=float), self.lower, self.upper)
        return (clipped - self.mean) / self.std

    def inverse(self, values: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return values * self.std + self.mean


class ConstantCalibrator:
    def __init__(self, positive_rate: float):
        self.positive_rate = float(np.clip(positive_rate, 0.0, 1.0))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n = len(X)
        positive = np.full(n, self.positive_rate, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


def fit_target_transform(values: pd.Series, winsor_quantile: float) -> TargetTransform:
    arr = np.asarray(values, dtype=float)
    lower = float(np.nanquantile(arr, winsor_quantile))
    upper = float(np.nanquantile(arr, 1.0 - winsor_quantile))
    clipped = np.clip(arr, lower, upper)
    mean = float(np.nanmean(clipped))
    std = float(np.nanstd(clipped))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return TargetTransform(lower=lower, upper=upper, mean=mean, std=std)


def regression_metrics_from_logits(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    rho = float("nan")
    if len(np.unique(labels)) > 1 and len(np.unique(preds)) > 1:
        rho = float(spearmanr(labels, preds).correlation)
    return {"rmse": rmse, "spearman": rho}


def _to_pair_dataset(df: pd.DataFrame, label_column: str = "target"):
    from datasets import Dataset

    subset = df[["question_text", "answer_text", label_column]].rename(columns={label_column: "label"})
    return Dataset.from_pandas(subset, preserve_index=False)


def build_rolling_splits(
    call_df: pd.DataFrame,
    min_train_calls: int,
    val_calls: int,
    test_calls: int,
    step_calls: int,
) -> list[dict[str, list[str] | int]]:
    ordered = (
        call_df[["call_id", "event_date"]]
        .drop_duplicates()
        .assign(event_date=lambda df: pd.to_datetime(df["event_date"]))
        .sort_values(["event_date", "call_id"])
        .reset_index(drop=True)
    )

    call_ids = ordered["call_id"].tolist()
    splits: list[dict[str, list[str] | int]] = []
    train_end = min_train_calls
    fold_idx = 0
    while train_end + val_calls + test_calls <= len(call_ids):
        val_end = train_end + val_calls
        test_end = val_end + test_calls
        splits.append(
            {
                "fold": fold_idx,
                "train_ids": call_ids[:train_end],
                "val_ids": call_ids[train_end:val_end],
                "test_ids": call_ids[val_end:test_end],
            }
        )
        train_end += step_calls
        fold_idx += 1
    return splits


def _tokenize_pair_batch(tokenizer, questions: list[str], answers: list[str], max_length: int):
    return tokenizer(
        questions,
        answers,
        truncation="longest_first",
        max_length=max_length,
    )


def train_pair_regressor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    max_length: int,
    model_dir: str | Path,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    eval_batch_size: int,
    num_train_epochs: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    seed: int,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )

    train_ds = _to_pair_dataset(train_df)
    val_ds = _to_pair_dataset(val_df)

    def tok(batch):
        return _tokenize_pair_batch(
            tokenizer=tokenizer,
            questions=batch["question_text"],
            answers=batch["answer_text"],
            max_length=max_length,
        )

    train_ds = train_ds.map(tok, batched=True).remove_columns(["question_text", "answer_text"])
    val_ds = val_ds.map(tok, batched=True).remove_columns(["question_text", "answer_text"])

    ensure_dir(model_dir)
    args = TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_rmse",
        greater_is_better=False,
        report_to="none",
        seed=seed,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=regression_metrics_from_logits,
    )
    train_result = trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    eval_metrics = trainer.evaluate()
    save_json(eval_metrics, Path(model_dir) / "val_pair_metrics.json")

    metadata = {
        "model_name": model_name,
        "max_length": max_length,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "seed": seed,
        "train_runtime": train_result.metrics.get("train_runtime"),
    }
    save_json(metadata, Path(model_dir) / "run_metadata.json")
    return tokenizer, trainer.model, eval_metrics


def predict_pair_scores(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    max_length: int,
    eval_batch_size: int,
) -> np.ndarray:
    ds = _to_pair_dataset(df)

    def tok(batch):
        return _tokenize_pair_batch(
            tokenizer=tokenizer,
            questions=batch["question_text"],
            answers=batch["answer_text"],
            max_length=max_length,
        )

    ds = ds.map(tok, batched=True).remove_columns(["question_text", "answer_text"])
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        args=TrainingArguments(
            output_dir="outputs/logs/qa_pair_predict",
            per_device_eval_batch_size=eval_batch_size,
            report_to="none",
        ),
    )
    pred_out = trainer.predict(ds)
    return np.asarray(pred_out.predictions).reshape(-1)


def aggregate_pair_scores(df: pd.DataFrame, score_col: str, pooling: str) -> pd.DataFrame:
    grouped = df.groupby(["call_id", "ticker", "event_date", "car_horizon", "label"], as_index=False)
    if pooling == "mean":
        out = grouped.agg(call_score=(score_col, "mean"), num_pairs=(score_col, "size"))
    elif pooling == "max":
        out = grouped.agg(call_score=(score_col, "max"), num_pairs=(score_col, "size"))
    else:
        raise ValueError(f"Unsupported pooling={pooling!r}")
    out["pooling"] = pooling
    return out


def fit_score_calibrator(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression | ConstantCalibrator:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return ConstantCalibrator(float(labels.mean()) if len(labels) else 0.5)
    model = LogisticRegression(random_state=42, solver="lbfgs")
    model.fit(np.asarray(scores).reshape(-1, 1), labels)
    return model


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not mask.any():
            continue
        accuracy = float(y_true[mask].mean())
        confidence = float(y_prob[mask].mean())
        ece += abs(accuracy - confidence) * (mask.sum() / total)
    return float(ece)


def evaluate_call_predictions(y_true_car: np.ndarray, call_scores: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true_car = np.asarray(y_true_car, dtype=float)
    call_scores = np.asarray(call_scores, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_true_sign = (y_true_car > 0.0).astype(int)

    class_metrics = classification_metrics(y_true_sign, y_prob)
    spearman = float("nan")
    pearson = float("nan")
    if len(np.unique(y_true_car)) > 1 and len(np.unique(call_scores)) > 1:
        spearman = float(spearmanr(y_true_car, call_scores).correlation)
        pearson = float(pearsonr(y_true_car, call_scores)[0])

    return {
        "auroc": float(class_metrics["auroc"]),
        "auprc": float(class_metrics["auprc"]),
        "accuracy": float(class_metrics["accuracy"]),
        "spearman": spearman,
        "pearson": pearson,
        "rmse": float(np.sqrt(np.mean((y_true_car - call_scores) ** 2))),
        "mae": float(np.mean(np.abs(y_true_car - call_scores))),
        "ece": expected_calibration_error(y_true_sign, y_prob),
        "brier": float(np.mean((y_true_sign - y_prob) ** 2)),
    }


def bootstrap_confidence_intervals(
    predictions_df: pd.DataFrame,
    score_col: str,
    prob_col: str,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    metrics_samples: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "accuracy": [],
        "spearman": [],
        "pearson": [],
        "rmse": [],
        "mae": [],
        "ece": [],
        "brier": [],
    }
    rng = np.random.default_rng(seed)
    n = len(predictions_df)
    if n == 0:
        return {key: {"lower": float("nan"), "upper": float("nan")} for key in metrics_samples}

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample = predictions_df.iloc[sample_idx]
        metrics = evaluate_call_predictions(
            y_true_car=sample["car_horizon"].to_numpy(),
            call_scores=sample[score_col].to_numpy(),
            y_prob=sample[prob_col].to_numpy(),
        )
        for key, value in metrics.items():
            if np.isfinite(value):
                metrics_samples[key].append(float(value))

    intervals: dict[str, dict[str, float]] = {}
    for key, values in metrics_samples.items():
        if not values:
            intervals[key] = {"lower": float("nan"), "upper": float("nan")}
            continue
        intervals[key] = {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
    return intervals


def _format_metric(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _format_metric_with_ci(value: float, ci: dict[str, float] | None) -> str:
    if ci is None or not np.isfinite(value):
        return _format_metric(value)
    return f"{_format_metric(value)} [{_format_metric(ci.get('lower', float('nan')))}, {_format_metric(ci.get('upper', float('nan')))}]"


def fit_upper_bound_model(train_df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    numeric_features = [
        "text_score_mean",
        "text_score_max",
        "pre_event_return_5d",
        "prior_momentum_5d",
        "prior_momentum_20d",
        "prior_momentum_60d",
        "relative_momentum_20d",
        "relative_momentum_60d",
        "volatility_20d",
        "volatility_60d",
        "market_volatility_20d",
        "beta_60d",
        "quality_score",
        "num_questions",
        "num_qa_turns",
        "year",
        "quarter",
        "num_pairs",
        "reported_eps",
        "estimated_eps",
        "eps_surprise",
        "eps_surprise_pct",
        "earnings_surprise",
        "reported_revenue",
        "estimated_revenue",
        "revenue_surprise",
        "revenue_surprise_pct",
        "eps12mtrailing_qavg",
        "eps12mtrailing_eoq",
        "eps12mfwd_qavg",
        "eps12mfwd_eoq",
        "eps_lt",
        "peforw_qavg",
        "peforw_eoq",
        "fwd_minus_trailing_eps_eoq",
        "fwd_minus_trailing_eps_qavg",
        "fwd_vs_trailing_eps_growth_eoq",
        "fwd_vs_trailing_eps_growth_qavg",
    ]
    # Keep post-event windows out of the default benchmark because they overlap
    # the PEAD label horizon and would leak target information.
    categorical_features = ["sector", "ticker", "source_id", "eps_beat_miss", "revenue_beat_miss"]

    available_numeric = [col for col in numeric_features if col in train_df.columns and not train_df[col].isna().all()]
    available_categorical = [col for col in categorical_features if col in train_df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                available_numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                available_categorical,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )
    model.fit(train_df[available_numeric + available_categorical], train_df["car_horizon"])
    return model, available_numeric, available_categorical


def write_markdown_report(
    path: str | Path,
    run_name: str,
    dataset_summary: dict[str, Any],
    split_summary: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    notes: list[str],
) -> None:
    ensure_dir(Path(path).parent)
    lines: list[str] = [
        "# QA Pair Regression Report",
        "",
        f"Run name: `{run_name}`",
        "",
        "## Dataset",
        "",
        f"- Pair filter profile: `{dataset_summary.get('pair_filter_profile', 'broad')}`",
        f"- Pair rows: `{dataset_summary['pair_rows']}`",
        f"- Calls with pairs: `{dataset_summary['calls_with_pairs']}`",
        f"- Pair rows before filtering: `{dataset_summary.get('pair_rows_before_filtering', dataset_summary['pair_rows'])}`",
        f"- Calls with pairs before filtering: `{dataset_summary.get('calls_with_pairs_before_filtering', dataset_summary['calls_with_pairs'])}`",
        f"- Pair retention rate: `{dataset_summary.get('pair_retention_rate', 1.0):.4f}`",
        f"- Mean pairs per call: `{dataset_summary['mean_pairs_per_call']:.2f}`",
        f"- Median pairs per call: `{dataset_summary['median_pairs_per_call']:.2f}`",
        f"- Mean question chars: `{dataset_summary['mean_question_chars']:.1f}`",
        f"- Mean answer chars: `{dataset_summary['mean_answer_chars']:.1f}`",
        f"- Pre-event 5d return coverage: `{dataset_summary.get('pre_event_return_5d_coverage', float('nan')):.4f}`",
        f"- Post-event 3d return coverage: `{dataset_summary.get('post_event_return_3d_coverage', float('nan')):.4f}`",
        f"- Management-role answer rate: `{dataset_summary.get('answer_has_management_role_rate', float('nan')):.4f}`",
        f"- Analyst-only answer rate: `{dataset_summary.get('answer_is_analyst_only_rate', float('nan')):.4f}`",
        f"- Answer question-mark rate: `{dataset_summary.get('answer_contains_question_mark_rate', float('nan')):.4f}`",
        f"- Positive rate: `{dataset_summary['positive_rate']:.4f}`",
        f"- Earnings surprise coverage: `{dataset_summary['earnings_surprise_coverage']:.4f}`",
        f"- Reported EPS coverage: `{dataset_summary.get('reported_eps_coverage', float('nan')):.4f}`",
        f"- Estimated EPS coverage: `{dataset_summary.get('estimated_eps_coverage', float('nan')):.4f}`",
        f"- EPS surprise coverage: `{dataset_summary.get('eps_surprise_coverage', float('nan')):.4f}`",
        f"- EPS surprise pct coverage: `{dataset_summary.get('eps_surprise_pct_coverage', float('nan')):.4f}`",
        f"- Reported revenue coverage: `{dataset_summary.get('reported_revenue_coverage', float('nan')):.4f}`",
        f"- Estimated revenue coverage: `{dataset_summary.get('estimated_revenue_coverage', float('nan')):.4f}`",
        f"- Revenue surprise coverage: `{dataset_summary.get('revenue_surprise_coverage', float('nan')):.4f}`",
        f"- EPS beat/miss coverage: `{dataset_summary.get('eps_beat_miss_coverage', float('nan')):.4f}`",
        f"- Revenue beat/miss coverage: `{dataset_summary.get('revenue_beat_miss_coverage', float('nan')):.4f}`",
        f"- Forward EPS coverage (glopardo): `{dataset_summary.get('glopardo_forward_eps_coverage', float('nan')):.4f}`",
        f"- Zero-pair calls excluded from rolling eval: `{dataset_summary.get('zero_pair_calls_excluded_in_rolling_eval', 0)}`",
        f"- Calls used in rolling eval: `{dataset_summary.get('rolling_eval_calls', dataset_summary['calls_with_pairs'])}`",
        "",
        "## Rolling Splits",
        "",
        "| Fold | Train Calls | Val Calls | Test Calls | Train Start | Train End | Test Start | Test End |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]

    for row in split_summary:
        lines.append(
            f"| {row['fold']} | {row['train_calls']} | {row['val_calls']} | {row['test_calls']} | "
            f"{row['train_start']} | {row['train_end']} | {row['test_start']} | {row['test_end']} |"
        )

    lines.extend(
        [
            "",
            "## Fold Results",
            "",
            "| Fold | Benchmark | Pooling | Val Spearman | Test AUROC | Test AUPRC | Test Spearman | Test ECE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['benchmark']} | {row['pooling']} | "
            f"{_format_metric(row['val_spearman'])} | {_format_metric(row['test_auroc'])} | "
            f"{_format_metric(row['test_auprc'])} | {_format_metric(row['test_spearman'])} | "
            f"{_format_metric(row['test_ece'])} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Benchmarks",
            "",
            "| Benchmark | AUROC | AUPRC | Spearman | Pearson | RMSE | MAE | ECE | Accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in benchmark_rows:
        lines.append(
            f"| {row['benchmark']} | "
            f"{_format_metric_with_ci(row['auroc'], row['ci'].get('auroc'))} | "
            f"{_format_metric_with_ci(row['auprc'], row['ci'].get('auprc'))} | "
            f"{_format_metric_with_ci(row['spearman'], row['ci'].get('spearman'))} | "
            f"{_format_metric_with_ci(row['pearson'], row['ci'].get('pearson'))} | "
            f"{_format_metric_with_ci(row['rmse'], row['ci'].get('rmse'))} | "
            f"{_format_metric_with_ci(row['mae'], row['ci'].get('mae'))} | "
            f"{_format_metric_with_ci(row['ece'], row['ci'].get('ece'))} | "
            f"{_format_metric_with_ci(row['accuracy'], row['ci'].get('accuracy'))} |"
        )

    lines.extend(["", "## Notes", ""])
    for note in notes:
        lines.append(f"- {note}")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config_path: str) -> dict[str, Any]:
    exp_cfg = load_yaml(config_path)
    model_cfg = load_yaml(exp_cfg["model_config"])
    rolling_cfg = exp_cfg.get("rolling", {})
    target_cfg = exp_cfg.get("target", {})
    aggregation_cfg = exp_cfg.get("aggregation", {})
    metrics_cfg = exp_cfg.get("metrics", {})

    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    pair_output_dir = Path(exp_cfg.get("pair_output_dir", "outputs/datasets/qa_pairs_mag7"))
    ensure_dir(pair_output_dir)
    model_output_dir = Path(exp_cfg.get("output_dir", "outputs/models/qa_pair_finbert_regression"))
    ensure_dir(model_output_dir)

    parsed_df = pd.read_csv(exp_cfg.get("parsed_calls_path", "data/interim/parsed_calls/parsed_calls.csv"))
    qa_summary_df = pd.read_csv(exp_cfg.get("qa_summary_path", "data/interim/qa_only/qa_dataset.csv"))
    metadata_df = pd.read_csv(exp_cfg.get("metadata_path", "data/raw/metadata/call_metadata.csv"))
    labels_df = pd.read_csv(exp_cfg.get("labels_path", "data/interim/labels/pead_labels.csv"))
    prices_df = pd.read_csv(exp_cfg.get("prices_path", "data/raw/prices/daily_returns.csv"))
    market_df = pd.read_csv(exp_cfg.get("market_path", "data/external/market_index/sp500_returns.csv"))
    earnings_path = Path(
        exp_cfg.get("earnings_fundamentals_path", "data/external/earnings_fundamentals/earnings_fundamentals.csv")
    )
    earnings_fundamentals_df = pd.read_csv(earnings_path) if earnings_path.exists() else None

    pair_df, call_features_df, dataset_summary = build_qa_pair_dataset(
        parsed_df=parsed_df,
        qa_summary_df=qa_summary_df,
        metadata_df=metadata_df,
        labels_df=labels_df,
        prices_df=prices_df,
        market_df=market_df,
        label_config_path=exp_cfg.get("label_config", "configs/data/pead_20d.yaml"),
        earnings_fundamentals_df=earnings_fundamentals_df,
        pair_filter_config=exp_cfg.get("pair_filters"),
    )
    zero_pair_calls = int((call_features_df["num_pairs"] == 0).sum())
    call_features_df = call_features_df[call_features_df["num_pairs"] > 0].copy()
    dataset_summary["zero_pair_calls_excluded_in_rolling_eval"] = zero_pair_calls
    dataset_summary["rolling_eval_calls"] = int(len(call_features_df))
    write_csv(pair_df, pair_output_dir / "qa_pair_dataset.csv")
    write_csv(call_features_df, pair_output_dir / "call_features.csv")
    save_json(dataset_summary, pair_output_dir / "summary.json")

    splits = build_rolling_splits(
        call_df=call_features_df,
        min_train_calls=int(rolling_cfg.get("min_train_calls", 200)),
        val_calls=int(rolling_cfg.get("val_calls", 48)),
        test_calls=int(rolling_cfg.get("test_calls", 48)),
        step_calls=int(rolling_cfg.get("step_calls", 48)),
    )
    if not splits:
        raise ValueError("No rolling splits were produced; adjust rolling config")

    split_summary: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    oof_predictions: list[pd.DataFrame] = []

    model_name = model_cfg.get("model_name", "ProsusAI/finbert")
    max_length = int(model_cfg.get("max_length", 256))
    learning_rate = float(model_cfg.get("learning_rate", 2e-5))
    weight_decay = float(model_cfg.get("weight_decay", 0.01))
    batch_size = int(model_cfg.get("batch_size", 8))
    eval_batch_size = int(model_cfg.get("eval_batch_size", 16))
    num_train_epochs = float(model_cfg.get("num_train_epochs", 2))
    warmup_ratio = float(model_cfg.get("warmup_ratio", 0.1))
    gradient_accumulation_steps = int(model_cfg.get("gradient_accumulation_steps", 1))
    winsor_quantile = float(target_cfg.get("winsorize_quantile", 0.05))
    pooling_options = aggregation_cfg.get("poolings", ["mean", "max"])
    bootstrap_samples = int(metrics_cfg.get("bootstrap_samples", 400))

    for split in splits:
        fold = int(split["fold"])
        fold_dir = model_output_dir / f"fold_{fold:02d}"
        ensure_dir(fold_dir)

        train_ids = set(split["train_ids"])
        val_ids = set(split["val_ids"])
        test_ids = set(split["test_ids"])

        train_calls = call_features_df[call_features_df["call_id"].isin(train_ids)].copy()
        val_calls = call_features_df[call_features_df["call_id"].isin(val_ids)].copy()
        test_calls = call_features_df[call_features_df["call_id"].isin(test_ids)].copy()

        split_summary.append(
            {
                "fold": fold,
                "train_calls": int(len(train_calls)),
                "val_calls": int(len(val_calls)),
                "test_calls": int(len(test_calls)),
                "train_start": train_calls["event_date"].min(),
                "train_end": train_calls["event_date"].max(),
                "test_start": test_calls["event_date"].min(),
                "test_end": test_calls["event_date"].max(),
            }
        )

        train_pairs = pair_df[pair_df["call_id"].isin(train_ids)].copy()
        val_pairs = pair_df[pair_df["call_id"].isin(val_ids)].copy()
        test_pairs = pair_df[pair_df["call_id"].isin(test_ids)].copy()

        target_transform = fit_target_transform(train_calls["car_horizon"], winsor_quantile=winsor_quantile)
        train_pairs["target"] = target_transform.transform(train_pairs["car_horizon"])
        val_pairs["target"] = target_transform.transform(val_pairs["car_horizon"])
        test_pairs["target"] = target_transform.transform(test_pairs["car_horizon"])

        tokenizer, model, pair_val_metrics = train_pair_regressor(
            train_df=train_pairs,
            val_df=val_pairs,
            model_name=model_name,
            max_length=max_length,
            model_dir=fold_dir / "text_model",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seed=seed + fold,
        )

        train_pairs["pair_score"] = target_transform.inverse(
            predict_pair_scores(train_pairs, tokenizer, model, max_length=max_length, eval_batch_size=eval_batch_size)
        )
        val_pairs["pair_score"] = target_transform.inverse(
            predict_pair_scores(val_pairs, tokenizer, model, max_length=max_length, eval_batch_size=eval_batch_size)
        )
        test_pairs["pair_score"] = target_transform.inverse(
            predict_pair_scores(test_pairs, tokenizer, model, max_length=max_length, eval_batch_size=eval_batch_size)
        )

        pool_results: list[dict[str, Any]] = []
        aggregate_maps: dict[str, dict[str, pd.DataFrame]] = {}
        for pooling in pooling_options:
            train_scores = aggregate_pair_scores(train_pairs, score_col="pair_score", pooling=pooling)
            val_scores = aggregate_pair_scores(val_pairs, score_col="pair_score", pooling=pooling)
            test_scores = aggregate_pair_scores(test_pairs, score_col="pair_score", pooling=pooling)
            calibrator = fit_score_calibrator(val_scores["call_score"].to_numpy(), val_scores["label"].to_numpy())
            val_prob = calibrator.predict_proba(val_scores["call_score"].to_numpy().reshape(-1, 1))[:, 1]
            test_prob = calibrator.predict_proba(test_scores["call_score"].to_numpy().reshape(-1, 1))[:, 1]
            val_metrics = evaluate_call_predictions(val_scores["car_horizon"], val_scores["call_score"], val_prob)
            test_metrics = evaluate_call_predictions(test_scores["car_horizon"], test_scores["call_score"], test_prob)
            pool_results.append(
                {
                    "pooling": pooling,
                    "calibrator": calibrator,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
            )
            aggregate_maps[pooling] = {"train": train_scores, "val": val_scores, "test": test_scores}

        selected = max(
            pool_results,
            key=lambda row: (
                -math.inf if not np.isfinite(row["val_metrics"]["spearman"]) else row["val_metrics"]["spearman"],
                -math.inf if not np.isfinite(row["val_metrics"]["auroc"]) else row["val_metrics"]["auroc"],
            ),
        )
        selected_pooling = str(selected["pooling"])
        selected_train = aggregate_maps[selected_pooling]["train"].rename(columns={"call_score": "text_score"})
        selected_val = aggregate_maps[selected_pooling]["val"].rename(columns={"call_score": "text_score"})
        selected_test = aggregate_maps[selected_pooling]["test"].rename(columns={"call_score": "text_score"})
        selected_val["probability"] = selected["calibrator"].predict_proba(
            selected_val["text_score"].to_numpy().reshape(-1, 1)
        )[:, 1]
        selected_test["probability"] = selected["calibrator"].predict_proba(
            selected_test["text_score"].to_numpy().reshape(-1, 1)
        )[:, 1]

        fold_rows.append(
            {
                "fold": fold,
                "benchmark": "text_only",
                "pooling": selected_pooling,
                "val_spearman": selected["val_metrics"]["spearman"],
                "test_auroc": selected["test_metrics"]["auroc"],
                "test_auprc": selected["test_metrics"]["auprc"],
                "test_spearman": selected["test_metrics"]["spearman"],
                "test_ece": selected["test_metrics"]["ece"],
            }
        )

        train_mean = aggregate_pair_scores(train_pairs, score_col="pair_score", pooling="mean").rename(
            columns={"call_score": "text_score_mean"}
        )
        train_max = aggregate_pair_scores(train_pairs, score_col="pair_score", pooling="max").rename(
            columns={"call_score": "text_score_max"}
        )
        val_mean = aggregate_pair_scores(val_pairs, score_col="pair_score", pooling="mean").rename(
            columns={"call_score": "text_score_mean"}
        )
        val_max = aggregate_pair_scores(val_pairs, score_col="pair_score", pooling="max").rename(
            columns={"call_score": "text_score_max"}
        )
        test_mean = aggregate_pair_scores(test_pairs, score_col="pair_score", pooling="mean").rename(
            columns={"call_score": "text_score_mean"}
        )
        test_max = aggregate_pair_scores(test_pairs, score_col="pair_score", pooling="max").rename(
            columns={"call_score": "text_score_max"}
        )

        train_call_model = (
            train_calls.merge(train_mean[["call_id", "text_score_mean"]], on="call_id", how="left")
            .merge(train_max[["call_id", "text_score_max"]], on="call_id", how="left")
            .copy()
        )
        val_call_model = (
            val_calls.merge(val_mean[["call_id", "text_score_mean"]], on="call_id", how="left")
            .merge(val_max[["call_id", "text_score_max"]], on="call_id", how="left")
            .copy()
        )
        test_call_model = (
            test_calls.merge(test_mean[["call_id", "text_score_mean"]], on="call_id", how="left")
            .merge(test_max[["call_id", "text_score_max"]], on="call_id", how="left")
            .copy()
        )

        upper_model, numeric_cols, categorical_cols = fit_upper_bound_model(train_call_model)
        feature_cols = numeric_cols + categorical_cols
        val_call_model["upper_bound_score"] = upper_model.predict(val_call_model[feature_cols])
        test_call_model["upper_bound_score"] = upper_model.predict(test_call_model[feature_cols])
        upper_calibrator = fit_score_calibrator(
            val_call_model["upper_bound_score"].to_numpy(),
            val_call_model["label"].to_numpy(),
        )
        val_call_model["upper_bound_probability"] = upper_calibrator.predict_proba(
            val_call_model["upper_bound_score"].to_numpy().reshape(-1, 1)
        )[:, 1]
        test_call_model["upper_bound_probability"] = upper_calibrator.predict_proba(
            test_call_model["upper_bound_score"].to_numpy().reshape(-1, 1)
        )[:, 1]

        upper_val_metrics = evaluate_call_predictions(
            val_call_model["car_horizon"],
            val_call_model["upper_bound_score"],
            val_call_model["upper_bound_probability"],
        )
        upper_test_metrics = evaluate_call_predictions(
            test_call_model["car_horizon"],
            test_call_model["upper_bound_score"],
            test_call_model["upper_bound_probability"],
        )
        fold_rows.append(
            {
                "fold": fold,
                "benchmark": "text_plus_tabular",
                "pooling": "mean+max",
                "val_spearman": upper_val_metrics["spearman"],
                "test_auroc": upper_test_metrics["auroc"],
                "test_auprc": upper_test_metrics["auprc"],
                "test_spearman": upper_test_metrics["spearman"],
                "test_ece": upper_test_metrics["ece"],
            }
        )

        text_preds = selected_test[["call_id", "ticker", "event_date", "car_horizon", "label", "text_score", "probability"]].copy()
        text_preds = text_preds.rename(columns={"text_score": "score"})
        text_preds["benchmark"] = "text_only"
        text_preds["fold"] = fold
        text_preds["selected_pooling"] = selected_pooling
        oof_predictions.append(text_preds)

        upper_preds = test_call_model[
            ["call_id", "ticker", "event_date", "car_horizon", "label", "upper_bound_score", "upper_bound_probability"]
        ].copy()
        upper_preds["benchmark"] = "text_plus_tabular"
        upper_preds["fold"] = fold
        upper_preds = upper_preds.rename(
            columns={"upper_bound_score": "score", "upper_bound_probability": "probability"}
        )
        oof_predictions.append(upper_preds)

        selected_test = selected_test.rename(columns={"text_score": "score"})
        write_csv(selected_test, fold_dir / "text_only_test_predictions.csv")
        write_csv(test_call_model, fold_dir / "upper_bound_test_predictions.csv")
        save_json(
            {
                "pair_val_metrics": pair_val_metrics,
                "text_only_test_metrics": selected["test_metrics"],
                "upper_bound_test_metrics": upper_test_metrics,
                "selected_pooling": selected_pooling,
            },
            fold_dir / "fold_summary.json",
        )

    combined = pd.concat(oof_predictions, ignore_index=True)
    text_only_df = combined[combined["benchmark"] == "text_only"].copy()
    upper_df = combined[combined["benchmark"] == "text_plus_tabular"].copy()

    benchmark_rows: list[dict[str, Any]] = []
    for benchmark_name, df in [
        ("text_only", text_only_df),
        ("text_plus_tabular", upper_df),
    ]:
        metrics = evaluate_call_predictions(df["car_horizon"], df["score"], df["probability"])
        ci = bootstrap_confidence_intervals(
            predictions_df=df,
            score_col="score",
            prob_col="probability",
            n_boot=bootstrap_samples,
            seed=seed,
        )
        benchmark_rows.append({"benchmark": benchmark_name, "ci": ci, **metrics})

    write_csv(combined, model_output_dir / "oof_call_predictions.csv")
    write_csv(pd.DataFrame(fold_rows), model_output_dir / "fold_metrics.csv")
    save_json({"dataset_summary": dataset_summary, "splits": split_summary, "benchmarks": benchmark_rows}, model_output_dir / "summary.json")

    notes = [
        "Pair-level text model is a FinBERT regression head trained on analyst-question plus following-answer pairs.",
        "Pool selection uses validation Spearman, with validation AUROC as the tie-breaker.",
        "Upper-bound model is a call-level Ridge regressor using aggregated text scores plus available tabular controls.",
        "Historical earnings surprise is not present in the local corpus, so surprise coverage is currently zero and the feature is excluded from fitting.",
        "Rolling evaluation uses non-overlapping test windows with expanding training history.",
    ]
    report_path = exp_cfg.get("report_path", "reports/qa_pair_regression_report.md")
    write_markdown_report(
        path=report_path,
        run_name=exp_cfg.get("run_name", "qa_pair_finbert_regression"),
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        fold_rows=fold_rows,
        benchmark_rows=benchmark_rows,
        notes=notes,
    )

    return {
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "benchmark_rows": benchmark_rows,
        "report_path": report_path,
        "output_dir": str(model_output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/qa_pair_regression.yaml")
    args = parser.parse_args()
    result = run_experiment(args.config)
    LOGGER.info("Finished QA pair regression experiment: %s", result["output_dir"])


if __name__ == "__main__":
    main()
