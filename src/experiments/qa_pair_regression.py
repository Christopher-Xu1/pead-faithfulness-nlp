from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
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
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)

BASE_CALL_NUMERIC_FEATURES = [
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
    "snapshot_log_market_cap",
    "snapshot_market_cap_percentile",
    "universe_calls_in_gold_corpus",
    "hist_log_market_cap",
    "hist_market_cap_percentile",
    "hist_market_cap_price_lag_days",
    "hist_market_cap_shares_staleness_days",
    "ticker_prior_call_count",
    "ticker_prior_call_count_365d",
    "ticker_days_since_prev_call",
    "ticker_mean_prior_call_gap_days",
    "reported_eps",
    "estimated_eps",
    "eps_surprise",
    "eps_surprise_pct",
    "earnings_surprise",
    "reported_revenue",
    "estimated_revenue",
    "revenue_surprise",
    "revenue_surprise_pct",
    "reported_capex",
    "estimated_capex",
    "capex_surprise",
    "capex_surprise_pct",
    "estimated_capex_is_proxy",
    "prior_capex_to_revenue_ratio",
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

BASE_CALL_CATEGORICAL_FEATURES = [
    "sector",
    "universe_sector",
    "universe_industry",
    "universe_included_by",
    "ticker",
    "source_id",
    "eps_beat_miss",
    "revenue_beat_miss",
    "capex_beat_miss",
]

BASIC_TEXT_FEATURES = ["text_score_mean", "text_score_max"]

RICH_TEXT_FEATURES = [
    "text_score_mean",
    "text_score_max",
    "text_score_min",
    "text_score_median",
    "text_score_std",
    "text_score_range",
    "text_score_q25",
    "text_score_q75",
    "text_score_q90",
    "text_score_top3_mean",
    "text_score_bottom3_mean",
    "text_score_recent3_mean",
    "text_score_recency_weighted_mean",
    "text_score_first",
    "text_score_last",
    "text_score_last_minus_first",
    "text_score_trend_slope",
    "pair_question_chars_mean",
    "pair_question_chars_max",
    "pair_answer_chars_mean",
    "pair_answer_chars_max",
    "pair_answer_turns_mean",
    "pair_answer_turns_max",
    "management_answer_rate",
    "mixed_answer_rate",
    "analyst_only_answer_rate",
    "answer_question_mark_rate",
]

PAIR_SEQUENCE_FEATURE_COLUMNS = [
    "pair_score",
    "pair_position_frac",
    "pair_from_end_frac",
    "question_char_len",
    "answer_char_len",
    "num_answer_turns",
    "answer_management_turn_count",
    "answer_analyst_turn_count",
    "answer_role_switch_count",
    "answer_management_turn_share",
    "answer_analyst_turn_share",
    "answer_has_management_role",
    "answer_has_analyst_role",
    "answer_has_mixed_roles",
    "answer_is_analyst_only",
    "answer_starts_with_management",
    "answer_ends_with_management",
    "answer_starts_with_analyst",
    "answer_ends_with_analyst",
    "snapshot_log_market_cap",
    "snapshot_market_cap_percentile",
    "universe_calls_in_gold_corpus",
    "hist_log_market_cap",
    "hist_market_cap_percentile",
    "hist_market_cap_price_lag_days",
    "hist_market_cap_shares_staleness_days",
    "estimated_capex_is_proxy",
    "prior_capex_to_revenue_ratio",
    "ticker_prior_call_count",
    "ticker_prior_call_count_365d",
    "ticker_days_since_prev_call",
    "ticker_mean_prior_call_gap_days",
]


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


def _pool_pair_scores(group: pd.DataFrame, score_col: str, pooling: str) -> float:
    ordered = group.sort_values("pair_index")
    scores = ordered[score_col].to_numpy(dtype=float)
    if len(scores) == 0:
        return float("nan")
    if pooling == "mean":
        return float(scores.mean())
    if pooling == "max":
        return float(scores.max())
    if pooling == "median":
        return float(np.median(scores))
    if pooling == "min":
        return float(scores.min())
    if pooling == "first":
        return float(scores[0])
    if pooling == "last":
        return float(scores[-1])
    if pooling == "top3_mean":
        k = min(3, len(scores))
        return float(np.sort(scores)[-k:].mean())
    if pooling == "bottom3_mean":
        k = min(3, len(scores))
        return float(np.sort(scores)[:k].mean())
    if pooling == "recency_weighted_mean":
        weights = np.linspace(1.0, 2.0, len(scores))
        return float(np.average(scores, weights=weights))
    raise ValueError(f"Unsupported pooling={pooling!r}")


def aggregate_pair_scores(df: pd.DataFrame, score_col: str, pooling: str) -> pd.DataFrame:
    group_cols = ["call_id", "ticker", "event_date", "car_horizon", "label"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, sort=False):
        call_id, ticker, event_date, car_horizon, label = keys
        rows.append(
            {
                "call_id": call_id,
                "ticker": ticker,
                "event_date": event_date,
                "car_horizon": car_horizon,
                "label": label,
                "call_score": _pool_pair_scores(group, score_col=score_col, pooling=pooling),
                "num_pairs": int(len(group)),
                "pooling": pooling,
            }
        )
    return pd.DataFrame(rows)


def _score_trend_slope(scores: np.ndarray) -> float:
    if len(scores) < 2 or np.allclose(scores, scores[0]):
        return 0.0
    x = np.arange(len(scores), dtype=float)
    slope = np.polyfit(x, scores, deg=1)[0]
    return float(slope)


def build_call_text_features(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    group_cols = ["call_id", "ticker", "event_date", "car_horizon", "label"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, sort=False):
        call_id, ticker, event_date, car_horizon, label = keys
        ordered = group.sort_values("pair_index")
        scores = ordered[score_col].to_numpy(dtype=float)
        question_chars = ordered["question_char_len"].to_numpy(dtype=float)
        answer_chars = ordered["answer_char_len"].to_numpy(dtype=float)
        answer_turns = ordered["num_answer_turns"].to_numpy(dtype=float)
        weights = np.linspace(1.0, 2.0, len(scores))
        recent_k = min(3, len(scores))
        top_k = min(3, len(scores))
        rows.append(
            {
                "call_id": call_id,
                "ticker": ticker,
                "event_date": event_date,
                "car_horizon": car_horizon,
                "label": label,
                "text_score_mean": float(scores.mean()),
                "text_score_max": float(scores.max()),
                "text_score_min": float(scores.min()),
                "text_score_median": float(np.median(scores)),
                "text_score_std": float(np.std(scores)),
                "text_score_range": float(scores.max() - scores.min()),
                "text_score_q25": float(np.quantile(scores, 0.25)),
                "text_score_q75": float(np.quantile(scores, 0.75)),
                "text_score_q90": float(np.quantile(scores, 0.90)),
                "text_score_top3_mean": float(np.sort(scores)[-top_k:].mean()),
                "text_score_bottom3_mean": float(np.sort(scores)[:top_k].mean()),
                "text_score_recent3_mean": float(scores[-recent_k:].mean()),
                "text_score_recency_weighted_mean": float(np.average(scores, weights=weights)),
                "text_score_first": float(scores[0]),
                "text_score_last": float(scores[-1]),
                "text_score_last_minus_first": float(scores[-1] - scores[0]),
                "text_score_trend_slope": _score_trend_slope(scores),
                "pair_question_chars_mean": float(question_chars.mean()),
                "pair_question_chars_max": float(question_chars.max()),
                "pair_answer_chars_mean": float(answer_chars.mean()),
                "pair_answer_chars_max": float(answer_chars.max()),
                "pair_answer_turns_mean": float(answer_turns.mean()),
                "pair_answer_turns_max": float(answer_turns.max()),
                "management_answer_rate": float(ordered["answer_has_management_role"].astype(float).mean()),
                "mixed_answer_rate": float(ordered["answer_has_mixed_roles"].astype(float).mean()),
                "analyst_only_answer_rate": float(ordered["answer_is_analyst_only"].astype(float).mean()),
                "answer_question_mark_rate": float(ordered["answer_contains_question_mark"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def _sequence_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_pair_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom = np.maximum(out["num_pairs"].to_numpy(dtype=float) - 1.0, 1.0)
    out["pair_position_frac"] = out["pair_index"].to_numpy(dtype=float) / denom
    out["pair_from_end_frac"] = 1.0 - out["pair_position_frac"]
    for column in PAIR_SEQUENCE_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(float)
    return out


def compute_sequence_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in feature_cols:
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[column] = {"mean": mean, "std": std}
    return stats


def build_call_sequence_records(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_stats: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group_cols = ["call_id", "ticker", "event_date", "car_horizon", "label"]
    for keys, group in df.groupby(group_cols, sort=False):
        call_id, ticker, event_date, car_horizon, label = keys
        ordered = group.sort_values("pair_index").reset_index(drop=True)
        feature_matrix = ordered[feature_cols].to_numpy(dtype=float)
        for idx, column in enumerate(feature_cols):
            stats = feature_stats[column]
            feature_matrix[:, idx] = (feature_matrix[:, idx] - stats["mean"]) / stats["std"]
        records.append(
            {
                "call_id": call_id,
                "ticker": ticker,
                "event_date": event_date,
                "car_horizon": float(car_horizon),
                "label": int(label),
                "features": feature_matrix.astype(np.float32),
            }
        )
    return records


class CallSequenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def _collate_call_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [item["features"].shape[0] for item in batch]
    max_len = max(lengths)
    input_dim = batch[0]["features"].shape[1]
    features = np.zeros((len(batch), max_len, input_dim), dtype=np.float32)
    labels = np.zeros(len(batch), dtype=np.float32)
    car = np.zeros(len(batch), dtype=np.float32)
    call_ids: list[str] = []
    tickers: list[str] = []
    event_dates: list[str] = []
    for idx, item in enumerate(batch):
        seq_len = item["features"].shape[0]
        features[idx, :seq_len, :] = item["features"]
        labels[idx] = float(item["label"])
        car[idx] = float(item["car_horizon"])
        call_ids.append(str(item["call_id"]))
        tickers.append(str(item["ticker"]))
        event_dates.append(str(item["event_date"]))
    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "car_horizon": torch.tensor(car, dtype=torch.float32),
        "call_id": call_ids,
        "ticker": tickers,
        "event_date": event_dates,
    }


class GRUSequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(features)
        packed = pack_padded_sequence(hidden, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_n = self.gru(packed)
        pooled = hidden_n[-1]
        return self.classifier(self.dropout(pooled)).squeeze(-1)


class AttentionPoolingSequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(features)
        scores = self.attention(hidden).squeeze(-1)
        mask = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def _build_sequence_model(model_kind: str, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if model_kind == "gru":
        return GRUSequenceClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_kind == "attention":
        return AttentionPoolingSequenceClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported sequence model kind={model_kind!r}")


def _evaluate_sequence_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["features"].to(device),
                batch["lengths"].to(device),
            )
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(batch["labels"].cpu().numpy())
    if not logits_list:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(logits_list), np.concatenate(labels_list)


def train_sequence_classifier(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    model_kind: str,
    feature_cols: list[str],
    sequence_cfg: dict[str, Any],
    seed: int,
) -> tuple[nn.Module, dict[str, dict[str, float]], dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prepared_train = prepare_pair_sequence_features(train_pairs)
    prepared_val = prepare_pair_sequence_features(val_pairs)
    feature_stats = compute_sequence_feature_stats(prepared_train, feature_cols=feature_cols)
    train_records = build_call_sequence_records(prepared_train, feature_cols=feature_cols, feature_stats=feature_stats)
    val_records = build_call_sequence_records(prepared_val, feature_cols=feature_cols, feature_stats=feature_stats)

    batch_size = int(sequence_cfg.get("batch_size", 16))
    train_loader = DataLoader(
        CallSequenceDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_call_sequences,
    )
    val_loader = DataLoader(
        CallSequenceDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_call_sequences,
    )

    hidden_dim = int(sequence_cfg.get("hidden_dim", 32))
    dropout = float(sequence_cfg.get("dropout", 0.1))
    learning_rate = float(sequence_cfg.get("learning_rate", 1e-3))
    weight_decay = float(sequence_cfg.get("weight_decay", 1e-4))
    num_epochs = int(sequence_cfg.get("num_epochs", 40))
    patience = int(sequence_cfg.get("patience", 6))

    device = _sequence_device()
    model = _build_sequence_model(model_kind=model_kind, input_dim=len(feature_cols), hidden_dim=hidden_dim, dropout=dropout)
    model.to(device)

    labels = np.asarray([record["label"] for record in train_records], dtype=float)
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device) if positives > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_score = -math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    best_metrics: dict[str, Any] = {}

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["features"].to(device), batch["lengths"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_logits, val_labels = _evaluate_sequence_classifier(model, val_loader, device=device)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics = probability_classification_metrics(val_labels, val_prob)
        val_score = val_metrics["auprc"] if np.isfinite(val_metrics["auprc"]) else -math.inf
        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "val_auprc": val_metrics["auprc"],
                "val_auroc": val_metrics["auroc"],
                "val_accuracy": val_metrics["accuracy"],
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "model_kind": model_kind,
        "feature_cols": feature_cols,
        "best_epoch": best_epoch,
        **best_metrics,
    }
    return model, feature_stats, metadata


def predict_sequence_classifier(
    df: pd.DataFrame,
    model: nn.Module,
    feature_cols: list[str],
    feature_stats: dict[str, dict[str, float]],
    batch_size: int = 16,
) -> pd.DataFrame:
    prepared = prepare_pair_sequence_features(df)
    records = build_call_sequence_records(prepared, feature_cols=feature_cols, feature_stats=feature_stats)
    loader = DataLoader(
        CallSequenceDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_call_sequences,
    )
    device = _sequence_device()
    model.to(device)
    model.eval()
    output_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"].to(device), batch["lengths"].to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            labels_np = batch["labels"].cpu().numpy()
            car_np = batch["car_horizon"].cpu().numpy()
            for idx in range(len(logits_np)):
                output_rows.append(
                    {
                        "call_id": batch["call_id"][idx],
                        "ticker": batch["ticker"][idx],
                        "event_date": batch["event_date"][idx],
                        "car_horizon": float(car_np[idx]),
                        "label": int(labels_np[idx]),
                        "sequence_score": float(logits_np[idx]),
                        "sequence_probability": float(probs[idx]),
                    }
                )
    return pd.DataFrame(output_rows)


def fit_score_calibrator(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression | ConstantCalibrator:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return ConstantCalibrator(float(labels.mean()) if len(labels) else 0.5)
    model = LogisticRegression(random_state=42, solver="lbfgs")
    model.fit(np.asarray(scores).reshape(-1, 1), labels)
    return model


def probability_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    threshold_arr = np.asarray(threshold, dtype=float)
    if threshold_arr.ndim == 0:
        threshold_arr = np.full(len(y_prob), float(threshold_arr), dtype=float)
    y_pred = (y_prob >= threshold_arr).astype(int)

    auroc = float("nan")
    auprc = float("nan")
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def select_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "accuracy",
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_prob) == 0:
        return 0.5, float("nan")

    candidate_thresholds = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0.05, 0.95, 19),
                    y_prob,
                    [0.25, 0.5, 0.75],
                ]
            ),
            0.0,
            1.0,
        )
    )

    best_threshold = 0.5
    best_score = -math.inf
    best_distance = math.inf
    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if objective == "accuracy":
            score = float(accuracy_score(y_true, y_pred))
        elif objective == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, y_pred))
        elif objective == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            raise ValueError(f"Unsupported threshold objective={objective!r}")

        distance = abs(float(threshold) - 0.5)
        if score > best_score or (math.isclose(score, best_score) and distance < best_distance):
            best_threshold = float(threshold)
            best_score = score
            best_distance = distance
    return best_threshold, best_score


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


def evaluate_call_predictions(
    y_true_car: np.ndarray,
    call_scores: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray = 0.5,
) -> dict[str, float]:
    y_true_car = np.asarray(y_true_car, dtype=float)
    call_scores = np.asarray(call_scores, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_true_sign = (y_true_car > 0.0).astype(int)

    class_metrics = probability_classification_metrics(y_true_sign, y_prob, threshold=threshold)
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
    threshold_col: str | None,
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
            threshold=sample[threshold_col].to_numpy() if threshold_col else 0.5,
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


def _available_feature_columns(
    train_df: pd.DataFrame,
    text_feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric_features = list(dict.fromkeys([*text_feature_columns, *BASE_CALL_NUMERIC_FEATURES]))
    available_numeric = [col for col in numeric_features if col in train_df.columns and not train_df[col].isna().all()]
    available_categorical = [col for col in BASE_CALL_CATEGORICAL_FEATURES if col in train_df.columns]
    return available_numeric, available_categorical


def fit_ridge_upper_bound_model(
    train_df: pd.DataFrame,
    text_feature_columns: list[str],
) -> tuple[Pipeline, list[str], list[str]]:
    available_numeric, available_categorical = _available_feature_columns(
        train_df=train_df,
        text_feature_columns=text_feature_columns,
    )
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


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        return np.ones(len(labels), dtype=float)
    total = counts.sum()
    class_weights = {label: total / (2.0 * count) for label, count in enumerate(counts)}
    return np.asarray([class_weights[int(label)] for label in labels], dtype=float)


def fit_boosted_classifier_model(
    train_df: pd.DataFrame,
    text_feature_columns: list[str],
    seed: int,
    classifier_cfg: dict[str, Any],
) -> tuple[Pipeline, list[str], list[str]]:
    available_numeric, available_categorical = _available_feature_columns(
        train_df=train_df,
        text_feature_columns=text_feature_columns,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                available_numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                available_categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=float(classifier_cfg.get("learning_rate", 0.05)),
        max_iter=int(classifier_cfg.get("max_iter", 250)),
        max_depth=int(classifier_cfg.get("max_depth", 3)),
        min_samples_leaf=int(classifier_cfg.get("min_samples_leaf", 10)),
        l2_regularization=float(classifier_cfg.get("l2_regularization", 1.0)),
        random_state=seed,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
    feature_cols = available_numeric + available_categorical
    sample_weight = _balanced_sample_weights(train_df["label"].to_numpy())
    model.fit(
        train_df[feature_cols],
        train_df["label"].to_numpy().astype(int),
        classifier__sample_weight=sample_weight,
    )
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
            "| Fold | Benchmark | Pooling | Threshold | Val Spearman | Test AUROC | Test AUPRC | Test Accuracy | Test Spearman | Test ECE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['benchmark']} | {row['pooling']} | "
            f"{_format_metric(row.get('threshold', float('nan')))} | "
            f"{_format_metric(row['val_spearman'])} | {_format_metric(row['test_auroc'])} | "
            f"{_format_metric(row['test_auprc'])} | {_format_metric(row['test_accuracy'])} | "
            f"{_format_metric(row['test_spearman'])} | {_format_metric(row['test_ece'])} |"
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
    threshold_cfg = exp_cfg.get("threshold_tuning", {})
    classifier_cfg = exp_cfg.get("tabular_classifier", {})
    sequence_cfg = exp_cfg.get("sequence_model", {})

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
    max_folds = rolling_cfg.get("max_folds")
    if max_folds is not None:
        splits = splits[: int(max_folds)]

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
    pooling_options = aggregation_cfg.get(
        "poolings",
        ["mean", "max", "median", "last", "top3_mean", "recency_weighted_mean"],
    )
    threshold_objective = str(threshold_cfg.get("metric", "accuracy"))
    bootstrap_samples = int(metrics_cfg.get("bootstrap_samples", 400))
    reuse_text_model_dir = exp_cfg.get("reuse_text_model_dir")

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

        pair_val_metrics: dict[str, Any]
        reuse_fold_model_dir = None
        if reuse_text_model_dir:
            candidate = Path(reuse_text_model_dir) / f"fold_{fold:02d}" / "text_model"
            if candidate.exists():
                reuse_fold_model_dir = candidate

        if reuse_fold_model_dir is not None:
            LOGGER.info("Reusing text model from %s", reuse_fold_model_dir)
            tokenizer = AutoTokenizer.from_pretrained(reuse_fold_model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(reuse_fold_model_dir)
            metrics_path = reuse_fold_model_dir / "val_pair_metrics.json"
            pair_val_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        else:
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
            tuned_threshold, _ = select_probability_threshold(
                val_scores["label"].to_numpy(),
                val_prob,
                objective=threshold_objective,
            )
            val_metrics = evaluate_call_predictions(
                val_scores["car_horizon"],
                val_scores["call_score"],
                val_prob,
                threshold=tuned_threshold,
            )
            test_metrics = evaluate_call_predictions(
                test_scores["car_horizon"],
                test_scores["call_score"],
                test_prob,
                threshold=tuned_threshold,
            )
            pool_results.append(
                {
                    "pooling": pooling,
                    "calibrator": calibrator,
                    "threshold": tuned_threshold,
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
        selected_threshold = float(selected["threshold"])
        selected_test = aggregate_maps[selected_pooling]["test"].rename(columns={"call_score": "text_score"}).copy()
        selected_test["probability"] = selected["calibrator"].predict_proba(
            selected_test["text_score"].to_numpy().reshape(-1, 1)
        )[:, 1]
        selected_test["threshold"] = selected_threshold

        fold_rows.append(
            {
                "fold": fold,
                "benchmark": "text_only",
                "pooling": selected_pooling,
                "threshold": selected_threshold,
                "val_spearman": selected["val_metrics"]["spearman"],
                "test_auroc": selected["test_metrics"]["auroc"],
                "test_auprc": selected["test_metrics"]["auprc"],
                "test_accuracy": selected["test_metrics"]["accuracy"],
                "test_spearman": selected["test_metrics"]["spearman"],
                "test_ece": selected["test_metrics"]["ece"],
            }
        )

        train_text_features = build_call_text_features(train_pairs, score_col="pair_score")
        val_text_features = build_call_text_features(val_pairs, score_col="pair_score")
        test_text_features = build_call_text_features(test_pairs, score_col="pair_score")
        rich_text_feature_columns = [
            col for col in train_text_features.columns if col not in {"call_id", "ticker", "event_date", "car_horizon", "label"}
        ]

        train_call_model = train_calls.merge(
            train_text_features[["call_id", *rich_text_feature_columns]],
            on="call_id",
            how="left",
        ).copy()
        val_call_model = val_calls.merge(
            val_text_features[["call_id", *rich_text_feature_columns]],
            on="call_id",
            how="left",
        ).copy()
        test_call_model = test_calls.merge(
            test_text_features[["call_id", *rich_text_feature_columns]],
            on="call_id",
            how="left",
        ).copy()

        sequence_summaries: dict[str, Any] = {}
        sequence_batch_size = int(sequence_cfg.get("batch_size", 16))
        sequence_model_kinds = sequence_cfg.get("models", ["gru", "attention"])
        for model_kind in sequence_model_kinds:
            sequence_model, sequence_feature_stats, sequence_metadata = train_sequence_classifier(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                model_kind=str(model_kind),
                feature_cols=PAIR_SEQUENCE_FEATURE_COLUMNS,
                sequence_cfg=sequence_cfg,
                seed=seed + fold,
            )
            train_sequence_scores = predict_sequence_classifier(
                train_pairs,
                model=sequence_model,
                feature_cols=PAIR_SEQUENCE_FEATURE_COLUMNS,
                feature_stats=sequence_feature_stats,
                batch_size=sequence_batch_size,
            ).rename(
                columns={
                    "sequence_score": f"text_sequence_score_{model_kind}",
                    "sequence_probability": f"text_sequence_probability_{model_kind}",
                }
            )
            val_sequence_scores = predict_sequence_classifier(
                val_pairs,
                model=sequence_model,
                feature_cols=PAIR_SEQUENCE_FEATURE_COLUMNS,
                feature_stats=sequence_feature_stats,
                batch_size=sequence_batch_size,
            ).rename(
                columns={
                    "sequence_score": f"text_sequence_score_{model_kind}",
                    "sequence_probability": f"text_sequence_probability_{model_kind}",
                }
            )
            test_sequence_scores = predict_sequence_classifier(
                test_pairs,
                model=sequence_model,
                feature_cols=PAIR_SEQUENCE_FEATURE_COLUMNS,
                feature_stats=sequence_feature_stats,
                batch_size=sequence_batch_size,
            ).rename(
                columns={
                    "sequence_score": f"text_sequence_score_{model_kind}",
                    "sequence_probability": f"text_sequence_probability_{model_kind}",
                }
            )

            score_col = f"text_sequence_score_{model_kind}"
            train_call_model = train_call_model.merge(train_sequence_scores[["call_id", score_col]], on="call_id", how="left")
            val_call_model = val_call_model.merge(val_sequence_scores[["call_id", score_col]], on="call_id", how="left")
            test_call_model = test_call_model.merge(test_sequence_scores[["call_id", score_col]], on="call_id", how="left")

            sequence_ridge_model, sequence_numeric_cols, sequence_categorical_cols = fit_ridge_upper_bound_model(
                train_call_model,
                text_feature_columns=[score_col],
            )
            sequence_feature_cols = sequence_numeric_cols + sequence_categorical_cols
            sequence_score_name = f"{model_kind}_ridge_score"
            sequence_prob_name = f"{model_kind}_ridge_probability"
            val_call_model[sequence_score_name] = sequence_ridge_model.predict(val_call_model[sequence_feature_cols])
            test_call_model[sequence_score_name] = sequence_ridge_model.predict(test_call_model[sequence_feature_cols])
            sequence_calibrator = fit_score_calibrator(
                val_call_model[sequence_score_name].to_numpy(),
                val_call_model["label"].to_numpy(),
            )
            val_call_model[sequence_prob_name] = sequence_calibrator.predict_proba(
                val_call_model[sequence_score_name].to_numpy().reshape(-1, 1)
            )[:, 1]
            test_call_model[sequence_prob_name] = sequence_calibrator.predict_proba(
                test_call_model[sequence_score_name].to_numpy().reshape(-1, 1)
            )[:, 1]
            sequence_threshold, _ = select_probability_threshold(
                val_call_model["label"].to_numpy(),
                val_call_model[sequence_prob_name].to_numpy(),
                objective=threshold_objective,
            )
            sequence_val_metrics = evaluate_call_predictions(
                val_call_model["car_horizon"],
                val_call_model[sequence_score_name],
                val_call_model[sequence_prob_name],
                threshold=sequence_threshold,
            )
            sequence_test_metrics = evaluate_call_predictions(
                test_call_model["car_horizon"],
                test_call_model[sequence_score_name],
                test_call_model[sequence_prob_name],
                threshold=sequence_threshold,
            )
            benchmark_name = f"text_tabular_ridge_{model_kind}_tuned"
            fold_rows.append(
                {
                    "fold": fold,
                    "benchmark": benchmark_name,
                    "pooling": "sequence",
                    "threshold": sequence_threshold,
                    "val_spearman": sequence_val_metrics["spearman"],
                    "test_auroc": sequence_test_metrics["auroc"],
                    "test_auprc": sequence_test_metrics["auprc"],
                    "test_accuracy": sequence_test_metrics["accuracy"],
                    "test_spearman": sequence_test_metrics["spearman"],
                    "test_ece": sequence_test_metrics["ece"],
                }
            )
            sequence_preds = test_call_model[
                ["call_id", "ticker", "event_date", "car_horizon", "label", sequence_score_name, sequence_prob_name]
            ].copy()
            sequence_preds["benchmark"] = benchmark_name
            sequence_preds["fold"] = fold
            sequence_preds["threshold"] = sequence_threshold
            sequence_preds = sequence_preds.rename(columns={sequence_score_name: "score", sequence_prob_name: "probability"})
            oof_predictions.append(sequence_preds)
            write_csv(sequence_preds, fold_dir / f"{model_kind}_ridge_tuned_test_predictions.csv")
            sequence_summaries[benchmark_name] = {
                "sequence_metadata": sequence_metadata,
                "test_metrics": sequence_test_metrics,
            }

        ridge_base_model, ridge_base_numeric, ridge_base_categorical = fit_ridge_upper_bound_model(
            train_call_model,
            text_feature_columns=BASIC_TEXT_FEATURES,
        )
        ridge_base_features = ridge_base_numeric + ridge_base_categorical
        val_call_model["ridge_base_score"] = ridge_base_model.predict(val_call_model[ridge_base_features])
        test_call_model["ridge_base_score"] = ridge_base_model.predict(test_call_model[ridge_base_features])
        ridge_base_calibrator = fit_score_calibrator(
            val_call_model["ridge_base_score"].to_numpy(),
            val_call_model["label"].to_numpy(),
        )
        val_call_model["ridge_base_probability"] = ridge_base_calibrator.predict_proba(
            val_call_model["ridge_base_score"].to_numpy().reshape(-1, 1)
        )[:, 1]
        test_call_model["ridge_base_probability"] = ridge_base_calibrator.predict_proba(
            test_call_model["ridge_base_score"].to_numpy().reshape(-1, 1)
        )[:, 1]
        ridge_base_tuned_threshold, _ = select_probability_threshold(
            val_call_model["label"].to_numpy(),
            val_call_model["ridge_base_probability"].to_numpy(),
            objective=threshold_objective,
        )
        ridge_base_val_metrics = evaluate_call_predictions(
            val_call_model["car_horizon"],
            val_call_model["ridge_base_score"],
            val_call_model["ridge_base_probability"],
            threshold=0.5,
        )
        ridge_base_test_metrics = evaluate_call_predictions(
            test_call_model["car_horizon"],
            test_call_model["ridge_base_score"],
            test_call_model["ridge_base_probability"],
            threshold=0.5,
        )
        ridge_base_tuned_val_metrics = evaluate_call_predictions(
            val_call_model["car_horizon"],
            val_call_model["ridge_base_score"],
            val_call_model["ridge_base_probability"],
            threshold=ridge_base_tuned_threshold,
        )
        ridge_base_tuned_test_metrics = evaluate_call_predictions(
            test_call_model["car_horizon"],
            test_call_model["ridge_base_score"],
            test_call_model["ridge_base_probability"],
            threshold=ridge_base_tuned_threshold,
        )

        fold_rows.extend(
            [
                {
                    "fold": fold,
                    "benchmark": "text_tabular_ridge_base",
                    "pooling": "mean+max",
                    "threshold": 0.5,
                    "val_spearman": ridge_base_val_metrics["spearman"],
                    "test_auroc": ridge_base_test_metrics["auroc"],
                    "test_auprc": ridge_base_test_metrics["auprc"],
                    "test_accuracy": ridge_base_test_metrics["accuracy"],
                    "test_spearman": ridge_base_test_metrics["spearman"],
                    "test_ece": ridge_base_test_metrics["ece"],
                },
                {
                    "fold": fold,
                    "benchmark": "text_tabular_ridge_base_tuned",
                    "pooling": "mean+max",
                    "threshold": ridge_base_tuned_threshold,
                    "val_spearman": ridge_base_tuned_val_metrics["spearman"],
                    "test_auroc": ridge_base_tuned_test_metrics["auroc"],
                    "test_auprc": ridge_base_tuned_test_metrics["auprc"],
                    "test_accuracy": ridge_base_tuned_test_metrics["accuracy"],
                    "test_spearman": ridge_base_tuned_test_metrics["spearman"],
                    "test_ece": ridge_base_tuned_test_metrics["ece"],
                },
            ]
        )

        ridge_rich_model, ridge_rich_numeric, ridge_rich_categorical = fit_ridge_upper_bound_model(
            train_call_model,
            text_feature_columns=RICH_TEXT_FEATURES,
        )
        ridge_rich_features = ridge_rich_numeric + ridge_rich_categorical
        val_call_model["ridge_rich_score"] = ridge_rich_model.predict(val_call_model[ridge_rich_features])
        test_call_model["ridge_rich_score"] = ridge_rich_model.predict(test_call_model[ridge_rich_features])
        ridge_rich_calibrator = fit_score_calibrator(
            val_call_model["ridge_rich_score"].to_numpy(),
            val_call_model["label"].to_numpy(),
        )
        val_call_model["ridge_rich_probability"] = ridge_rich_calibrator.predict_proba(
            val_call_model["ridge_rich_score"].to_numpy().reshape(-1, 1)
        )[:, 1]
        test_call_model["ridge_rich_probability"] = ridge_rich_calibrator.predict_proba(
            test_call_model["ridge_rich_score"].to_numpy().reshape(-1, 1)
        )[:, 1]
        ridge_rich_threshold, _ = select_probability_threshold(
            val_call_model["label"].to_numpy(),
            val_call_model["ridge_rich_probability"].to_numpy(),
            objective=threshold_objective,
        )
        ridge_rich_val_metrics = evaluate_call_predictions(
            val_call_model["car_horizon"],
            val_call_model["ridge_rich_score"],
            val_call_model["ridge_rich_probability"],
            threshold=ridge_rich_threshold,
        )
        ridge_rich_test_metrics = evaluate_call_predictions(
            test_call_model["car_horizon"],
            test_call_model["ridge_rich_score"],
            test_call_model["ridge_rich_probability"],
            threshold=ridge_rich_threshold,
        )
        fold_rows.append(
            {
                "fold": fold,
                "benchmark": "text_tabular_ridge_rich_tuned",
                "pooling": "rich_agg",
                "threshold": ridge_rich_threshold,
                "val_spearman": ridge_rich_val_metrics["spearman"],
                "test_auroc": ridge_rich_test_metrics["auroc"],
                "test_auprc": ridge_rich_test_metrics["auprc"],
                "test_accuracy": ridge_rich_test_metrics["accuracy"],
                "test_spearman": ridge_rich_test_metrics["spearman"],
                "test_ece": ridge_rich_test_metrics["ece"],
            }
        )

        boosted_model, boosted_numeric, boosted_categorical = fit_boosted_classifier_model(
            train_call_model,
            text_feature_columns=RICH_TEXT_FEATURES,
            seed=seed + fold,
            classifier_cfg=classifier_cfg,
        )
        boosted_features = boosted_numeric + boosted_categorical
        val_call_model["boosted_probability"] = boosted_model.predict_proba(val_call_model[boosted_features])[:, 1]
        test_call_model["boosted_probability"] = boosted_model.predict_proba(test_call_model[boosted_features])[:, 1]
        val_call_model["boosted_score"] = val_call_model["boosted_probability"] - 0.5
        test_call_model["boosted_score"] = test_call_model["boosted_probability"] - 0.5
        boosted_threshold, _ = select_probability_threshold(
            val_call_model["label"].to_numpy(),
            val_call_model["boosted_probability"].to_numpy(),
            objective=threshold_objective,
        )
        boosted_val_metrics = evaluate_call_predictions(
            val_call_model["car_horizon"],
            val_call_model["boosted_score"],
            val_call_model["boosted_probability"],
            threshold=boosted_threshold,
        )
        boosted_test_metrics = evaluate_call_predictions(
            test_call_model["car_horizon"],
            test_call_model["boosted_score"],
            test_call_model["boosted_probability"],
            threshold=boosted_threshold,
        )
        fold_rows.append(
            {
                "fold": fold,
                "benchmark": "text_tabular_boosted_rich_tuned",
                "pooling": "rich_agg",
                "threshold": boosted_threshold,
                "val_spearman": boosted_val_metrics["spearman"],
                "test_auroc": boosted_test_metrics["auroc"],
                "test_auprc": boosted_test_metrics["auprc"],
                "test_accuracy": boosted_test_metrics["accuracy"],
                "test_spearman": boosted_test_metrics["spearman"],
                "test_ece": boosted_test_metrics["ece"],
            }
        )

        text_preds = selected_test[["call_id", "ticker", "event_date", "car_horizon", "label", "text_score", "probability", "threshold"]].copy()
        text_preds = text_preds.rename(columns={"text_score": "score"})
        text_preds["benchmark"] = "text_only"
        text_preds["fold"] = fold
        text_preds["selected_pooling"] = selected_pooling
        oof_predictions.append(text_preds)

        ridge_base_preds = test_call_model[
            ["call_id", "ticker", "event_date", "car_horizon", "label", "ridge_base_score", "ridge_base_probability"]
        ].copy()
        ridge_base_preds["benchmark"] = "text_tabular_ridge_base"
        ridge_base_preds["fold"] = fold
        ridge_base_preds["threshold"] = 0.5
        ridge_base_preds = ridge_base_preds.rename(
            columns={"ridge_base_score": "score", "ridge_base_probability": "probability"}
        )
        oof_predictions.append(ridge_base_preds)

        ridge_base_tuned_preds = ridge_base_preds.copy()
        ridge_base_tuned_preds["benchmark"] = "text_tabular_ridge_base_tuned"
        ridge_base_tuned_preds["threshold"] = ridge_base_tuned_threshold
        oof_predictions.append(ridge_base_tuned_preds)

        ridge_rich_preds = test_call_model[
            ["call_id", "ticker", "event_date", "car_horizon", "label", "ridge_rich_score", "ridge_rich_probability"]
        ].copy()
        ridge_rich_preds["benchmark"] = "text_tabular_ridge_rich_tuned"
        ridge_rich_preds["fold"] = fold
        ridge_rich_preds["threshold"] = ridge_rich_threshold
        ridge_rich_preds = ridge_rich_preds.rename(
            columns={"ridge_rich_score": "score", "ridge_rich_probability": "probability"}
        )
        oof_predictions.append(ridge_rich_preds)

        boosted_preds = test_call_model[
            ["call_id", "ticker", "event_date", "car_horizon", "label", "boosted_score", "boosted_probability"]
        ].copy()
        boosted_preds["benchmark"] = "text_tabular_boosted_rich_tuned"
        boosted_preds["fold"] = fold
        boosted_preds["threshold"] = boosted_threshold
        boosted_preds = boosted_preds.rename(columns={"boosted_score": "score", "boosted_probability": "probability"})
        oof_predictions.append(boosted_preds)

        write_csv(text_preds, fold_dir / "text_only_test_predictions.csv")
        write_csv(ridge_base_preds, fold_dir / "ridge_base_test_predictions.csv")
        write_csv(ridge_base_tuned_preds, fold_dir / "ridge_base_tuned_test_predictions.csv")
        write_csv(ridge_rich_preds, fold_dir / "ridge_rich_test_predictions.csv")
        write_csv(boosted_preds, fold_dir / "boosted_rich_test_predictions.csv")
        save_json(
            {
                "pair_val_metrics": pair_val_metrics,
                "text_only_test_metrics": selected["test_metrics"],
                "ridge_base_test_metrics": ridge_base_test_metrics,
                "ridge_base_tuned_test_metrics": ridge_base_tuned_test_metrics,
                "ridge_rich_test_metrics": ridge_rich_test_metrics,
                "boosted_rich_test_metrics": boosted_test_metrics,
                "sequence_model_summaries": sequence_summaries,
                "selected_pooling": selected_pooling,
                "threshold_objective": threshold_objective,
            },
            fold_dir / "fold_summary.json",
        )

    combined = pd.concat(oof_predictions, ignore_index=True)

    benchmark_order = [
        "text_only",
        "text_tabular_ridge_base",
        "text_tabular_ridge_base_tuned",
        "text_tabular_ridge_gru_tuned",
        "text_tabular_ridge_attention_tuned",
        "text_tabular_ridge_rich_tuned",
        "text_tabular_boosted_rich_tuned",
    ]
    benchmark_rows: list[dict[str, Any]] = []
    for benchmark_name in benchmark_order:
        df = combined[combined["benchmark"] == benchmark_name].copy()
        if df.empty:
            continue
        metrics = evaluate_call_predictions(
            df["car_horizon"],
            df["score"],
            df["probability"],
            threshold=df["threshold"].to_numpy(),
        )
        ci = bootstrap_confidence_intervals(
            predictions_df=df,
            score_col="score",
            prob_col="probability",
            threshold_col="threshold",
            n_boot=bootstrap_samples,
            seed=seed,
        )
        benchmark_rows.append({"benchmark": benchmark_name, "ci": ci, **metrics})

    write_csv(combined, model_output_dir / "oof_call_predictions.csv")
    write_csv(pd.DataFrame(fold_rows), model_output_dir / "fold_metrics.csv")
    benchmark_export_rows = [{key: value for key, value in row.items() if key != "ci"} for row in benchmark_rows]
    benchmark_export_df = pd.DataFrame(benchmark_export_rows)
    if not benchmark_export_df.empty and "text_tabular_ridge_base" in benchmark_export_df["benchmark"].tolist():
        base_row = benchmark_export_df[benchmark_export_df["benchmark"] == "text_tabular_ridge_base"].iloc[0]
        for metric_name in ["auroc", "auprc", "accuracy", "spearman", "ece"]:
            benchmark_export_df[f"delta_{metric_name}_vs_ridge_base"] = (
                benchmark_export_df[metric_name] - float(base_row[metric_name])
            )
    write_csv(benchmark_export_df, model_output_dir / "benchmark_summary.csv")
    save_json({"dataset_summary": dataset_summary, "splits": split_summary, "benchmarks": benchmark_rows}, model_output_dir / "summary.json")

    notes = [
        "Pair-level text model is a FinBERT regression head trained on analyst-question plus following-answer pairs.",
        "Pool selection uses validation Spearman, with validation AUROC as the tie-breaker.",
        "The benchmark ladder compares the original mean+max Ridge baseline, the same model with tuned thresholds, a richer order-aware aggregation feature set, and a boosted classifier on the rich feature set.",
        "Sequence benchmarks train compact call-level classifiers over ordered QA-pair feature sequences, including pair score, pair position, answer lengths, and coarse speaker-role features.",
        "Richer text aggregation includes distributional and order-aware features such as quantiles, recent-pair averages, first/last deltas, and score trend slope.",
        f"Thresholds are selected on each validation fold to maximize `{threshold_objective}` rather than using a fixed 0.5 cutoff.",
        "Event-level earnings surprise and beat/miss features are used when available in the local fundamentals table.",
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
