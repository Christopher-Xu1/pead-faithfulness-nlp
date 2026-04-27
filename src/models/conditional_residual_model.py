from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.utils.io import ensure_dir, save_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def infer_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class NumericFeatureScaler:
    feature_columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_columns: list[str]) -> "NumericFeatureScaler":
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for column in feature_columns:
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(values.median()) if not values.dropna().empty else 0.0
            imputed = values.fillna(median)
            mean = float(imputed.mean())
            std = float(imputed.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            medians[column] = median
            means[column] = mean
            stds[column] = std
        return cls(feature_columns=list(feature_columns), medians=medians, means=means, stds=stds)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        matrix: list[np.ndarray] = []
        for column in self.feature_columns:
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            values = values.fillna(self.medians[column]).astype(float)
            values = (values - self.means[column]) / self.stds[column]
            matrix.append(values.to_numpy(dtype=np.float32))
        if not matrix:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.column_stack(matrix).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
        }


class PairResidualDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        numeric_scaler: NumericFeatureScaler,
        conditioning_columns: list[str],
        target_column: str = "residual_target",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.questions = self.df["question_text"].fillna("").astype(str).tolist()
        self.answers = self.df["answer_text"].fillna("").astype(str).tolist()
        self.numeric_features = numeric_scaler.transform(self.df[conditioning_columns])
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.target_column = target_column
        self.has_labels = target_column in self.df.columns
        self.labels = (
            pd.to_numeric(self.df[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if self.has_labels
            else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.questions[idx],
            self.answers[idx],
            truncation="longest_first",
            max_length=self.max_length,
            padding=False,
        )
        item: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "numeric_features": self.numeric_features[idx],
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"]
        if self.has_labels and self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class PairResidualCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        token_features = []
        for item in batch:
            token_item = {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            if "token_type_ids" in item:
                token_item["token_type_ids"] = item["token_type_ids"]
            token_features.append(token_item)
        padded = self.tokenizer.pad(token_features, return_tensors="pt")
        padded["numeric_features"] = torch.tensor(
            np.stack([item["numeric_features"] for item in batch]),
            dtype=torch.float32,
        )
        if "labels" in batch[0]:
            padded["labels"] = torch.tensor([item["labels"] for item in batch], dtype=torch.float32)
        return padded


class ConditionalResidualFinBERTRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        numeric_feature_dim: int,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = int(self.encoder.config.hidden_size)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + numeric_feature_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_features: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0]
        combined = torch.cat([pooled, numeric_features], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)
        result = {"predictions": predictions}
        if labels is not None:
            result["loss"] = nn.functional.mse_loss(predictions, labels)
        return result


@dataclass
class PairModelBundle:
    model: ConditionalResidualFinBERTRegressor
    tokenizer: Any
    numeric_scaler: NumericFeatureScaler
    conditioning_columns: list[str]
    max_length: int
    eval_batch_size: int


def _evaluate_pair_model(
    model: ConditionalResidualFinBERTRegressor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.get("labels")
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids").to(device) if batch.get("token_type_ids") is not None else None,
                numeric_features=batch["numeric_features"].to(device),
                labels=labels.to(device) if labels is not None else None,
            )
            if "loss" in outputs:
                losses.append(float(outputs["loss"].detach().cpu().item()))
            predictions.append(outputs["predictions"].detach().cpu().numpy())
            if labels is not None:
                targets.append(labels.detach().cpu().numpy())
    if not predictions or not targets:
        return {"mse": float("nan"), "rmse": float("nan")}
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    mse = float(np.mean((pred - target) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "loss": float(np.mean(losses)) if losses else mse,
    }


def train_conditional_residual_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    conditioning_columns: list[str],
    max_length: int,
    output_dir: str | Path,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    eval_batch_size: int,
    num_train_epochs: int,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    head_hidden_dim: int,
    dropout: float,
    patience: int,
    seed: int,
    freeze_encoder: bool = False,
) -> tuple[PairModelBundle, dict[str, Any]]:
    if train_df.empty or val_df.empty:
        raise ValueError("Conditional residual model requires non-empty train and validation pair datasets")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = ensure_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    numeric_scaler = NumericFeatureScaler.fit(train_df, conditioning_columns)

    train_dataset = PairResidualDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        numeric_scaler=numeric_scaler,
        conditioning_columns=conditioning_columns,
    )
    val_dataset = PairResidualDataset(
        df=val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        numeric_scaler=numeric_scaler,
        conditioning_columns=conditioning_columns,
    )
    collator = PairResidualCollator(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator)

    device = infer_torch_device()
    model = ConditionalResidualFinBERTRegressor(
        model_name=model_name,
        numeric_feature_dim=len(conditioning_columns),
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, gradient_accumulation_steps)))
    total_train_steps = max(1, steps_per_epoch * int(num_train_epochs))
    warmup_steps = int(total_train_steps * float(warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    best_state: dict[str, Any] | None = None
    best_val_rmse = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(int(num_train_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids").to(device) if batch.get("token_type_ids") is not None else None,
                numeric_features=batch["numeric_features"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs["loss"] / max(1, gradient_accumulation_steps)
            loss.backward()
            epoch_losses.append(float(outputs["loss"].detach().cpu().item()))

            should_step = (step + 1) % max(1, gradient_accumulation_steps) == 0 or (step + 1) == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_metrics = _evaluate_pair_model(model=model, loader=val_loader, device=device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "val_mse": float(val_metrics["mse"]),
                "val_rmse": float(val_metrics["rmse"]),
            }
        )
        LOGGER.info(
            "Conditional residual epoch=%d train_loss=%.4f val_rmse=%.4f",
            epoch,
            float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            float(val_metrics["rmse"]),
        )
        if float(val_metrics["rmse"]) < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    metadata = {
        "model_name": model_name,
        "conditioning_columns": conditioning_columns,
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "eval_batch_size": int(eval_batch_size),
        "num_train_epochs": int(num_train_epochs),
        "warmup_ratio": float(warmup_ratio),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "head_hidden_dim": int(head_hidden_dim),
        "dropout": float(dropout),
        "patience": int(patience),
        "freeze_encoder": bool(freeze_encoder),
        "best_val_rmse": float(best_val_rmse),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "history": history,
        "numeric_scaler": numeric_scaler.to_dict(),
    }
    save_json(metadata, output_dir / "training_metadata.json")

    bundle = PairModelBundle(
        model=model,
        tokenizer=tokenizer,
        numeric_scaler=numeric_scaler,
        conditioning_columns=list(conditioning_columns),
        max_length=int(max_length),
        eval_batch_size=int(eval_batch_size),
    )
    return bundle, metadata


def predict_pair_residuals(
    df: pd.DataFrame,
    model_bundle: PairModelBundle,
) -> np.ndarray:
    dataset = PairResidualDataset(
        df=df,
        tokenizer=model_bundle.tokenizer,
        max_length=model_bundle.max_length,
        numeric_scaler=model_bundle.numeric_scaler,
        conditioning_columns=model_bundle.conditioning_columns,
        target_column="__missing_target__",
    )
    loader = DataLoader(
        dataset,
        batch_size=model_bundle.eval_batch_size,
        shuffle=False,
        collate_fn=PairResidualCollator(model_bundle.tokenizer),
    )
    device = infer_torch_device()
    model_bundle.model.to(device)
    model_bundle.model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model_bundle.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids").to(device) if batch.get("token_type_ids") is not None else None,
                numeric_features=batch["numeric_features"].to(device),
            )
            predictions.append(outputs["predictions"].detach().cpu().numpy())
    if not predictions:
        return np.array([], dtype=float)
    return np.concatenate(predictions)
