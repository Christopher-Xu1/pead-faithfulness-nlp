from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging_utils import get_logger
from src.utils.metrics import classification_metrics_from_logits
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _to_hf_dataset(df: pd.DataFrame, text_column: str = "text", label_column: str = "label"):
    from datasets import Dataset

    subset = df[[text_column, label_column]].rename(columns={text_column: "text", label_column: "label"})
    return Dataset.from_pandas(subset, preserve_index=False)


def _tokenize_batch(tokenizer, texts: list[str], max_length: int, text_packing: str = "raw") -> dict[str, list[list[int]]]:
    if text_packing == "raw":
        return tokenizer(texts, truncation=True, max_length=max_length)

    if text_packing != "head_tail":
        raise ValueError(f"Unsupported text_packing={text_packing!r}")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    available = max(1, max_length - special_tokens)
    packed: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": []}
    for text in texts:
        token_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        if len(token_ids) > available:
            head = available // 2
            tail = available - head
            token_ids = token_ids[:head] + token_ids[-tail:]
        encoded = tokenizer.prepare_for_model(token_ids, truncation=False, padding=False)
        for key, value in encoded.items():
            packed.setdefault(key, []).append(value)
    return packed


def train_from_config(config_path: str) -> None:
    exp_cfg = load_yaml(config_path)
    model_cfg = load_yaml(exp_cfg["model_config"])
    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    train_df = pd.read_csv(exp_cfg["train_path"])
    val_df = pd.read_csv(exp_cfg["val_path"])
    text_column = exp_cfg.get("text_column", "text")
    text_packing = exp_cfg.get("text_packing", "raw")

    model_name = model_cfg.get("model_name", "roberta-base")
    max_length = int(model_cfg.get("max_length", 512))

    from src.models.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    train_ds = _to_hf_dataset(train_df, text_column=text_column)
    val_ds = _to_hf_dataset(val_df, text_column=text_column)

    def tok(batch):
        return _tokenize_batch(
            tokenizer=tokenizer,
            texts=batch["text"],
            max_length=max_length,
            text_packing=text_packing,
        )

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    output_dir = exp_cfg.get("output_dir", "outputs/models/default_run")
    ensure_dir(output_dir)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(model_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(model_cfg.get("weight_decay", 0.01)),
        per_device_train_batch_size=int(model_cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(model_cfg.get("eval_batch_size", 16)),
        num_train_epochs=float(model_cfg.get("num_train_epochs", 3)),
        warmup_ratio=float(model_cfg.get("warmup_ratio", 0.1)),
        gradient_accumulation_steps=int(model_cfg.get("gradient_accumulation_steps", 1)),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{model_cfg.get('metric_for_best_model', 'auroc')}",
        greater_is_better=True,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=classification_metrics_from_logits,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_metrics = trainer.evaluate()
    save_json(eval_metrics, Path(output_dir) / "val_metrics.json")

    metadata = {
        "seed": seed,
        "config_path": config_path,
        "model_name": model_name,
        "max_length": max_length,
        "text_column": text_column,
        "text_packing": text_packing,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "run_name": exp_cfg.get("run_name", "default_run"),
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
    }
    save_json(metadata, Path(output_dir) / "run_metadata.json")
    LOGGER.info("Saved model and metadata to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/baseline_pead.yaml")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
