from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging_utils import get_logger
from src.utils.metrics import classification_metrics, softmax

LOGGER = get_logger(__name__)


def _to_hf_dataset(df: pd.DataFrame):
    from datasets import Dataset

    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def evaluate_from_config(config_path: str) -> dict[str, float]:
    exp_cfg = load_yaml(config_path)
    model_cfg = load_yaml(exp_cfg["model_config"])

    model_dir = exp_cfg.get("output_dir", "outputs/models/default_run")
    test_path = exp_cfg.get("test_path", "data/processed/test.csv")
    run_name = exp_cfg.get("run_name", "default_run")

    test_df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    test_ds = _to_hf_dataset(test_df)
    max_length = int(model_cfg.get("max_length", 512))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    test_ds = test_ds.map(tok, batched=True)
    test_ds = test_ds.remove_columns(["text"])

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    pred_out = trainer.predict(test_ds)
    probs = softmax(pred_out.predictions)[:, 1]
    labels = pred_out.label_ids

    metrics = classification_metrics(labels, probs)

    metrics_dir = Path("outputs/metrics") / run_name
    ensure_dir(metrics_dir)

    pred_df = test_df.copy()
    pred_df["prob_1"] = probs
    pred_df["pred_label"] = (probs >= 0.5).astype(int)
    pred_df.to_csv(metrics_dir / "test_predictions.csv", index=False)

    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="quantile")
    calibration = {
        "fraction_positives": [float(x) for x in frac_pos],
        "mean_predicted_value": [float(x) for x in mean_pred],
    }

    save_json(metrics, metrics_dir / "test_metrics.json")
    save_json(calibration, metrics_dir / "calibration.json")

    LOGGER.info("Saved test predictions and metrics to %s", metrics_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/baseline_pead.yaml")
    args = parser.parse_args()
    _ = evaluate_from_config(args.config)


if __name__ == "__main__":
    main()
