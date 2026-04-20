from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import load_json, load_yaml, write_csv


def collect_one(config_path: str | Path) -> dict[str, object]:
    exp_cfg = load_yaml(config_path)
    model_cfg = load_yaml(exp_cfg["model_config"])

    run_name = exp_cfg.get("run_name", Path(config_path).stem)
    model_dir = Path(exp_cfg.get("output_dir", "outputs/models/default_run"))
    metrics_dir = Path("outputs/metrics") / run_name
    val_metrics = load_json(model_dir / "val_metrics.json")
    test_metrics = load_json(metrics_dir / "test_metrics.json")

    row: dict[str, object] = {
        "config_path": str(config_path),
        "run_name": run_name,
        "model_name": model_cfg.get("model_name"),
        "max_length": model_cfg.get("max_length"),
        "seed": exp_cfg.get("seed", 42),
        "text_column": exp_cfg.get("text_column", "text"),
        "text_packing": exp_cfg.get("text_packing", "raw"),
        "label_config": exp_cfg.get("label_config"),
        "train_path": exp_cfg["train_path"],
        "val_path": exp_cfg["val_path"],
        "test_path": exp_cfg["test_path"],
        "val_auroc": val_metrics.get("eval_auroc"),
        "val_auprc": val_metrics.get("eval_auprc"),
        "val_accuracy": val_metrics.get("eval_accuracy"),
        "test_auroc": test_metrics.get("auroc"),
        "test_auprc": test_metrics.get("auprc"),
        "test_accuracy": test_metrics.get("accuracy"),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="reports/experiment_metrics.csv")
    args = parser.parse_args()

    rows = [collect_one(config_path) for config_path in args.configs]
    df = pd.DataFrame(rows).sort_values(["run_name"]).reset_index(drop=True)
    write_csv(df, args.output)


if __name__ == "__main__":
    main()
