from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.build_dataset import build_dataset
from src.data.compute_pead import _compute_event_labels
from src.data.split_dataset import time_based_split
from src.utils.io import ensure_dir, load_yaml, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_variant_dataset(
    qa_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    label_config_path: str | Path,
    min_abs_car: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    label_cfg = load_yaml(label_config_path)
    labels_df = _compute_event_labels(
        metadata=metadata_df[["call_id", "ticker", "event_date"]],
        prices=prices_df,
        market=market_df,
        horizon=int(label_cfg.get("pead_horizon", 20)),
        event_lag_days=int(label_cfg.get("event_lag_days", 1)),
        label_threshold=float(label_cfg.get("label_threshold", 0.0)),
    )

    dataset_df = build_dataset(qa_df, labels_df)
    dataset_df["car_horizon"] = dataset_df["car_horizon"].astype(float)
    if min_abs_car > 0:
        dataset_df = dataset_df[dataset_df["car_horizon"].abs() >= float(min_abs_car)].copy()
    dataset_df = dataset_df.sort_values("event_date").reset_index(drop=True)

    train_df, val_df, test_df = time_based_split(dataset_df)
    summary = {
        "label_config": str(label_config_path),
        "min_abs_car": float(min_abs_car),
        "dataset_rows": int(len(dataset_df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rate": float(train_df["label"].mean()) if len(train_df) else 0.0,
        "val_positive_rate": float(val_df["label"].mean()) if len(val_df) else 0.0,
        "test_positive_rate": float(test_df["label"].mean()) if len(test_df) else 0.0,
    }
    return dataset_df, train_df, val_df, test_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-input", default="data/interim/qa_only/qa_dataset.csv")
    parser.add_argument("--metadata-input", default="data/raw/metadata/call_metadata.csv")
    parser.add_argument("--prices-input", default="data/raw/prices/daily_returns.csv")
    parser.add_argument("--market-input", default="data/external/market_index/sp500_returns.csv")
    parser.add_argument("--label-config", default="configs/data/pead_20d.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-abs-car", type=float, default=0.0)
    args = parser.parse_args()

    qa_df = pd.read_csv(args.qa_input)
    metadata_df = pd.read_csv(args.metadata_input)
    prices_df = pd.read_csv(args.prices_input)
    market_df = pd.read_csv(args.market_input)

    dataset_df, train_df, val_df, test_df, summary = build_variant_dataset(
        qa_df=qa_df,
        metadata_df=metadata_df,
        prices_df=prices_df,
        market_df=market_df,
        label_config_path=args.label_config,
        min_abs_car=args.min_abs_car,
    )

    out_dir = ensure_dir(args.output_dir)
    write_csv(dataset_df, out_dir / "dataset.csv")
    write_csv(train_df, out_dir / "train.csv")
    write_csv(val_df, out_dir / "val.csv")
    write_csv(test_df, out_dir / "test.csv")
    save_json(summary, out_dir / "summary.json")
    LOGGER.info("Saved variant dataset to %s", out_dir)


if __name__ == "__main__":
    main()
