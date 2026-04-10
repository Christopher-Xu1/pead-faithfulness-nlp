from __future__ import annotations

import argparse

import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["event_date"] = pd.to_datetime(data["event_date"])
    data = data.sort_values("event_date").reset_index(drop=True)

    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/dataset.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _ = args.seed
    df = pd.read_csv(args.input)
    train, val, test = time_based_split(df)

    ensure_dir(args.output_dir)
    train.to_csv(f"{args.output_dir}/train.csv", index=False)
    val.to_csv(f"{args.output_dir}/val.csv", index=False)
    test.to_csv(f"{args.output_dir}/test.csv", index=False)

    LOGGER.info("Saved splits to %s", args.output_dir)
    LOGGER.info("train=%d val=%d test=%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
