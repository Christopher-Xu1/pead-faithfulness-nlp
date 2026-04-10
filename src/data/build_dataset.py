from __future__ import annotations

import argparse

import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_dataset(qa_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    qa_df = qa_df.copy()
    labels_df = labels_df.copy()
    qa_df["event_date"] = pd.to_datetime(qa_df["event_date"]).dt.strftime("%Y-%m-%d")
    labels_df["event_date"] = pd.to_datetime(labels_df["event_date"]).dt.strftime("%Y-%m-%d")

    merged = qa_df.merge(
        labels_df[["call_id", "ticker", "event_date", "car_horizon", "label"]],
        on=["call_id", "ticker", "event_date"],
        how="inner",
    )
    merged = merged.dropna(subset=["text", "label"]).reset_index(drop=True)
    merged["label"] = merged["label"].astype(int)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-input", default="data/interim/qa_only/qa_dataset.csv")
    parser.add_argument("--labels-input", default="data/interim/labels/pead_labels.csv")
    parser.add_argument("--output", default="data/processed/dataset.csv")
    args = parser.parse_args()

    qa_df = pd.read_csv(args.qa_input)
    labels_df = pd.read_csv(args.labels_input)
    ds = build_dataset(qa_df, labels_df)

    ensure_dir("data/processed")
    ds.to_csv(args.output, index=False)
    LOGGER.info("Saved processed dataset to %s", args.output)


if __name__ == "__main__":
    main()
